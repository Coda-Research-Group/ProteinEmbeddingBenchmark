import argparse
import os
import subprocess
import time
from pathlib import Path

import pandas as pd
import torch
from rdkit import Chem
from torchdrug import data, layers, models, transforms
from torchdrug.data import DataLoader
from torchdrug.layers import geometry

# define model
WEIGHTS_PATH = "/scripts/gearnet/mc_gearnet_edge.pth"


class AlphaFoldDB(data.ProteinDataset):
    def __init__(self, path, transform):
        self.path = path
        self.files = [f"{self.path}/{f}" for f in os.listdir(self.path)]
        self.files = [file for file in self.files if file.endswith(".cif")]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        subprocess.run(
            [
                "python",
                "/scripts/gearnet/mmcif_to_pdb.py",
                "--ciffile",
                self.files[index],
            ]
        )
        mol = Chem.MolFromPDBFile(self.files[index] + ".pdb", sanitize=False)
        if mol is None:
            raise ValueError("RDKit cannot read PDB file `%s`" % self.files[index])
        protein = data.Protein.from_molecule(
            mol, "position", "length", "symbol", None, False
        )
        if hasattr(protein, "residue_feature"):
            with protein.residue():
                protein.residue_feature = protein.residue_feature.to_dense()
        item = {"graph": protein, "protein": self.files[index].split("-")[1]}
        if self.transform:
            item = self.transform(item)
        return item


def _untar(tar_file, input_dir, scratch_dir):
    """Helper function to paralelize untaring"""
    tar_file_path = str(os.path.join(input_dir, tar_file))
    subprocess.run(
        ["tar", "-xf", tar_file_path, "-C", scratch_dir, "--wildcards", "*.cif.gz"]
    )


def extract(input_dir, scratch_dir, index):
    """Extracts CHUNK of proteins on persistent storage from tars and moves them to the zip folder"""
    scrach_dir_loc = str(os.path.join(scratch_dir))
    if not os.path.exists(scrach_dir_loc):
        os.mkdir(scrach_dir_loc)

    # first untar and move to zip folder
    with open(index, "r") as index_f:
        print(f"Taking from index file {index}")
        proteins = index_f.readlines()
        for tar_protein_file in proteins:
            tar_file = tar_protein_file.split(",")[0].strip()
            _untar(tar_file, input_dir, scrach_dir_loc)

    # then unzip proteins itself
    subprocess.run(["gzip", "-dfr", str(scrach_dir_loc)])

    return scrach_dir_loc


def run(input_path, output_path, batch_size, dev):
    """Run GeaNET embeddings on a directory of .cif files
    Args:
        input_path (str): Path to directory containing cif files
        output_path (str): Path to save embeddings as .pkl
        batch_size (int): Batch size for processing
        dev (str): cpu or cuda

    Returns:
        None, saves embeddings as .pkl
    """
    transform = transforms.ProteinView(view="residue")

    # instantiate model
    graph_construction_model = layers.GraphConstruction(
        node_layers=[geometry.AlphaCarbonNode()],
        edge_layers=[
            geometry.SpatialEdge(radius=10.0, min_distance=5),
            geometry.KNNEdge(k=10, min_distance=5),
            geometry.SequentialEdge(max_distance=2),
        ],
        edge_feature="gearnet",
    )
    gearnet_edge = models.GearNet(
        input_dim=21,
        hidden_dims=[512, 512, 512, 512, 512, 512],
        num_relation=7,
        edge_input_dim=59,
        num_angle_bin=8,
        batch_norm=True,
        concat_hidden=True,
        short_cut=True,
        readout="sum",
    )

    device = torch.device(dev)
    net = torch.load(WEIGHTS_PATH, map_location=device)

    gearnet_edge.load_state_dict(net)
    gearnet_edge.eval().to(device)

    # instantiate the dataset
    dataset = AlphaFoldDB(input_path, transform)

    # instantiate the dataloder
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SequentialSampler(dataset),
    )

    # run
    embeddings = {}
    for sample in iter(loader):
        start_time = time.time()
        proteins = graph_construction_model(sample["graph"]).to(device)
        output = gearnet_edge(
            proteins, proteins.node_feature.float(), all_loss=None, metric=None
        )
        batch_embeddings = output["graph_feature"].detach().cpu().numpy()
        for i, protein in enumerate(sample["protein"]):
            embeddings[protein] = batch_embeddings[i]
        print(f"Time to process batch is {time.time() - start_time} s")

    df = pd.DataFrame(list(embeddings.items()), columns=["protein", "embedding"])
    df.to_pickle(Path(output_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--scratch_dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--position", type=int, required=False, default=0)
    parser.add_argument("--batch", type=int, required=False, default=4)
    parser.add_argument("--device", type=str, required=False, default="cpu")
    parser.add_argument("--cache", type=bool, required=False, default=False)
    parser.add_argument("--index", type=str, required=False, default=False)

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    scratch_dir = Path(args.scratch_dir)
    assert input_path.exists()

    if not output_path.exists():
        os.mkdir(output_path)

    extracted_data = args.scratch_dir
    if not args.cache:
        extracted_data = extract(input_path, scratch_dir, args.index)

    run(
        extracted_data,
        os.path.join(output_path, f"cif2gearnet-result-{args.position}.pkl"),
        batch_size=args.batch,
        dev=args.device,
    )
