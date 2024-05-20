import argparse
import os
import subprocess
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
from tqdm import tqdm

pd.options.mode.chained_assignment = None

DST_TRESHHOLD = 20.0


def _untar(tar_file, input_dir, scratch_dir):
    """Helper function to paralelize untaring"""
    tar_file_path = str(os.path.join(input_dir, tar_file))
    subprocess.run(
        ["tar", "-xf", tar_file_path, "-C", scratch_dir, "--wildcards", "*.cif.gz"]
    )
    # !f'tar -xf {tar_file_path} -C {scratch_dir}'


def extract(input_dir, scratch_dir, index):
    """Extracts CHUNK of proteins on persistent storage from tars and moves them to the zip folder"""
    print(
        f'{datetime.now().strftime("%m/%d/%Y, %H:%M:%S")} - Started extrating data tar/gz'
    )
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
    print(f'{datetime.now().strftime("%m/%d/%Y, %H:%M:%S")} - Unzip started.')
    subprocess.run(["gzip", "-dfr", str(scrach_dir_loc)])
    print(f'{datetime.now().strftime("%m/%d/%Y, %H:%M:%S")} - Unzip ended.')

    return scrach_dir_loc


def run(pdb_path, output_path, granularity=10):
    """Calcuklate all protein descriptors

    Args:
        pdb_path (str): path to PDB
        output_path (str): output file
        granularity (int): granularity of the descriptors
    """
    print(f'{datetime.now().strftime("%m/%d/%Y, %H:%M:%S")} - Started')
    proteins = os.listdir(pdb_path)
    proteins = [file for file in proteins if file.endswith(".cif")]
    print(f'{datetime.now().strftime("%m/%d/%Y, %H:%M:%S")} - Loaded {len(proteins)}')

    results = []
    data = []
    index = []
    print(f"Total proteins to process {len(proteins)}")
    for i, protein in enumerate(proteins):
        if i % 100 == 0:
            print(
                f'{datetime.now().strftime("%m/%d/%Y, %H:%M:%S")} - Processing "{i}-th" protein ({protein})'
            )
        # result = pool.apply_async(process_protein, (pdb_path + protein, granularity))
        protein_location = os.path.join(pdb_path, protein)
        ix, dt = process_protein(protein_location, granularity)
        # index[i], data[i] = process_protein(protein_location, granularity)
        # print(index[i])
        index.extend(ix)
        data.extend(dt)
        # print(data[i])

    print(f'{datetime.now().strftime("%m/%d/%Y, %H:%M:%S")} - Processing started')
    t = time()
    # data = [n for sublist in [result.get('data') for result in results] for n in sublist]
    # index = [n for sublist in [result.get('index') for result in results] for n in sublist]
    df = pd.DataFrame(index=index, data=data)
    df.to_pickle(Path(output_path))
    t = time() - t
    print(f"Processing took {t:.1f} seconds")


def process_protein(protein, granularity):
    """Create protein descriptor from file

    Args:
        protein (str): path to protein file
        granularity (int): descriptor granularity
        fstart (_type_): filename protein id start index
        fend (_type_): filename protein id end index

    Returns:
        dict: protein chain id and the descriptor
    """
    protein_chains = read_and_extract(protein, granularity)

    data_list = []
    index_list = []
    for chain, df in protein_chains:
        desc = create_descriptor(df, granularity)
        data_list.append(desc)
        index_list.append(f"{protein.split('/')[-1].split('-')[1].upper()}")

    return index_list, data_list


def create_descriptor(chain_df, granularity):
    """Create protein descriptor from extracted data

    Args:
        chain_df (DataFrame): extracted protein data
        granularity (int): granularity of the descriptor
    """

    def compute_matrix(row):
        dist = np.linalg.norm(
            np.array([row["x_x"], row["y_x"], row["z_x"]])
            - np.array([row["x_y"], row["y_y"], row["z_y"]])
        )
        return (DST_TRESHHOLD - dist) / DST_TRESHHOLD if dist <= DST_TRESHHOLD else 0.0

    chain_df["key"] = 0
    chain_df = chain_df.sort_values("normalized_rs")
    chain_df = pd.merge(chain_df, chain_df, on="key", how="left")
    chain_df["dist"] = chain_df.apply(lambda row: compute_matrix(row), axis=1)

    chain_df = chain_df.pivot(
        index="normalized_rs_x", columns="normalized_rs_y", values="dist"
    )
    nparray = chain_df.to_numpy(dtype="float16")
    shape = nparray.shape[0]
    nparray = np.pad(nparray, (0, granularity - shape), "constant")
    nup = nparray[np.triu_indices(nparray.shape[0], k=1)]
    return nup


def read_and_extract(protein_file, granularity):
    """Extract protein descriptor data from PDB gz file

    Args:
        protein_file (str): path to protein file
        granularity (int): descriptor granularity
    """

    def remap(n, min_, max_):
        if max_ - min_ >= granularity:
            return int((n - min_) / (max_ - min_) * (granularity - 1)) + 1
        return n - min_ + 1

    df = pd.DataFrame(
        columns=["atom", "residue", "chain", "residue_sequence", "x", "y", "z"]
    )

    atoms = []
    residues = []
    chains = []
    residue_sequences = []
    xs = []
    ys = []
    zs = []

    # print(f'Opening file {protein_file} at {time()}.')
    with open(protein_file, "rt") as file:
        model = True
        for line in file:
            words = line.split()
            if len(words) == 0:
                continue
            if model and line[0:4] == "ATOM":
                atoms.append(words[3])
                residues.append(words[5])
                chains.append(words[6])
                if words[6] != "A":
                    print("Chain is not A")
                residue_sequences.append(words[8])
                xs.append(words[10])
                ys.append(words[11])
                zs.append(words[12])
    # print(f'File {protein_file} read at {time()}.')

    if len(residue_sequences) == 0:
        return []

    coded_residue_sequences = []
    index = 1
    last = residue_sequences[0]
    for rs in residue_sequences:
        if rs == last:
            coded_residue_sequences.append(index)
        else:
            index += 1
            coded_residue_sequences.append(index)
            last = rs

    df = pd.DataFrame(
        {
            "atom": atoms,
            "residue": residues,
            "chain": chains,
            "residue_sequence": coded_residue_sequences,
            "x": xs,
            "y": ys,
            "z": zs,
        }
    )
    df = df.astype({"residue_sequence": int, "x": float, "y": float, "z": float})
    chains = df["chain"].unique()
    tables = []
    for chain in chains:
        table = df[df["chain"] == chain]
        min_ = np.min(table["residue_sequence"])
        max_ = np.max(table["residue_sequence"])
        table.loc[:, "normalized_rs"] = table.loc[:, "residue_sequence"].apply(
            lambda x: remap(x, min_, max_)
        )
        table = table.drop(["residue_sequence"], axis=1)
        table = table.groupby(["chain", "normalized_rs"])
        table = table[["x", "y", "z"]].mean().reset_index()
        table = table.sort_values(["chain", "normalized_rs"])
        tables.append((chain, table))

    # print(f'File {protein_file} processed at {time()}.')

    return tables


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--scratch_dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--granularity", type=int, required=False, default=10)
    # parser.add_argument("--chunk", type=int, required=False, default=10)
    parser.add_argument("--position", type=int, required=False, default=0)
    parser.add_argument("--cache", type=bool, required=False, default=False)
    parser.add_argument("--index", type=str, required=False, default=False)

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    scratch_dir = Path(args.scratch_dir)
    assert input_path.exists()

    # start_index = args.position*args.chunk
    # end_index = start_index+args.chunk-1

    extracted_data = args.scratch_dir
    if not args.cache:
        extracted_data = extract(input_path, scratch_dir, args.index)

    run(
        extracted_data,
        os.path.join(output_path, f"cif2emb-result-{args.position}"),
        args.granularity,
    )
