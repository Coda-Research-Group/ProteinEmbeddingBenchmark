import argparse
import os
import subprocess
from pathlib import Path

import pandas as pd
from encode import get_raw_feature_tensor, load_model


def _untar(tar_file, input_dir, scratch_dir):
    """Helper function to paralelize untaring"""
    tar_file_path = str(os.path.join(input_dir, tar_file))

    subprocess.run(
        ["tar", "-xf", tar_file_path, "-C", scratch_dir, "--wildcards", "*.cif.gz"]
    )
    # subprocess.run(
    #    ["tar", "-xf", tar_file_path, "-C", scratch_dir, "--wildcards", "*.cif.gz"]
    # )
    # !f'tar -xf {tar_file_path} -C {scratch_dir}'


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


def chunks(lst, n) -> list:
    """Yield successive n-sized chunks from lst.

    Args:
        lst (list): List to be divided
        n (int): size of list to be yielded

    Yields:
        list: list of size n"""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def run(input_path, output_path, batch_size, fold_number=0):
    """Run GraSR embedding on a directory of cif files

    Args:
        input_path (str): Path to directory containing cif files
        output_path (str): Path to save embeddings as .pkl
        fold_number (int, optional): Fold number of the model. Accepts values 0-4.
        batch_size (int, optional): Batch size

    Returns:
        None, saves embeddings as .pkl
    """
    model = load_model(f"./grasr/saved_models/grasr_fold{fold_number}.pkl")
    model.eval()
    embeddings = {}

    proteins = os.listdir(input_path)
    proteins = [file for file in proteins if file.endswith(".cif")]

    for protein_batch in chunks(proteins, batch_size):
        x, ld, am = get_raw_feature_tensor(
            [f"{input_path}/{protein}" for protein in protein_batch]
        )
        embeddings_result = model((x, x, ld, ld, am, am), True).detach().numpy()

        for key, value in zip(protein_batch, embeddings_result):
            embeddings[key.split("-")[1]] = value

    df = pd.DataFrame(list(embeddings.items()), columns=["protein", "embedding"])
    df.to_pickle(Path(output_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--scratch_dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=False, default=64)
    parser.add_argument("--position", type=int, required=False, default=0)
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
        os.path.join(output_path, f"cif2grasr-result-{args.position}"),
        args.batch_size,
    )
