import argparse
import os
import subprocess
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd


def extract(input_dir, scratch_dir, index, position):
    """Extract one tarball of ESM proteins
    
    Args:
        input_dir (str): Path to directory containing tarballs
        scratch_dir (str): Path to scratch directory
        index (str): Path to index file
        position (int): Position in index file
    """
    scrach_dir_loc = str(os.path.join(scratch_dir))

    # select which tarball to extract
    with open(index, "r") as index_f:
        print(f"Taking from index file {index}")
        file = index_f.readlines()[position].strip()

    # extract to scratch directory
    tar_file_path = str(os.path.join(input_dir, file))
    subprocess.run(["tar", "--no-overwrite-dir", "-xvmf", tar_file_path, "-C", scratch_dir])

    return scrach_dir_loc


def plytoobj(filename):
    obj_filename = filename[:-4] + ".obj"
    obj_file = open(obj_filename, "w")

    with open(filename) as ply_file:
        ply_file_content = ply_file.read().split("\n")[:-1]

        for content in ply_file_content:
            content_info = content.split()
            if len(content_info) == 6:
                vertex_info = "v " + " ".join(content_info[0:3])
                obj_file.write(vertex_info + "\n")
            elif len(content_info) == 7:
                vertex1, vertex2, vertex3 = map(int, content_info[1:4])
                vertex1, vertex2, vertex3 = vertex1 + 1, vertex2 + 1, vertex3 + 1
                face_info = (
                    "f " + str(vertex1) + " " + str(vertex2) + " " + str(vertex3)
                )
                obj_file.write(face_info + "\n")

        obj_file.close()


def process_protein(path, protein_file):
    """Process a single protein file

    Args:
        path (str): Path to protein file
        protein_file (str): Name of protein file
    """
    start_time = time.time()
    os.mkdir(os.path.join(path, "tmp"))
    
    # convert to ply -> obj -> grid -> zernike
    subprocess.run(
        [
            "/scripts/3d-af-surfer/bin/EDTSurf",
            "-i",
            f"{path}{protein_file}",
            "-h",
            "2",
            "-f",
            "1",
            "-o",
            f"{path}tmp/{protein_file}",
        ]
    )
    # the C script can potentially fail, check if the output file exists
    if not os.path.exists(f"{path}tmp/{protein_file}.ply"):
        subprocess.run(["rm", "-r", f"{path}tmp/"])
        return None

    plytoobj(f"{path}tmp/{protein_file}.ply")
    subprocess.run(
        ["/scripts/3d-af-surfer/bin/obj2grid", "-g", "64", f"{path}tmp/{protein_file}.obj"]
    )
    subprocess.run(
        [
            "/scripts/3d-af-surfer/bin/map2zernike",
            f"{path}tmp/{protein_file}.obj.grid",
            "-c",
            "0.5",
        ]
    )

    # convert to vector
    with open(f"{path}tmp/{protein_file}.obj.grid.inv") as f:
        embedding = [float(x.strip()) for x in f.readlines()]

    # clean up
    subprocess.run(["rm", "-r", f"{path}tmp/"])

    print(f"Processed {protein_file} in {time.time() - start_time} seconds")
    return embedding[1:]


def process_esm_minibatch(input_path, folder, output_path):
    """Process a batch of proteins in a folder
    
    Args:
        input_path (str): Path to protein folder
        folder (str): Name of protein folder
    """
    index = []
    data = []

    # if proteins end with gz, unzip them
    proteins = os.listdir(f"{input_path}/{folder}")
    if proteins[0].endswith(".gz"):
        print("Unzipping files...")
        for file in proteins:
            subprocess.run(["gunzip", f"{input_path}/{folder}/{file}"])

    proteins = [file for file in os.listdir(f"{input_path}/{folder}") if file.endswith(".pdb")]
    print("Starting processing", len(proteins), "files...")

    for protein in proteins:
        result = process_protein(f"{input_path}/{folder}/", protein)

        # if EDTSurf failed, store protein name
        if not result:
            with open(f"{output_path.split('esm2surfer')[0]}failed_proteins.txt", "a") as f:
                f.write(protein + " folder: " + folder + " path: " + output_path + "\n")
        else:  
            index.append(protein.split(".")[0])
            data.append(result)
        
    return index, data

def run(input_path, output_path, processes=8):
    """Run Invariant-3d-coordinates embedding on a nested directory of pdb files

    Args:
        input_path (str): Path to directory containing pdb files
        output_path (str): Path to save embeddings as parquet

    Returns:
        None, saves embeddings as parquet
    """
    index = []
    data = []

    protein_folders = os.listdir(input_path)

    # process each protein folder in parallel
    with Pool(processes) as p:
        results = p.starmap(process_esm_minibatch, [(input_path, folder, output_path) for folder in protein_folders])
    
    for result in results:
        index.extend(result[0])
        data.extend(result[1])
        
    df = pd.DataFrame(index=index, data=data, dtype="float32")
    df.to_parquet(output_path, compression="gzip")


if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--scratch_dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--position", type=int, required=False, default=0)
    parser.add_argument("--processes", type=int, required=False, default=32)
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
        extracted_data = extract(input_path, scratch_dir, args.index, args.position)

    run(
        extracted_data,
        os.path.join(output_path, f"esm2surfer-{args.position}.parquet"),
        args.processes,
    )
    print("Finished in", time.time() - start_time, "seconds")
