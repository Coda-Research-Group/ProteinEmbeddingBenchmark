import argparse
from copy import copy
import os
import subprocess
from time import sleep, time
import requests
import pandas as pd
from random import sample
from scipy.spatial.distance import euclidean, cosine
from sklearn.isotonic import spearmanr
from multiprocessing import Pool
import numpy as np

from tqdm import tqdm
import json


PROTEIN_INDEX = pd.read_csv("/embeddings/proteins-index.csv", sep="\t", header=None)
PROTEIN_PATH = "/proteins"
FOLDSEEK_DIR = "foldseek-files/"
SCRATCH_DIR = "tmp/"
WORK_PATH = "data/"
US_ALIGN = "USalign/USalign"
JOB_CIF2EMB_PATH = "/embeddings/configs/"
print("Loaded protein csv index and configs")


def get_protein_tm_score(path, protein_x, output):
    """Compute TM-score for one protein with all proteins in the folder.

    Args:
        path (str): path to folder with cif or pdb proteins
        protein_x (str): name of protein to compute distances from
        output (str): path to output json file

    Returns:
        dict: sorted dictionary of TM-scores from protein_x to all other proteins in the folder.
            Dictionary is also saved as json during processing.
    """
    t = time()
    tm_score_dict = {}
    proteins = os.listdir(path)
    proteins = [file for file in proteins if file.endswith(".cif")]

    for protein_y in proteins:
        result = subprocess.run(
            [US_ALIGN, os.path.join(path, protein_x), os.path.join(path, protein_y)], capture_output=True
        )
        result = result.stdout.decode("utf-8")
        tm_score = float(result.split("TM-score=")[1].split()[0])
        tm_score_dict[protein_y.split("-")[1]] = tm_score

    # save distances sorted from most to least similar
    with open(output, "w") as f:
        sorted_dict = {
            k: 1 - v
            for k, v in sorted(
                tm_score_dict.items(), key=lambda item: item[1], reverse=True
            )
        }
        json.dump(sorted_dict, f)

    t = time() - t
    print(f"Ground truth for {protein_x} was computed in {t}")

    return sorted_dict


def get_embedding_distance(df, protein_name, distance_type, output) -> dict:
    """
    Compute distance between one protein and all proteins in the folder.

    Args:
        file (str): path to pickle file
        protein_name (str): name of protein to compute distances from
        distance_type (function): any distance function from scipy.spatial.distance - e.g. `cosine`,
            `euclidean`. List of them can be found at
            https://docs.scipy.org/doc/scipy/reference/spatial.distance.html .

    Returns:
        dict: sorted dictionary of distances from protein_name to all other proteins in the file.
            Dictionary is also saved as json during processing.
    """
    t = time()
    protein_x = df[df["protein"] == protein_name].iloc[0]

    # iterate through data frame and compute distances
    distance_dict = {}
    for _, protein_y in df.iterrows():
        dist = distance_type(protein_x["embedding"], protein_y["embedding"])
        distance_dict[protein_y["protein"]] = float(dist)

    # normalize distances and convert to similarity
    max_dist = max(distance_dict.values())
    min_dist = min(distance_dict.values())
    if max_dist == min_dist:
        max_dist = 1
        print("Max and min distance are the same, setting max to 1")
    similarity_dict = {
        k: (v - min_dist) / (max_dist - min_dist) for k, v in distance_dict.items()
    }

    # save distances sorted from most to least similar
    with open(output, "w") as f:
        sorted_dict = {
            k: v for k, v in sorted(similarity_dict.items(), key=lambda item: item[1])
        }
        json.dump(sorted_dict, f)

    t = time() - t
    print(
        f"Distance of type {distance_type.__name__} for {protein_name} was computed in {t}s"
    )

    return sorted_dict


def map_job_to_tar(protein_dict):
    """
    Map jobs to tar files present in protein dict

    Args:
    protein_dict (dict): a dictionary with structure tar_file: [protein1, protein2, ...]
    job_path (str): path to the folder with jobs

    Returns:
    dict: a dictionary with structure job: tar_file
    """
    print("Mapping jobs to tar files present in protein dict")
    job_to_tar = {}
    files = os.listdir(JOB_CIF2EMB_PATH)
    files = [file for file in files if file.startswith("job-")]
    for file in tqdm(files, desc=f"Mapping {len(files)} jobs to tar files"):
        with open(os.path.join(JOB_CIF2EMB_PATH, file), "r") as f:
            content = f.read()
            for tar in protein_dict.keys():
                if tar in content:
                    job_to_tar[f.name.split("-")[-1]] = tar
                    break
            if len(job_to_tar) == len(protein_dict):
                break
    return job_to_tar


def construct_embedding_dataframe(
    protein_dict, job_to_tar, embedding_path, embedding_name, suffix="", lmi=False
):
    """
    Get a dataframe with embeddings for all proteins in the protein_dict

    Args:
    protein_dict (dict): a dictionary with structure tar_file: [protein1, protein2, ...]
    embedding_path (str): path to the folder with embeddings

    Returns:
    pd.DataFrame: a dataframe with embeddings
    """
    embedding_df = pd.DataFrame(columns=["protein", "embedding"])
    all_proteins = sum(list(protein_dict.values()), [])
    for appendix, tar in tqdm(
        job_to_tar.items(), desc=f"Retrieving {embedding_name}embeddings from files"
    ):
        # get all lines which are in protein_dict[tar]
        df = pd.read_pickle(
            os.path.join(embedding_path, embedding_name + appendix + suffix)
        )
        if lmi:
            df = pd.DataFrame(
                [(protein, list(value)) for protein, value in list(df.iterrows())],
                columns=["protein", "embedding"],
            )

        df = df[df["protein"].isin([protein.split("-")[1] for protein in all_proteins])]
        embedding_df = pd.concat([embedding_df, df])

    print(embedding_df.head())
    return embedding_df


def _find_protein_in_index(protein_name):
    """
    Find a protein in the protein index file

    Args:
    protein_name (str): the name of the protein to find

    Returns:
    str: the name of tar file containing protein
    """
    return (
        PROTEIN_INDEX[PROTEIN_INDEX[0].str.contains(protein_name)][0]
        .values[0]
        .split(",")[0]
    )


def get_specific_proteins(proteins):
    """
    Get specific proteins from the protein index file

    Args:
    proteins (list): a list of proteins to get

    Returns:
    dict: a dictionary with structure tar_file: [protein1, protein2, ... protein5]
    """
    print("Getting specific proteins")
    proteins_dict = {}
    for protein in proteins:
        tar = _find_protein_in_index(protein)
        if tar in proteins_dict:
            proteins_dict[tar].append(protein)
        else:
            proteins_dict[tar] = [protein]
    return proteins_dict


def get_random_proteins(n):
    """
    Get n random proteins from the protein index file

    Args:
    n (int): the number of proteins to get

    Returns:
    dict: a dictionary with structure tar_file: [protein1, protein2, ... protein5]
    """
    print("Getting random proteins, n =", n)
    proteins = {}
    amount = 0

    while amount < n:
        protein_sample = PROTEIN_INDEX.sample(5)
        protein_sample = protein_sample[0].values[0].split(",")
        # get only tar files with more than 5 samples
        if len(protein_sample[1:]) < 5:
            continue
        proteins[protein_sample[0]] = sample(protein_sample[1:], 5)
        amount += 5

    return proteins


def extend_sample_with_foldseek(protein_dict, foldseek_file, limit=500):
    """
    Extend the sample with similar proteins from FoldSeek

    Args:
    protein_dict (dict): a dictionary with structure tar_file: [protein1, protein2, ...]

    Returns:
    list: a list of proteins used for the LMI query, these can be used for evaluation
    dict: dictionary extended by similar proteins
    """
    # open file
    foldseek_proteins = [f"AF-{foldseek_file}-F1-model_v4"]
    with open(os.path.join(FOLDSEEK_DIR, foldseek_file), "r") as f:
        f = f.read().split()
        foldseek_proteins += [w for w in f if w.startswith("AF-")]
        if limit + 1 < len(foldseek_proteins):
            foldseek_proteins = foldseek_proteins[:limit + 1]
        foldseek_proteins = [w.split("-model")[0] for w in foldseek_proteins]

    print("Added", len(foldseek_proteins), "similar proteins from FoldSeek")

    # add the similar proteins to the protein_dict
    for protein in tqdm(foldseek_proteins, desc="Locating proteins in index"):
        tar = _find_protein_in_index(protein)
        if tar in protein_dict:
            protein_dict[tar].append(protein)
        else:
            protein_dict[tar] = [protein]

    print("Current sample is", len(protein_dict), "tar files")

    return foldseek_proteins[0], protein_dict


def extract_proteins(protein_dict, protein_foldername):
    """
    Prepare all needed proteins to working directory

    Args:
        protein_dict (dict): a dictionary with structure tar_file: [protein1, protein2, ...]
        protein_foldername (str): temporary foldername to store results
    """
    print("Extracting proteins to working directory")
    os.makedirs(os.path.join(SCRATCH_DIR, protein_foldername), exist_ok=True)
    os.makedirs(os.path.join(WORK_PATH, protein_foldername), exist_ok=True)

    tars = [os.path.join(PROTEIN_PATH, tar) for tar in protein_dict.keys()]
    print("Untaring", len(tars), "files")

    for protein_tar, proteins in tqdm(protein_dict.items(), desc="Extracting proteins"):
        # untar and unzip needed files
        subprocess.run(
            ["tar", "-xf", os.path.join(PROTEIN_PATH, protein_tar), "-C", os.path.join(SCRATCH_DIR, protein_foldername)]
        )
        proteins_to_unzip = [
            os.path.join(SCRATCH_DIR, protein_foldername, protein + "-model_v3.cif.gz")
            for protein in proteins
        ]
        subprocess.run(["gzip", "-dfr"] + proteins_to_unzip)
        # check if all files are retrieved
        if not all(
            [
                os.path.exists(os.path.join(SCRATCH_DIR, protein_foldername, protein + "-model_v3.cif"))
                for protein in proteins
            ]
        ):
            print("Error while unzipping")
            break
        # move the files to the working directory
        proteins_to_move = [
            os.path.join(SCRATCH_DIR, protein_foldername, protein + "-model_v3.cif") for protein in proteins
        ]
        subprocess.run(["mv"] + proteins_to_move + [os.path.join(WORK_PATH, protein_foldername)])
        subprocess.run(["rm", "-r", os.path.join(SCRATCH_DIR, protein_foldername)])
        os.makedirs(os.path.join(SCRATCH_DIR, protein_foldername), exist_ok=True)
    
    # finally remove temp folder
    subprocess.run(["rm", "-r", os.path.join(SCRATCH_DIR, protein_foldername)])


def run(random, foldseek_file):
    """
    Run the whole pipeline

    Args:
    random (int): the number of random proteins to get
    foldseek_file (str): the name of the file with similar proteins from FoldSeek
    """
    start_time = time()
    # prepare set of proteins and protein to be tested against
    protein_dict = get_random_proteins(random)

    protein, protein_dict = extend_sample_with_foldseek(
        protein_dict, foldseek_file, random
    )

    # extract proteins to working directory
    extract_proteins(protein_dict, protein)

    # compute ground truth and similarity for all proteins in the sample
    get_protein_tm_score(
        os.path.join(WORK_PATH, protein), protein + "-model_v3.cif", f"results/{protein}-distances.json"
    )

    # find which jobs contain embeddings and create dataframes with them
    job_to_tar = map_job_to_tar(protein_dict)
    zirke_embd_df = construct_embedding_dataframe(
        protein_dict,
        job_to_tar,
        "/embeddings/3dzd-embedding",
        "cif23dzd-result-",
        ".pkl",
    )
    grasr_embd_df = construct_embedding_dataframe(
        protein_dict, job_to_tar, "/embeddings/grasr-embedding", "cif2grasr-result-"
    )
    pca_grasr_embd_df = construct_embedding_dataframe(
        protein_dict,
        job_to_tar,
        "/embeddings/pca-grasr-embedding",
        "pca-cif2grasr-result-",
    )
    lmi_embd_df = construct_embedding_dataframe(
        protein_dict,
        job_to_tar,
        "/data/proteins/ng-granularity-10",
        "cif2emb-result-",
        lmi=True,
    )
    lmi_30_embd_df = construct_embedding_dataframe(
        protein_dict,
        job_to_tar,
        "/embeddings/ng-granularity-30",
        "cif2emb-result-",
        lmi=True,
    )

    configs = [
        {
            "name": "zirke-cosine",
            "df": zirke_embd_df,
            "metric": cosine,
        },
        {
            "name": "zirke-euclidean",
            "df": zirke_embd_df,
            "metric": euclidean,
        },
        {
            "name": "grasr-cosine",
            "df": grasr_embd_df,
            "metric": cosine,
        },
        {
            "name": "grasr-euclidean",
            "df": grasr_embd_df,
            "metric": euclidean,
        },
        {
            "name": "pca-cosine",
            "df": pca_grasr_embd_df,
            "metric": cosine,
        },
        {
            "name": "pca-euclidean",
            "df": pca_grasr_embd_df,
            "metric": euclidean,
        },
        {
            "name": "lmi-cosine",
            "df": lmi_embd_df,
            "metric": cosine,
        },
        {
            "name": "lmi-euclidean",
            "df": lmi_embd_df,
            "metric": euclidean,
        },
        {
            "name": "lmi-30-cosine",
            "df": lmi_30_embd_df,
            "metric": cosine,
        },
        {
            "name": "lmi-30-euclidean",
            "df": lmi_30_embd_df,
            "metric": euclidean,
        },
    ]

    for config in configs:
        get_embedding_distance(
            config["df"],
            protein.split("-")[1],
            config["metric"],
            f"results/{protein}-distances-{config['name']}.json",
        )
    subprocess.run(["rm", "-r", os.path.join(WORK_PATH, protein)])

    print("Total time:", time() - start_time)


if __name__ == "__main__":
    #parser = argparse.ArgumentParser()
    #parser.add_argument("--random", type=int, required=False, default=100)
    #parser.add_argument("--foldseek-file", type=str, required=True)

    #args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    proteins = os.listdir(FOLDSEEK_DIR)

    processed_proteins = [file.split("-dist")[0] for file in os.listdir("results") if file.endswith("distances.json")]
    processed_proteins = [protein.split("-")[1] for protein in processed_proteins]
    proteins = list(set(proteins) - set(processed_proteins))
    for protein in tqdm(proteins, desc="Collecting protein data"):
        run(1000, protein)
