import argparse
import os
from pathlib import Path

import pandas as pd
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm


def fit_pca(embeddings_path, name, n_components=50):
    """
    Iteratively fit PCA on all embeddings in the folder.

    Args:
    embeddings_path (str): path to the folder with embeddings
    name (str): base name of the embedding pkl file

    Returns:
    sklearn.decomposition.IncrementalPCA: fitted PCA model
    """
    ipca = IncrementalPCA(n_components)

    embeddings = os.listdir(embeddings_path)[:1000]
    embeddings = [file for file in embeddings if name in file]

    for emb in tqdm(embeddings, desc="Fitting PCA"):
        data = pd.read_pickle(os.path.join(embeddings_path, emb))
        data = list(value for value in data["embedding"].values)
        ipca.partial_fit(data)
    return ipca


def transform_embeddings(pca, embeddings_path, name, output_path):
    """
    Transform embeddings with PCA and save them to the output folder.

    Args:
    pca (sklearn.decomposition.IncrementalPCA): fitted PCA model
    embeddings_path (str): path to the folder with embeddings
    name (str): base name of the embedding pkl file
    output_path (str): path to the output folder
    """
    os.makedirs(output_path, exist_ok=True)
    embeddings = os.listdir(embeddings_path)
    embeddings = [file for file in embeddings if name in file]

    for emb in tqdm(embeddings, desc="Transforming and saving embeddings"):
        unpickled_df = pd.read_pickle(os.path.join(embeddings_path, emb))
        raw_embeddings = list(value for value in unpickled_df["embedding"].values)
        embeddings_transformed = pca.transform(raw_embeddings)
        pca_embeddings = {}
        for protein, embedding in zip(
            unpickled_df["protein"].values, embeddings_transformed
        ):
            pca_embeddings[protein] = embedding

        df = pd.DataFrame(
            list(pca_embeddings.items()), columns=["protein", "embedding"]
        )
        df.to_pickle(os.path.join(output_path, "pca-" + emb))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--input", "-i", required=True, help="Path to folder with original embedding"
    )
    argparser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Path to folder where PCA embedding will be saved",
    )
    argparser.add_argument(
        "--name",
        "-n",
        default="cif2emb",
        help="Base name of the existing embedding pickle file",
    )
    argparser.add_argument(
        "--components", "-c", default=50, type=int, help="Number of PCA components"
    )
    args = argparser.parse_args()

    assert os.path.exists(args.input)
    print("Fitting PCA on", args.input)

    ipca = fit_pca(
        embeddings_path=args.input, name=args.name, n_components=args.components
    )
    transform_embeddings(
        pca=ipca, embeddings_path=args.input, name=args.name, output_path=args.output
    )

    print("PCA embedding is saved in", args.output)
