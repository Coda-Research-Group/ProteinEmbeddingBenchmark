import os
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from Bio import SeqIO
from Bio.PDB.MMCIFParser import MMCIFParser
from deepfrier.layers import FuncPredictor, GraphConv, MultiGraphConv, SumPooling
from deepfrier.utils import seq2onehot
from keras.models import Model

cmap_thresh = 10.0
parser = MMCIFParser()


def load_cif(protein):
    """Load a cif file with biopython library

    Args:
        protein (str): cif file name
    """
    # Generate (diagonalized) C_alpha distance matrix from a cif file
    structure = parser.get_structure(protein.split("/")[-1].split(".")[0], protein)
    residues = [r for r in structure.get_residues()]

    # sequence from atom lines
    records = SeqIO.parse(protein, "cif-atom")
    seqs = [str(r.seq) for r in records]

    distances = np.empty((len(residues), len(residues)))
    for x in range(len(residues)):
        for y in range(len(residues)):
            one = residues[x]["CA"].get_coord()
            two = residues[y]["CA"].get_coord()
            distances[x, y] = np.linalg.norm(one - two)

    return distances, seqs[0]


def read_and_extract(protein):
    """Read file and extract C_alpha distance matrix"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        D, seq = load_cif(protein)

    A = np.double(D < cmap_thresh)
    S = seq2onehot(seq)
    S = S.reshape(1, *S.shape)
    A = A.reshape(1, *A.shape)

    protein_name = protein.split("/")[-1].split(".")[0]

    return protein_name, [A, S]


def run(input_path, output_path):
    """Run DeepFRI on a directory of cif files"""
    model = tf.keras.models.load_model(
        "DeepFRI-MERGED_MultiGraphConv_3x512_fcd_1024_ca_10A_molecular_function.hdf5",
        custom_objects={
            "MultiGraphConv": MultiGraphConv,
            "GraphConv": GraphConv,
            "FuncPredictor": FuncPredictor,
            "SumPooling": SumPooling,
        },
    )
    encoder = Model(model.input, model.layers[-2].output)
    embeddings = {}

    proteins = os.listdir(input_path)
    proteins = [file for file in proteins if file.endswith(".cif")]
    with Pool() as pool:
        for protein, result in pool.imap_unordered(
            read_and_extract, [f"{input_path}{protein}" for protein in proteins]
        ):
            embeddings[protein] = encoder.predict(result)

    df = pd.DataFrame(list(embeddings.items()), columns=["protein", "embedding"])
    df.to_pickle(Path(output_path))
