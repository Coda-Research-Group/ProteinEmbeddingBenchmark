# Protein Embedding Computational and Benchmarking Tool

Protein Embedding Computational and Benchmarking Tool hosts resources needed
to transform mmCIF proteins to 5 state-of-the-art protein vector representations
(embeddings), and a benchmarking tool to test how well the produced vectors
follow the similarity of original protein files. Protein similarity is measured
in TM-score, and the benchmarked metrics of vector space similarity are Euclidean
and Cosine distance. The repository is divided in 4 modules.

## Convertor Container Image

The `convertor_container_image` module contains Dockerfile and resources which can be built
into a Container image used for transforming mmCIF protein files into embeddings. The code
provides embedding implementations used in LMI, DeepFRI, GearNET, GRaSR, and 3D-af-Surfer.

To transform a folder containing mmCIF proteins into embeddings, execute corresponding
run() method of selected embedding method. Example how to use converting method in your code:

```
from convertor_container_image.3d-af-surfer.cif23dzd import run

run(<input_folder>, <output_file>)


# <input_folder>: folder with .mmCIF files
# <output_file>: pickle file where embeddings will be stored
```

## Embedding Computational Pipeline

If you have access to an infrastructure capable of large scale processing of protein data,
the `embedding_computational_pipeline` module serves as an automated pipeline to convert
proteins from https://alphafold.ebi.ac.uk/download (~23TB of archives) into a selected
embedding type. After deploying contents of the module into a pod in a Kuberneted cluster,
one can innitiate the pipeline by executing following command:

```
python3 job_orchestrator.py 
    --template job_templates/grasr-job.yaml.jinja2
    --start_id 0
    --jobs_number 25339
    --max_concurrent_jobs 50
    --namespace <your-kubernetes-cluster>
```

There are 5 available jinja templates to choose from. The data will be batched and processed
in form of Kubernetes jobs.

## Open-source results available for download

If you do not have access to an infrastructure to run the conversion code, we open-sourced
the results for 5 precomputed embedding types, and hosted them on _. Each embedding method was
computed on AFDB v2, and contains 214m protein embeddings.

## Embedding Benchmark Tool

The `embedding_benchmark_tool` module provides an analysis conducted on a dataset constructed
from 100 protein subsets (proteins from all samples form together a dataset of size 143,738),
as well as more detailed analysis of one selected protein. It also stores a script to create
your own test dataset. The main point of analysis was to assess which embedding corresponds the
closest to similarity ranking by TM-score.

Example of the benchmark results. Full results are available under `evaluate_dataset.ipynb`...

```
    Emb. Method             AUC   Corr. coeff. st.dev.
1.  3d-af-Surfer (C)        0.810 0.595        0.220
2.  3d-af-Surfer (E)        0.803 0.593        0.217
3.  PCA-reduced GraSR (C)   0.776 0.569        0.217
4.  GraSR (C)               0.765 0.556        0.213
5.  GraSR (E)               0.765 0.556        0.213
6.  PCA-reduced GraSR (E)   0.765 0.556        0.213
7.  LMI-30 (E)              0.679 0.254        0.185
8.  LMI-10 (C)              0.669 0.284        0.192
9.  LMI-10 (E)              0.666 0.255        0.206
10. LMI-30 (C)              0.614 0.172        0.253


# (C) = Cosine distance
# (E) = Euclidean distance
```