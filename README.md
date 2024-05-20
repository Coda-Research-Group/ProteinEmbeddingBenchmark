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

run(<input_folder>, <output_file>.pkl)


# <input_folder>: folder with .mmCIF files, you can use our test folder sample_mmcif_proteins
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
the results for 5 precomputed embedding types, and hosted them on the [Czech National Repository](https://data.narodni-repozitar.cz/). Each embedding method was
computed on AFDB v3, and contains 214m protein embeddings.


|Embedding           |Type     |Dimensionality     |Size  |Download link|
|--------------------|---------|------------------:|-----:|------------|
|3D-af-Surfer        |Geometric           |120  |161 GiB  |[10.48700/datst.tbws0-hj147](https://doi.org/10.48700/datst.tbws0-hj147)|
|GraSR               |Neural network      |400  |149 GiB  |[10.48700/datst.br8aq-db495](https://doi.org/10.48700/datst.br8aq-db495)|
|PCA-reduced GraSR   |PCA reduction        |50   |79 GiB  |[10.48700/datst.rec6m-2sq83](https://doi.org/10.48700/datst.rec6m-2sq83)|
|LMI-10              |Geometric            |45    |8 GiB  |[10.48700/datst.0y0y6-v0783](https://doi.org/10.48700/datst.0y0y6-v0783)|
|LMI-30              |Geometric           |435   |67 GiB  |[10.48700/datst.tbws0-hj147](https://doi.org/10.48700/datst.tbws0-hj147)|


## Embedding Benchmark Tool

The `embedding_benchmark_tool` module provides an analysis conducted on a dataset constructed
from 100 protein subsets (proteins from all samples form together a dataset of size 143,738),
as well as more detailed analysis of one selected protein. It also stores a script to create
your own test dataset. The main point of analysis was to assess which embedding corresponds the
closest to similarity ranking by *TM-score*.

Example of the benchmark results. Full results are available under `evaluate_dataset.ipynb`...


|Rank |Embedding Method    |AUC   |Corr. coeff. |F1|
|-----|--------------------|-----:|------------:|-:|
|1.  |3d-af-Surfer (C)        |0.822 |0.595     |0.743|
|2.  |3d-af-Surfer (E)        |0.815 |0.593     |0.757|
|3.  |PCA-reduced GraSR (C)   |0.777 |0.569     |0.712|
|4.  |GraSR (C)               |0.766 |0.556     |0.712|
|5.  |GraSR (E)               |0.766 |0.556     |0.712|
|6.  |PCA-reduced GraSR (E)   |0.766 |0.556     |0.712|
|7.  |LMI-30 (C)              |0.615 |0.172     |0.549|
|8.  |LMI-10 (C)              |0.669 |0.284     |0.573|
|9.  |LMI-30 (E)              |0.674 |0.254     |0.514|
|10. |LMI-10 (E)              |0.661 |0.255     |0.516|

```
# (C) = Vector distance computed with Cosine distance
# (E) = Vector distance computed with Euclidean distance
```