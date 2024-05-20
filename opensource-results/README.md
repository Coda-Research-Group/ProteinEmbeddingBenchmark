# Precomputed vector representations of AFDB proteins

This archive contains protein embeddings for 214M million proteins from AlphaFold's
database v3 (https://alphafold.ebi.ac.uk/download). Protein data are divided into 25,339
batches of approximately 10,000 proteins in each. All data are stored in pickle format
and can be opened as pandas Data Frame.

Example how to unpickle the first batch:

```
import pandas as pd

df = pd.read_pickle("<embedding_type>-result-0")
df.head(5)


    protein	    embedding
0	A0A0U3I784	[0.4768706157267759, 0.10517105532942905, 0.11...
1	A0A5S3UPL9	[-0.08769398609953678, 0.10037438174737087, -0...
2	A0A0U3I785	[-0.10973752976170348, -0.173122438762918, 0.2...
3	A0A4Q7E244	[-0.3485142771166243, 0.011767758365625623, -0...
4	A0A5S3UXP0	[0.4017669408697772, -0.137390934144526, 0.259...
```


To find a specific protein, use batch-to-protein-index.csv, which lists contents of each
batch. Example how to open the index file:

```
import pandas as pd

batch_info = pd.read_csv("batch-to-protein-index.csv", sep="\t", header=None)
batch_info.head(5)


    batch content
0	0	  A0A0U3I784,A0A5S3UPL9,A0A0U3I785,A0A4Q7E244,A0...
1	1	  A0A2N3QEE5,A0A2N3QEE8,A0A2N3QEE9,A0A2N3QEF0,A0...
2	2	  T1IGR7,T1IGR8,T1IGR9,T1IGS0,T1IGS1,T1IGS2,T1IG...
3	3	  A0A3L6SLG2,A0A3L6SLG3,A0A3L6SLG4,A0A3L6SLG5,A0...
4	4	  A0A1R3HCG5,A0A1R3HCG6,A0A1R3HCG8,A0A1R3HCG9,A0...
```


For more information, visit our GitHub repository at https://github.com/Coda-Research-Group/ProteinEmbeddingBenchmark.