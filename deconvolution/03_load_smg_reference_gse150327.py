#!/usr/bin/env python
"""Build a reference AnnData (raw counts + CellType) from the R-exported MM bundle."""
import os
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp
import anndata as ad

EXPORT = "data/reference/adult_export"
OUT = "data/reference/adult_ref.h5ad"

# counts.mtx is genes x cells -> transpose to cells x genes
m = scipy.io.mmread(os.path.join(EXPORT, "counts.mtx")).tocsr().T.tocsr()
genes = pd.read_csv(os.path.join(EXPORT, "genes.tsv"), header=None)[0].values
bcs = pd.read_csv(os.path.join(EXPORT, "barcodes.tsv"), header=None)[0].values
meta = pd.read_csv(os.path.join(EXPORT, "metadata.csv"))
meta = meta.set_index("barcode").loc[bcs]

adata = ad.AnnData(X=sp.csr_matrix(m, dtype=np.float32),
                   obs=meta.copy(),
                   var=pd.DataFrame(index=genes))
adata.var_names_make_unique()
adata.obs["CellType"] = adata.obs["CellType"].astype("category")

print(adata)
print("\nCell types:\n", adata.obs["CellType"].value_counts())
adata.write(OUT)
print("\nWrote", OUT)
