#!/usr/bin/env python
"""Combined FULL reference: SMG iScience (14 granular types) + ALL annotated
GSE216476 cells (8 SMG/SLG-mix types), batch key = study. Raw counts, common
genes. GSE216476 types get a ' (mix)' suffix to mark the cross-gland source
(except the SLG-specific mucous acinar, kept as-is)."""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, scanpy as sc, anndata as ad
import scipy.sparse as sp

SMG = "data/reference/adult_ref.h5ad"
G2 = "data/reference/GSE216476/slg_annotated_full.h5ad"
OUT = "data/reference/combined_full_ref.h5ad"

smg = sc.read_h5ad(SMG)
smg.obs["study"] = "SMG_iScience"
smg.obs["CellType"] = smg.obs["CellType"].astype(str)
smg.X = sp.csr_matrix(smg.X)

g2 = sc.read_h5ad(G2)
g2 = ad.AnnData(X=sp.csr_matrix(g2.layers["counts"]),
                obs=g2.obs[["CellType"]].copy(), var=pd.DataFrame(index=g2.var_names))
g2.obs["CellType"] = g2.obs["CellType"].astype(str)
# suffix gland-mix types (keep the SLG-specific name intact)
g2.obs["CellType"] = g2.obs["CellType"].apply(
    lambda c: c if c == "Mucous acinar (SLG)" else f"{c} (mix)")
g2.obs["study"] = "GSE216476_mix"

common = smg.var_names.intersection(g2.var_names)
print(f"SMG {smg.n_obs} cells, GSE216476 {g2.n_obs} cells; common genes {len(common)}", flush=True)
smg = smg[:, common].copy(); g2 = g2[:, common].copy()

comb = ad.concat([smg, g2], join="outer", index_unique="-")
comb.obs["CellType"] = comb.obs["CellType"].astype("category")
comb.obs["study"] = comb.obs["study"].astype("category")
comb.X = sp.csr_matrix(comb.X)
print("\nCombined FULL reference:", comb.shape, "| cell types:", comb.obs["CellType"].nunique(), flush=True)
print(comb.obs["CellType"].value_counts().to_string(), flush=True)
comb.write(OUT)
print("\nWrote", OUT, flush=True)
