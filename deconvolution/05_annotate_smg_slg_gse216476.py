#!/usr/bin/env python
"""Full annotation of GSE216476 (SMG+SLG mix): assign every leiden cluster to a
cell type, report composition, and make a UMAP. Saves an annotated h5ad (raw
counts kept) for building the full combined reference."""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, scanpy as sc
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

IN = "data/reference/GSE216476/slg_clustered.h5ad"
OUTH5 = "data/reference/GSE216476/slg_annotated_full.h5ad"
OUT = "results/combined_ref_qc"; os.makedirs(OUT, exist_ok=True)

# leiden cluster -> cell type (from marker diagnostics in step 10)
CLUSTER_MAP = {
    "0": "Serous acinar", "3": "Serous acinar", "4": "Serous acinar", "8": "Serous acinar",
    "5": "Seromucous acinar",
    "12": "Mucous acinar (SLG)",
    "1": "Ductal", "6": "Ductal", "10": "Ductal", "11": "Ductal",
    "15": "Myoepithelial",
    "2": "Immune", "9": "Immune", "14": "Immune",
    "7": "Vascular",
    "13": "Mesenchymal",
}
adata = sc.read_h5ad(IN)
adata.obs["CellType"] = adata.obs["leiden"].map(CLUSTER_MAP).astype("category")
assert adata.obs["CellType"].isna().sum() == 0, "unmapped clusters!"

# composition
comp = adata.obs["CellType"].value_counts()
comp_pct = (comp / comp.sum() * 100).round(2)
print("=== GSE216476 (SMG+SLG) full composition ===", flush=True)
print(pd.DataFrame({"n_cells": comp, "pct": comp_pct}).to_string(), flush=True)
pd.DataFrame({"n_cells": comp, "pct": comp_pct}).to_csv(f"{OUT}/GSE216476_full_composition.csv")

# UMAP (recompute embedding on HVG)
emb = adata[:, adata.var["highly_variable"]].copy()
sc.pp.scale(emb, max_value=10); sc.tl.pca(emb, n_comps=30)
sc.pp.neighbors(emb, n_neighbors=15, n_pcs=30); sc.tl.umap(emb)
adata.obsm["X_umap"] = emb.obsm["X_umap"]
sc.pl.umap(adata, color="CellType", show=False, size=12, legend_loc="right margin",
           title="GSE216476 (SMG+SLG) — full annotation")
plt.savefig(f"{OUT}/GSE216476_full_umap.png", dpi=140, bbox_inches="tight"); plt.close()

adata.write(OUTH5)
print("\nWrote", OUTH5, "and composition/UMAP to", OUT, flush=True)
