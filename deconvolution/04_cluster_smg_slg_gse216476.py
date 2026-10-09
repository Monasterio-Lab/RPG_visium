#!/usr/bin/env python
"""Cluster GSE216476 (mouse SMG+SLG scRNA-seq, no author labels): QC, Leiden clustering (resolution 0.5) and per-cluster marker means for annotation. Cluster → cell-type assignment is done in 05_annotate_smg_slg_gse216476.py. Raw counts kept in .layers[‘counts’]."""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, scanpy as sc

TENX = "data/reference/GSE216476/tenx"
OUT = "data/reference/GSE216476/slg_clustered.h5ad"
sc.settings.verbosity = 1

adata = sc.read_10x_mtx(TENX, var_names="gene_symbols")
adata.var_names_make_unique()
print("raw:", adata.shape, flush=True)

# --- QC ---
adata.var["mt"] = adata.var_names.str.startswith(("mt-", "Mt-"))
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True, percent_top=None)
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
adata = adata[adata.obs["pct_counts_mt"] < 25].copy()
adata = adata[adata.obs["n_genes_by_counts"] < 7500].copy()
print("after QC:", adata.shape, flush=True)

adata.layers["counts"] = adata.X.copy()   # keep raw counts for the reference

# --- standard clustering ---
sc.pp.normalize_total(adata, target_sum=1e4); sc.pp.log1p(adata)
adata.raw = adata
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
adata_hvg = adata[:, adata.var["highly_variable"]].copy()
sc.pp.scale(adata_hvg, max_value=10)
sc.tl.pca(adata_hvg, n_comps=30)
sc.pp.neighbors(adata_hvg, n_neighbors=15, n_pcs=30)
sc.tl.leiden(adata_hvg, resolution=0.5, flavor="igraph", n_iterations=2, directed=False)
adata.obs["leiden"] = adata_hvg.obs["leiden"]
print("clusters:", adata.obs["leiden"].value_counts().to_dict(), flush=True)

# --- per-cluster marker means (lineage diagnostics) ---
markers = {
    "Mucous-acinar(SLG)": ["A2ml1", "Muc19", "Pigr"],
    "Serous-acinar":      ["Bpifa2", "Smgc", "Dcpp1", "Prol1"],
    "Ductal":             ["Krt8", "Krt18", "Krt19", "Krt7"],
    "Myoepithelial":      ["Acta2", "Myh11"],
    "Immune":             ["Ptprc", "C1qa", "Cd52", "Cd3e"],
    "Vascular":           ["Pecam1", "Emcn", "Plvap"],
    "Mesenchymal":        ["Col1a1", "Col3a1", "Dcn", "Vim"],
}
flat = [g for v in markers.values() for g in v if g in adata.raw.var_names]
expr = sc.get.obs_df(adata, keys=flat + ["leiden"], use_raw=True)
print("\n=== mean log-norm expression per cluster ===", flush=True)
print(expr.groupby("leiden", observed=True).mean().round(2).to_string(), flush=True)

# --- top marker genes per cluster ---
sc.tl.rank_genes_groups(adata, "leiden", method="wilcoxon", n_genes=10, use_raw=True)
top = sc.get.rank_genes_groups_df(adata, group=None)
print("\n=== top genes per cluster ===", flush=True)
for cl in sorted(adata.obs["leiden"].unique(), key=int):
    g = top.loc[top["group"] == cl, "names"].head(10).tolist()
    print(f"cluster {cl}: {', '.join(g)}", flush=True)

adata.write(OUT)
print("\nWrote", OUT, flush=True)
