#!/usr/bin/env python
"""Spatial Leiden clustering of the D1 section (submandibular + sublingual gland).
Produces a spatial cluster map, a UMAP, per-cluster top markers and a panel of
canonical SMG/SLG genes so the clusters can be labelled SMG vs SLG by hand.
Outputs to results/D1_cluster/ and figures to results/figures_d1/."""
import os, sys, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd, scanpy as sc
import matplotlib; matplotlib.use("Agg")
matplotlib.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})
import matplotlib.pyplot as plt
from visium_io import load_visium

OUTS = "SMG:SLG_Outs/D1/outs"
LIB = "D1"
RES = "results/D1_cluster"; FIG = "results/figures_d1"
os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)
RESOLUTION = float(os.environ.get("LEIDEN_RES", "0.6"))

adata = load_visium(OUTS, LIB, img_key="hires")
adata.layers["counts"] = adata.X.copy()
print(f"loaded D1: {adata.n_obs} spots x {adata.n_vars} genes", flush=True)

# QC
adata.var["mt"] = adata.var_names.str.startswith("mt-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True, percent_top=None)
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
adata = adata[adata.obs["pct_counts_mt"] < 30].copy()
print(f"after QC: {adata.n_obs} spots x {adata.n_vars} genes", flush=True)

# normalise / HVG / PCA / neighbours / Leiden
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.raw = adata
sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat")
adata.layers["lognorm"] = adata.X.copy()
hvg = adata[:, adata.var["highly_variable"]].copy()
sc.pp.scale(hvg, max_value=10)
sc.tl.pca(hvg, n_comps=30, svd_solver="arpack")
adata.obsm["X_pca"] = hvg.obsm["X_pca"]
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
sc.tl.leiden(adata, resolution=RESOLUTION, key_added="leiden", flavor="igraph",
             n_iterations=2, directed=False)
sc.tl.umap(adata)
n_clusters = adata.obs["leiden"].nunique()
print(f"Leiden (res={RESOLUTION}) -> {n_clusters} clusters", flush=True)

# per-cluster top markers (Wilcoxon)
sc.tl.rank_genes_groups(adata, "leiden", method="wilcoxon")
rg = adata.uns["rank_genes_groups"]; groups = rg["names"].dtype.names
top = pd.DataFrame({g: [rg["names"][g][i] for i in range(15)] for g in groups})
top.to_csv(f"{RES}/D1_cluster_top_markers.csv", index=False)

# canonical SMG / SLG markers for hand-labelling
MARK = {"SLG mucous (Muc19/A2ml1)": ["Muc19", "A2ml1", "Mucl3"],
        "SMG serous (Bpifa2/Smgc/Dcpp1)": ["Bpifa2", "Smgc", "Dcpp1"],
        "Ductal (Klk1/Krt8)": ["Klk1", "Krt8", "Krt19"],
        "Myoepithelial (Acta2/Krt5)": ["Acta2", "Krt5"]}
present = {k: [g for g in v if g in adata.raw.var_names] for k, v in MARK.items()}
mscore = {}
for k, genes in present.items():
    if genes:
        sc.tl.score_genes(adata, genes, score_name=k, use_raw=True)
        mscore[k] = genes

# mean marker-score per cluster -> helps decide which cluster is which gland
prof = adata.obs.groupby("leiden")[list(mscore)].mean()
prof.to_csv(f"{RES}/D1_cluster_marker_scores.csv")
print("\nmean marker score per cluster:\n", prof.round(3).to_string(), flush=True)

adata.write(f"{RES}/D1_clustered.h5ad")
adata.obs[["leiden"]].rename_axis("barcode").to_csv(f"{RES}/D1_spot_clusters.csv")

# ---- figures ----
sc.pl.spatial(adata, color="leiden", img_key="hires", size=1.5, show=False,
              title=f"D1 Leiden clusters (res={RESOLUTION})")
plt.savefig(f"{FIG}/D1_spatial_clusters.png", dpi=200, bbox_inches="tight"); plt.close()

sc.pl.umap(adata, color="leiden", show=False, title="D1 Leiden clusters")
plt.savefig(f"{FIG}/D1_umap_clusters.png", dpi=200, bbox_inches="tight"); plt.close()

# spatial marker panels (one per canonical marker set, summed score)
keys = list(mscore)
fig, axes = plt.subplots(1, len(keys), figsize=(6 * len(keys), 6))
axes = np.atleast_1d(axes)
for ax, k in zip(axes, keys):
    sc.pl.spatial(adata, color=k, img_key="hires", size=1.5, ax=ax, show=False,
                  title=k, cmap="magma")
plt.savefig(f"{FIG}/D1_spatial_markers.png", dpi=200, bbox_inches="tight"); plt.close()

# dotplot of canonical genes across clusters
allgenes = [g for genes in mscore.values() for g in genes]
sc.pl.dotplot(adata, allgenes, groupby="leiden", use_raw=True, standard_scale="var",
              show=False, figsize=(max(8, len(allgenes) * 0.6), 0.5 * n_clusters + 2),
              title="Canonical SMG/SLG markers per D1 cluster")
plt.savefig(f"{FIG}/D1_marker_dotplot.png", dpi=200, bbox_inches="tight"); plt.close()

print("\nWrote:", flush=True)
print(f"  {RES}/D1_clustered.h5ad, D1_spot_clusters.csv, D1_cluster_top_markers.csv, D1_cluster_marker_scores.csv")
print(f"  {FIG}/D1_spatial_clusters.png, D1_umap_clusters.png, D1_spatial_markers.png, D1_marker_dotplot.png")
