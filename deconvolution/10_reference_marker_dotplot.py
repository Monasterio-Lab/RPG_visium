#!/usr/bin/env python
"""Marker genes per cell type for the SMG+SLG+Parotid combined reference:
Wilcoxon rank_genes_groups (one-vs-rest), top markers per type -> dotplot.
Documents the genes that define each of the 27 reference cell types."""
import os, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np, pandas as pd, scanpy as sc
import matplotlib; matplotlib.use("Agg")
matplotlib.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
import matplotlib.pyplot as plt

REF = "data/reference/combined_parotid_ref.h5ad"
OUT = "results/figures_parotid"; os.makedirs(OUT, exist_ok=True)
NTOP = 3

ad = sc.read_h5ad(REF)
ad.obs["CellType"] = ad.obs["CellType"].astype("category")
sc.pp.normalize_total(ad, target_sum=1e4); sc.pp.log1p(ad)
sc.pp.filter_genes(ad, min_cells=10)

sc.tl.rank_genes_groups(ad, "CellType", method="wilcoxon", n_genes=30)
res = ad.uns["rank_genes_groups"]; groups = list(res["names"].dtype.names)

# order cell types by lineage for readability
LIN_ORDER = ["acinar", "serous", "mucous", "smgc", "bpifa2", "gct", "duct", "myoep",
             "mesench", "stromal", "vascular", "endothel", "immune", "macroph",
             "nk", "erythro"]
def lin_rank(ct):
    cl = ct.lower()
    for i, k in enumerate(LIN_ORDER):
        if k in cl: return i
    return len(LIN_ORDER)
order = sorted(groups, key=lambda g: (lin_rank(g), g))

# top markers per type (dedup, keep first occurrence)
marker_map, seen = {}, set()
for g in order:
    names = [res["names"][g][i] for i in range(len(res["names"][g]))]
    picks = [n for n in names if n not in seen][:NTOP]
    for n in picks: seen.add(n)
    marker_map[g] = picks if picks else names[:NTOP]
pd.DataFrame({g: marker_map[g] for g in order}).to_csv(f"{OUT}/parotid_ref_markers_top.csv", index=False)

ad.obs["CellType"] = ad.obs["CellType"].cat.reorder_categories(order)
dp = sc.pl.dotplot(ad, marker_map, groupby="CellType", return_fig=True, cmap="Reds",
                   standard_scale="var", figsize=(0.32 * sum(len(v) for v in marker_map.values()) + 4,
                                                   0.42 * len(order) + 2))
dp.add_totals().style(dot_edge_color="black", dot_edge_lw=0.2)
for ext in ["png", "svg", "pdf"]:
    dp.savefig(f"{OUT}/parotid_ref_marker_dotplot.{ext}",
               dpi=160 if ext == "png" else None, bbox_inches="tight")
print(f"{len(order)} cell types; wrote parotid_ref_marker_dotplot.png/svg/pdf + markers_top.csv", flush=True)
