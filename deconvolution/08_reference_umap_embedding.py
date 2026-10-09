#!/usr/bin/env python
"""UMAP of the new SMG+SLG+Parotid combined reference (combined_parotid_ref),
integrated WITH and WITHOUT Harmony, coloured by cell-type annotation and by
dataset (3 studies). Dataset legend uses the citation format.
"""
import os, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np, scanpy as sc, harmonypy
import matplotlib; matplotlib.use("Agg")
matplotlib.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

REF = "data/reference/combined_parotid_ref.h5ad"
OUT = "results/figures_parotid"; os.makedirs(OUT, exist_ok=True)

# study -> (colour, citation label)
STUDY = {
    "GSE216476_mix":     ("#4C72B0", "GSE216476 — SMG/SLG (Huang et al. 2022)"),
    "SMG_iScience":      ("#C44E52", "GSE150327 — SMG (Hauser et al. 2020)"),
    "GSE223516_parotid": ("#55A868", "GSE223516 — Parotid (Rheinheimer et al. 2023)"),
}

ad = sc.read_h5ad(REF)
ad.obs["study"] = ad.obs["study"].astype("category")
ad.obs["CellType"] = ad.obs["CellType"].astype("category")
sc.pp.normalize_total(ad, target_sum=1e4); sc.pp.log1p(ad)
sc.pp.highly_variable_genes(ad, n_top_genes=2000, batch_key="study")
emb = ad[:, ad.var["highly_variable"]].copy()
sc.pp.scale(emb, max_value=10); sc.tl.pca(emb, n_comps=50)

# uncorrected UMAP
sc.pp.neighbors(emb, n_neighbors=15, n_pcs=30, use_rep="X_pca")
sc.tl.umap(emb); emb.obsm["X_umap_uncorr"] = emb.obsm["X_umap"].copy()
# harmony UMAP
ho = harmonypy.run_harmony(emb.obsm["X_pca"], emb.obs, ["study"])
Z = np.asarray(ho.Z_corr); emb.obsm["X_pca_harmony"] = Z if Z.shape[0] == emb.n_obs else Z.T
sc.pp.neighbors(emb, n_neighbors=15, n_pcs=30, use_rep="X_pca_harmony")
sc.tl.umap(emb); emb.obsm["X_umap_harm"] = emb.obsm["X_umap"].copy()
emb.write(f"{OUT}/parotid_ref_embedded.h5ad")

cts = sorted(ad.obs["CellType"].cat.categories)
palette = (list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("tab20b").colors))
ctcol = {c: palette[i % len(palette)] for i, c in enumerate(cts)}

def scatter(ax, basis, color, title):
    XY = emb.obsm[basis]
    if color == "study":
        for s, (col, _) in STUDY.items():
            m = (emb.obs["study"] == s).to_numpy()
            ax.scatter(XY[m, 0], XY[m, 1], s=5, c=col, linewidths=0, alpha=0.7)
    else:
        for c in cts:
            m = (emb.obs["CellType"] == c).to_numpy()
            ax.scatter(XY[m, 0], XY[m, 1], s=5, c=[ctcol[c]], linewidths=0, alpha=0.8)
        # on-top labels at each cell type's centroid (median, robust to outliers)
        for c in cts:
            m = (emb.obs["CellType"] == c).to_numpy()
            if m.sum() < 5: continue
            cx, cy = np.median(XY[m, 0]), np.median(XY[m, 1])
            ax.text(cx, cy, c, fontsize=6.2, fontweight="bold", ha="center", va="center",
                    color="#111", zorder=6,
                    path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")

fig = plt.figure(figsize=(17, 13))
import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(2, 2, hspace=0.16, wspace=0.12, left=0.04, right=0.74, top=0.93, bottom=0.05)
scatter(fig.add_subplot(gs[0, 0]), "X_umap_uncorr", "CellType", "Uncorrected — cell-type annotation")
scatter(fig.add_subplot(gs[0, 1]), "X_umap_uncorr", "study", "Uncorrected — dataset")
scatter(fig.add_subplot(gs[1, 0]), "X_umap_harm", "CellType", "Harmony-integrated — cell-type annotation")
scatter(fig.add_subplot(gs[1, 1]), "X_umap_harm", "study", "Harmony-integrated — dataset")

# dataset legend (citation format), top-right
axd = fig.add_axes([0.76, 0.62, 0.23, 0.30]); axd.axis("off")
axd.text(0.0, 1.0, "dataset", fontsize=13, va="top", ha="left")
h = [Line2D([0], [0], marker="o", color="w", markerfacecolor=col, markersize=13, label=lab)
     for s, (col, lab) in STUDY.items()]
axd.legend(handles=h, loc="upper left", bbox_to_anchor=(0.0, 0.92), frameon=False,
           fontsize=11, handletextpad=0.6, labelspacing=1.4, borderpad=0)
# cell-type legend (right margin)
axc = fig.add_axes([0.76, 0.04, 0.23, 0.55]); axc.axis("off")
axc.text(0.0, 1.0, "cell type", fontsize=13, va="top", ha="left")
hc = [Line2D([0], [0], marker="o", color="w", markerfacecolor=ctcol[c], markersize=9, label=c) for c in cts]
axc.legend(handles=hc, loc="upper left", bbox_to_anchor=(0.0, 0.96), frameon=False,
           fontsize=8.2, handletextpad=0.5, labelspacing=0.32, ncol=1, borderpad=0)

fig.suptitle("SMG + SLG + Parotid combined salivary reference — integrated UMAP (3 datasets)",
             fontsize=15, fontweight="bold", y=0.975)
for ext in ["png", "svg", "pdf"]:
    fig.savefig(f"{OUT}/parotid_ref_umap.{ext}", dpi=160 if ext == "png" else None,
                bbox_inches="tight", facecolor="white")
print(f"{ad.n_obs} cells, {len(cts)} cell types, studies {list(STUDY)}", flush=True)
print("wrote parotid_ref_umap.png/svg/pdf", flush=True)
