#!/usr/bin/env python
"""Reference UMAP, 3 colourings x (uncorrected + Harmony): by dataset, by cell
type (lineage-shaded), and by LINEAGE. Cell-type legend grouped under the
lineage classifications used for the per-spot scatterpies. Loads the saved
embedding (parotid_ref_embedded.h5ad) — no recompute."""
import os, sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "analysis")
import numpy as np, pandas as pd, scanpy as sc
import matplotlib; matplotlib.use("Agg")
matplotlib.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.colors import to_rgb
from rename_celltypes import rename_lineage

OUT = "results/figures_parotid"
emb = sc.read_h5ad(f"{OUT}/parotid_ref_embedded.h5ad")
if "X_umap_uncorr" not in emb.obsm:  # (re)build if missing
    os.system(f"{sys.executable} analysis/08_reference_umap_embedding.py")
    emb = sc.read_h5ad(f"{OUT}/parotid_ref_embedded.h5ad")

STUDY = {"GSE216476_mix": ("#4C72B0", "GSE216476 — SMG/SLG (Huang et al. 2022)"),
         "SMG_iScience": ("#C44E52", "GSE150327 — SMG (Hauser et al. 2020)"),
         "GSE223516_parotid": ("#55A868", "GSE223516 — Parotid (Rheinheimer et al. 2023)")}
LIN_ORDER = ["Acinar-Serous", "Acinar-Mucous", "Ductal", "Myoepithelial",
             "Stromal/Mesench", "Endothelial/Vasc", "Immune", "Others"]
LINCOL = {"Acinar-Serous": "#E15759", "Acinar-Mucous": "#4E79A7", "Ductal": "#59A14F",
          "Myoepithelial": "#76B7B2", "Stromal/Mesench": "#B07AA1", "Endothelial/Vasc": "#EDC948",
          "Immune": "#F28E2B", "Others": "#BAB0AC"}   # matches composition/scatterpie lineages
LINEAGE = rename_lineage({
 "Acinar-Serous": ["Bpifa2+", "Serous acinar (mix)", "Acinar", "Seromucous acinar (mix)", "Serous acinar (parotid)"],
 "Acinar-Mucous": ["Mucous acinar (SLG)", "Smgc+"],
 "Ductal": ["Ductal (mix)", "Striated duct", "Basal duct", "Intercalated duct", "Ascl3+ duct", "GCT", "Ductal (parotid)"],
 "Myoepithelial": ["Myoepithelial", "Myoepithelial (mix)"],
 "Stromal/Mesench": ["Mesenchymal (mix)", "Stromal", "Mesenchymal (parotid)"],
 "Endothelial/Vasc": ["Endothelial", "Vascular (mix)", "Endothelial (parotid)"],
 "Immune": ["Immune (mix)", "Macrophages", "NK cells", "Immune (parotid)"],
 "Others": ["Erythroid"]})
ct2lin = {c: L for L, cs in LINEAGE.items() for c in cs}

cts_present = list(emb.obs["CellType"].astype(str).unique())
missing = [c for c in cts_present if c not in ct2lin]
if missing: print("UNMAPPED:", missing, flush=True)
emb.obs["lineage"] = emb.obs["CellType"].astype(str).map(ct2lin).fillna("Others")

def shades(hexc, n):
    base = np.array(to_rgb(hexc))
    if n == 1: return [tuple(base)]
    out = []
    for f in np.linspace(0.34, -0.36, n):  # dark -> light around base
        c = base * (1 - f) if f >= 0 else base * (1 + f) + np.array([1, 1, 1]) * (-f)
        out.append(tuple(np.clip(c, 0, 1)))
    return out

# order cell types within each lineage by abundance; assign shades
counts = emb.obs["CellType"].value_counts()
ctcol, ct_by_lin = {}, {}
for L in LIN_ORDER:
    cs = [c for c in LINEAGE[L] if c in cts_present]
    cs = sorted(cs, key=lambda c: -counts.get(c, 0))
    ct_by_lin[L] = cs
    for c, col in zip(cs, shades(LINCOL[L], len(cs))): ctcol[c] = col

def panel(ax, basis, mode, title):
    XY = emb.obsm[basis]
    if mode == "study":
        for s, (col, _) in STUDY.items():
            m = (emb.obs["study"] == s).to_numpy()
            ax.scatter(XY[m, 0], XY[m, 1], s=4, c=col, linewidths=0, alpha=0.65, rasterized=True)
    elif mode == "lineage":
        for L in LIN_ORDER:
            m = (emb.obs["lineage"] == L).to_numpy()
            if m.any(): ax.scatter(XY[m, 0], XY[m, 1], s=4, c=[LINCOL[L]], linewidths=0, alpha=0.8, rasterized=True)
        for L in LIN_ORDER:
            m = (emb.obs["lineage"] == L).to_numpy()
            if m.sum() > 30:
                ax.text(np.median(XY[m, 0]), np.median(XY[m, 1]), L, fontsize=8, fontweight="bold",
                        ha="center", va="center", color="#111", path_effects=[pe.withStroke(linewidth=2.4, foreground="white")])
    else:  # cell type (lineage-shaded)
        for c in cts_present:
            m = (emb.obs["CellType"] == c).to_numpy()
            ax.scatter(XY[m, 0], XY[m, 1], s=4, c=[ctcol[c]], linewidths=0, alpha=0.85, rasterized=True)
        for c in cts_present:
            m = (emb.obs["CellType"] == c).to_numpy()
            if m.sum() > 30:
                ax.text(np.median(XY[m, 0]), np.median(XY[m, 1]), c, fontsize=5.0, fontweight="bold",
                        ha="center", va="center", color="#111", path_effects=[pe.withStroke(linewidth=1.6, foreground="white")])
    ax.set_title(title, fontsize=11.5, fontweight="bold"); ax.set_xticks([]); ax.set_yticks([])

fig = plt.figure(figsize=(21, 12.5))
gs = fig.add_gridspec(2, 3, left=0.025, right=0.72, top=0.92, bottom=0.05, hspace=0.13, wspace=0.06)
rows = [("X_umap_uncorr", "Uncorrected"), ("X_umap_harm", "Harmony-integrated")]
for r, (basis, rlab) in enumerate(rows):
    panel(fig.add_subplot(gs[r, 0]), basis, "study", f"{rlab} — by dataset")
    panel(fig.add_subplot(gs[r, 1]), basis, "celltype", f"{rlab} — by cell type")
    panel(fig.add_subplot(gs[r, 2]), basis, "lineage", f"{rlab} — by lineage")

# ---- dataset legend ----
axd = fig.add_axes([0.735, 0.80, 0.26, 0.12]); axd.axis("off")
axd.text(0, 1, "dataset", fontsize=12.5, va="top", fontweight="bold")
axd.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor=col, markersize=12, label=lab)
                    for _, (col, lab) in STUDY.items()], loc="upper left", bbox_to_anchor=(0, 0.88),
           frameon=False, fontsize=10, labelspacing=1.2, handletextpad=0.5)

# ---- grouped cell-type legend (under lineage classifications) ----
axl = fig.add_axes([0.735, 0.04, 0.26, 0.72]); axl.axis("off"); axl.set_xlim(0, 1); axl.set_ylim(0, 1)
axl.text(0, 1.0, "cell type — grouped by lineage", fontsize=12.5, va="top", fontweight="bold")
y = 0.955; dy = 1.0 / (len(cts_present) + len(LIN_ORDER) + 2)
for L in LIN_ORDER:
    cs = ct_by_lin[L]
    if not cs: continue
    axl.add_patch(Rectangle((0.0, y - 0.012), 0.03, 0.02, color=LINCOL[L], transform=axl.transAxes, clip_on=False))
    axl.text(0.05, y, L, fontsize=10.2, fontweight="bold", color=LINCOL[L], va="center", transform=axl.transAxes)
    y -= dy * 1.05
    for c in cs:
        axl.add_patch(Rectangle((0.06, y - 0.010), 0.024, 0.017, color=ctcol[c], transform=axl.transAxes, clip_on=False))
        axl.text(0.10, y, c, fontsize=8.3, va="center", transform=axl.transAxes)
        y -= dy
    y -= dy * 0.25

fig.suptitle("SMG + SLG + Parotid reference — integrated UMAP by dataset, cell type, and lineage",
             fontsize=15, fontweight="bold", y=0.975)
for ext in ["png", "svg", "pdf"]:
    fig.savefig(f"{OUT}/parotid_ref_umap.{ext}", dpi=160 if ext == "png" else None, bbox_inches="tight", facecolor="white")
print("wrote parotid_ref_umap.png/svg/pdf (3 colourings, lineage-grouped legend)", flush=True)
