#!/usr/bin/env python
"""Stacked ('piled') composition bars per gland (RPG / SMG / SLG), two panels:
(A) overall lineage level, (B) cell-type level (coloured by lineage shade).
From the decontaminated parotid-augmented cell2location proportions."""
import os, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
matplotlib.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.colors import to_rgb

OUT = "results/figures_parotid"; os.makedirs(OUT, exist_ok=True)
GLANDS = ["SLG", "RPG", "SMG"]

# ---- composition table (mean proportion %) ----
roi = pd.read_csv("roi_myregion.csv"); rpgbc = set(roi[roi.Gland == "RPG"]["Barcode"])
A = pd.read_csv("results/A1_combined_parotid/A1_cell_type_proportions.csv", index_col=0)
D = pd.read_csv("results/D1_combined_parotid/D1_cell_type_proportions.csv", index_col=0)
gl = pd.read_csv("results/D1_cluster/D1_spot_glands.csv").set_index("barcode")["gland"]
comp = pd.DataFrame({
    "RPG": A.loc[[b for b in A.index if b in rpgbc]].mean() * 100,
    "SMG": D.loc[[b for b in D.index if b in set(gl[gl == "SMG"].index)]].mean() * 100,
    "SLG": D.loc[[b for b in D.index if b in set(gl[gl == "SLG"].index)]].mean() * 100,
}).fillna(0)

LIN = {
 "Acinar-Serous": ["Bpifa2+", "Serous acinar (mix)", "Acinar", "Seromucous acinar (mix)", "Serous acinar (parotid)"],
 "Acinar-Mucous": ["Mucous acinar (SLG)", "Smgc+"],
 "Ductal": ["Ductal (mix)", "GCT", "Striated duct", "Basal duct", "Intercalated duct", "Ascl3+ duct", "Ductal (parotid)"],
 "Myoepithelial": ["Myoepithelial (mix)", "Myoepithelial"],
 "Stromal/Mesench": ["Mesenchymal (parotid)", "Mesenchymal (mix)", "Stromal"],
 "Endothelial/Vasc": ["Endothelial (parotid)", "Endothelial", "Vascular (mix)"],
 "Immune": ["Macrophages", "Immune (parotid)", "NK cells", "Immune (mix)"],
 "Others": ["Erythroid"],
}
from rename_celltypes import rename_lineage
LIN = rename_lineage(LIN)
LINCOL = {"Acinar-Serous": "#E15759", "Acinar-Mucous": "#4E79A7", "Ductal": "#59A14F",
          "Myoepithelial": "#76B7B2", "Stromal/Mesench": "#B07AA1", "Endothelial/Vasc": "#EDC948",
          "Immune": "#F28E2B", "Others": "#BAB0AC"}
lin_order = list(LIN)

# lineage-level table
lintab = pd.DataFrame({L: comp.reindex([c for c in cs if c in comp.index]).sum()
                       for L, cs in LIN.items()}).T.reindex(lin_order)

def shade(hexc, f):  # blend toward white by fraction f
    r = np.array(to_rgb(hexc)); return tuple(r * (1 - f) + np.array([1, 1, 1]) * f)

# ---- figure ----
fig = plt.figure(figsize=(15, 8.5))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.15], wspace=0.55)
x = np.arange(len(GLANDS)); W = 0.62

# Panel A: lineage
axA = fig.add_subplot(gs[0])
bottom = np.zeros(len(GLANDS))
for L in lin_order:
    vals = lintab.loc[L, GLANDS].values.astype(float)
    axA.bar(x, vals, W, bottom=bottom, color=LINCOL[L], edgecolor="white", linewidth=0.6)
    for xi, (v, b) in enumerate(zip(vals, bottom)):
        if v >= 3:
            axA.text(xi, b + v / 2, f"{v:.0f}", ha="center", va="center", fontsize=8.5,
                     color="white" if L in ("Acinar-Serous", "Acinar-Mucous", "Ductal", "Stromal/Mesench") else "#222",
                     fontweight="bold")
    bottom += vals
axA.set_xticks(x); axA.set_xticklabels(GLANDS, fontsize=12, fontweight="bold")
axA.set_ylabel("% of composition (mean over spots)", fontsize=11)
axA.set_ylim(0, 100); axA.set_title("A  By lineage", fontsize=13, fontweight="bold", loc="left")
axA.legend(handles=[Patch(facecolor=LINCOL[L], label=L) for L in lin_order],
           bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False, fontsize=9.5, title="lineage")
for s in ["top", "right"]: axA.spines[s].set_visible(False)

# Panel B: cell type (shaded by lineage)
axB = fig.add_subplot(gs[1])
bottom = np.zeros(len(GLANDS)); handles = []
for L in lin_order:
    cts = [c for c in LIN[L] if c in comp.index]
    n = len(cts)
    for i, c in enumerate(cts):
        col = shade(LINCOL[L], 0.10 + 0.62 * (i / max(n - 1, 1)))
        vals = comp.loc[c, GLANDS].values.astype(float)
        axB.bar(x, vals, W, bottom=bottom, color=col, edgecolor="white", linewidth=0.4)
        for xi, (v, b) in enumerate(zip(vals, bottom)):
            if v >= 4:
                axB.text(xi, b + v / 2, f"{v:.0f}", ha="center", va="center", fontsize=7,
                         color="white" if (i == 0 and L in ("Acinar-Serous", "Acinar-Mucous", "Ductal")) else "#222")
        bottom += vals
        handles.append(Patch(facecolor=col, label=c))
axB.set_xticks(x); axB.set_xticklabels(GLANDS, fontsize=12, fontweight="bold")
axB.set_ylim(0, 100); axB.set_title("B  By cell type", fontsize=13, fontweight="bold", loc="left")
axB.legend(handles=handles, bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False,
           fontsize=7.6, ncol=1, labelspacing=0.28, title="cell type (grouped by lineage)")
for s in ["top", "right"]: axB.spines[s].set_visible(False)

fig.suptitle("RPG vs SMG vs SLG — cell composition (decontaminated parotid-augmented deconvolution)",
             fontsize=14, fontweight="bold", y=0.99)
for ext in ["png", "svg", "pdf"]:
    fig.savefig(f"{OUT}/composition_stacked_RPG_SMG_SLG.{ext}", dpi=200 if ext == "png" else None,
                bbox_inches="tight", facecolor="white")
print("wrote composition_stacked_RPG_SMG_SLG.png/svg/pdf", flush=True)
