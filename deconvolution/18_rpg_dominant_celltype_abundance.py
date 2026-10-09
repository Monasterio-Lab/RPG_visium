#!/usr/bin/env python
"""(1) Dominant cell type per RPG spot on the H&E, and (2) the per-cell-type
abundance grid for the RPG ROI — both from the parotid-augmented deconvolution."""
import os, re, json, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np, pandas as pd, scanpy as sc
from scipy.spatial import cKDTree
from PIL import Image; Image.MAX_IMAGE_PIXELS = None
import matplotlib; matplotlib.use("Agg")
matplotlib.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection

RES = "results/A1_combined_parotid"; SPDIR = "VISIUM_outs/V10S15-395_A1/spatial"
OUT = "results/figures_parotid"; os.makedirs(OUT, exist_ok=True)
BC_RE = re.compile(r"^[ACGT]{8,}-\d+$")

prop = pd.read_csv(f"{RES}/A1_cell_type_proportions.csv", index_col=0)
abund = pd.read_csv(f"{RES}/A1_cell_abundances_q05.csv", index_col=0)
roi = pd.read_csv("roi_myregion.csv")
bcc = next(c for c in roi.columns if roi[c].astype(str).str.match(BC_RE).mean() > 0.8)
ROI_BC = set(b if "-" in str(b) else f"{b}-1" for b in roi[bcc].astype(str).str.strip())

ad = sc.read_h5ad(f"{RES}/A1_cell2location.h5ad")
sf = json.load(open(f"{SPDIR}/scalefactors_json.json"))
HE = np.asarray(Image.open(f"{SPDIR}/tissue_hires_image.png").convert("RGB")); H_hi, W_hi = HE.shape[:2]
lo = Image.open(f"{SPDIR}/tissue_lowres_image.png")
fw = lo.size[0] / sf["tissue_lowres_scalef"]; fh = lo.size[1] / sf["tissue_lowres_scalef"]
sx, sy = W_hi / fw, H_hi / fh
XY = ad.obsm["spatial"] * np.array([sx, sy])
in_roi = np.array([b in ROI_BC for b in ad.obs_names]); idx = np.where(in_roi)[0]
bcs = ad.obs_names[idx]; xy = XY[idx]

# ---------- (1) dominant cell type per spot ----------
dom = prop.loc[bcs].idxmax(axis=1)
present = dom.value_counts()
cats = present.index.tolist()
pal = (list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("tab20b").colors)
       + list(plt.get_cmap("tab20c").colors))
cmap = {c: pal[i % len(pal)] for i, c in enumerate(cats)}
pad = 6 * np.median(np.diff(np.sort(xy[:, 0])) + 1)
x0, x1 = xy[:, 0].min() - 40, xy[:, 0].max() + 40
y0, y1 = xy[:, 1].min() - 40, xy[:, 1].max() + 40
fig, ax = plt.subplots(figsize=(11, 9))
cx0, cx1 = max(int(x0), 0), min(int(x1), W_hi); cy0, cy1 = max(int(y0), 0), min(int(y1), H_hi)
ax.imshow(HE[cy0:cy1, cx0:cx1], extent=[cx0, cx1, cy1, cy0])
for (gx, gy), b in zip(xy, bcs):
    ax.add_patch(plt.Circle((gx, gy), 26, color=cmap[dom[b]], ec="white", lw=0.4))
ax.set_xlim(x0, x1); ax.set_ylim(y1, y0); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values(): s.set_visible(False)
ax.set_title("Dominant cell type per RPG spot (parotid-augmented deconvolution)", fontsize=13)
h = [Line2D([0], [0], marker="o", color="w", markerfacecolor=cmap[c], markeredgecolor="white",
            markersize=10, label=f"{c}  ({present[c]})") for c in cats]
ax.legend(handles=h, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=9,
          title="dominant cell type (n spots)")
for ext in ["png", "svg", "pdf"]:
    fig.savefig(f"{OUT}/RPG_dominant_celltype.{ext}", dpi=250 if ext == "png" else None,
                bbox_inches="tight", facecolor="white")
plt.close(); print("dominant types in RPG:", cats, flush=True)

# ---------- (1b) dominant cell type — NO H&E, grey context spots ----------
radius = 26   # original dot size (matches the H&E dominant map), non-overlapping
padc = 0.35 * max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]))
gx0, gx1 = xy[:, 0].min() - padc, xy[:, 0].max() + padc
gy0, gy1 = xy[:, 1].min() - padc, xy[:, 1].max() + padc
inwin = (XY[:, 0] >= gx0) & (XY[:, 0] <= gx1) & (XY[:, 1] >= gy0) & (XY[:, 1] <= gy1)
ctx_xy = XY[inwin & ~in_roi]
fig, ax = plt.subplots(figsize=(11, 9)); ax.set_facecolor("white")
ax.add_collection(PatchCollection([Circle((cx, cy), radius) for cx, cy in ctx_xy],
                                  facecolor="#dcdcdc", edgecolor="none"))
for (gx, gy), b in zip(xy, bcs):
    ax.add_patch(Circle((gx, gy), radius, facecolor=cmap[dom[b]], edgecolor="white", lw=0.4))
ax.set_xlim(gx0, gx1); ax.set_ylim(gy1, gy0); ax.set_aspect("equal")
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values(): s.set_visible(False)
ax.set_title("Dominant cell type per RPG spot", fontsize=18, fontweight="bold")
leg = ax.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor=cmap[c],
          markeredgecolor="white", markersize=17, label=f"{c}  ({present[c]})") for c in cats],
          loc="upper right", frameon=True, framealpha=0.72, facecolor="white", edgecolor="none",
          fontsize=16, title="dominant cell type (n spots)", title_fontsize=17,
          labelspacing=0.55, handletextpad=0.5, borderpad=0.8)
leg.set_zorder(10)
for ext in ["png", "svg", "pdf"]:
    fig.savefig(f"{OUT}/RPG_dominant_celltype_nobg.{ext}", dpi=200 if ext == "png" else None,
                bbox_inches="tight", facecolor="white")
plt.close(); print("wrote RPG_dominant_celltype_nobg.*", flush=True)

# ---------- (2) abundance grid (ROI only) — manual render, SAME orientation as H&E ----------
cts = list(abund.columns)
radius = 0.62 * np.median(cKDTree(XY).query(XY, k=2)[0][:, 1])  # true-size spots (bigger dots)
ncol = 5; nrow = int(np.ceil(len(cts) / ncol))
figg, axes = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 3.5 * nrow))
axes = np.atleast_1d(axes).ravel()
for axx, c in zip(axes, cts):
    axx.imshow(HE[cy0:cy1, cx0:cx1], extent=[cx0, cx1, cy1, cy0])
    v = abund.loc[bcs, c].values
    vmax = np.percentile(v, 99) if (v > 0).any() else 1.0
    pc = PatchCollection([Circle((gx, gy), radius) for gx, gy in xy], cmap="magma")
    pc.set_array(v); pc.set_clim(0, max(vmax, 1e-6)); pc.set_edgecolor("none")
    axx.add_collection(pc)
    axx.set_xlim(x0, x1); axx.set_ylim(y1, y0); axx.set_aspect("equal")
    axx.set_xticks([]); axx.set_yticks([])
    for s in axx.spines.values(): s.set_visible(False)
    axx.set_title(c, fontsize=9)
    figg.colorbar(pc, ax=axx, fraction=0.046, pad=0.02).ax.tick_params(labelsize=6)
for axx in axes[len(cts):]: axx.axis("off")
figg.suptitle("RPG ROI — per-cell-type abundance (parotid-augmented deconvolution)",
              fontsize=14, fontweight="bold", y=0.995)
plt.tight_layout()
figg.savefig(f"{OUT}/RPG_abundance_grid.png", dpi=150, bbox_inches="tight", facecolor="white")
figg.savefig(f"{OUT}/RPG_abundance_grid.svg", bbox_inches="tight", facecolor="white")
figg.savefig(f"{OUT}/RPG_abundance_grid.pdf", bbox_inches="tight", facecolor="white")
plt.close("all")

# ---------- (2b) same grid WITHOUT H&E: grey context dots + coloured ROI ----------
padc = 0.35 * max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]))
gx0, gx1 = xy[:, 0].min() - padc, xy[:, 0].max() + padc
gy0, gy1 = xy[:, 1].min() - padc, xy[:, 1].max() + padc
inwin = (XY[:, 0] >= gx0) & (XY[:, 0] <= gx1) & (XY[:, 1] >= gy0) & (XY[:, 1] <= gy1)
ctx_xy = XY[inwin & ~in_roi]                       # surrounding (non-ROI) spots
figh, axesh = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 3.5 * nrow))
axesh = np.atleast_1d(axesh).ravel()
for axx, c in zip(axesh, cts):
    axx.set_facecolor("white")
    axx.add_collection(PatchCollection([Circle((gx, gy), radius) for gx, gy in ctx_xy],
                                       facecolor="#dcdcdc", edgecolor="none"))
    v = abund.loc[bcs, c].values
    vmax = np.percentile(v, 99) if (v > 0).any() else 1.0
    pc = PatchCollection([Circle((gx, gy), radius) for gx, gy in xy], cmap="magma")
    pc.set_array(v); pc.set_clim(0, max(vmax, 1e-6)); pc.set_edgecolor("none")
    axx.add_collection(pc)
    axx.set_xlim(gx0, gx1); axx.set_ylim(gy1, gy0); axx.set_aspect("equal")
    axx.set_xticks([]); axx.set_yticks([])
    for s in axx.spines.values(): s.set_visible(False)
    axx.set_title(c, fontsize=9)
    figh.colorbar(pc, ax=axx, fraction=0.046, pad=0.02).ax.tick_params(labelsize=6)
for axx in axesh[len(cts):]: axx.axis("off")
figh.suptitle("RPG ROI — per-cell-type abundance (no H&E; grey = surrounding spots)",
              fontsize=14, fontweight="bold", y=0.995)
plt.tight_layout()
figh.savefig(f"{OUT}/RPG_abundance_grid_nobg.png", dpi=150, bbox_inches="tight", facecolor="white")
figh.savefig(f"{OUT}/RPG_abundance_grid_nobg.svg", bbox_inches="tight", facecolor="white")
figh.savefig(f"{OUT}/RPG_abundance_grid_nobg.pdf", bbox_inches="tight", facecolor="white")
plt.close("all")
print(f"wrote RPG_dominant_celltype.*, RPG_abundance_grid.* and _nobg.* ({len(cts)} types)", flush=True)
