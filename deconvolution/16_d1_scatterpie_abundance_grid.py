#!/usr/bin/env python
"""D1 (SMG+SLG) Visium — abundance grid + lineage scatterpie from the
parotid-augmented deconvolution (results/D1_combined_parotid). Same aesthetics
as the RPG/A1 figures. SMG/SLG regions labelled from D1_spot_glands.csv."""
import os, re, json, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np, pandas as pd, scanpy as sc
from scipy.spatial import cKDTree
from PIL import Image; Image.MAX_IMAGE_PIXELS = None
import matplotlib; matplotlib.use("Agg")
matplotlib.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patheffects as pe
from matplotlib.patches import Wedge, Patch, Circle
from matplotlib.collections import PatchCollection
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

RES = "results/D1_combined_parotid"
SPDIR = "SMG:SLG_Outs/D1/outs/spatial"
OUT = "results/figures_parotid"; os.makedirs(OUT, exist_ok=True)
SPOT_UM = 55.0

LINEAGE = {
 "Acinar-Serous": ["Bpifa2+", "Serous acinar (mix)", "Acinar", "Seromucous acinar (mix)", "Serous acinar (parotid)"],
 "Acinar-Mucous": ["Mucous acinar (SLG)", "Smgc+"],
 "Ductal": ["Ductal (mix)", "Striated duct", "Basal duct", "Intercalated duct", "Ascl3+ duct", "GCT",
            "Myoepithelial", "Myoepithelial (mix)", "Ductal (parotid)"],
 "Stromal/Mesench": ["Mesenchymal (mix)", "Stromal", "Mesenchymal (parotid)", "Endothelial",
                     "Vascular (mix)", "Endothelial (parotid)"],
 "Immune": ["Immune (mix)", "Macrophages", "NK cells", "Immune (parotid)"],
}
from rename_celltypes import rename_lineage
LINEAGE = rename_lineage(LINEAGE)
CATS = ["Acinar-Serous", "Acinar-Mucous", "Ductal", "Stromal/Mesench", "Immune", "Others"]
COL = dict(zip(CATS, ["#E15759", "#4E79A7", "#59A14F", "#B07AA1", "#F28E2B", "#CFCFCF"]))

prop = pd.read_csv(f"{RES}/D1_cell_type_proportions.csv", index_col=0)
abund = pd.read_csv(f"{RES}/D1_cell_abundances_q05.csv", index_col=0)
ad = sc.read_h5ad(f"{RES}/D1_cell2location.h5ad")
assigned = sum(LINEAGE.values(), [])
lin = pd.DataFrame({g: prop[[c for c in m if c in prop.columns]].sum(axis=1) for g, m in LINEAGE.items()},
                   index=prop.index)
lin["Others"] = prop.drop(columns=[c for c in assigned if c in prop.columns]).sum(axis=1)
lin = lin[CATS].div(lin[CATS].sum(axis=1), axis=0)

sf = json.load(open(f"{SPDIR}/scalefactors_json.json"))
HE = np.asarray(Image.open(f"{SPDIR}/tissue_hires_image.png").convert("RGB")); H_hi, W_hi = HE.shape[:2]
lo = Image.open(f"{SPDIR}/tissue_lowres_image.png")
fw = lo.size[0] / sf["tissue_lowres_scalef"]; fh = lo.size[1] / sf["tissue_lowres_scalef"]
sx, sy = W_hi / fw, H_hi / fh
XY = ad.obsm["spatial"] * np.array([sx, sy])
UM_PER_PX = (SPOT_UM / sf["spot_diameter_fullres"]) / sx
bcs = ad.obs_names
radius = 0.5 * np.median(cKDTree(XY).query(XY, k=2)[0][:, 1])

# gland labels
gl = pd.read_csv("results/D1_cluster/D1_spot_glands.csv").set_index("barcode")["gland"].reindex(bcs)
x0, x1 = XY[:, 0].min() - 8 * radius, XY[:, 0].max() + 8 * radius
y0, y1 = XY[:, 1].min() - 8 * radius, XY[:, 1].max() + 8 * radius
cx0, cx1 = max(int(x0), 0), min(int(x1), W_hi); cy0, cy1 = max(int(y0), 0), min(int(y1), H_hi)

# ---------- scatterpie ----------
fig = plt.figure(figsize=(15, 12)); ax = fig.add_axes([0.02, 0.03, 0.73, 0.92])
ax.imshow(HE[cy0:cy1, cx0:cx1], extent=[cx0, cx1, cy1, cy0])
for (gx, gy), b in zip(XY, bcs):
    ang = 90.0; m = lin.loc[b, CATS].values; m = m / m.sum()
    for ci, c in enumerate(CATS):
        f = m[ci]
        if f <= 0: continue
        t2 = ang - f * 360.0
        ax.add_patch(Wedge((gx, gy), radius, t2, ang, facecolor=COL[c], edgecolor="white", linewidth=0.08))
        ang = t2
for g, col in [("SMG", "#1b1b1b"), ("SLG", "#1b1b1b")]:
    m = (gl == g).to_numpy()
    if m.any():
        ax.text(XY[m, 0].mean(), XY[m, 1].mean(), g, fontsize=22, fontweight="bold",
                color="white", ha="center", va="center",
                path_effects=[pe.withStroke(linewidth=3, foreground="black")])
ax.set_xlim(x0, x1); ax.set_ylim(y1, y0); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values(): s.set_visible(False)
ax.set_title("D1 (SMG + SLG) — per-spot lineage composition (parotid-augmented deconvolution)", fontsize=14)
ax.legend(handles=[Patch(facecolor=COL[c], label=c) for c in CATS], loc="upper left",
          bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=11, title="lineage")
ax.add_artist(AnchoredSizeBar(ax.transData, 500 / UM_PER_PX, "500 µm", "lower right", pad=0.3,
              color="white", frameon=False, size_vertical=radius * 0.3, fontproperties=fm.FontProperties(size=12)))

# ---- magnified representative pies: SLG (top-right) + SMG (bottom-left) ----
from matplotlib.patches import ConnectionPatch
def comp(bc):
    v = lin.loc[bc, CATS].values; return v / v.sum()
def rep_mean(gland):        # central spot closest to the gland mean composition (SLG)
    idx = np.where((gl == gland).to_numpy())[0]
    d = np.linalg.norm(XY[idx] - XY[idx].mean(0), axis=1); central = idx[d <= np.percentile(d, 50)]
    Lall = lin.loc[bcs[idx], CATS].values; mean = (Lall / Lall.sum(1, keepdims=True)).mean(0)
    Lc = lin.loc[bcs[central], CATS].values; Lc = Lc / Lc.sum(1, keepdims=True)
    j = central[np.argmin(np.linalg.norm(Lc - mean, axis=1))]; return bcs[j], XY[j]
def subzone(gland, score, frac=0.30):   # representative of the highest-`score` sub-area
    idx = np.where((gl == gland).to_numpy())[0]
    s = score(lin.loc[bcs[idx], CATS]); k = max(3, int(len(idx) * frac))
    top = idx[np.argsort(s)[-k:]]; j = top[np.linalg.norm(XY[top] - XY[top].mean(0), axis=1).argmin()]
    return bcs[j], XY[j]
# SLG: one representative (top-right).  SMG: two sub-zones (duct-rich + acinar-serous-rich), bottom-left.
picks = [("SLG — representative", *rep_mean("SLG"), [0.71, 0.71, 0.27, 0.27]),
         ("SMG — acinar-serous-rich", *subzone("SMG", lambda L: (L["Acinar-Serous"] - L["Ductal"]).values), [0.02, 0.28, 0.22, 0.22]),
         ("SMG — duct-rich", *subzone("SMG", lambda L: (L["Ductal"] - L["Acinar-Serous"]).values), [0.02, 0.03, 0.22, 0.22])]
for label, bc, xy, loc in picks:
    ax.add_patch(Circle(tuple(xy), radius * 1.9, fill=False, ec="white", lw=2.6, zorder=6))
    axp = ax.inset_axes(loc)
    axp.pie(comp(bc), colors=[COL[c] for c in CATS], startangle=90, counterclock=False,
            wedgeprops=dict(edgecolor="white", linewidth=1.3))
    axp.add_patch(Circle((0, 0), 1.06, facecolor="white", edgecolor="#333", lw=1.4, zorder=0))
    axp.set_title(label, fontsize=11.5, fontweight="bold")
    fig.add_artist(ConnectionPatch(xyA=tuple(xy), coordsA=ax.transData, xyB=(0, 0), coordsB=axp.transData,
                                   lw=2.2, ls=(0, (5, 4)), color="white", zorder=5))
for ext in ["png", "svg", "pdf"]:
    fig.savefig(f"{OUT}/D1_scatterpie_parotid.{ext}", dpi=300 if ext == "png" else None,
                bbox_inches="tight", facecolor="white")
plt.close(); print("wrote D1_scatterpie_parotid.*", flush=True)

# ---------- abundance grid (H&E + no-bg) ----------
cts = list(abund.columns); ncol = 5; nrow = int(np.ceil(len(cts) / ncol))
def grid(bg, tag):
    fg, axes = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 3.5 * nrow)); axes = np.atleast_1d(axes).ravel()
    for axx, c in zip(axes, cts):
        if bg: axx.imshow(HE[cy0:cy1, cx0:cx1], extent=[cx0, cx1, cy1, cy0])
        else: axx.set_facecolor("white")
        v = abund.loc[bcs, c].values; vmax = np.percentile(v, 99) if (v > 0).any() else 1.0
        pc = PatchCollection([Circle((gx, gy), radius) for gx, gy in XY], cmap="magma")
        pc.set_array(v); pc.set_clim(0, max(vmax, 1e-6)); pc.set_edgecolor("none")
        axx.add_collection(pc)
        axx.set_xlim(x0, x1); axx.set_ylim(y1, y0); axx.set_aspect("equal")
        axx.set_xticks([]); axx.set_yticks([])
        for s in axx.spines.values(): s.set_visible(False)
        axx.set_title(c, fontsize=9)
        fg.colorbar(pc, ax=axx, fraction=0.046, pad=0.02).ax.tick_params(labelsize=6)
    for axx in axes[len(cts):]: axx.axis("off")
    fg.suptitle(f"D1 (SMG+SLG) — per-cell-type abundance{'' if bg else ' (no H&E)'}",
                fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    for ext in ["png", "svg", "pdf"]:
        fg.savefig(f"{OUT}/{tag}.{ext}", dpi=150 if ext == "png" else None, bbox_inches="tight", facecolor="white")
    plt.close()
grid(True, "D1_abundance_grid")
grid(False, "D1_abundance_grid_nobg")
print(f"wrote D1_abundance_grid.* and _nobg.* ({len(cts)} types)", flush=True)
print("D1 mean proportion top:", lin.mean().round(3).to_dict(), flush=True)
