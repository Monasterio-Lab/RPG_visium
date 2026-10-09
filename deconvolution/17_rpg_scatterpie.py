#!/usr/bin/env python
"""Per-spot lineage pie-charts (scatterpie) over the A1 H&E for the RPG ROI,
using the PAROTID-augmented deconvolution. Identifies the 3 RPG glandular
regions (KMeans), and zooms 4 representative spots into enlarged pies:
upper-centre, upper-peripheral, centre region, distal (immune-rich) region.
Lineage grouping per the requested legend (Acinar-Serous/Mucous, Ductal,
Stromal/Mesench, Immune, Others)."""
import os, re, json, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np, pandas as pd, scanpy as sc
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from PIL import Image; Image.MAX_IMAGE_PIXELS = None
import matplotlib; matplotlib.use("Agg")
matplotlib.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Wedge, Patch, ConnectionPatch
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

RES = "results/A1_combined_parotid"
SPDIR = "VISIUM_outs/V10S15-395_A1/spatial"
OUT = "results/figures_parotid"; os.makedirs(OUT, exist_ok=True)
SPOT_UM = 55.0; BC_RE = re.compile(r"^[ACGT]{8,}-\d+$")

# 28 reference cell types -> 6 lineages
LINEAGE = {
 "Acinar-Serous": ["Bpifa2+", "Serous acinar (mix)", "Acinar", "Seromucous acinar (mix)",
                   "Serous acinar (parotid)"],
 "Acinar-Mucous": ["Mucous acinar (SLG)", "Smgc+"],
 "Ductal": ["Ductal (mix)", "Striated duct", "Basal duct", "Intercalated duct", "Ascl3+ duct",
            "GCT", "Myoepithelial", "Myoepithelial (mix)", "Ductal (parotid)"],
 "Stromal/Mesench": ["Mesenchymal (mix)", "Stromal", "Mesenchymal (parotid)", "Endothelial",
                     "Vascular (mix)", "Endothelial (parotid)"],
 "Immune": ["Immune (mix)", "Macrophages", "NK cells", "Immune (parotid)"],
}
from rename_celltypes import rename_lineage
LINEAGE = rename_lineage(LINEAGE)
CATS = ["Acinar-Serous", "Acinar-Mucous", "Ductal", "Stromal/Mesench", "Immune", "Others"]
COL = dict(zip(CATS, ["#E15759", "#4E79A7", "#59A14F", "#B07AA1", "#F28E2B", "#CFCFCF"]))

prop = pd.read_csv(f"{RES}/A1_cell_type_proportions.csv", index_col=0)
assigned = sum(LINEAGE.values(), [])
lin = pd.DataFrame({g: prop[[c for c in m if c in prop.columns]].sum(axis=1) for g, m in LINEAGE.items()},
                   index=prop.index)
lin["Others"] = prop.drop(columns=[c for c in assigned if c in prop.columns]).sum(axis=1)
lin = lin[CATS].div(lin[CATS].sum(axis=1), axis=0)

roi = pd.read_csv("roi_myregion.csv")
bcc = next(c for c in roi.columns if roi[c].astype(str).str.match(BC_RE).mean() > 0.8)
ROI_BC = set(b if "-" in str(b) else f"{b}-1" for b in roi[bcc].astype(str).str.strip())

ad = sc.read_h5ad(f"{RES}/A1_cell2location.h5ad")
sf = json.load(open(f"{SPDIR}/scalefactors_json.json"))
HE = np.asarray(Image.open(f"{SPDIR}/tissue_hires_image.png").convert("RGB"))
H_hi, W_hi = HE.shape[:2]
lo = Image.open(f"{SPDIR}/tissue_lowres_image.png")
fullres_w = lo.size[0] / sf["tissue_lowres_scalef"]; fullres_h = lo.size[1] / sf["tissue_lowres_scalef"]
sx, sy = W_hi / fullres_w, H_hi / fullres_h
XY = ad.obsm["spatial"] * np.array([sx, sy])
UM_PER_PX = (SPOT_UM / sf["spot_diameter_fullres"]) / sx
in_roi = np.array([b in ROI_BC for b in ad.obs_names])
roi_idx = np.where(in_roi)[0]
roi_xy = XY[roi_idx]
roi_bcs = ad.obs_names[roi_idx]

# --- 3 RPG regions via KMeans on spot coords ---
km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(roi_xy)
reg = km.labels_
sizes = pd.Series(reg).value_counts()
big = sizes.idxmax()                                   # largest = Upper
others = [r for r in sizes.index if r != big]
imm = {r: lin.loc[roi_bcs[reg == r], "Immune"].mean() for r in others}
distal = max(imm, key=imm.get); centre = [r for r in others if r != distal][0]
RNAME = {big: "Upper (largest)", centre: "Centre", distal: "Distal (immune-rich)"}
print("region sizes:", sizes.to_dict(), "| immune frac:", {RNAME[r]: round(v,3) for r,v in imm.items()}, flush=True)

def comp(bc):
    v = lin.loc[bc, CATS].values; return v / v.sum()
def pick_centre(region):
    idx = np.where(reg == region)[0]; pts = roi_xy[idx]; c = pts.mean(0)
    j = idx[np.linalg.norm(pts - c, axis=1).argmin()]
    return roi_bcs[j], roi_xy[j]
def pick_subzone(region, score, frac=0.30):
    """representative spot of the highest-`score` sub-zone WITHIN a region
    (centroid of the top-frac spots -> nearest spot, so it sits inside)."""
    idx = np.where(reg == region)[0]
    s = score(lin.loc[roi_bcs[idx]])
    k = max(3, int(len(idx) * frac))
    top = idx[np.argsort(s)[-k:]]
    c = roi_xy[top].mean(0)
    j = top[np.linalg.norm(roi_xy[top] - c, axis=1).argmin()]
    return roi_bcs[j], roi_xy[j]

# the two clear sub-zones inside the big upper region: mucous-acinar core & ductal-rich
mucous = pick_subzone(big, lambda L: (L["Acinar-Mucous"] - L["Ductal"]).values)
# ductal zoom = the user-selected ductal spot (green-circled, lower-centre of upper region)
_dbc = "ACTACCAGCTCTCTGG-1"; _di = int(np.where(roi_bcs == _dbc)[0][0])
ductal = (roi_bcs[_di], roi_xy[_di])
reps = [("Upper — Acinar-Mucous core", *mucous),
        ("Upper — Ductal-rich zone", *ductal),
        ("Centre region", *pick_centre(centre)),
        ("Distal (immune-rich)", *pick_centre(distal))]

# ---------- figure ----------
radius = 0.46 * np.median(cKDTree(XY).query(XY, k=2)[0][:, 1])
pad = 6 * radius
x0, x1 = roi_xy[:, 0].min() - pad, roi_xy[:, 0].max() + pad
y0, y1 = roi_xy[:, 1].min() - pad, roi_xy[:, 1].max() + pad
fig = plt.figure(figsize=(17, 11))
axm = fig.add_axes([0.03, 0.05, 0.60, 0.9])
cx0, cx1 = max(int(x0), 0), min(int(x1), W_hi); cy0, cy1 = max(int(y0), 0), min(int(y1), H_hi)
axm.imshow(HE[cy0:cy1, cx0:cx1], extent=[cx0, cx1, cy1, cy0])
for gi, bc in zip(roi_idx, roi_bcs):
    x, y = XY[gi]; ang = 90.0; m = comp(bc)
    for ci, c in enumerate(CATS):
        f = m[ci]
        if f <= 0: continue
        t2 = ang - f * 360.0
        axm.add_patch(Wedge((x, y), radius, t2, ang, facecolor=COL[c], edgecolor="white", linewidth=0.15))
        ang = t2
axm.set_xlim(x0, x1); axm.set_ylim(y1, y0); axm.set_aspect("equal")
axm.set_xticks([]); axm.set_yticks([])
for s in axm.spines.values(): s.set_visible(False)
axm.set_title("RPG region — per-spot lineage composition (parotid-augmented deconvolution)", fontsize=13)
axm.legend(handles=[Patch(facecolor=COL[c], label=c) for c in CATS], loc="lower left",
           bbox_to_anchor=(0.0, 0.0), frameon=True, framealpha=0.85, fontsize=10, title="lineage")
axm.add_artist(AnchoredSizeBar(axm.transData, 500 / UM_PER_PX, "500 µm", "lower right", pad=0.3,
               color="white", frameon=False, size_vertical=radius * 0.3,
               fontproperties=fm.FontProperties(size=11)))
# mark representative spots
for _, bc, xy in reps:
    axm.add_patch(plt.Circle(tuple(xy), radius * 1.5, fill=False, ec="white", lw=2.2))

# ---------- 4 zoomed inset pies on the right, with connectors ----------
ys = [0.74, 0.52, 0.30, 0.08]
for (label, bc, xy), yy in zip(reps, ys):
    axp = fig.add_axes([0.70, yy, 0.18, 0.18])
    m = comp(bc)
    axp.pie(m, colors=[COL[c] for c in CATS], startangle=90, counterclock=False,
            wedgeprops=dict(edgecolor="white", linewidth=1.2))
    axp.set_title(label, fontsize=11, fontweight="bold")
    con = ConnectionPatch(xyA=tuple(xy), coordsA=axm.transData,
                          xyB=(0, 0), coordsB=axp.transData,
                          lw=2, ls=(0, (5, 4)), color="white", zorder=5)
    fig.add_artist(con)
fig.suptitle("RPG lineage composition across the three glandular regions (zoom: representative spots)",
             fontsize=14, fontweight="bold", y=0.985)
for ext in ["png", "svg", "pdf"]:
    fig.savefig(f"{OUT}/RPG_scatterpie_parotid.{ext}", dpi=300 if ext == "png" else None,
                bbox_inches="tight", facecolor="white")
print("representative spots:", [(l, b) for l, b, _ in reps], flush=True)
print("wrote RPG_scatterpie_parotid.png/svg/pdf", flush=True)
