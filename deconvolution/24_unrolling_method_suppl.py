"""Supplementary Fig. 12 (minimalist): how the RPG unrolling was performed.
Left: the A1 H&E with Visium spots at their TRUE size, coloured by the unrolled
coordinate, with the fitted medial axis (dashed) and each spot's perpendicular
projection onto it (thin lines). Right: the unrolled track (per-spot position strip
+ per-gene fraction-of-peak heatmap). """
import os, json, warnings, sys; warnings.filterwarnings("ignore"); sys.path.insert(0, "analysis")
import numpy as np, pandas as pd, scanpy as sc
from sklearn.cluster import DBSCAN
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree
from statsmodels.nonparametric.smoothers_lowess import lowess
from PIL import Image; Image.MAX_IMAGE_PIXELS = None
import matplotlib; matplotlib.use("Agg")
matplotlib.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42,
                            "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"]})
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from visium_io import load_visium_a1

OUT = "results/figures_parotid"; SPDIR = "VISIUM_outs/V10S15-395_A1/spatial"
GREEN, ORANGE, BLUE = "#2E7D32", "#E07B00", "#1F5FBF"
CORE = ["Aqp5", "Gp2", "Muc5b", "Cldn10", "Kcnn4", "Sbpl"]   # Sbpl: flat (1.6x), expressed throughout
PROX = ["Muc19", "Dcpp1", "Dcpp2", "Smgc", "Bpifb2", "Bpifb6", "Wfdc12", "Cgref1"]
DIST = ["Lyz2", "Bpifb1", "Lipf"]

# ---------- unrolled coordinate + perpendicular ----------
# start from the SAME 100 glandularity-cleaned RPG spots used for the pseudo-bulk
# comparison (Suppl. Fig. 10A / Fig. 2J), so the regionalization is coherent with the
# cleaning; DBSCAN then drops the few spatially isolated nasal spots that remain.
a1 = load_visium_a1(); sc.pp.normalize_total(a1, target_sum=1e4); sc.pp.log1p(a1)
gl = sc.read_h5ad("results/glands_glandular/glands_glandular.h5ad")
kept100 = set(b[:-3] if b.endswith("-A1") else b for b in gl.obs_names[gl.obs["gland"] == "RPG"])
inroi = np.array([b in kept100 for b in a1.obs_names])
rbc0 = a1.obs_names[inroi]; rxy0 = a1.obsm["spatial"].astype(float)[inroi]
print(f"starting from {inroi.sum()} glandularity-cleaned RPG spots", flush=True)
nn = np.median(cKDTree(rxy0).query(rxy0, k=2)[0][:, 1])
lab = DBSCAN(eps=1.7 * nn, min_samples=5).fit(rxy0).labels_
cores = list(pd.Series(lab[lab >= 0]).value_counts().index[:3]); cp = {c: rxy0[lab == c] for c in cores}
assign = np.full(len(rxy0), -9)
for i, p in enumerate(rxy0):
    dm, cb = 1e18, -9
    for c in cores:
        d = np.min(np.linalg.norm(cp[c] - p, axis=1))
        if d < dm: dm, cb = d, c
    assign[i] = cb if dm <= 1.7 * nn else -9
keep = assign != -9; rbc = rbc0[keep]; rxy = rxy0[keep]; regid = assign[keep]
prop = pd.read_csv("results/A1_combined_parotid/A1_cell_type_proportions.csv", index_col=0)
imm = [c for c in prop.columns if "immune" in c.lower() or "macrophage" in c.lower() or c.startswith("NK")]
sz = pd.Series(regid).value_counts(); big = sz.idxmax(); oth = [c for c in cores if c != big]
immv = {c: prop.loc[rbc[regid == c], imm].sum(1).mean() for c in oth}
dist = max(oth, key=lambda c: immv[c]); cent = [c for c in oth if c != dist][0]
rname = {big: "Proximal", cent: "Central", dist: "Distal"}; region = np.array([rname[c] for c in regid])
C = {r: rxy[region == r].mean(0) for r in ["Proximal", "Central", "Distal"]}
pp = rxy[region == "Proximal"]; dP = C["Proximal"] - C["Central"]; dP /= np.linalg.norm(dP) + 1e-9
ptip = pp[np.argmax((pp - C["Central"]) @ dP)]
dpp = rxy[region == "Distal"]; dD = C["Distal"] - C["Central"]; dD /= np.linalg.norm(dD) + 1e-9
dtip = dpp[np.argmax((dpp - C["Central"]) @ dD)]
anc = np.array([ptip, C["Proximal"], C["Central"], C["Distal"], dtip])
tck, _ = splprep([anc[:, 0], anc[:, 1]], k=2, s=0); gg = np.linspace(0, 1, 800); cx, cy = splev(gg, tck)
arc = np.concatenate([[0], np.cumsum(np.hypot(np.diff(cx), np.diff(cy)))])
_, idx = cKDTree(np.c_[cx, cy]).query(rxy); s = arc[idx]
if s[region == "Proximal"].mean() > s[region == "Distal"].mean(): s = s.max() - s
foot = np.c_[cx[idx], cy[idx]]; rollc = a1.obsm["spatial"].astype(float).mean(0)
perp = np.sign(((rollc - foot) * (rxy - foot)).sum(1)) * np.linalg.norm(rxy - foot, axis=1) / nn
order_rn = sorted(["Proximal", "Central", "Distal"], key=lambda r: s[region == r].mean())
coord = np.zeros(len(s)); off = 0.0
for r in order_rn:
    m = region == r; coord[m] = off + (s[m] - s[m].min()); off += s[m].max() - s[m].min()
coord /= off
b1, b2 = coord[region == "Proximal"].max(), coord[region == "Central"].max()

def expr(g):
    if g not in a1.var_names: return None
    x = a1[rbc, g].X; return np.asarray(x.todense()).ravel() if hasattr(x, "todense") else np.asarray(x).ravel()

grid = np.linspace(0, 1, 240); rows, labels, lcol, sep = [], [], [], []
for genes, col in [(CORE, GREEN), (PROX, ORANGE), (DIST, BLUE)]:
    prof = {}
    for g in genes:
        v = expr(g)
        if v is None: continue
        sm = lowess(v, coord, frac=0.5, return_sorted=True)
        y = np.interp(grid, sm[:, 0], sm[:, 1]); y = y / (y.max() + 1e-9)
        prof[g] = (y, grid[np.argmax(y)])
    for g in sorted(prof, key=lambda k: prof[k][1]):
        rows.append(prof[g][0]); labels.append(g); lcol.append(col)
    sep.append(len(rows))
Z = np.array(rows); nrow = len(Z)
REGSPAN = [("Proximal", 0, b1, ORANGE), ("Central", b1, b2, GREEN), ("Distal", b2, 1.0, BLUE)]

# ================= FIGURE =================
sf = json.load(open(f"{SPDIR}/scalefactors_json.json"))
HE = np.asarray(Image.open(f"{SPDIR}/tissue_hires_image.png").convert("RGB")); Hh, Wh = HE.shape[:2]
loi = Image.open(f"{SPDIR}/tissue_lowres_image.png")
sx = Wh / (loi.size[0] / sf["tissue_lowres_scalef"]); sy = Hh / (loi.size[1] / sf["tissue_lowres_scalef"])
hxy = rxy * np.array([sx, sy]); footh = foot * np.array([sx, sy])
rad = 0.5 * sf["spot_diameter_fullres"] * sx           # TRUE Visium spot size
pad = 9 * nn * sx
x0, x1 = hxy[:, 0].min() - pad, hxy[:, 0].max() + pad; y0, y1 = hxy[:, 1].min() - pad, hxy[:, 1].max() + pad
cx0, cx1 = max(int(x0), 0), min(int(x1), Wh); cy0, cy1 = max(int(y0), 0), min(int(y1), Hh)

fig = plt.figure(figsize=(18.5, 11))
gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 1.15, 0.02], height_ratios=[0.30, 1.0], wspace=0.10, hspace=0.06)

# ---- (left) H&E: how the unrolling was performed ----
axh = fig.add_subplot(gs[:, 0]); axh.imshow(HE[cy0:cy1, cx0:cx1], extent=[cx0, cx1, cy1, cy0])
axh.plot(np.c_[hxy[:, 0], footh[:, 0], np.full(len(hxy), np.nan)].ravel(),
         np.c_[hxy[:, 1], footh[:, 1], np.full(len(hxy), np.nan)].ravel(),
         color="#101010", lw=1.0, alpha=0.8, zorder=2)                       # perpendicular projections
pc = PatchCollection([Circle((x, y), rad) for x, y in hxy], cmap="viridis")
pc.set_array(coord); pc.set_clim(0, 1); pc.set_edgecolor("white"); pc.set_linewidth(0.2); pc.set_zorder(3)
axh.add_collection(pc)
axh.plot(cx * sx, cy * sy, color="white", lw=2.4, ls="--", zorder=4)
RCOL = {"Proximal": ORANGE, "Central": GREEN, "Distal": BLUE}; RNUM = {"Proximal": "1", "Central": "2", "Distal": "3"}
rollc_h = rollc * np.array([sx, sy])
for r in ["Proximal", "Central", "Distal"]:                                   # numbered badges OUTSIDE the spots
    fx = rxy[region == r] * np.array([sx, sy]); cen = fx.mean(0)
    fr = np.max(np.linalg.norm(fx - cen, axis=1))
    d = cen - rollc_h; d = d / (np.linalg.norm(d) + 1e-9)
    bp = cen + d * (fr + 2.2 * nn * sx)
    axh.plot([cen[0], bp[0]], [cen[1], bp[1]], color=RCOL[r], lw=1.6, zorder=5)
    axh.text(bp[0], bp[1], RNUM[r], ha="center", va="center", fontsize=15, fontweight="bold", color="white",
             bbox=dict(boxstyle="circle,pad=0.3", fc=RCOL[r], ec="white", lw=1.6), zorder=6)
# metric grid + scale bar (Visium spot pitch = 100 um; nn fullres px = 100 um)
spitch = nn * sx                                    # hires px per 100 um
for xg in np.arange(x0, x1, 5 * spitch): axh.axvline(xg, color="white", lw=0.5, alpha=0.35, zorder=1)
for yg in np.arange(y0, y1, 5 * spitch): axh.axhline(yg, color="white", lw=0.5, alpha=0.35, zorder=1)
bx, by = x0 + 0.06 * (x1 - x0), y1 - 0.08 * (y1 - y0)
axh.plot([bx, bx + 5 * spitch], [by, by], color="black", lw=4, solid_capstyle="butt", zorder=7)
axh.text(bx + 2.5 * spitch, by - 0.013 * (y1 - y0), "500 um", ha="center", va="bottom", fontsize=10.5, fontweight="bold", zorder=7)
axh.set_xlim(x0, x1); axh.set_ylim(y1, y0); axh.axis("off")
axh.set_title("How the unrolling was performed  (grid = 500 um)\nglandularity-cleaned spots projected onto the fitted medial axis", fontsize=13, fontweight="bold")
cbx = fig.add_axes([0.045, 0.07, 0.14, 0.015])
cbh = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap="viridis"), cax=cbx, orientation="horizontal")
cbh.set_ticks([0, 1]); cbh.set_ticklabels(["proximal", "distal"]); cbh.set_label("unrolled coordinate", fontsize=10)

# ---- (right-top) per-spot strip (minimal) ----
axs = fig.add_subplot(gs[0, 1])
for rn, lo, hi, c in REGSPAN:
    axs.axvspan(lo, hi, color=c, alpha=0.05, zorder=0)
    axs.text((lo + hi) / 2, 1.04, rn, transform=axs.get_xaxis_transform(), ha="center", va="bottom",
             fontsize=11, color=c, fontweight="bold")
sc_objs = []
for rn, _, _, c in REGSPAN:
    m = region == rn
    sc_objs.append((axs.scatter(coord[m], perp[m], s=16, c=c, edgecolors="white", linewidths=0.3, zorder=3), int(m.sum())))
axs.axhline(0, color="#999", lw=0.8, ls="--")
for b in (b1, b2): axs.axvline(b, color="#444", lw=1.0, ls=(0, (3, 3)))
axs.set_ylim(-6, 6); axs.set_yticks([-4, -2, 0, 2, 4]); axs.set_ylabel("across-axis\n(spot pitch)", fontsize=10)
axs.set_xlim(0, 1); axs.set_xticks(np.arange(0, 1.01, 0.2))
axs.grid(True, color="#dddddd", lw=0.6); axs.set_axisbelow(True)
axs.tick_params(labelbottom=False, labelsize=9)
for sp in ["top"]: axs.spines[sp].set_visible(False)
axs2 = axs.twinx(); axs2.set_ylim(-600, 600); axs2.set_yticks([-400, -200, 0, 200, 400])   # 1 spot pitch = 100 um
axs2.set_ylabel("um", fontsize=9, rotation=-90, labelpad=12); axs2.tick_params(labelsize=8)
axs2.spines["top"].set_visible(False)
# along-axis scale bar: the unrolled coordinate is linear in arc length (1/off per px),
# so 500 um = 5*nn/off in coordinate units (valid within each region)
barw = 5 * nn / off
axs.add_patch(Rectangle((0.02, -6.0), barw + 0.12, 1.7, fc="white", ec="none", alpha=0.85, zorder=4))
axs.plot([0.035, 0.035 + barw], [-5.3, -5.3], color="black", lw=3.5, solid_capstyle="butt", zorder=5)
axs.text(0.035 + barw / 2, -4.7, "500 um", ha="center", va="bottom", fontsize=8.5, fontweight="bold", zorder=5)

# ---- (right-bottom) heatmap (minimal) ----
axg = fig.add_subplot(gs[1, 1], sharex=axs)
im = axg.imshow(Z, aspect="auto", cmap="Spectral_r", extent=[0, 1, nrow, 0], vmin=0, vmax=1, interpolation="nearest")
for xg in np.arange(0.2, 1.0, 0.2): axg.axvline(xg, color="white", lw=0.5, alpha=0.45, zorder=3)   # coordinate grid
for b in (b1, b2): axg.axvline(b, color="#222", lw=1.0, ls=(0, (3, 3)))
for y in sep[:-1]: axg.axhline(y, color="white", lw=2.0)
axg.set_xticks(np.arange(0, 1.01, 0.2))
axg.set_yticks(np.arange(nrow) + 0.5); axg.set_yticklabels(labels, fontsize=11, fontstyle="italic")
for t, c in zip(axg.get_yticklabels(), lcol): t.set_color(c)
grp = {(0, sep[0]): (GREEN, "Core"), (sep[0], sep[1]): (ORANGE, "Proximal"), (sep[1], sep[2]): (BLUE, "Distal")}
for (y0g, y1g), (c, name) in grp.items():
    axg.plot([-0.062, -0.062], [y0g + 0.12, y1g - 0.12], transform=axg.get_yaxis_transform(),
             color=c, lw=3.2, clip_on=False, solid_capstyle="round")
    axg.text(-0.088, (y0g + y1g) / 2, name, transform=axg.get_yaxis_transform(), rotation=90,
             ha="center", va="center", fontsize=10.5, color=c, fontweight="bold")
axg.set_xlim(0, 1); axg.tick_params(labelsize=9)
axg.set_xlabel("unrolled coordinate  (proximal 0 to distal 1)", fontsize=11)
for sp in ["top", "right", "left"]: axg.spines[sp].set_visible(False)
cax = fig.add_subplot(gs[1, 2]); cb = fig.colorbar(im, cax=cax)
cb.set_label("expression (fraction of gene's peak)", fontsize=10); cb.ax.tick_params(labelsize=9)

# size the strip markers to the TRUE Visium spot diameter (55 um = 0.55 spot pitch)
# on the across-axis (y in spot pitch, range 12), computed from the axis geometry
fig.canvas.draw()
bbh = axs.get_window_extent().height * 72.0 / fig.dpi        # strip axis height in points
s_true = (0.55 * bbh / 12.0) ** 2                            # marker area for a 0.55-pitch diameter
for o, n_ in sc_objs: o.set_sizes([s_true] * n_)
for ext in ["png", "svg", "pdf"]:
    fig.savefig(f"{OUT}/RPG_unrolling_method_suppl.{ext}", dpi=200 if ext == "png" else None, bbox_inches="tight", facecolor="white")
print("wrote RPG_unrolling_method_suppl.png/svg/pdf; true spot radius(px)=%.1f" % rad, flush=True)
