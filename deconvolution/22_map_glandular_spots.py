#!/usr/bin/env python
"""Map the glandular RPG spots selected by 21_glandularity_filter_umap.py (100 with GLAND_THR=-2.0) back onto the A1 H&E, to show where they sit within the RPG ROI.

Three panels: (1) whole section with RPG ROI + selected spots; (2) zoom on RPG
showing kept (glandular) vs dropped (mucosa/muscle) spots; (3) zoom coloured by
the glandularity score (sal - muc - mus) with the kept spots outlined.
Spots are aligned to the hi-res image by the same image-size scale used in
analysis/20 (the stored hires_scalef is stale)."""
import os, re, json, sys, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd, scanpy as sc
from PIL import Image; Image.MAX_IMAGE_PIXELS = None
import matplotlib; matplotlib.use("Agg")
matplotlib.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from visium_io import load_visium_a1

SPDIR = "VISIUM_outs/V10S15-395_A1/spatial"
FIG = "results/figures_glands_glandular"; os.makedirs(FIG, exist_ok=True)
BC_RE = re.compile(r"^[ACGT]{8,}-?\d*$")

# selected glandular RPG barcodes (strip the -A1 concat suffix)
gl = sc.read_h5ad("results/glands_glandular/glands_glandular.h5ad")
sel = {n[:-3] if n.endswith("-A1") else n for n in gl.obs_names[gl.obs.gland == "RPG"]}
print(f"selected glandular RPG spots: {len(sel)}", flush=True)

# RPG ROI barcodes
roi = pd.read_csv("roi_myregion.csv")
bcc = next(c for c in roi.columns if roi[c].astype(str).str.match(BC_RE).mean() > 0.8)
ROI = set(b if "-" in str(b) else f"{b}-1" for b in roi[bcc].astype(str).str.strip())

# A1 spots + scores (recompute the glandularity score on all ROI spots)
a1 = load_visium_a1()
SAL = ['Aqp5','Bhlha15','Bpifa2','Smgc','Dcpp1','Dcpp2','Dcpp3','Muc19','A2ml1','Pip','Prol1',
       'Mucl2','Car6','Klk1','Egf','Wfdc18','Cyp2f2','Fxyd2','Smr2','Scgb2b27','Scgb1b27']
MUC = ['Muc5b','Tff1','Tff2','Gkn1','Gkn2','Pgc','Gif','Chia1','Krt13','Krt4','Krt6a','Krt76',
       'Krtdap','Sbpl','Krt15','Krt16','Crnn','Sprr1a','Lgals7','Dsp','Bpifb2','Bpifb6','Aqp3','Krt5','Krt14']
MUS = ['Acta1','Actn2','Actn3','Myh1','Myh2','Myh3','Myh4','Myh7','Myh8','Myl1','Mylpf','Mybpc1',
       'Mybpc2','Tnnt1','Tnnt3','Tnni1','Tnni2','Tnnc2','Ttn','Ckm','Ckmt2','Des','Mb','Pgam2',
       'Pvalb','Atp2a1','Casq1','Ryr1','Neb','Tcap','Myoz1','Tpm2','Ldb3','Smpx','Klhl41','Pygm','Eno3']
a1n = a1.copy(); sc.pp.normalize_total(a1n, target_sum=1e4); sc.pp.log1p(a1n)
for nm, gs in [("sal", SAL), ("muc", MUC), ("mus", MUS)]:
    sc.tl.score_genes(a1n, [g for g in gs if g in a1n.var_names], score_name=nm)
a1.obs["glandscore"] = a1n.obs["sal"] - a1n.obs["muc"] - a1n.obs["mus"]

# hi-res H&E + spot->image scale (stored hires_scalef is stale)
sf = json.load(open(f"{SPDIR}/scalefactors_json.json"))
HE = np.asarray(Image.open(f"{SPDIR}/tissue_hires_image.png").convert("RGB"))
H_hi, W_hi = HE.shape[:2]
lo = Image.open(f"{SPDIR}/tissue_lowres_image.png")
fullres_w, fullres_h = lo.size[0] / sf["tissue_lowres_scalef"], lo.size[1] / sf["tissue_lowres_scalef"]
sx, sy = W_hi / fullres_w, H_hi / fullres_h
XY = a1.obsm["spatial"] * np.array([sx, sy])

names = a1.obs_names.to_numpy()
in_roi = np.array([b in ROI for b in names])
in_sel = np.array([b in sel for b in names])
roi_drop = in_roi & ~in_sel
print(f"A1 spots {a1.n_obs} | ROI {in_roi.sum()} | selected {in_sel.sum()} | dropped {roi_drop.sum()}", flush=True)

# RPG bounding box for zoom panels
rx, ry = XY[in_roi, 0], XY[in_roi, 1]
pad = 0.18 * max(rx.ptp(), ry.ptp())
zx0, zx1 = rx.min() - pad, rx.max() + pad
zy0, zy1 = ry.min() - pad, ry.max() + pad


def base(ax, crop=False):
    ax.imshow(HE, extent=[0, W_hi, H_hi, 0])
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if crop:
        ax.set_xlim(zx0, zx1); ax.set_ylim(zy1, zy0)


fig, axes = plt.subplots(1, 3, figsize=(24, 9))

# (1) whole section
ax = axes[0]; base(ax)
ax.scatter(XY[~in_roi, 0], XY[~in_roi, 1], s=3, c="#BBBBBB", alpha=0.35, linewidths=0)
ax.scatter(XY[roi_drop, 0], XY[roi_drop, 1], s=14, c="#F28E2B", alpha=0.9, linewidths=0, label=f"RPG dropped ({roi_drop.sum()})")
ax.scatter(XY[in_sel, 0], XY[in_sel, 1], s=20, c="#59A14F", edgecolor="k", linewidths=0.3, label=f"glandular kept ({in_sel.sum()})")
from matplotlib.patches import Rectangle
ax.add_patch(Rectangle((zx0, zy0), zx1 - zx0, zy1 - zy0, fill=False, ec="k", lw=1.2, ls="--"))
ax.set_title("A1 section — selected glandular RPG spots", fontsize=13)
ax.legend(loc="upper right", fontsize=10, framealpha=0.9)

# (2) zoom: kept vs dropped
ax = axes[1]; base(ax, crop=True)
ax.scatter(XY[roi_drop, 0], XY[roi_drop, 1], s=70, c="#F28E2B", alpha=0.9, edgecolor="k", linewidths=0.3, label="dropped (mucosa/muscle)")
ax.scatter(XY[in_sel, 0], XY[in_sel, 1], s=70, c="#59A14F", edgecolor="k", linewidths=0.4, label="glandular (kept)")
ax.set_title("RPG ROI — kept vs dropped", fontsize=13)
ax.legend(loc="upper right", fontsize=10, framealpha=0.9)

# (3) zoom: glandularity gradient, kept outlined
ax = axes[2]; base(ax, crop=True)
scsc = ax.scatter(XY[in_roi, 0], XY[in_roi, 1], s=70, c=a1.obs["glandscore"][in_roi],
                  cmap="viridis", edgecolor="none")
ax.scatter(XY[in_sel, 0], XY[in_sel, 1], s=130, facecolors="none", edgecolor="#59A14F", linewidths=1.4)
cb = fig.colorbar(scsc, ax=ax, fraction=0.046, pad=0.02); cb.set_label("glandularity (sal − muc − mus)")
ax.set_title("RPG ROI — glandularity score (kept outlined)", fontsize=13)

plt.tight_layout()
plt.savefig(f"{FIG}/glandular_spots_spatial.svg", bbox_inches="tight")
plt.savefig(f"{FIG}/glandular_spots_spatial.png", dpi=200, bbox_inches="tight")
plt.close()

# export the selected barcodes + scores
out = pd.DataFrame({"barcode": names[in_roi], "selected_glandular": in_sel[in_roi],
                    "glandscore": a1.obs["glandscore"][in_roi].values}).sort_values("glandscore", ascending=False)
out.to_csv("results/glands_glandular/RPG_spot_selection.csv", index=False)
print("wrote glandular_spots_spatial.svg/.png and RPG_spot_selection.csv", flush=True)
