#!/usr/bin/env python
"""Data-driven salivary enrichment score (no hand-picked genes): derive the
signature from the top one-vs-rest DE markers of the six SMG/SLG acinar cell types in the reference (ducts excluded), then score A1 spots with it and compare RPG vs the rest of the section."""
import os, re, sys, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd, scanpy as sc
from scipy.stats import mannwhitneyu, spearmanr
from scipy.spatial import cKDTree
from PIL import Image; Image.MAX_IMAGE_PIXELS = None
import matplotlib; matplotlib.use("Agg")
matplotlib.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from visium_io import load_visium_a1

REF = "data/reference/combined_parotid_ref.h5ad"; SPDIR = "VISIUM_outs/V10S15-395_A1/spatial"
OUT = "results/figures_parotid"; BC_RE = re.compile(r"^[ACGT]{8,}-?\d*$")
TOP_PER_TYPE = 20
# SMG/SLG ACINAR (secretory) cell types — the salivary-SPECIFIC program (foregut
# has no serous/mucous salivary acini). Ducts excluded: their top markers are
# pan-epithelial keratins/tight-junctions also present in oesophagus/stomach.
GLAND_EPI = ["Bpifa2+ (SMG)", "Serous acinar (SMG/SLG)", "Acinar (SMG)", "Seromucous acinar (SMG/SLG)",
             "Mucous acinar (SLG)", "Smgc+ (SMG)"]

# ---- derive signature by DE (one-vs-rest Wilcoxon), keeping SPECIFIC markers ----
ref = sc.read_h5ad(REF); ref.obs["CellType"] = ref.obs["CellType"].astype(str)
sc.pp.normalize_total(ref, target_sum=1e4); sc.pp.log1p(ref)
sc.tl.rank_genes_groups(ref, "CellType", method="wilcoxon", pts=True, n_genes=400)
groups = list(ref.uns["rank_genes_groups"]["names"].dtype.names)
gland_types = [g for g in GLAND_EPI if g in groups]
degdf = sc.get.rank_genes_groups_df(ref, group=None)
print("gland-epithelial types used:", len(gland_types), flush=True)

a1 = load_visium_a1(); sc.pp.normalize_total(a1, target_sum=1e4); sc.pp.log1p(a1)
# drop housekeeping / metabolic / ambient (not salivary identity)
HK = re.compile(r"^(Rp[sl]\d|mt-|Hb[ab]-|Cox\d|Uqcr|Nduf|Atp5|Atp1[ab]|Hsp|Hist\d|H[0-9]af|H3f|H1f|"
                r"Malat1|Gm\d|Rik$|Actg|Actb|Ptma|Naca|Eef|Ppia|Serf|Prdx|Tmsb|Fau|Tpt1|Ubc|Snhg|"
                r"Gas5|Chchd|Nme\d|Taldo|Hmgn|Jun$|Mif$|Sfn$|Ndufa|Serp|Sec61|Chchd)")
sig = set()
for g in gland_types:
    d = degdf[(degdf.group == g) & (degdf.logfoldchanges > 1.5) &
              (degdf.pct_nz_group > 0.40) & (degdf.pct_nz_reference < 0.35) &
              (~degdf.names.str.match(HK)) & (degdf.names.isin(a1.var_names))]
    sig.update(d.sort_values("logfoldchanges", ascending=False)["names"].head(TOP_PER_TYPE))
sig = sorted(sig)
pd.Series(sig, name="gene").to_csv(f"{OUT}/datadriven_salivary_signature.csv", index=False)
print(f"data-driven signature: {len(sig)} unique genes (top {TOP_PER_TYPE}/type)", flush=True)
print(sig, flush=True)

# ---- score A1 ----
sc.tl.score_genes(a1, sig, score_name="sal_dd")
roi = pd.read_csv("roi_myregion.csv")
bcc = next(c for c in roi.columns if roi[c].astype(str).str.match(BC_RE).mean() > 0.8)
ROI_BC = set(b if "-" in str(b) else f"{b}-1" for b in roi[bcc].astype(str).str.strip())
grp = np.array(["RPG" if b in ROI_BC else "Non-glandular" for b in a1.obs_names])
dd = a1.obs["sal_dd"].values
rm, nm = dd[grp == "RPG"].mean(), dd[grp == "Non-glandular"].mean()
_, p = mannwhitneyu(dd[grp == "RPG"], dd[grp == "Non-glandular"], alternative="greater")
print(f"data-driven score: RPG {rm:.3f} vs Non-glandular {nm:.3f}  (MWU p={p:.2e})", flush=True)

# ---- figure: spatial map + violin + concordance ----
sf = __import__("json").load(open(f"{SPDIR}/scalefactors_json.json"))
HE = np.asarray(Image.open(f"{SPDIR}/tissue_hires_image.png").convert("RGB")); H, W = HE.shape[:2]
lo = Image.open(f"{SPDIR}/tissue_lowres_image.png")
fw = lo.size[0] / sf["tissue_lowres_scalef"]; fh = lo.size[1] / sf["tissue_lowres_scalef"]
sx, sy = W / fw, H / fh; XY = a1.obsm["spatial"] * np.array([sx, sy])
radius = 0.62 * np.median(cKDTree(XY).query(XY, k=2)[0][:, 1]); in_roi = grp == "RPG"

fig = plt.figure(figsize=(15, 8)); gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 0.7], wspace=0.22)
ax = fig.add_subplot(gs[0]); ax.imshow(HE)
pc = PatchCollection([Circle((x, y), radius) for x, y in XY], cmap="magma")
pc.set_array(dd); pc.set_clim(0, np.percentile(dd, 99)); pc.set_edgecolor("none"); ax.add_collection(pc)
ax.add_collection(PatchCollection([Circle((x, y), radius * 1.15) for x, y in XY[in_roi]],
                                  facecolor="none", edgecolor="#39FF14", linewidths=0.5))
ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values(): s.set_visible(False)
ax.set_title("Data-driven salivary score (DE markers of SMG/SLG epithelium)\ngreen = RPG ROI", fontsize=12.5, fontweight="bold")
fig.colorbar(pc, ax=ax, fraction=0.045, pad=0.02)

axv = fig.add_subplot(gs[1])
dv = [dd[grp == "RPG"], dd[grp == "Non-glandular"]]
parts = axv.violinplot(dv, showmeans=True, showextrema=False)
for b, c in zip(parts["bodies"], ["#59A14F", "#BAB0AC"]): b.set_facecolor(c); b.set_alpha(0.6)
jr = np.random.default_rng(0)                       # jittered individual spots
for i, (d, c) in enumerate(zip(dv, ["#2e6b34", "#6b6b6b"]), start=1):
    axv.scatter(jr.normal(i, 0.055, size=len(d)), d, s=2.5, c=c, alpha=0.30, linewidths=0, zorder=3)
axv.set_xticks([1, 2]); axv.set_xticklabels(["RPG", "Non-\nglandular"], fontsize=11)
axv.set_ylabel("data-driven salivary score"); axv.set_title(f"RPG vs rest\nMWU p={p:.1e}", fontsize=12, fontweight="bold")
for s in ["top", "right"]: axv.spines[s].set_visible(False)

fig.suptitle("Whole A1 section — RPG salivary enrichment by a DATA-DRIVEN signature", fontsize=15, fontweight="bold", y=0.99)
for ext in ["png", "svg", "pdf"]:
    fig.savefig(f"{OUT}/wholesection_salivary_datadriven.{ext}", dpi=180 if ext == "png" else None,
                bbox_inches="tight", facecolor="white")
print("wrote wholesection_salivary_datadriven.png/svg/pdf + signature csv", flush=True)
