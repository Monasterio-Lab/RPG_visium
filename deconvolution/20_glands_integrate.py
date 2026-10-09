#!/usr/bin/env python
"""Integrate the three salivary regions for gland-vs-gland comparison:
  RPG  - suspected salivary ROI in section A1 (roi_myregion.csv)
  SMG  - submandibular main body of section D1 (D1_spot_glands.csv)
  SLG  - sublingual top lobe of section D1
Produces: combined object, PCA + UMAP (uncorrected and Harmony, coloured by
gland and by section), pseudobulk gland-correlation heatmap, and DEGs
(three-way one-vs-rest + all pairwise).

CAVEAT baked into the outputs: RPG is the only region from section A1, so
RPG-vs-(SMG/SLG) contrasts are confounded with section/batch; SMG-vs-SLG is
within-section and clean. Harmony-corrected embeddings are provided alongside the
uncorrected ones for transparency."""
import os, re, sys, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd, scanpy as sc, anndata as ad
import matplotlib; matplotlib.use("Agg")
matplotlib.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})
import matplotlib.pyplot as plt
import seaborn as sns
from visium_io import load_visium, load_visium_a1

RES = "results/glands_compare"; FIG = "results/figures_glands"
os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)
PAL = {"RPG": "#59A14F", "SMG": "#4E79A7", "SLG": "#E15759"}
SECPAL = {"A1": "#9C755F", "D1": "#EDC948"}
BC_RE = re.compile(r"^[ACGT]{8,}-?\d*$")

# ---- RPG spots from A1 ----
a1 = load_visium_a1()
roi = pd.read_csv("roi_myregion.csv")
bcc = next(c for c in roi.columns if roi[c].astype(str).str.match(BC_RE).mean() > 0.8)
ROI_BC = set(b if "-" in str(b) else f"{b}-1" for b in roi[bcc].astype(str).str.strip())
rpg = a1[[b in ROI_BC for b in a1.obs_names]].copy()
rpg.obs["gland"] = "RPG"; rpg.obs["section"] = "A1"
print(f"RPG spots: {rpg.n_obs}", flush=True)

# ---- SMG / SLG spots from D1 ----
d1 = load_visium("SMG:SLG_Outs/D1/outs", "D1", img_key="hires")
gl = pd.read_csv("results/D1_cluster/D1_spot_glands.csv", index_col="barcode")
d1 = d1[[b in gl.index for b in d1.obs_names]].copy()
d1.obs["gland"] = gl.loc[d1.obs_names, "gland"].values
d1.obs["section"] = "D1"
print("D1 spots:", d1.obs['gland'].value_counts().to_dict(), flush=True)

# ---- concatenate on shared genes (same mm10 reference) ----
for a in (rpg, d1):
    a.var_names_make_unique()
    if "counts" not in a.layers:
        a.layers["counts"] = a.X.copy()
shared = rpg.var_names.intersection(d1.var_names)
print(f"shared genes: {len(shared)}", flush=True)
comb = ad.concat([rpg[:, shared], d1[:, shared]], join="inner", index_unique="-",
                 label="batch_src", keys=["A1", "D1"])
comb.obs["gland"] = comb.obs["gland"].astype("category")
comb.obs["section"] = comb.obs["section"].astype("category")
comb.X = comb.layers["counts"].copy()
print(f"combined: {comb.n_obs} spots x {comb.n_vars} genes", flush=True)
print(comb.obs["gland"].value_counts().to_string(), flush=True)

# ---- normalise / HVG / PCA ----
sc.pp.normalize_total(comb, target_sum=1e4); sc.pp.log1p(comb)
comb.layers["lognorm"] = comb.X.copy()
comb.raw = comb
sc.pp.highly_variable_genes(comb, n_top_genes=2000, flavor="seurat", batch_key="section")
hvg = comb[:, comb.var["highly_variable"]].copy()
sc.pp.scale(hvg, max_value=10)
sc.tl.pca(hvg, n_comps=30, svd_solver="arpack")
comb.obsm["X_pca"] = hvg.obsm["X_pca"]
comb.uns["pca"] = hvg.uns["pca"]

# ---- Harmony batch correction on section (call harmonypy directly; its v2.0.0
# output orientation trips up scanpy's wrapper, so handle the shape robustly) ----
import harmonypy
ho = harmonypy.run_harmony(comb.obsm["X_pca"], comb.obs, ["section"])
Z = np.asarray(ho.Z_corr)
if Z.shape[0] != comb.n_obs:
    Z = Z.T
comb.obsm["X_pca_harmony"] = Z

# ---- UMAPs: uncorrected and Harmony ----
sc.pp.neighbors(comb, n_neighbors=15, n_pcs=30, use_rep="X_pca")
sc.tl.umap(comb); comb.obsm["X_umap_uncorr"] = comb.obsm["X_umap"].copy()
sc.pp.neighbors(comb, n_neighbors=15, n_pcs=30, use_rep="X_pca_harmony")
sc.tl.umap(comb); comb.obsm["X_umap_harmony"] = comb.obsm["X_umap"].copy()

comb.write(f"{RES}/glands_combined.h5ad")

# ---- DEG: three-way one-vs-rest + pairwise (Wilcoxon on lognorm) ----
def dump_rank(adata, groupby, groups, tag):
    sc.tl.rank_genes_groups(adata, groupby, groups=groups, method="wilcoxon",
                            use_raw=True, pts=True)
    df = sc.get.rank_genes_groups_df(adata, group=None)
    df.to_csv(f"{RES}/DEG_{tag}.csv", index=False)
    return df

deg3 = dump_rank(comb, "gland", list(PAL), "3way_one_vs_rest")
for a, b in [("SMG", "SLG"), ("RPG", "SMG"), ("RPG", "SLG")]:
    sub = comb[comb.obs["gland"].isin([a, b])].copy()
    sub.obs["gland"] = sub.obs["gland"].cat.remove_unused_categories()
    sc.tl.rank_genes_groups(sub, "gland", groups=[a], reference=b,
                            method="wilcoxon", use_raw=True, pts=True)
    sc.get.rank_genes_groups_df(sub, group=a).to_csv(
        f"{RES}/DEG_{a}_vs_{b}.csv", index=False)
    print(f"DEG {a} vs {b}: written", flush=True)

# ---- pseudobulk gland correlation (similarity) ----
def pseudobulk(adata, key):
    X = adata.raw.to_adata().to_df()
    return X.groupby(adata.obs[key].values).mean()
pb = pseudobulk(comb, "gland")
hv = comb.var_names[comb.var["highly_variable"]]
corr = pb[hv].T.corr(method="spearman")
corr.to_csv(f"{RES}/gland_pseudobulk_spearman.csv")

# ================= FIGURES =================
def emb_panel(basis, key, pal, fname, title):
    sc.pl.embedding(comb, basis=basis, color=key, palette=pal, show=False,
                    title=title, size=18)
    plt.savefig(f"{FIG}/{fname}.svg", bbox_inches="tight")
    plt.savefig(f"{FIG}/{fname}.png", dpi=200, bbox_inches="tight"); plt.close()

emb_panel("X_pca", "gland", PAL, "PCA_by_gland_uncorrected", "PCA (uncorrected) — by gland")
emb_panel("X_pca", "section", SECPAL, "PCA_by_section_uncorrected", "PCA (uncorrected) — by section")
emb_panel("X_pca_harmony", "gland", PAL, "PCA_by_gland_harmony", "PCA (Harmony) — by gland")
comb.obsm["X_umap"] = comb.obsm["X_umap_uncorr"]
emb_panel("X_umap", "gland", PAL, "UMAP_by_gland_uncorrected", "UMAP (uncorrected) — by gland")
emb_panel("X_umap", "section", SECPAL, "UMAP_by_section_uncorrected", "UMAP (uncorrected) — by section")
comb.obsm["X_umap"] = comb.obsm["X_umap_harmony"]
emb_panel("X_umap", "gland", PAL, "UMAP_by_gland_harmony", "UMAP (Harmony, section-corrected) — by gland")
emb_panel("X_umap", "section", SECPAL, "UMAP_by_section_harmony", "UMAP (Harmony) — by section")

# correlation heatmap
plt.figure(figsize=(4.2, 3.6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="rocket_r", vmin=corr.values.min(),
            vmax=1, square=True, cbar_kws={"label": "Spearman r (HVG pseudobulk)"})
plt.title("Gland transcriptome similarity"); plt.tight_layout()
plt.savefig(f"{FIG}/gland_pseudobulk_corr.svg", bbox_inches="tight")
plt.savefig(f"{FIG}/gland_pseudobulk_corr.png", dpi=200, bbox_inches="tight"); plt.close()

# top-DEG dotplot per gland (one-vs-rest, ribo/mito excluded, by logFC)
deg = deg3[(deg3.pvals_adj < 0.05) & (deg3.logfoldchanges > 1)].copy()
deg = deg[~deg.names.str.match(r"^(Rp[sl]|mt-)")]
topg = {g: deg[deg.group == g].sort_values("logfoldchanges", ascending=False)
        .names.head(8).tolist() for g in PAL}
sc.pl.dotplot(comb, topg, groupby="gland", use_raw=True, standard_scale="var",
              show=False, figsize=(14, 3.2),
              title="Top gland-enriched genes (one-vs-rest, logFC>1, ribo/mito excl.)")
plt.savefig(f"{FIG}/gland_top_DEG_dotplot.svg", bbox_inches="tight")
plt.savefig(f"{FIG}/gland_top_DEG_dotplot.png", dpi=200, bbox_inches="tight"); plt.close()

# DEG counts summary
rows = []
for g in PAL:
    d = deg3[(deg3.group == g) & (deg3.pvals_adj < 0.05) & (deg3.logfoldchanges > 1)]
    rows.append({"gland": g, "n_spots": int((comb.obs.gland == g).sum()),
                 "n_enriched_genes(padj<0.05,logFC>1)": len(d)})
summary = pd.DataFrame(rows)
summary.to_csv(f"{RES}/DEG_summary.csv", index=False)
print("\n", summary.to_string(index=False), flush=True)
print("\nGland pseudobulk Spearman:\n", corr.round(3).to_string(), flush=True)
print("\nTop one-vs-rest genes:", {g: v[:6] for g, v in topg.items()}, flush=True)
print("\nDone. Figures in", FIG, "| tables in", RES, flush=True)
