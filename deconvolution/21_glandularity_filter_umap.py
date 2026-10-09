#!/usr/bin/env python
"""Restrict RPG to its most salivary-glandular spots and re-run the full
comparison set (PCA+loadings, Harmony UMAP, pseudobulk correlation, DEG, both
Venns, GO enrichment).

Why this exact strategy: RPG has NO spatially separable pure-gland pocket — the
mucosal/foregut score dominates the salivary score on every spot (mean 1.98 vs
0.51) and salivary cells are admixed throughout. So 'genuinely glandular' is
approached two ways together:
  (1) spot-level: keep RPG spots with glandularity index (salivary − mucosal − muscle) > GLAND_THR; paper run GLAND_THR=-2.0 → 100 of 161 spots
  (2) gene-level: drop skeletal-muscle and mucosal/squamous/foregut genes
SMG and SLG are kept whole. The section/depth batch confound still remains
(RPG = section A1 only, ~5x deeper)."""
import os, sys, json, time, urllib.request, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd, scanpy as sc
from visium_io import load_visium_a1
import matplotlib; matplotlib.use("Agg")
matplotlib.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn3, venn3_circles

RES = "results/glands_glandular"; FIG = "results/figures_glands_glandular"
os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)
PAL = {"RPG": "#59A14F", "SMG": "#4E79A7", "SLG": "#E15759"}
GLAND_N = int(os.environ.get("GLAND_N", "40"))
GLAND_THR = os.environ.get("GLAND_THR")  # if set, threshold on glandscore instead of top-N
PADJ, LFC = 0.05, 1.0
RIBO_MT = r"^(Rp[sl]|mt-)"

SAL = ['Aqp5','Bhlha15','Bpifa2','Smgc','Dcpp1','Dcpp2','Dcpp3','Muc19','A2ml1','Pip','Prol1',
       'Mucl2','Car6','Klk1','Egf','Wfdc18','Cyp2f2','Fxyd2','Smr2','Scgb2b27','Scgb1b27']
MUC = ['Muc5b','Tff1','Tff2','Gkn1','Gkn2','Pgc','Gif','Chia1','Krt13','Krt4','Krt6a','Krt76',
       'Krtdap','Sbpl','Krt15','Krt16','Crnn','Sprr1a','Lgals7','Dsp','Bpifb2','Bpifb6','Aqp3','Krt5','Krt14']
MUS = ['Acta1','Actn2','Actn3','Myh1','Myh2','Myh3','Myh4','Myh7','Myh8','Myl1','Mylpf','Mybpc1',
       'Mybpc2','Tnnt1','Tnnt3','Tnni1','Tnni2','Tnnc2','Ttn','Ckm','Ckmt2','Des','Mb','Pgam2',
       'Pvalb','Atp2a1','Casq1','Ryr1','Neb','Tcap','Myoz1','Tpm2','Ldb3','Smpx','Klhl41','Pygm','Eno3']

comb = sc.read_h5ad("results/glands_compare/glands_combined.h5ad")

# canonical glandularity score: computed on the FULL A1 section (identical to the
# spatial map / threshold diagnostic), then applied to the combined RPG spots so a
# chosen threshold means exactly what the map showed.
a1 = load_visium_a1(); sc.pp.normalize_total(a1, target_sum=1e4); sc.pp.log1p(a1)
for nm, gs in [("sal", SAL), ("muc", MUC), ("mus", MUS)]:
    sc.tl.score_genes(a1, [g for g in gs if g in a1.var_names], score_name=nm)
a1gs = a1.obs["sal"] - a1.obs["muc"] - a1.obs["mus"]   # indexed by A1 barcode

# (1) select most-glandular RPG spots (threshold on glandscore, else top-N)
rpg = comb.obs.index[comb.obs.gland == "RPG"]
rpg_a1bc = pd.Index([n[:-3] if n.endswith("-A1") else n for n in rpg])
comb.obs["glandscore"] = np.nan
comb.obs.loc[rpg, "glandscore"] = a1gs.reindex(rpg_a1bc).to_numpy()
if GLAND_THR is not None:
    keep_rpg = comb.obs.loc[rpg].index[comb.obs.loc[rpg, "glandscore"] > float(GLAND_THR)]
    sel_desc = f"glandscore > {GLAND_THR}"
else:
    keep_rpg = comb.obs.loc[rpg].nlargest(GLAND_N, "glandscore").index
    sel_desc = f"top {GLAND_N}"
drop = [b for b in rpg if b not in set(keep_rpg)]
comb = comb[~comb.obs_names.isin(drop)].copy()
kept_bc = pd.Index([n[:-3] if n.endswith("-A1") else n for n in keep_rpg])
print(f"RPG: kept {len(keep_rpg)}/{len(rpg)} glandular spots ({sel_desc}; "
      f"sal={a1.obs.loc[kept_bc,'sal'].mean():.2f}, muc={a1.obs.loc[kept_bc,'muc'].mean():.2f}, "
      f"mus={a1.obs.loc[kept_bc,'mus'].mean():.2f})", flush=True)

# (2) drop muscle + mucosal/squamous/foregut genes
contam = sorted(set(MUS) | set(MUC))
keep_genes = ~comb.var_names.isin(contam)
removed = int((~keep_genes).sum())
comb = comb[:, keep_genes].copy(); comb.raw = comb
print(f"removed {removed} contaminant genes; {comb.n_obs} spots x {comb.n_vars} genes", flush=True)
print("gland counts:", comb.obs.gland.value_counts().to_dict(), flush=True)

# ---- recompute embedding ----
sc.pp.highly_variable_genes(comb, n_top_genes=2000, flavor="seurat", batch_key="section")
hvg = comb[:, comb.var["highly_variable"]].copy()
sc.pp.scale(hvg, max_value=10); sc.tl.pca(hvg, n_comps=30, svd_solver="arpack")
comb.obsm["X_pca"] = hvg.obsm["X_pca"]; vr = hvg.uns["pca"]["variance_ratio"]
loadings = hvg.varm["PCs"]; hvg_names = hvg.var_names.to_numpy()
import harmonypy
ho = harmonypy.run_harmony(comb.obsm["X_pca"], comb.obs, ["section"])
Z = np.asarray(ho.Z_corr); comb.obsm["X_pca_harmony"] = Z if Z.shape[0] == comb.n_obs else Z.T
sc.pp.neighbors(comb, n_neighbors=15, n_pcs=30, use_rep="X_pca_harmony"); sc.tl.umap(comb)
comb.write(f"{RES}/glands_glandular.h5ad")

# ---- PCA variance + loadings + gland separation ----
pc_var = pd.DataFrame({"PC": [f"PC{i+1}" for i in range(30)], "pct_variance": vr * 100,
                       "cumulative_pct": np.cumsum(vr) * 100})
Hh = comb.obsm["X_pca_harmony"]; gl = comb.obs["gland"].astype(str).to_numpy()
sep = pd.DataFrame({g: Hh[gl == g].mean(0) for g in PAL}, index=pc_var.PC)
sep["separation"] = sep[list(PAL)].max(1) - sep[list(PAL)].min(1)
pc_var = pc_var.join(sep, on="PC"); pc_var.to_csv(f"{RES}/PCA_variance_and_gland_means.csv", index=False)
rows = []
for i in range(30):
    o = np.argsort(loadings[:, i])
    for r in range(20):
        rows.append({"PC": f"PC{i+1}", "rank": r+1,
                     "gene_pos": hvg_names[o[::-1][r]], "loading_pos": loadings[o[::-1][r], i],
                     "gene_neg": hvg_names[o[r]], "loading_neg": loadings[o[r], i]})
pd.DataFrame(rows).to_csv(f"{RES}/PCA_loadings_top_genes.csv", index=False)

# ---- pseudobulk correlation ----
pb = comb.raw.to_adata().to_df().groupby(comb.obs["gland"].values).mean()
hv = comb.var_names[comb.var["highly_variable"]]
corr = pb[hv].T.corr(method="spearman"); corr.to_csv(f"{RES}/gland_pseudobulk_spearman.csv")

# ---- DEG: 3-way one-vs-rest + pairwise ----
sc.tl.rank_genes_groups(comb, "gland", method="wilcoxon", use_raw=True, pts=True)
deg3 = sc.get.rank_genes_groups_df(comb, group=None); deg3.to_csv(f"{RES}/DEG_3way_one_vs_rest.csv", index=False)
def pair(a, b):
    sub = comb[comb.obs.gland.isin([a, b])].copy(); sub.obs.gland = sub.obs.gland.cat.remove_unused_categories()
    sc.tl.rank_genes_groups(sub, "gland", groups=[a], reference=b, method="wilcoxon", use_raw=True, pts=True)
    d = sc.get.rank_genes_groups_df(sub, group=a); d.to_csv(f"{RES}/DEG_{a}_vs_{b}.csv", index=False); return d
P = {(a, b): pair(a, b) for a, b in [("SMG", "SLG"), ("RPG", "SMG"), ("RPG", "SLG")]}

# ---- expressed-gene matched Venn ----
import scipy.sparse as sp
counts = comb.layers["counts"]; counts = counts if sp.issparse(counts) else sp.csr_matrix(counts)
genes = comb.var_names.to_numpy()
det = {g: np.asarray((counts[(comb.obs.gland == g).to_numpy()] > 0).mean(0)).ravel() for g in PAL}
TOPK = 1000
mset = {g: set(pd.Series(det[g], index=genes).nlargest(TOPK).index) for g in PAL}
fig, ax = plt.subplots(figsize=(8, 7))
v = venn3([mset["RPG"], mset["SMG"], mset["SLG"]], set_labels=list(PAL), ax=ax,
          set_colors=[PAL[g] for g in PAL], alpha=0.55); venn3_circles([mset[g] for g in PAL], ax=ax, lw=1)
for t, g in zip(v.set_labels, PAL):
    if t: t.set_color(PAL[g]); t.set_fontweight("bold"); t.set_fontsize(14)
ax.set_title(f"Glandular-RPG depth-matched gene repertoire (top {TOPK}/gland)", fontsize=12)
plt.tight_layout(); plt.savefig(f"{FIG}/gene_venn_matched.svg", bbox_inches="tight")
plt.savefig(f"{FIG}/gene_venn_matched.png", dpi=200, bbox_inches="tight"); plt.close()
R, S, L = (mset[g] for g in PAL)
gvenn = {"core_all3": R & S & L, "SMG_SLG_only": (S & L) - R, "RPG_SMG_only": (R & S) - L,
         "RPG_SLG_only": (R & L) - S, "RPG_unique": R - S - L, "SMG_unique": S - R - L, "SLG_unique": L - R - S}
for k, gs in gvenn.items():
    pd.Series(sorted(gs), name="gene").to_csv(f"{RES}/genevenn_{k}.csv", index=False)

# ---- DEG/marker Venn (pairwise contrasts) + signatures ----
def up(df, pos=True):
    d = df[(df.pvals_adj < PADJ) & (~df.names.str.match(RIBO_MT))]
    return set(d[d.logfoldchanges > LFC].names) if pos else set(d[d.logfoldchanges < -LFC].names)
uRS, uSR = up(P[("RPG", "SMG")]), up(P[("RPG", "SMG")], False)
uRL, uLR = up(P[("RPG", "SLG")]), up(P[("RPG", "SLG")], False)
uSL, uLS = up(P[("SMG", "SLG")]), up(P[("SMG", "SLG")], False)
D_RS, D_RL, D_SL = uRS | uSR, uRL | uLR, uSL | uLS
fig, ax = plt.subplots(figsize=(8.5, 7))
v = venn3([D_RS, D_RL, D_SL], set_labels=["RPG vs SMG", "RPG vs SLG", "SMG vs SLG"], ax=ax,
          set_colors=["#8C6BB1", "#41AB5D", "#EF6548"], alpha=0.5); venn3_circles([D_RS, D_RL, D_SL], ax=ax, lw=1)
for t in (v.set_labels or []):
    if t: t.set_fontweight("bold"); t.set_fontsize(12)
ax.set_title(f"Glandular-RPG DE per contrast (padj<{PADJ}, |logFC|>{LFC:.0f})", fontsize=12)
plt.tight_layout(); plt.savefig(f"{FIG}/DEG_venn.svg", bbox_inches="tight")
plt.savefig(f"{FIG}/DEG_venn.png", dpi=200, bbox_inches="tight"); plt.close()
sigs = {"RPG_up": uRS & uRL, "SMG_up": uSR & uSL, "SLG_up": uLR & uLS,
        "salivary_shared_up": uSR & uLR, "SMG_vs_SLG_up": uSL, "SLG_vs_SMG_up": uLS}
for k, gs in sigs.items():
    pd.Series(sorted(gs), name="gene").to_csv(f"{RES}/sig_{k}.csv", index=False)

# ---- GO enrichment (Enrichr) on key signatures ----
ENR = "https://maayanlab.cloud/Enrichr"
def _mp(f):
    b = "----b7MA4YWxkTrZu"; o = []
    for k, vv in f.items(): o += ["--"+b, f'Content-Disposition: form-data; name="{k}"', "", vv]
    o += ["--"+b+"--", ""]; return "\r\n".join(o).encode(), b
def go_bp(genes, tag):
    genes = sorted({g.upper() for g in genes})
    if len(genes) < 5: return pd.DataFrame()
    body, b = _mp({"list": "\n".join(genes), "description": tag})
    rq = urllib.request.Request(ENR+"/addList", data=body); rq.add_header("Content-Type", f"multipart/form-data; boundary={b}")
    uid = json.loads(urllib.request.urlopen(rq, timeout=90).read())["userListId"]; time.sleep(0.4)
    lib = "GO_Biological_Process_2021"
    rows = json.loads(urllib.request.urlopen(f"{ENR}/enrich?userListId={uid}&backgroundType={lib}", timeout=90).read()).get(lib, [])
    return pd.DataFrame([{"term": r[1], "adj_pval": r[6], "n": len(r[5]), "genes": ";".join(r[5])} for r in rows])
go = {}
for tag in ["RPG_up", "SLG_vs_SMG_up", "SMG_vs_SLG_up", "salivary_shared_up"]:
    g = pd.read_csv(f"{RES}/sig_{tag}.csv")["gene"].astype(str).tolist() if os.path.exists(f"{RES}/sig_{tag}.csv") else []
    go[tag] = go_bp(g, tag)
    if len(go[tag]): go[tag].to_csv(f"{RES}/GO_{tag}.csv", index=False)

# ================= summary figures =================
fig, ax = plt.subplots(figsize=(4.4, 3.7))
sns.heatmap(corr.loc[list(PAL), list(PAL)], annot=True, fmt=".2f", cmap="rocket_r", vmin=0.5, vmax=1,
            square=True, cbar_kws={"label": "Spearman r"}); ax.set_title("Glandular-RPG similarity")
plt.tight_layout(); plt.savefig(f"{FIG}/pseudobulk_corr.svg", bbox_inches="tight")
plt.savefig(f"{FIG}/pseudobulk_corr.png", dpi=200, bbox_inches="tight"); plt.close()
sc.pl.umap(comb, color="gland", palette=PAL, show=False, size=22, title="UMAP (Harmony) — glandular RPG")
plt.savefig(f"{FIG}/UMAP_by_gland_harmony.svg", bbox_inches="tight")
plt.savefig(f"{FIG}/UMAP_by_gland_harmony.png", dpi=200, bbox_inches="tight"); plt.close()
panel = ["RPG_up", "SLG_vs_SMG_up", "SMG_vs_SLG_up", "salivary_shared_up"]
fig, axes = plt.subplots(2, 2, figsize=(18, 10)); axes = axes.ravel()
for ax, tag in zip(axes, panel):
    d = go.get(tag, pd.DataFrame())
    if len(d) == 0: ax.set_visible(False); continue
    b = d.sort_values("adj_pval").head(10).iloc[::-1]; b["nlp"] = -np.log10(b.adj_pval.clip(lower=1e-300))
    ax.barh(range(len(b)), b.nlp, color=PAL.get(tag.split("_")[0], "#666"))
    ax.set_yticks(range(len(b))); ax.set_yticklabels(b.term.str.replace(r"\s*\(GO:\d+\)", "", regex=True).str.slice(0, 52), fontsize=9)
    ax.axvline(-np.log10(0.05), color="r", ls="--", lw=0.8); ax.set_xlabel("-log10 adj p"); ax.set_title(tag, fontsize=11)
fig.suptitle("Glandular-RPG GO Biological Process enrichment", fontsize=14)
plt.tight_layout(); plt.savefig(f"{FIG}/GO_enrichment.svg", bbox_inches="tight")
plt.savefig(f"{FIG}/GO_enrichment.png", dpi=200, bbox_inches="tight"); plt.close()

# ---- console summary ----
print("\n=== pseudobulk Spearman (glandular RPG) ===\n", corr.loc[list(PAL), list(PAL)].round(3).to_string(), flush=True)
print("\nPC weights: PC1 %.1f%%  PC2 %.1f%%  PC3 %.1f%%" % tuple(vr[:3]*100), flush=True)
print("gland separation top PCs:\n", pc_var[["PC", "pct_variance", "separation"]].head(4).round(2).to_string(index=False), flush=True)
print("\nmatched-Venn:", {k: len(v) for k, v in gvenn.items()}, flush=True)
print("DEG-contrast sizes:", {"RPGvsSMG": len(D_RS), "RPGvsSLG": len(D_RL), "SMGvsSLG": len(D_SL)}, flush=True)
print("signature sizes:", {k: len(v) for k, v in sigs.items()}, flush=True)
rpg_up = deg3[(deg3.group == "RPG") & (deg3.pvals_adj < .05) & (deg3.logfoldchanges > 1) & (~deg3.names.str.match(RIBO_MT))]
print("\nglandular-RPG top one-vs-rest genes:", rpg_up.sort_values("logfoldchanges", ascending=False).names.head(15).tolist(), flush=True)
for tag in panel:
    d = go.get(tag, pd.DataFrame())
    if len(d):
        t = d.sort_values("adj_pval").iloc[0]; print(f"top GO {tag}: {t.term[:55]} (adjp={t.adj_pval:.1e})", flush=True)
print("\nWrote outputs to", RES, "and", FIG, flush=True)
