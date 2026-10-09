#!/usr/bin/env python
"""Rigorous ambient-RNA decontamination of the parotid (GSE223516 control) with
a DecontX-style cluster-aware EM (Yang et al. 2020, Genome Biol), then re-annotate
on the decontaminated counts and rebuild the combined SMG+SLG+Parotid reference.

Model: observed counts of cell c ~ Multinomial( theta_c * phi_{z(c)} + (1-theta_c) * b ),
where phi_z = native expression distribution of cluster z, b = ambient (soup)
distribution, theta_c = native fraction. EM estimates theta_c and phi_z; the
decontaminated count is x_cg * P(native). Needs no empty droplets.
"""
import os, glob, tarfile, gzip, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np, pandas as pd, scanpy as sc, anndata as ad
import scipy.sparse as sp, scipy.io
import matplotlib; matplotlib.use("Agg")
matplotlib.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})
import matplotlib.pyplot as plt

SCR = ("/private/tmp/claude-501/-Users-gustavomonasterio-Bioinformatics-Claude-"
       "ORES-Deconvolution/07b953b0-c262-40ca-9e22-fc00d39fbd86/scratchpad")
TAR = f"{SCR}/parotid_GSE223516_RAW.tar"; WORK = f"{SCR}/parotid_work"
PREFIX = "GSM6957905_NonIR-PAR"
FIG = "results/figures_parotid"; os.makedirs(FIG, exist_ok=True)
OUT_PAR = "data/reference/parotid_ref.h5ad"
OUT_COMB = "data/reference/combined_parotid_ref.h5ad"
COMBINED_FULL = "data/reference/combined_full_ref.h5ad"

# ---------- load control 10x (raw counts) ----------
os.makedirs(WORK, exist_ok=True)
if not glob.glob(f"{WORK}/*"):
    with tarfile.open(TAR) as t: t.extractall(WORK)
def op(f): return gzip.open(f, "rt") if f.endswith(".gz") else open(f)
mtx = glob.glob(f"{WORK}/{PREFIX}*matrix.mtx*")[0]
X = scipy.io.mmread(gzip.open(mtx, "rb") if mtx.endswith(".gz") else open(mtx, "rb")).T.tocsr()
feats = pd.read_csv(op(glob.glob(f"{WORK}/{PREFIX}*features*")[0]), header=None, sep="\t")
sym = (feats[1] if feats.shape[1] > 1 else feats[0]).astype(str).values
bcs = pd.read_csv(op(glob.glob(f"{WORK}/{PREFIX}*barcodes*")[0]), header=None)[0].astype(str).values
a = ad.AnnData(sp.csr_matrix(X)); a.var_names = pd.Index(sym); a.var_names_make_unique()
a.obs_names = pd.Index([f"PAR_{b}" for b in bcs])
a.var["mt"] = a.var_names.str.match(r"^mt-")
sc.pp.calculate_qc_metrics(a, qc_vars=["mt"], inplace=True, percent_top=None)
a = a[(a.obs.n_genes_by_counts >= 200) & (a.obs.pct_counts_mt < 30)].copy()
sc.pp.filter_genes(a, min_cells=3)
print(f"parotid QC: {a.shape}", flush=True)

# ---------- clusters (for the native profiles) ----------
w = a.copy(); sc.pp.normalize_total(w, target_sum=1e4); sc.pp.log1p(w)
sc.pp.highly_variable_genes(w, n_top_genes=2000)
wh = w[:, w.var.highly_variable].copy(); sc.pp.scale(wh, max_value=10)
sc.tl.pca(wh, n_comps=30); sc.pp.neighbors(wh, n_neighbors=15, n_pcs=30)
sc.tl.leiden(wh, resolution=1.5); z = wh.obs["leiden"].astype(int).to_numpy()
K = z.max() + 1

# ---------- DecontX-style EM ----------
Xd = np.asarray(a.X.todense(), dtype=np.float64)          # cells x genes counts
Ncg = Xd.sum(1, keepdims=True)                            # per-cell total
b = Xd.sum(0); b = b / b.sum()                            # ambient (soup) distribution
# cluster native profiles (init from cluster-summed counts, +pseudocount)
def cluster_phi():
    phi = np.zeros((K, Xd.shape[1]))
    for k in range(K):
        s = Xd[z == k].sum(0) + 1e-6
        phi[k] = s / s.sum()
    return phi
phi = cluster_phi()
theta = np.full(Xd.shape[0], 0.90)                        # native fraction / cell
eps = 1e-12
for it in range(60):
    phi_c = phi[z]                                        # native profile per cell
    pn = theta[:, None] * phi_c
    pc = (1 - theta)[:, None] * b[None, :]
    r = pn / (pn + pc + eps)                              # P(native) per (cell,gene)
    # M-step
    theta_new = (Xd * r).sum(1) / (Ncg[:, 0] + eps)
    theta_new = np.clip(theta_new, 0.02, 0.999)
    Ynat = Xd * r
    phi = np.vstack([(Ynat[z == k].sum(0) + 1e-6) for k in range(K)])
    phi = phi / phi.sum(1, keepdims=True)
    d = np.abs(theta_new - theta).mean(); theta = theta_new
    if it % 10 == 0 or d < 1e-4:
        print(f"  EM iter {it}: mean contamination={1-theta.mean():.3f} (delta {d:.2e})", flush=True)
    if d < 1e-4: break

# decontaminated counts = native part, rounded
Y = np.rint(Xd * r).astype(np.int32)
a.layers["counts_raw"] = sp.csr_matrix(a.X)
a.X = sp.csr_matrix(Y)
a.obs["contamination"] = 1 - theta
a.obs["leiden"] = wh.obs["leiden"].values
print(f"median per-cell contamination removed: {np.median(1-theta):.2%}", flush=True)

# ---------- validate: Bpifa2 in immune (Ptprc+) cells before/after ----------
def det_in(mat, gene, mask):
    if gene not in a.var_names: return np.nan
    j = a.var_names.get_loc(gene); v = np.asarray(mat[mask, j].todense()).ravel() if sp.issparse(mat) else mat[mask, j]
    return (v > 0).mean()
imm_mask = np.asarray(sp.csr_matrix(a.layers["counts_raw"])[:, a.var_names.get_loc("Ptprc")].todense()).ravel() > 0 \
           if "Ptprc" in a.var_names else np.zeros(a.n_obs, bool)
for g in ["Bpifa2", "Amy1", "Car6", "Dcpp1", "Ptprc"]:
    if g in a.var_names:
        before = det_in(sp.csr_matrix(a.layers["counts_raw"]), g, imm_mask)
        after = det_in(a.X, g, imm_mask)
        print(f"  [immune cells] {g}: detection {before*100:.0f}% -> {after*100:.0f}%", flush=True)

# ---------- re-annotate on DECONTAMINATED counts (Ptprc-first gating) ----------
wc = a.copy(); sc.pp.normalize_total(wc, target_sum=1e4); sc.pp.log1p(wc)
clusters = sorted(pd.unique(a.obs["leiden"]), key=int)
def detg(gene):
    if gene not in wc.var_names: return pd.Series(0.0, index=clusters)
    v = np.asarray(wc[:, gene].X.todense()).ravel()
    return pd.Series((v > 0), index=wc.obs_names).groupby(a.obs["leiden"]).mean().reindex(clusters)
D = {g: detg(g) for g in ["Ptprc", "Pecam1", "Col1a1", "Acta2", "Krt14", "Krt5", "Klk1", "Krt19", "Aqp5"]}
def assign(cl):
    if D["Ptprc"][cl] > 0.35: return "Immune (parotid)"
    if D["Pecam1"][cl] > 0.30: return "Endothelial (parotid)"
    if D["Col1a1"][cl] > 0.30: return "Mesenchymal (parotid)"
    if D["Acta2"][cl] > 0.30 and max(D["Krt14"][cl], D["Krt5"][cl]) > 0.25: return "Myoepithelial (parotid)"
    if D["Klk1"][cl] > 0.30 or (D["Krt19"][cl] > 0.40 and D["Aqp5"][cl] < 0.30): return "Ductal (parotid)"
    return "Serous acinar (parotid)"
cl2ct = {cl: assign(cl) for cl in clusters}
a.obs["CellType"] = a.obs["leiden"].map(cl2ct).astype(str)
print("\ndecontaminated parotid types:", a.obs["CellType"].value_counts().to_dict(), flush=True)

# ---------- targeted ambient scrub (DecontX under-removes near-uniform ambient) ----------
# acinar SECRETED-PRODUCT genes are made only by epithelial secretory cells; their
# counts in non-acinar cells are 100% ambient -> zero them there (keep in acinar).
SECRETED = ["Bpifa2", "Bpifa1", "Amy1", "Amy2a5", "Car6", "Dcpp1", "Dcpp2", "Dcpp3",
            "Smgc", "Muc19", "Muc10", "Pip", "Prol1", "Prb1", "Prr27", "Smr3a", "Smr2", "Obp1a"]
sec_idx = [a.var_names.get_loc(g) for g in SECRETED if g in a.var_names]
nonac = np.where((a.obs["CellType"] != "Serous acinar (parotid)").values)[0]
Xl = a.X.tolil(); Xl[np.ix_(nonac, sec_idx)] = 0; a.X = Xl.tocsr()
print(f"scrubbed {len(sec_idx)} secreted-product genes from {len(nonac)} non-acinar cells", flush=True)
# re-validate Bpifa2 in immune cells
if "Bpifa2" in a.var_names:
    im2 = (a.obs["CellType"] == "Immune (parotid)").values
    bp = np.asarray(a.X[im2, a.var_names.get_loc("Bpifa2")].todense()).ravel()
    print(f"[after scrub] Bpifa2 in Immune(parotid): mean count {bp.mean():.2f}, detection {(bp>0).mean()*100:.0f}%", flush=True)

# ---------- write parotid ref (decontaminated counts) + combined ----------
raw = ad.AnnData(sp.csr_matrix(a.X), obs=a.obs[["CellType"]].copy(),
                 var=pd.DataFrame(index=a.var_names))
raw.obs["sample"] = "parotid_control"; raw.obs["study"] = "GSE223516_parotid"
raw.write(OUT_PAR)
comb = sc.read_h5ad(COMBINED_FULL)
comb.obs["CellType"] = comb.obs["CellType"].astype(str); comb.obs["study"] = comb.obs["study"].astype(str)
comb.obs["sample"] = comb.obs.get("sample", comb.obs["study"]).astype(str)
comb.obs = comb.obs[["CellType", "sample", "study"]]
common = comb.var_names.intersection(raw.var_names)
merged = ad.concat([comb[:, common].copy(), raw[:, common].copy()], join="outer", index_unique="-")
merged.obs["CellType"] = merged.obs["CellType"].astype("category")
merged.obs["study"] = merged.obs["study"].astype("category"); merged.X = sp.csr_matrix(merged.X)
merged.write(OUT_COMB)
print(f"\nwrote {OUT_COMB}: {merged.shape}, {merged.obs.CellType.nunique()} types", flush=True)
print("done", flush=True)
