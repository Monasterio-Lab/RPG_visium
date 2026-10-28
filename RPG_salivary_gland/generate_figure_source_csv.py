#!/usr/bin/env python
"""Source-data CSVs for Fig. 2B, 2E-2J and Suppl. Figs. 1, 5B, 7A, 8, 9, 10, 11, 12.

Reads saved outputs  and the A1 Visium data (V10S15-395_A1). 

Usage:
    python generate_figure_source_csv.py [--data <repo>/data] [--out <repo>/figure_source_csv]
    --data holds VISIUM_outs/V10S15-395_A1, VISIUM_outs/V13U19-349_D1 (GEO) and the Zenodo folders
    (annotations, cell2location, integration, reference, nmf, HPA). Defaults assume this script sits in
    <repo>/RPG_salivary_gland/.
Requires: scanpy 1.12.1, pandas, numpy, scipy, scikit-learn, statsmodels, Pillow.
"""
import argparse, json, os, re
import numpy as np, pandas as pd, scanpy as sc
from PIL import Image
from scipy.spatial import cKDTree
Image.MAX_IMAGE_PIXELS = None

# ---------------------------------------------------------------- helpers
def newname(ct):  # = analysis/rename_celltypes.py
    ct = str(ct)
    if ct.endswith("(mix)"): return ct[:-5].rstrip() + " (SMG/SLG)"
    if ct.endswith("(parotid)"): return ct[:-9].rstrip() + " (Parotid)"
    if ct.endswith("(SLG)"): return ct
    return ct + " (SMG)"
def rename_lineage(d): return {L: [newname(c) for c in cs] for L, cs in d.items()}

# 6-lineage grouping used for the scatter-pies
LIN6 = rename_lineage({
 "Acinar-Serous": ["Bpifa2+", "Serous acinar (mix)", "Acinar", "Seromucous acinar (mix)", "Serous acinar (parotid)"],
 "Acinar-Mucous": ["Mucous acinar (SLG)", "Smgc+"],
 "Ductal": ["Ductal (mix)", "Striated duct", "Basal duct", "Intercalated duct", "Ascl3+ duct", "GCT",
            "Myoepithelial", "Myoepithelial (mix)", "Ductal (parotid)"],
 "Stromal/Mesench": ["Mesenchymal (mix)", "Stromal", "Mesenchymal (parotid)", "Endothelial",
                     "Vascular (mix)", "Endothelial (parotid)"],
 "Immune": ["Immune (mix)", "Macrophages", "NK cells", "Immune (parotid)"]})
CATS6 = ["Acinar-Serous", "Acinar-Mucous", "Ductal", "Stromal/Mesench", "Immune", "Others"]
# 8-lineage grouping
LIN8 = rename_lineage({
 "Acinar-Serous": ["Bpifa2+", "Serous acinar (mix)", "Acinar", "Seromucous acinar (mix)", "Serous acinar (parotid)"],
 "Acinar-Mucous": ["Mucous acinar (SLG)", "Smgc+"],
 "Ductal": ["Ductal (mix)", "GCT", "Striated duct", "Basal duct", "Intercalated duct", "Ascl3+ duct", "Ductal (parotid)"],
 "Myoepithelial": ["Myoepithelial (mix)", "Myoepithelial"],
 "Stromal/Mesench": ["Mesenchymal (parotid)", "Mesenchymal (mix)", "Stromal"],
 "Endothelial/Vasc": ["Endothelial (parotid)", "Endothelial", "Vascular (mix)"],
 "Immune": ["Macrophages", "Immune (parotid)", "NK cells", "Immune (mix)"],
 "Others": ["Erythroid"]})

def lin6(prop):
    assigned = sum(LIN6.values(), [])
    lin = pd.DataFrame({g: prop[[c for c in m if c in prop.columns]].sum(axis=1) for g, m in LIN6.items()}, index=prop.index)
    lin["Others"] = prop.drop(columns=[c for c in assigned if c in prop.columns]).sum(axis=1)
    return lin[CATS6].div(lin[CATS6].sum(axis=1), axis=0)

POSCOLS = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"]

class Section:
    """Spot positions + the hires-image scaling the plotting scripts use for spot geometry."""
    def __init__(self, h5, positions, spdir, header):
        a = sc.read_10x_h5(h5); a.var_names_make_unique(); self.adata = a
        pos = pd.read_csv(positions) if header else pd.read_csv(positions, header=None, names=POSCOLS)
        self.pos = pos.set_index("barcode").loc[a.obs_names]
        sf = json.load(open(f"{spdir}/scalefactors_json.json"))
        Wh, Hh = Image.open(f"{spdir}/tissue_hires_image.png").size
        lo = Image.open(f"{spdir}/tissue_lowres_image.png").size
        fw, fh = lo[0] / sf["tissue_lowres_scalef"], lo[1] / sf["tissue_lowres_scalef"]
        self.sx, self.sy = Wh / fw, Hh / fh
        self.spatial = self.pos[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy().astype(float)  # = obsm['spatial']
        self.XY = self.spatial * np.array([self.sx, self.sy])                                         # plot coords
    def coords(self, bcs):
        p = self.pos.loc[bcs, ["array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"]]
        return p.reset_index().rename(columns={"index": "barcode"})

def save(df, out, name, info):
    path = os.path.join(out, name); df.to_csv(path, index=False, float_format="%.6g")
    info.append((name, df.shape)); print(f"wrote {name}: {df.shape}", flush=True)

# ---------------------------------------------------------------- panels
def fig2B(a, nmf, out, info):
    """Fig. 2B"""
    d = pd.read_csv(nmf)
    cols = ["spaceranger_barcode", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres",
            "factor_n6", "factor_n27", "factor_n33", "factor_n36"]
    save(d[cols].rename(columns={"spaceranger_barcode": "barcode"}), out, "Fig2B.csv", info)

def fig2E_S9(d1, H, out, info):
    """Fig. 2E and Suppl. Fig. 9"""
    prop = pd.read_csv(f"{H}/cell2location/V13U19-349_D1/V13U19-349_D1_cell_type_proportions.csv", index_col=0)
    abund = pd.read_csv(f"{H}/cell2location/V13U19-349_D1/V13U19-349_D1_cell_abundances_q05.csv", index_col=0)
    bcs = prop.index; assert list(bcs) == list(d1.adata.obs_names)
    lin = lin6(prop)
    gl = pd.read_csv(f"{H}/annotations/V13U19-349_D1_gland_assignment.csv").set_index("barcode")["gland"].reindex(bcs)
    XY = d1.XY
    def rep_mean(gland):
        idx = np.where((gl == gland).to_numpy())[0]
        d = np.linalg.norm(XY[idx] - XY[idx].mean(0), axis=1); central = idx[d <= np.percentile(d, 50)]
        Lall = lin.loc[bcs[idx], CATS6].values; mean = (Lall / Lall.sum(1, keepdims=True)).mean(0)
        Lc = lin.loc[bcs[central], CATS6].values; Lc = Lc / Lc.sum(1, keepdims=True)
        return bcs[central[np.argmin(np.linalg.norm(Lc - mean, axis=1))]]
    def subzone(gland, score, frac=0.30):
        idx = np.where((gl == gland).to_numpy())[0]
        s = score(lin.loc[bcs[idx], CATS6]); k = max(3, int(len(idx) * frac))
        top = idx[np.argsort(s)[-k:]]; return bcs[top[np.linalg.norm(XY[top] - XY[top].mean(0), axis=1).argmin()]]
    picks = {rep_mean("SLG"): "SLG representative (acinar-mucous-rich)",
             subzone("SMG", lambda L: (L["Acinar-Serous"] - L["Ductal"]).values): "SMG acinar-serous-rich",
             subzone("SMG", lambda L: (L["Ductal"] - L["Acinar-Serous"]).values): "SMG duct-rich"}
    df = d1.coords(bcs)
    df["gland"] = gl.fillna("not assigned (failed D1 QC)").values
    for c in CATS6: df[c] = lin[c].values
    df["enlarged_pie"] = [picks.get(b, "") for b in bcs]
    save(df, out, "Fig2E.csv", info)
    df = d1.coords(bcs); df["gland"] = gl.fillna("not assigned (failed D1 QC)").values
    df = pd.concat([df, abund.loc[bcs].reset_index(drop=True)], axis=1)
    save(df, out, "FigS9.csv", info)
    return picks

def roi_set(H):
    return set(pd.read_csv(f"{H}/annotations/V10S15-395_A1_RPG_ROI_loupe.csv")["Barcode"])

def fig2F(a1, H, out, info):
    """Fig. 2F"""
    from sklearn.cluster import KMeans
    prop = pd.read_csv(f"{H}/cell2location/V10S15-395_A1/V10S15-395_A1_cell_type_proportions.csv", index_col=0)
    assert list(prop.index) == list(a1.adata.obs_names)
    lin = lin6(prop); ROI = roi_set(H)
    names = a1.adata.obs_names; in_roi = np.array([b in ROI for b in names]); idx = np.where(in_roi)[0]
    roi_xy, roi_bcs = a1.XY[idx], names[idx]
    km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(roi_xy); reg = km.labels_
    sizes = pd.Series(reg).value_counts(); big = sizes.idxmax(); others = [r for r in sizes.index if r != big]
    imm = {r: lin.loc[roi_bcs[reg == r], "Immune"].mean() for r in others}
    distal = max(imm, key=imm.get); centre = [r for r in others if r != distal][0]
    RN = {big: "Upper (largest)", centre: "Centre", distal: "Distal (immune-rich)"}
    def pick_centre(region):
        i = np.where(reg == region)[0]; p = roi_xy[i]; return roi_bcs[i[np.linalg.norm(p - p.mean(0), axis=1).argmin()]]
    def pick_subzone(region, score, frac=0.30):
        i = np.where(reg == region)[0]; s = score(lin.loc[roi_bcs[i]]); k = max(3, int(len(i) * frac))
        top = i[np.argsort(s)[-k:]]; return roi_bcs[top[np.linalg.norm(roi_xy[top] - roi_xy[top].mean(0), axis=1).argmin()]]
    picks = {pick_subzone(big, lambda L: (L["Acinar-Mucous"] - L["Ductal"]).values): "Upper - Acinar-Mucous core",
             "ACTACCAGCTCTCTGG-1": "Upper - Ductal-rich zone (spot hard-coded in the script)",
             pick_centre(centre): "Centre region", pick_centre(distal): "Distal (immune-rich)"}
    df = a1.coords(roi_bcs); df["area_kmeans"] = [RN[r] for r in reg]
    for c in CATS6: df[c] = lin.loc[roi_bcs, c].values
    df["enlarged_pie"] = [picks.get(b, "") for b in roi_bcs]
    save(df, out, "Fig2F.csv", info)
    return df

def fig2G(H, out, info):
    """Fig. 2G"""
    ROI = roi_set(H)
    A = pd.read_csv(f"{H}/cell2location/V10S15-395_A1/V10S15-395_A1_cell_type_proportions.csv", index_col=0)
    D = pd.read_csv(f"{H}/cell2location/V13U19-349_D1/V13U19-349_D1_cell_type_proportions.csv", index_col=0)
    gl = pd.read_csv(f"{H}/annotations/V13U19-349_D1_gland_assignment.csv").set_index("barcode")["gland"]
    comp = pd.DataFrame({"SLG": D.loc[[b for b in D.index if b in set(gl[gl == "SLG"].index)]].mean() * 100,
                         "RPG": A.loc[[b for b in A.index if b in ROI]].mean() * 100,
                         "SMG": D.loc[[b for b in D.index if b in set(gl[gl == "SMG"].index)]].mean() * 100}).fillna(0)
    lin = pd.DataFrame({L: comp.reindex([c for c in cs if c in comp.index]).sum() for L, cs in LIN8.items()}).T
    n = {"SLG": int((gl == "SLG").sum()), "RPG": len(ROI), "SMG": int((gl == "SMG").sum())}
    df = lin.reset_index().rename(columns={"index": "lineage"})
    df.columns = ["lineage"] + [f"{g} (% mean over {n[g]} spots)" for g in ["SLG", "RPG", "SMG"]]
    save(df, out, "Fig2G.csv", info)
    return lin

def fig2HI_S10(a1, H, out, info):
    """Fig. 2H, 2I and Suppl. Fig. 10"""
    prop = pd.read_csv(f"{H}/cell2location/V10S15-395_A1/V10S15-395_A1_cell_type_proportions.csv", index_col=0)
    abund = pd.read_csv(f"{H}/cell2location/V10S15-395_A1/V10S15-395_A1_cell_abundances_q05.csv", index_col=0)
    ROI = roi_set(H); names = a1.adata.obs_names; XY = a1.XY
    in_roi = np.array([b in ROI for b in names]); idx = np.where(in_roi)[0]; bcs = names[idx]; xy = XY[idx]
    padc = 0.35 * max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]))
    gx0, gx1 = xy[:, 0].min() - padc, xy[:, 0].max() + padc; gy0, gy1 = xy[:, 1].min() - padc, xy[:, 1].max() + padc
    inwin = (XY[:, 0] >= gx0) & (XY[:, 0] <= gx1) & (XY[:, 1] >= gy0) & (XY[:, 1] <= gy1)
    ctx = names[inwin & ~in_roi]
    dom = prop.loc[bcs].idxmax(axis=1)
    df = pd.concat([a1.coords(bcs).assign(spot_type="RPG ROI", dominant_cell_type=dom.values),
                    a1.coords(ctx).assign(spot_type="context (grey)", dominant_cell_type="")], ignore_index=True)
    save(df, out, "Fig2H.csv", info)
    four = ["Mucous acinar (SLG)", "Bpifa2+ (SMG)", "Basal duct (SMG)", "Striated duct (SMG)"]
    roi_part = pd.concat([a1.coords(bcs).assign(spot_type="RPG ROI"), abund.loc[bcs].reset_index(drop=True)], axis=1)
    ctx_part = a1.coords(ctx).assign(spot_type="context (grey)")
    save(pd.concat([roi_part[list(roi_part.columns[:6]) + four], ctx_part], ignore_index=True), out,
         "Fig2I.csv", info)
    save(pd.concat([roi_part, ctx_part], ignore_index=True), out, "FigS10.csv", info)
    return dom

def fig2J(H, out, info):
    """Fig. 2J"""
    g = sc.read_h5ad(f"{H}/integration/glandular_spots_RPG_SLG_SMG.h5ad")
    df = pd.DataFrame({"spot_id": g.obs_names, "section": g.obs["section"].astype(str).values,
                       "gland": g.obs["gland"].astype(str).values,
                       "UMAP1": g.obsm["X_umap"][:, 0], "UMAP2": g.obsm["X_umap"][:, 1]})
    df["barcode"] = [n.rsplit("-", 1)[0] if n.endswith(("-A1", "-D1")) else n for n in df.spot_id]
    save(df[["barcode", "section", "gland", "UMAP1", "UMAP2"]], out, "Fig2J.csv", info)
    return df

def s1(a1, out, info):
    """Suppl. Fig. 1"""
    a = a1.adata
    umi = np.asarray(a.X.sum(1)).ravel(); genes = np.asarray((a.X > 0).sum(1)).ravel()
    mt = a.var_names.str.match(r"^mt-"); mito = np.asarray(a[:, mt].X.sum(1)).ravel() / np.maximum(umi, 1) * 100
    df = a1.coords(a.obs_names); df["UMIs"] = umi; df["log10_UMIs_plus1"] = np.log10(umi + 1)
    df["genes_detected"] = genes; df["mito_percent"] = mito
    save(df, out, "FigS1.csv", info)
    return df

def s5B(d1, out, info):
    """Suppl. Fig. 5B"""
    a = d1.adata
    umi = np.asarray(a.X.sum(1)).ravel(); genes = np.asarray((a.X > 0).sum(1)).ravel()
    mt = a.var_names.str.match(r"^mt-"); mito = np.asarray(a[:, mt].X.sum(1)).ravel() / np.maximum(umi, 1) * 100
    df = d1.coords(a.obs_names); df["UMIs"] = umi; df["log10_UMIs_plus1"] = np.log10(umi + 1)
    df["genes_detected"] = genes; df["mito_percent"] = mito
    save(df, out, "FigS5B.csv", info)
    return df

def s7A(H, out, info):
    """Suppl. Fig. 7A"""
    R = pd.read_csv(f"{H}/HPA/hpa_rna_nTPM_wide.csv", index_col=0)
    P = pd.read_csv(f"{H}/HPA/hpa_protein_IHC_wide.csv", index_col=0)
    LV = {"Not detected": 0, "Low": 1, "Medium": 2, "High": 3}
    Pn = P.apply(lambda s: s.map(LV))
    keep = (R.max(1) >= 1) | (Pn.max(1) >= 1); R, P, Pn = R[keep], P[keep], Pn[keep]
    rows = []
    for g in R.columns:
        gmax = max(float(np.log10(R[g].fillna(0).values + 1.0).max()), 1e-9)
        for org in R.index:
            v = R.loc[org, g]
            rows.append({"organ": org, "gene": g, "RNA_nTPM": v,
                         "RNA_plotted_log10_nTPM_plus1_scaled_to_gene_max": np.log10(v + 1) / gmax if pd.notna(v) else np.nan,
                         "protein_IHC_level": P.loc[org, g]})
    save(pd.DataFrame(rows), out, "FigS7A.csv", info)

def s8(H, out, info):
    """Suppl. Fig. 8A/B and 8C"""
    ref = sc.read_h5ad(f"{H}/reference/scRNA_reference_SMG_SLG_parotid.h5ad")
    ct2lin = {c: L for L, cs in LIN8.items() for c in cs}
    STUDY = {"GSE216476_mix": "GSE216476 - SMG/SLG (Huang et al. 2022)", "SMG_iScience": "GSE150327 - SMG (Hauser et al. 2020)",
             "GSE223516_parotid": "GSE223516 - Parotid (Rheinheimer et al. 2023)"}
    df = pd.DataFrame({"cell_id": ref.obs_names, "dataset": ref.obs["study"].astype(str).map(STUDY).values,
                       "cell_type": ref.obs["CellType"].astype(str).values})
    df["lineage"] = df.cell_type.map(ct2lin).fillna("Others")
    save(df, out, "FigS8AB.csv", info)
    # 8C: replicate 10_reference_marker_dotplot
    ad = ref.copy(); ad.obs["CellType"] = ad.obs["CellType"].astype("category")
    sc.pp.normalize_total(ad, target_sum=1e4); sc.pp.log1p(ad); sc.pp.filter_genes(ad, min_cells=10)
    sc.tl.rank_genes_groups(ad, "CellType", method="wilcoxon", n_genes=30)
    res = ad.uns["rank_genes_groups"]; groups = list(res["names"].dtype.names)
    LIN_ORDER = ["acinar", "serous", "mucous", "smgc", "bpifa2", "gct", "duct", "myoep", "mesench", "stromal",
                 "vascular", "endothel", "immune", "macroph", "nk", "erythro"]
    def lin_rank(ct):
        cl = ct.lower()
        for i, k in enumerate(LIN_ORDER):
            if k in cl: return i
        return len(LIN_ORDER)
    order = sorted(groups, key=lambda g: (lin_rank(g), g))
    marker_map, seen = {}, set()
    for g in order:
        names = [res["names"][g][i] for i in range(len(res["names"][g]))]
        picks = [n for n in names if n not in seen][:3]
        for n in picks: seen.add(n)
        marker_map[g] = picks if picks else names[:3]
    genes = [x for g in order for x in marker_map[g]]
    X = ad[:, genes].X; X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    ct = ad.obs["CellType"].astype(str).values
    rows = []
    for t in order:
        m = ct == t; sub = X[m]
        for j, gname in enumerate(genes):
            rows.append({"cell_type": t, "n_cells": int(m.sum()), "gene": gname,
                         "marker_of": [k for k, v in marker_map.items() if gname in v][0],
                         "fraction_expressing": float((sub[:, j] > 0).mean()), "mean_expression": float(sub[:, j].mean())})
    d = pd.DataFrame(rows)
    rng = d.groupby("gene")["mean_expression"].agg(["min", "max"])
    d["mean_expression_scaled_per_gene"] = [(r.mean_expression - rng.loc[r.gene, "min"]) / max(rng.loc[r.gene, "max"] - rng.loc[r.gene, "min"], 1e-12) for r in d.itertuples()]
    save(d, out, "FigS8C.csv", info)
    return ref, marker_map, order

SAL = ['Aqp5','Bhlha15','Bpifa2','Smgc','Dcpp1','Dcpp2','Dcpp3','Muc19','A2ml1','Pip','Prol1',
       'Mucl2','Car6','Klk1','Egf','Wfdc18','Cyp2f2','Fxyd2','Smr2','Scgb2b27','Scgb1b27']
MUC = ['Muc5b','Tff1','Tff2','Gkn1','Gkn2','Pgc','Gif','Chia1','Krt13','Krt4','Krt6a','Krt76',
       'Krtdap','Sbpl','Krt15','Krt16','Crnn','Sprr1a','Lgals7','Dsp','Bpifb2','Bpifb6','Aqp3','Krt5','Krt14']
MUS = ['Acta1','Actn2','Actn3','Myh1','Myh2','Myh3','Myh4','Myh7','Myh8','Myl1','Mylpf','Mybpc1',
       'Mybpc2','Tnnt1','Tnnt3','Tnni1','Tnni2','Tnnc2','Ttn','Ckm','Ckmt2','Des','Mb','Pgam2',
       'Pvalb','Atp2a1','Casq1','Ryr1','Neb','Tcap','Myoz1','Tpm2','Ldb3','Smpx','Klhl41','Pygm','Eno3']

def s11A(a1, H, out, info):
    """Suppl. Fig. 11A"""
    a = a1.adata.copy(); sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
    for nm, gs in [("sal", SAL), ("muc", MUC), ("mus", MUS)]:
        sc.tl.score_genes(a, [g for g in gs if g in a.var_names], score_name=nm)
    gscore = a.obs["sal"] - a.obs["muc"] - a.obs["mus"]
    g = sc.read_h5ad(f"{H}/integration/glandular_spots_RPG_SLG_SMG.h5ad", backed="r")
    kept = {n[:-3] if n.endswith("-A1") else n for n in g.obs_names[g.obs["gland"] == "RPG"]}
    ROI = roi_set(H); names = a.obs_names
    df = a1.coords(names)
    df["category"] = ["RPG kept (glandular)" if b in kept else ("RPG dropped (mucosa/muscle)" if b in ROI else "outside ROI") for b in names]
    df["salivary_score"] = a.obs["sal"].values; df["mucosal_foregut_score"] = a.obs["muc"].values
    df["muscle_score"] = a.obs["mus"].values; df["glandularity_index"] = gscore.values
    save(df, out, "FigS11A.csv", info)
    return df, kept

def s11B(a1, ref, H, out, info):
    """Suppl. Fig. 11B"""
    from scipy.stats import mannwhitneyu
    r = ref.copy(); r.obs["CellType"] = r.obs["CellType"].astype(str)
    sc.pp.normalize_total(r, target_sum=1e4); sc.pp.log1p(r)
    sc.tl.rank_genes_groups(r, "CellType", method="wilcoxon", pts=True, n_genes=400)
    groups = list(r.uns["rank_genes_groups"]["names"].dtype.names)
    GLAND_EPI = ["Bpifa2+ (SMG)", "Serous acinar (SMG/SLG)", "Acinar (SMG)", "Seromucous acinar (SMG/SLG)", "Mucous acinar (SLG)", "Smgc+ (SMG)"]
    gland_types = [g for g in GLAND_EPI if g in groups]
    degdf = sc.get.rank_genes_groups_df(r, group=None)
    a = a1.adata.copy(); sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
    HK = re.compile(r"^(Rp[sl]\d|mt-|Hb[ab]-|Cox\d|Uqcr|Nduf|Atp5|Atp1[ab]|Hsp|Hist\d|H[0-9]af|H3f|H1f|"
                    r"Malat1|Gm\d|Rik$|Actg|Actb|Ptma|Naca|Eef|Ppia|Serf|Prdx|Tmsb|Fau|Tpt1|Ubc|Snhg|"
                    r"Gas5|Chchd|Nme\d|Taldo|Hmgn|Jun$|Mif$|Sfn$|Ndufa|Serp|Sec61|Chchd)")
    sig = set()
    for g in gland_types:
        d = degdf[(degdf.group == g) & (degdf.logfoldchanges > 1.5) & (degdf.pct_nz_group > 0.40) &
                  (degdf.pct_nz_reference < 0.35) & (~degdf.names.str.match(HK)) & (degdf.names.isin(a.var_names))]
        sig.update(d.sort_values("logfoldchanges", ascending=False)["names"].head(20))
    sig = sorted(sig)
    sc.tl.score_genes(a, sig, score_name="sal_dd")
    ROI = roi_set(H); names = a.obs_names
    df = a1.coords(names); df["group"] = ["RPG" if b in ROI else "Non-glandular" for b in names]
    df["datadriven_salivary_score"] = a.obs["sal_dd"].values
    save(df, out, "FigS11B.csv", info)
    rr, nn = df[df.group == "RPG"].datadriven_salivary_score, df[df.group != "RPG"].datadriven_salivary_score
    U, p = mannwhitneyu(rr, nn, alternative="greater")
    return df, sig, dict(n_rpg=len(rr), n_rest=len(nn), mean_rpg=rr.mean(), mean_rest=nn.mean(), auc=U / (len(rr) * len(nn)), p=p)

def s12AB(a1, H, out, info):
    """Suppl. Fig. 12A/B"""
    from sklearn.cluster import DBSCAN
    from scipy.interpolate import splprep, splev
    from statsmodels.nonparametric.smoothers_lowess import lowess
    CORE = ["Aqp5", "Gp2", "Muc5b", "Cldn10", "Kcnn4", "Sbpl"]
    PROX = ["Muc19", "Dcpp1", "Dcpp2", "Smgc", "Bpifb2", "Bpifb6", "Wfdc12", "Cgref1"]
    DIST = ["Lyz2", "Bpifb1", "Lipf"]
    a = a1.adata.copy(); sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
    g = sc.read_h5ad(f"{H}/integration/glandular_spots_RPG_SLG_SMG.h5ad", backed="r")
    kept100 = set(b[:-3] if b.endswith("-A1") else b for b in g.obs_names[g.obs["gland"] == "RPG"])
    inroi = np.array([b in kept100 for b in a.obs_names])
    rbc0 = a.obs_names[inroi]; rxy0 = a1.spatial[inroi]
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
    prop = pd.read_csv(f"{H}/cell2location/V10S15-395_A1/V10S15-395_A1_cell_type_proportions.csv", index_col=0)
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
    foot = np.c_[cx[idx], cy[idx]]; rollc = a1.spatial.mean(0)
    perp = np.sign(((rollc - foot) * (rxy - foot)).sum(1)) * np.linalg.norm(rxy - foot, axis=1) / nn
    order_rn = sorted(["Proximal", "Central", "Distal"], key=lambda r: s[region == r].mean())
    coord = np.zeros(len(s)); off = 0.0
    for r in order_rn:
        m = region == r; coord[m] = off + (s[m] - s[m].min()); off += s[m].max() - s[m].min()
    coord /= off
    b1, b2 = coord[region == "Proximal"].max(), coord[region == "Central"].max()
    df = a1.coords(rbc); df["region"] = region; df["unrolled_coordinate"] = coord
    df["across_axis_offset_spot_pitch"] = perp
    df["axis_foot_pxl_col_fullres"] = foot[:, 0]; df["axis_foot_pxl_row_fullres"] = foot[:, 1]
    save(df, out, "FigS12A_B_unrolled_spots.csv", info)
    save(pd.DataFrame({"axis_point": np.arange(800), "pxl_col_in_fullres": cx, "pxl_row_in_fullres": cy}), out,
         "FigS12A_medial_axis.csv", info)
    grid = np.linspace(0, 1, 240); prof = {}
    for genes, grp in [(CORE, "core"), (PROX, "proximal"), (DIST, "distal")]:
        for gname in genes:
            if gname not in a.var_names: continue
            x = a[rbc, gname].X; v = np.asarray(x.todense()).ravel() if hasattr(x, "todense") else np.asarray(x).ravel()
            sm = lowess(v, coord, frac=0.5, return_sorted=True)
            y = np.interp(grid, sm[:, 0], sm[:, 1]); prof[(grp, gname)] = y / (y.max() + 1e-9)
    H_ = pd.DataFrame({"unrolled_coordinate": grid})
    for (grp, gname), y in prof.items(): H_[f"{grp}:{gname}"] = y
    save(H_, out, "FigS12B_heatmap.csv", info)
    return df, (b1, b2), sz

def s12C(d1, out, info):
    """Suppl. Fig. 12C"""
    a = d1.adata.copy(); sc.pp.normalize_total(a, target_sum=1e4)
    df = d1.coords(a.obs_names)
    for gname in ["Sbpl", "Gp2", "Bpifb2", "Bpifb6", "Bpifb1", "Lipf"]:
        if gname in a.var_names:
            x = a[:, gname].X; x = np.asarray(x.todense()).ravel() if hasattr(x, "todense") else np.asarray(x).ravel()
        else:
            x = np.zeros(a.n_obs)
        df[f"{gname}_log2_norm_plus1"] = np.log2(x + 1)
    save(df, out, "FigS12C.csv", info)
    return df

# ---------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
    ap.add_argument("--data", default=f"{ROOT}/data"); ap.add_argument("--out", default=f"{ROOT}/figure_source_csv")
    ap.add_argument("--only", default="")
    A = ap.parse_args(); os.makedirs(A.out, exist_ok=True); H = A.data; info = []
    A1DIR = f"{H}/VISIUM_outs/V10S15-395_A1"; D1DIR = f"{H}/VISIUM_outs/V13U19-349_D1"
    NMF = f"{H}/nmf/se.VAC.d0_NMF_spot_activity.csv"
    a1 = Section(f"{A1DIR}/filtered_feature_bc_matrix.h5", f"{A1DIR}/spatial/tissue_positions_list.csv", f"{A1DIR}/spatial", header=False)
    d1 = Section(f"{D1DIR}/filtered_feature_bc_matrix.h5", f"{D1DIR}/spatial/tissue_positions.csv", f"{D1DIR}/spatial", header=True)
    s1(a1, A.out, info)
    fig2B(a1, NMF, A.out, info); fig2E_S9(d1, H, A.out, info); fig2F(a1, H, A.out, info); fig2G(H, A.out, info)
    fig2HI_S10(a1, H, A.out, info); fig2J(H, A.out, info); s5B(d1, A.out, info); s7A(H, A.out, info)
    ref, _, _ = s8(H, A.out, info); s11A(a1, H, A.out, info); s11B(a1, ref, H, A.out, info)
    s12AB(a1, H, A.out, info); s12C(d1, A.out, info)
