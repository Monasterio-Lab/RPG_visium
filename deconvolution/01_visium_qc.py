"""Per-DATASET Visium QC figures (they appear in different parts of the paper):
one self-contained 2x2 panel each for A1 (ORES) and D1 (SMG/SLG) — spatial UMI
homogeneity map + genes/UMIs/mito per-spot distributions."""
import os, sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "analysis")
import numpy as np, scanpy as sc
import matplotlib; matplotlib.use("Agg")
matplotlib.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
import matplotlib.pyplot as plt
from visium_io import load_visium_a1, load_visium

OUT = "results/figures_parotid"; os.makedirs(OUT, exist_ok=True)
COL = {"A1 (ORES)": "#4C72B0", "D1 (SMG/SLG)": "#C44E52"}

def load(name, kind):
    a = load_visium_a1() if kind == "a1" else load_visium("SMG:SLG_Outs/D1/outs", "D1", img_key="hires")
    a.var_names_make_unique()
    a.obs["umi"] = np.asarray(a.X.sum(1)).ravel()
    a.obs["genes"] = np.asarray((a.X > 0).sum(1)).ravel()
    mt = a.var_names.str.match(r"^mt-")
    a.obs["mito"] = np.asarray(a[:, mt].X.sum(1)).ravel() / np.maximum(a.obs["umi"], 1) * 100
    return a

def qc_fig(ad, name, kind, tag):
    c = COL[name]
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.24)

    # spatial UMI homogeneity
    ax = fig.add_subplot(gs[0, 0]); xy = ad.obsm["spatial"].astype(float); v = np.log10(ad.obs["umi"] + 1)
    scb = ax.scatter(xy[:, 0], xy[:, 1], s=9, c=v, cmap="viridis",
                     vmin=np.percentile(v, 2), vmax=np.percentile(v, 98), linewidths=0)
    ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    cv = ad.obs["umi"].std() / ad.obs["umi"].mean()
    ax.set_title(f"UMIs per spot (log10) — spatial\nhomogeneity CV = {cv:.2f}", fontsize=12, fontweight="bold")
    cb = fig.colorbar(scb, ax=ax, fraction=0.046, pad=0.02); cb.set_label("log10 UMI", fontsize=9)

    def hist(ax, col, xlab, title):
        d = ad.obs[col].values; med = np.median(d)
        ax.hist(d, bins=45, color=c, alpha=0.8, edgecolor="white", linewidth=0.3)
        ax.axvline(med, color="#222", ls="--", lw=1.4)
        ax.text(0.97, 0.95, f"median = {med:.0f}" if col != "mito" else f"median = {med:.1f}%",
                transform=ax.transAxes, ha="right", va="top", fontsize=10.5, fontweight="bold")
        ax.set_xlabel(xlab, fontsize=11); ax.set_ylabel("spots", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        for s in ["top", "right"]: ax.spines[s].set_visible(False)
    hist(fig.add_subplot(gs[0, 1]), "genes", "genes per spot", "Genes per spot")
    hist(fig.add_subplot(gs[1, 0]), "umi", "UMIs per spot", "UMIs per spot")
    hist(fig.add_subplot(gs[1, 1]), "mito", "mitochondrial %", "Mitochondrial content")

    n = ad.n_obs
    fig.suptitle(f"{name} — Visium QC  (n = {n:,} spots under tissue)", fontsize=15, fontweight="bold", y=0.98)
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(f"{OUT}/{tag}.{ext}", dpi=200 if ext == "png" else None, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {tag}: n={n}, median genes={int(np.median(ad.obs['genes']))}, "
          f"median UMI={int(np.median(ad.obs['umi']))}, median mito={np.median(ad.obs['mito']):.2f}%", flush=True)

qc_fig(load("A1 (ORES)", "a1"), "A1 (ORES)", "a1", "visium_qc_A1_ORES")
qc_fig(load("D1 (SMG/SLG)", "d1"), "D1 (SMG/SLG)", "d1", "visium_qc_D1_SMG_SLG")
