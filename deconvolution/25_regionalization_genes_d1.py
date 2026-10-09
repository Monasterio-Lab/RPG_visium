"""Regionalization genes mapped on the SMG/SLG (D1) section in the same Loupe-style
format as the ORES panel (column groups core/proximal/distal, boxed gene names,
'Log2 Exp' colorbars). Colorbar maxima match the ORES panel for direct comparison;
the maps are near-empty because these foregut/airway genes are absent from SMG/SLG."""
import warnings, sys; warnings.filterwarnings("ignore"); sys.path.insert(0, "analysis")
import numpy as np, scanpy as sc
import matplotlib; matplotlib.use("Agg")
matplotlib.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42,
                            "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"]})
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from visium_io import load_visium
OUT = "results/figures_parotid"; BLUE = "#5B8FD4"
# (gene, colorbar max) matching the ORES panel; layout: columns = groups, rows = 2 genes
COLS = [("glandular core", [("Sbpl", 10), ("Gp2", 8)]),
        ("proximal", [("Bpifb2", 9), ("Bpifb6", 4)]),
        ("distal", [("Bpifb1", 12), ("Lipf", 8)])]

d = load_visium("SMG:SLG_Outs/D1/outs", "D1", img_key="hires"); d.var_names_make_unique()
sc.pp.normalize_total(d, target_sum=1e4)
XY = d.obsm["spatial"].astype(float)
def log2e(g):
    if g not in d.var_names: return np.zeros(d.n_obs)
    x = d[:, g].X; x = np.asarray(x.todense()).ravel() if hasattr(x, "todense") else np.asarray(x).ravel()
    return np.log2(x + 1)

fig = plt.figure(figsize=(13, 7.6))
# column / row geometry (figure coords)
col_x = [0.10, 0.43, 0.76]; cw = 0.19
row_y = [0.50, 0.07]; rh = 0.34
box = dict(boxstyle="round,pad=0.25", fc="white", ec=BLUE, lw=1.6)
for ci, (grp, genes) in enumerate(COLS):
    x0 = col_x[ci]
    # vertical group-label box (spans both rows)
    fig.text(x0 - 0.058, 0.455, grp, rotation=90, ha="center", va="center", fontsize=15, bbox=box)
    for ri, (g, vmax) in enumerate(genes):
        y0 = row_y[ri]; v = log2e(g)
        ax = fig.add_axes([x0 + 0.05, y0, cw, rh])
        ax.scatter(XY[:, 0], XY[:, 1], c=v, s=5, cmap="jet", vmin=0, vmax=vmax, linewidths=0)
        ax.set_aspect("equal"); ax.invert_yaxis(); ax.axis("off")
        ax.text(0.5, 1.06, g, transform=ax.transAxes, ha="center", va="bottom", fontsize=15, fontstyle="italic", bbox=box)
        # colorbar to the LEFT with 'Log2 Exp'
        cax = fig.add_axes([x0 + 0.012, y0 + 0.05, 0.014, rh - 0.13])
        cb = fig.colorbar(ScalarMappable(norm=Normalize(0, vmax), cmap="jet"), cax=cax)
        cb.set_ticks([0, vmax]); cb.ax.tick_params(labelsize=10)
        cax.set_ylabel("Log2 Exp", fontsize=11, labelpad=-6)
# header
fig.text(0.55, 0.965, "Regionalization genes", ha="center", va="center", fontsize=20, bbox=box)
fig.add_artist(Line2D([0.14, 0.96], [0.925, 0.925], color="black", lw=2.0))
fig.text(0.5, 0.005, "SMG/SLG (D1) section — same colorbar scale as the ORES panel", ha="center", fontsize=9, color="#666")
for ext in ["png", "svg", "pdf"]:
    fig.savefig(f"{OUT}/regionalization_genes_D1_loupe.{ext}", dpi=200 if ext == "png" else None, bbox_inches="tight", facecolor="white")
print("wrote regionalization_genes_D1_loupe; D1 max log2:", {g: round(float(log2e(g).max()), 1) for _, gs in COLS for g, _ in gs}, flush=True)
