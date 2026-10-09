#!/usr/bin/env python
"""Stage A: estimate per-cell-type expression signatures with cell2location's
negative-binomial regression model. CPU by default; GPU with ACCELERATOR=gpu. Paper run: REF_H5AD=data/reference/combined_parotid_ref.h5ad BATCH_KEY=study SIG_OUTDIR=results/reference_signatures_parotid.

NOTE: the os.environ guards MUST be set before numpy/torch import to avoid the
macOS duplicate-OpenMP deadlock (torch + MKL both ship libomp)."""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, scanpy as sc, torch, scvi, cell2location
from cell2location.models import RegressionModel
from cell2location.utils.filtering import filter_genes
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

torch.set_num_threads(4)
scvi.settings.seed = 0
ACCEL = os.environ.get("ACCELERATOR", "cpu")  # "gpu" on the server

REF = os.environ.get("REF_H5AD", "data/reference/adult_ref.h5ad")
OUTDIR = os.environ.get("SIG_OUTDIR", "results/reference_signatures")
BATCH_KEY = os.environ.get("BATCH_KEY") or None  # e.g. "study" for combined refs
os.makedirs(OUTDIR, exist_ok=True)

adata_ref = sc.read_h5ad(REF)

# --- gene filtering (cell2location recommended permissive thresholds) ---
selected = filter_genes(adata_ref, cell_count_cutoff=5,
                         cell_percentage_cutoff2=0.03, nonz_mean_cutoff=1.12)
adata_ref = adata_ref[:, selected].copy()
print(f"Genes after filtering: {adata_ref.n_vars}", flush=True)

# --- train NB regression model ---
if BATCH_KEY:
    print(f"Using batch_key={BATCH_KEY}", flush=True)
    RegressionModel.setup_anndata(adata=adata_ref, labels_key="CellType", batch_key=BATCH_KEY)
else:
    RegressionModel.setup_anndata(adata=adata_ref, labels_key="CellType")
mod = RegressionModel(adata_ref)
mod.train(max_epochs=250, accelerator=ACCEL)

mod.plot_history(20)
plt.savefig(f"{OUTDIR}/regression_train_history.png", dpi=120, bbox_inches="tight"); plt.close()

# --- export posterior / signatures ---
adata_ref = mod.export_posterior(
    adata_ref, sample_kwargs={"num_samples": 1000, "batch_size": 2500})
mod.save(f"{OUTDIR}/regression_model", overwrite=True)

keys = [f"means_per_cluster_mu_fg_{i}" for i in adata_ref.uns["mod"]["factor_names"]]
if "means_per_cluster_mu_fg" in adata_ref.varm:
    inf_aver = adata_ref.varm["means_per_cluster_mu_fg"][keys].copy()
else:
    inf_aver = adata_ref.var[keys].copy()
inf_aver.columns = adata_ref.uns["mod"]["factor_names"]
inf_aver.to_csv(f"{OUTDIR}/cell_type_signatures.csv")
adata_ref.write(f"{OUTDIR}/adata_ref_signatures.h5ad")
print("Signatures shape (genes x cell types):", inf_aver.shape, flush=True)
print("Cell types:", list(inf_aver.columns), flush=True)
print("Stage A done.", flush=True)
