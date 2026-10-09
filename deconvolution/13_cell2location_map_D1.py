#!/usr/bin/env python
"""Stage B for the D1 (SMG+SLG) Visium section: map the parotid-augmented
salivary reference onto D1 with cell2location. Mirror of 12_cell2location_map_A1.py but uses the
general load_visium loader (D1 has a valid hires image + headered positions)."""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
import warnings; warnings.filterwarnings("ignore")
import sys; sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd, scanpy as sc, torch, scvi, cell2location
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from visium_io import load_visium

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "8")))
scvi.settings.seed = 0
ACCEL = os.environ.get("ACCELERATOR", "cpu")
MAX_EPOCHS = int(os.environ.get("MAX_EPOCHS", "30000"))
N_CELLS_PER_LOCATION = int(os.environ.get("N_CELLS_PER_LOCATION", "20"))
DETECTION_ALPHA = float(os.environ.get("DETECTION_ALPHA", "20"))

VIS_DIR = os.environ.get("VIS_DIR", "VISIUM_outs/D1/outs")
LIBRARY = os.environ.get("LIBRARY_ID", "D1")
SIG_CSV = os.environ.get("SIG_CSV", "results/reference_signatures_parotid/cell_type_signatures.csv")
OUTDIR = os.environ.get("MAP_OUTDIR", "results/D1_combined_parotid")
os.makedirs(OUTDIR, exist_ok=True)

inf_aver = pd.read_csv(SIG_CSV, index_col=0)
print("Reference signatures:", inf_aver.shape, flush=True)

adata_vis = load_visium(VIS_DIR, library_id=LIBRARY, img_key="hires")
adata_vis.obs["sample"] = LIBRARY
print(f"Visium {LIBRARY}:", adata_vis.shape, flush=True)

adata_vis.var["MT"] = adata_vis.var_names.str.startswith(("mt-", "Mt-", "MT-"))
adata_vis.obsm["MT"] = adata_vis[:, adata_vis.var["MT"].values].X.toarray()
adata_vis = adata_vis[:, ~adata_vis.var["MT"].values].copy()

shared = [g for g in adata_vis.var_names if g in set(inf_aver.index)]
print(f"Shared genes Visium∩reference: {len(shared)}", flush=True)
adata_vis = adata_vis[:, shared].copy()
inf_aver = inf_aver.loc[shared, :].copy()

cell2location.models.Cell2location.setup_anndata(adata=adata_vis)
mod = cell2location.models.Cell2location(
    adata_vis, cell_state_df=inf_aver,
    N_cells_per_location=N_CELLS_PER_LOCATION, detection_alpha=DETECTION_ALPHA)
mod.view_anndata_setup()
mod.train(max_epochs=MAX_EPOCHS, batch_size=None, train_size=1, accelerator=ACCEL)
mod.plot_history(1000); plt.legend(labels=["full data training"])
plt.savefig(f"{OUTDIR}/mapping_train_history.png", dpi=120, bbox_inches="tight"); plt.close()

adata_vis = mod.export_posterior(
    adata_vis, sample_kwargs={"num_samples": 1000, "batch_size": mod.adata.n_obs})
mod.save(f"{OUTDIR}/cell2location_model", overwrite=True)
adata_vis.write(f"{OUTDIR}/{LIBRARY}_cell2location.h5ad")

abund = adata_vis.obsm["q05_cell_abundance_w_sf"].copy()
abund.columns = [c.replace("q05cell_abundance_w_sf_", "") for c in abund.columns]
abund.index = adata_vis.obs_names
abund.to_csv(f"{OUTDIR}/{LIBRARY}_cell_abundances_q05.csv")
prop = abund.div(abund.sum(axis=1), axis=0)
prop.to_csv(f"{OUTDIR}/{LIBRARY}_cell_type_proportions.csv")
print("Mean proportion per cell type:", flush=True)
print(prop.mean().sort_values(ascending=False).head(15), flush=True)
print("Stage B (D1) done.", flush=True)
