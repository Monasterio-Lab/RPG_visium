#!/usr/bin/env python
"""Map the D1 Leiden clusters to the two anatomical glands and render the split
on the H&E for confirmation. Default proposal: cluster 2 (top acinar lobe) = SLG,
clusters 0/1/3 (duct-rich main body) = SMG. Override with env GLAND_MAP, e.g.
GLAND_MAP='SLG:2;SMG:0,1,3'."""
import os, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np, pandas as pd, scanpy as sc
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = "results/D1_cluster"; FIG = "results/figures_d1"
adata = sc.read_h5ad(f"{RES}/D1_clustered.h5ad")

spec = os.environ.get("GLAND_MAP", "SLG:2;SMG:0,1,3")
gland_of = {}
for part in spec.split(";"):
    g, cls = part.split(":")
    for c in cls.split(","):
        gland_of[c.strip()] = g.strip()
adata.obs["gland"] = adata.obs["leiden"].astype(str).map(gland_of).astype("category")
print("spots per gland:\n", adata.obs["gland"].value_counts().to_string(), flush=True)

adata.obs[["leiden", "gland"]].rename_axis("barcode").to_csv(f"{RES}/D1_spot_glands.csv")

pal = {"SMG": "#4E79A7", "SLG": "#E15759"}
sc.pl.spatial(adata, color="gland", img_key="hires", size=1.5, show=False,
              palette=pal, title="D1 proposed gland assignment (confirm before DEG)")
plt.savefig(f"{FIG}/D1_gland_assignment.png", dpi=200, bbox_inches="tight"); plt.close()
print("wrote", f"{FIG}/D1_gland_assignment.png", flush=True)
