"""Robust Visium loader for A1 that avoids the corrupted hires PNG.

Reads the filtered matrix + tissue positions manually and attaches the working
lowres image into adata.uns['spatial'] so scanpy's sc.pl.spatial works with
img_key='lowres'. Coordinate convention matches scanpy.read_visium:
obsm['spatial'] = [pxl_col_in_fullres, pxl_row_in_fullres]."""
import json, numpy as np, pandas as pd, scanpy as sc
from PIL import Image


def load_visium(outs_dir, library_id, img_key="hires"):
    """General Visium loader. Handles both Space Ranger position formats
    (`tissue_positions_list.csv` headerless and `tissue_positions.csv` with a
    header) and attaches the requested image to uns['spatial']. Coordinate
    convention matches scanpy: obsm['spatial'] = [pxl_col, pxl_row]."""
    import os
    adata = sc.read_10x_h5(f"{outs_dir}/filtered_feature_bc_matrix.h5")
    adata.var_names_make_unique()
    spdir = f"{outs_dir}/spatial"
    cols = ["barcode", "in_tissue", "array_row", "array_col",
            "pxl_row_in_fullres", "pxl_col_in_fullres"]
    if os.path.exists(f"{spdir}/tissue_positions.csv"):
        pos = pd.read_csv(f"{spdir}/tissue_positions.csv").set_index("barcode")
    else:
        pos = pd.read_csv(f"{spdir}/tissue_positions_list.csv",
                          header=None, names=cols).set_index("barcode")
    pos = pos.loc[adata.obs_names]
    adata.obs[["in_tissue", "array_row", "array_col"]] = pos[["in_tissue", "array_row", "array_col"]]
    adata.obsm["spatial"] = pos[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy()
    with open(f"{spdir}/scalefactors_json.json") as f:
        sf = json.load(f)
    img_file = "tissue_hires_image.png" if img_key == "hires" else "tissue_lowres_image.png"
    img = np.asarray(Image.open(f"{spdir}/{img_file}").convert("RGB")).astype(np.float32)
    img = img / (65535.0 if img.max() > 255 else 255.0)
    adata.uns["spatial"] = {library_id: {
        "images": {img_key: img}, "scalefactors": sf,
        "metadata": {"source_image_path": f"{spdir}/{img_file}"}}}
    adata.obs["library_id"] = library_id
    return adata


def load_visium_a1(vis_dir="VISIUM_outs/V10S15-395_A1", library_id="A1"):
    adata = sc.read_10x_h5(f"{vis_dir}/filtered_feature_bc_matrix.h5")
    adata.var_names_make_unique()

    # tissue positions (Space Ranger v1: no header)
    cols = ["barcode", "in_tissue", "array_row", "array_col",
            "pxl_row_in_fullres", "pxl_col_in_fullres"]
    pos = pd.read_csv(f"{vis_dir}/spatial/tissue_positions_list.csv",
                      header=None, names=cols).set_index("barcode")
    pos = pos.loc[adata.obs_names]  # align to filtered (in-tissue) barcodes
    adata.obs[["in_tissue", "array_row", "array_col"]] = pos[["in_tissue", "array_row", "array_col"]]
    adata.obsm["spatial"] = pos[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy()

    # scalefactors
    with open(f"{vis_dir}/spatial/scalefactors_json.json") as f:
        sf = json.load(f)

    # working lowres image -> float [0,1]
    img = np.asarray(Image.open(f"{vis_dir}/spatial/tissue_lowres_image.png").convert("RGB"))
    img = img.astype(np.float32)
    img = img / (65535.0 if img.max() > 255 else 255.0)

    adata.uns["spatial"] = {library_id: {
        "images": {"lowres": img},
        "scalefactors": sf,
        "metadata": {"source_image_path": f"{vis_dir}/spatial/tissue_lowres_image.png"},
    }}
    adata.obs["library_id"] = library_id
    return adata
