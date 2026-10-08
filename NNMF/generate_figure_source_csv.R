# Create source-data CSVs for the NMF panels of the ORES manuscript
# (Fig. 1D, 1F, 1I and Fig. 2A) from the saved Seurat object `se.VAC.d0`.

suppressPackageStartupMessages(library(SeuratObject))

object_path <- "../../manuscript/visium-ores-rolls/data/se.VAC.d0"
spots_path  <- "../../manuscript/visium-ores-rolls/data/data_for_geo/V10S15-395_A1/spatial/tissue_positions_list_curated.csv"
outdir      <- "../../manuscript/visium-ores-rolls/source_data"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

# Panels, factors (manuscript numbering n1..n55) and number of genes shown in the figure
panels <- list(
  Fig1D = list(factors = c(1, 41, 3, 46),      n_genes = 6),   # stomach
  Fig1F = list(factors = c(43, 17, 7, 2, 10),  n_genes = 6),   # esophagus
  Fig1I = list(factors = c(20, 31, 35, 48),    n_genes = 6),   # oral mucosa
  Fig2A = list(factors = c(6, 27, 33, 36),     n_genes = 20)   # pharyngeal glandular region
)

se <- readRDS(object_path)
loadings <- se@reductions$NMF@feature.loadings   # genes x factors (factor_1 ... factor_55)
activity <- se@reductions$NMF@cell.embeddings    # spots x factors
colnames(loadings) <- sub("^factor_", "factor_n", colnames(loadings))
colnames(activity) <- sub("^factor_", "factor_n", colnames(activity))

# Spot coordinates from the curated Space Ranger position file (no header)
pos <- read.csv(spots_path, header = FALSE,
                col.names = c("barcode", "in_tissue", "array_row", "array_col",
                              "pxl_row_in_fullres", "pxl_col_in_fullres"))
rownames(pos) <- pos$barcode
sr_barcode <- sub("_1$", "", rownames(activity))   # drop STutility "_1" section suffix
coords <- pos[sr_barcode, ]
stopifnot(all(coords$in_tissue == 1))

fmt <- function(x) signif(x, 6)

for (panel in names(panels)) {
  spec <- panels[[panel]]
  cols <- paste0("factor_n", spec$factors)

  # Spatial maps: all spots of the section, activity of each factor shown
  spatial <- data.frame(
    barcode = sr_barcode,
    array_row = coords$array_row,
    array_col = coords$array_col,
    pxl_row_in_fullres = coords$pxl_row_in_fullres,
    pxl_col_in_fullres = coords$pxl_col_in_fullres,
    fmt(activity[, cols, drop = FALSE]),
    check.names = FALSE, row.names = NULL
  )
  write.csv(spatial, file.path(outdir, paste0(panel, "_spatial.csv")), row.names = FALSE, quote = FALSE)

  # Gene weight bar charts: top n genes per factor, ranked by weight
  top <- do.call(rbind, lapply(cols, function(cl) {
    w <- sort(loadings[, cl], decreasing = TRUE)[seq_len(spec$n_genes)]
    data.frame(factor = cl, rank = seq_along(w), gene = names(w), weight = fmt(unname(w)))
  }))
  write.csv(top, file.path(outdir, paste0(panel, "_top_genes.csv")), row.names = FALSE, quote = FALSE)
}

sessionInfo()