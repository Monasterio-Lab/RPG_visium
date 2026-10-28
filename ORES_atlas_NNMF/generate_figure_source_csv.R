# Create source-data CSVs for the NMF panels of the ORES manuscript
# (Fig. 1D, 1F, 1H, Fig. 2A and Suppl. Figs. 2, 3, 4B-D) from the saved Seurat object `se.VAC.d0`.

suppressPackageStartupMessages(library(SeuratObject))

object_path <- "../data/nmf/se.VAC.d0"
spots_path  <- "../data/VISIUM_outs/V10S15-395_A1/spatial/tissue_positions_list_curated.csv"
outdir      <- "../figure_source_csv"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

# Panels, factors (manuscript numbering n1..n55) and number of genes shown in the figure.
# Panels with n_genes = NA show spatial maps only.
panels <- list(
  Fig1D = list(factors = c(1, 41, 3, 46),      n_genes = 6),   # stomach
  Fig1F = list(factors = c(43, 17, 7, 2, 10),  n_genes = 6),   # esophagus
  Fig1H = list(factors = c(20, 31, 35, 48),    n_genes = 6),   # palatal (oral) mucosa
  Fig2A = list(factors = c(6, 27, 33, 36),     n_genes = 20),  # pharyngeal glandular region
  # Suppl. Fig. 2: factor maps per region (03_factor_plots.Rmd)
  S2A = list(factors = c(1, 3, 4, 5, 9, 18, 22, 26, 34, 39, 41, 46), n_genes = NA),  # stomach
  S2B = list(factors = c(2, 7, 10, 15, 17, 29, 40, 43, 51),          n_genes = NA),  # esophagus
  S2C = list(factors = c(20, 31, 35, 42, 45, 48),                    n_genes = NA),  # oral mucosa
  S2D = list(factors = c(13, 16, 28, 30, 32, 54, 55),                n_genes = NA),  # nasal mucosa
  # Suppl. Fig. 4 B-D: factor overlays with top-40 gene weights
  S4B = list(factors = 31, n_genes = 40),
  S4C = list(factors = 20, n_genes = 40),
  S4D = list(factors = 35, n_genes = 40)
)

# Suppl. Fig. 3: heatmap of the top genes per factor, factor groups as in 03_factor_plots.Rmd
# (oral, nasal, pharyngeal, esophagus, stomach)
s3_groups <- list(
  "oral mucosa"  = c(20, 31, 35, 42, 45, 48),
  "nasal mucosa" = c(13, 16, 28, 30, 32, 54, 55),
  "pharyngeal"   = c(6, 24, 37, 53),
  "esophagus"    = c(2, 7, 10, 15, 17, 29, 40, 43, 51),
  "stomach"      = c(1, 3, 4, 5, 9, 18, 22, 26, 34, 39, 41, 46)
)
s3_top_n <- 4

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
  write.csv(spatial, file.path(outdir, paste0("Fig", panel, "_spatial.csv")), row.names = FALSE, quote = FALSE)

  # Gene weight bar charts: top n genes per factor, ranked by weight
  if (is.na(spec$n_genes)) {
    message(sprintf("%s: %d spots, factors %s", panel, nrow(spatial), paste(cols, collapse = ", ")))
    next
  }
  top <- do.call(rbind, lapply(cols, function(cl) {
    w <- sort(loadings[, cl], decreasing = TRUE)[seq_len(spec$n_genes)]
    data.frame(factor = cl, rank = seq_along(w), gene = names(w), weight = fmt(unname(w)))
  }))
  write.csv(top, file.path(outdir, paste0("Fig", panel, "_top_genes.csv")), row.names = FALSE, quote = FALSE)

  message(sprintf("%s: %d spots, factors %s", panel, nrow(spatial), paste(cols, collapse = ", ")))
}

# Suppl. Fig. 3 heatmap: union of the top s3_top_n genes of every factor (in plotting order),
# each factor's weights divided by that factor's maximum weight (as in 03_factor_plots.Rmd)
s3_factors <- paste0("factor_n", unlist(s3_groups, use.names = FALSE))
s3_genes <- unique(unlist(lapply(s3_factors, function(cl) {
  names(sort(loadings[, cl], decreasing = TRUE))[seq_len(s3_top_n)]
})))
heat <- loadings[s3_genes, s3_factors, drop = FALSE]
heat <- sweep(heat, 2, apply(heat, 2, max), "/")
write.csv(data.frame(gene = s3_genes, fmt(heat), check.names = FALSE, row.names = NULL),
          file.path(outdir, "FigS3_heatmap.csv"), row.names = FALSE, quote = FALSE)
write.csv(data.frame(factor = s3_factors,
                     region = rep(names(s3_groups), lengths(s3_groups))),
          file.path(outdir, "FigS3_factor_regions.csv"), row.names = FALSE, quote = FALSE)
message(sprintf("S3: %d genes x %d factors", length(s3_genes), length(s3_factors)))

sessionInfo()
