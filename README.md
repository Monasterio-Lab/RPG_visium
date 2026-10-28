# Spatial transcriptomics of the retropharyngeal gland

This repository contains the code to reproduce the spatial transcriptomics results of the manuscript
*Spatial transcriptomics of the upper aerodigestive tract reveals previously unrecognised retropharyngeal salivary gland*.

Two 10x Genomics Visium sections were analysed:

| Section | Tissue | Slide / capture area | Space Ranger |
|---|---|---|---|
| A1 ("VAC d0") | ORES roll (oral–esophageal–stomach) | V10S15-395, A1 | 1.2.1 |
| D1 | submandibular + sublingual gland (SMG/SLG) | V13U19-349, D1 | 3.0.0 |

Both were processed with the `refdata-gex-mm10-2020-A` reference.

## Repository structure

```
.
├── ORES_atlas_NNMF/                    NMF atlas of the ORES section
│   ├── code/
│   │   ├── 01_spot_curation.Rmd        manual curation of tissue spots
│   │   ├── 02_Upper_digestive_tract_Visium_analysis_VAC_d0.Rmd   SCTransform + NMF
│   │   └── 03_factor_plots.Rmd         factor maps and gene-weight plots
    ├── package_versions.txt            R package versions
│   └── generate_figure_source_csv.R    source data for the NMF figure panels
└── RPG_salivary_gland/                 retropharyngeal gland analysis
    ├── code/                           scripts 01–26 + helpers
    ├── generate_figure_source_csv.py   source data for the gland figure panels
    └── pip_freeze.txt                  Python environment
```

## Code

| Folder | Figures |
|---|---|
| `ORES_atlas_NNMF/` | Fig. 1D–I, Fig. 2A, Suppl. Figs. 2, 3, 4B–D |
| `RPG_salivary_gland/` | Fig. 2B, 2E–J; Suppl. Figs. 1, 5B, 7A, 8–12 |


### RPG_salivary_gland

| Scripts | Content |
|---|---|
| 01, 02 | Visium QC of A1 and D1 |
| 03–07 | build the single-cell reference: GSE150327 (03), GSE216476 clustering and annotation (04, 05), merge (06), GSE223516 parotid with ambient-RNA removal and final reference (07) |
| 08–10 | reference UMAP and marker dot plot |
| 11–13 | cell2location: reference signatures (11), mapping A1 (12) and D1 (13) |
| 14, 15 | D1 clustering and assignment of clusters to SMG / SLG |
| 16–19 | cell-type composition |
| 20, 21 | integration of RPG, SMG and SLG spots; glandularity filter; Harmony UMAP |
| 22–25 | glandular spots, salivary score, unrolling, regionalization genes |
| 26 | Human Protein Atlas RNA and protein expression |

### ORES_atlas_NNMF
| Scripts | Content |
|---|---|
| 01 | manual curation of spots in STutility |
| 02 | SCTransform and NMF with 55 factors |
| 03 | factor maps and gene-weight plots |


## Data

**GEO**: raw sequencing data (FASTQ), Space Ranger outputs and the full-resolution H&E image of both sections.

**Zenodo**:

| Folder | Content |
|---|---|
| `annotations/` | `V10S15-395_A1_RPG_ROI_loupe.csv`: RPG region (161 spots). `V13U19-349_D1_gland_assignment.csv`: SMG / SLG label per D1 spot |
| `cell2location/<section>/` | cell-type abundances and proportions per spot |
| `reference/` | `scRNA_reference_SMG_SLG_parotid.h5ad` (single-cell reference) and `cell2location_reference_signatures.csv` |
| `integration/` | `glandular_spots_RPG_SLG_SMG.h5ad`: integrated RPG, SMG and SLG spots after glandularity filter |
| `nmf/` | `se.VAC.d0`: Seurat object with the NMF results, plus CSV exports of the spot activities and gene loadings |
| `HPA/` | Human Protein Atlas RNA (nTPM) and protein (IHC) tables |

**Public single-cell data** used to build the reference: GSE150327 (Hauser et al. 2020), GSE216476 (Huang et al. 2022) and GSE223516 (Rheinheimer et al. 2023).

## Reproducibility notes

- **NMF:** no seed; use the deposited object.
- **cell2location and UMAP:** seeds are set, but training on different hardware or library versions gives slightly different numbers. The deposited cell2location outputs and integrated object are the ones used in the paper.
- **Human Protein Atlas:** code queries proteinatlas.org live, so values can change between HPA releases. The deposited `HPA/` tables are the ones used in the paper.
