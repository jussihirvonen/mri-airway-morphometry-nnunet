# Upper Airway Morphometry in Acute Neck Infections

Analysis code for the study *"Upper Airway Morphometry in Acute Neck
Infections: Clinical Associations and Deep Learning–Based Segmentation"*
(Viertonen et al., Turku University Hospital).

MRI-based morphometry of the pharyngeal airway in 322 patients with acute
neck infection: five prespecified features (luminal volume, minimum
cross-sectional area, compactness, maximum midline shift, LCSA50),
cluster-permutation airway profile analysis, clinical severity associations,
and nnU-Net-based automated segmentation with feature-level agreement.

**No patient data are included in this repository.** Imaging data,
segmentation masks, and clinical variables cannot be shared publicly under
the study permissions. The code is published for methodological transparency.
The repository also contains no server names, no study-specific folder names,
no reader names, and no case identifiers; everything machine- or study-specific
is read from an untracked local configuration file (see *Local layout*).

## Layout

- `final_analysis/` — analysis pipeline. `rerun_all.sh` runs the core chain
  (feature extraction → clinical merge → univariate and model analyses →
  figures). Numbered scripts `09`–`28` cover segmentation agreement,
  interobserver analyses (`13_inter_rater.py` expert reader, `14_novice_reader.py`
  novice reader), risk stratification, airway profile analysis (VBA), and
  provenance recomputations; `29`–`42` are the revision analyses. See
  `CANONICAL_PROVENANCE.md` for the manuscript-element-to-script map.
- `final_analysis/r/` — R replication pipeline (ICC, severity comparisons,
  origin-stratified analysis).
- `manuscript_drafts/` — table and figure builders: `build_tables_figures.py`
  (Tables 1–4, severity heatmap, risk stratification), `build_figure1_flowchart.py`
  (patient flow, counts derived from source data with assertions),
  `build_figure_metrics3d.py` (feature definitions on one rendered lumen),
  `build_figure_segmentation.py` (manual vs nnU-Net examples),
  `build_figure_airway3d.py` (3D lumen renderings), `build_revision_tables.py`
  (revised Tables 3–4 and Supplementary Tables S1–S6).
- `nnunet_setup/` — nnU-Net dataset preparation and training.
  `01_make_splits_local.py` builds the five folds stratified by abscess status
  (deterministic, seed 42, so the fold membership is regenerated rather than
  stored), `02_stage_dataset005_local.py` stages images and masks into nnU-Net
  layout, `03_train_launch.sh` is the preprocessing and training launcher used
  for the reported model (single GPU); `04_train_launch_2gpu.sh` and
  `05_stage_to_gpu_host.sh` are a two-GPU variant of the same launcher and its
  staging script. `dataset.json` is the nnU-Net dataset descriptor.

## Paths

No absolute paths are hard-coded. Every script resolves the project root from
its own location, so the tree runs in place once the data folders sit beside the
scripts. Override with environment variables when the data live elsewhere:

- `AIRWAY_ROOT` — the airway project folder (contains `final_analysis/`,
  `manuscript_drafts/`, `nnunet_setup/`, and `paths_local.json`).
- `CLINICAL_MASTER_XLSX` — the clinical master workbook; overrides the
  `clinical_master_xlsx` entry of `paths_local.json`.
- `NNUNET_HOME` — nnU-Net working directory on the GPU host, used by the
  training launcher and watcher (default: `$HOME/nnunet`).
- `REMOTE` — ssh alias of the GPU host, required by `05_stage_to_gpu_host.sh`.

## Local layout

Mask folders, the clinical and review workbooks, the parent-study tables, the
image folders, the submission folder, and the cases used as figure examples are
study-specific and are not part of the repository. Scripts
that need them read an untracked `paths_local.json` in `AIRWAY_ROOT` (ignored by
git) and fall back to the generic defaults shown here:

```json
{
  "reference_masks": "reference_masks",
  "expert_masks": "expert_masks",
  "novice_masks_round1": "novice_masks/round1",
  "novice_masks_round2": "novice_masks/round2",
  "expert_review_xlsx": "expert_review_log.xlsx",
  "expert_review_sheet": "review",
  "submission_dir": "submission",
  "clinical_master_xlsx": "clinical_master.xlsx",
  "clinical_master_xlsx_copy": "clinical_master.xlsx",
  "accession_map_csv": "accession_map.csv",
  "abscess_clinical_csv": "abscess_clinical.csv",
  "shape_features_csv": "shape_features.csv",
  "nifti_pos_dir": "images/positive",
  "nifti_neg_dir": "images/negative",
  "examples": {
    "overlay": {"Good": "<case>", "Intermediate": "<case>", "Poor": "<case>"},
    "profile_method": "<case>", "profile_normal": "<case>", "profile_abnormal": "<case>",
    "airway3d_normal": "<case>", "airway3d_stenotic": "<case>",
    "metrics3d_case": "<case>", "metrics3d_normal": "<case>",
    "segmentation_representative": "<case>", "segmentation_stenosis": "<case>",
    "midline_debug": []
  }
}
```

Paths are relative to `AIRWAY_ROOT` and may point outside it with `../`. The two
workbook entries name the clinical master workbook and the copy kept beside the
abscess tables; they may be the same file. The `examples` entries are only needed by
the figure scripts that render representative cases; without them those scripts
stop with a message naming the missing key.

## Requirements

Python 3.11 (numpy, scipy, pandas, nibabel, matplotlib, pingouin, openpyxl,
python-docx); R 4.3 (dplyr, pROC, irr, boot, logistf). nnU-Net v2 for model
training (default 3d_fullres configuration, single-channel post-contrast T1).

## Status

Private during peer review. Public release with a Zenodo DOI at acceptance.
