# Airway morphometry study — canonical dataset, scripts, and results

Written 2026-08-15 after the full pre-submission numbers audit. This file is
the single authoritative map of what is final. If a file is not listed here,
treat it as exploratory or superseded. Project root:
the project folder (the directory that contains `final_analysis/`, `manuscript_drafts/`, and `nnunet_setup/`)

Manuscript: Google Docs "ILM-morfometria" (live). Local audit snapshot and condensation redline: dated copies kept with the manuscript drafts.

## 1. Dataset (final, corrected 2026-06-22)

| Item | Location | Status |
|---|---|---|
| Reference masks (reference annotator), n=322 | reference-mask folder (`paths_local.json`: reference_masks) | FINAL. All single-component after 2026-06-22 corrections (9 connects by the annotator, 23 auto-cleaned deletes). Pre-correction originals: `final_analysis/_pre_correction_backup_2026-06-22/` |
| Expert interobserver masks (expert reader), n=30 | expert-reader folder (`case<NNN>airway.nii.gz`; key expert_masks) | FINAL (two cases use the corrected reference where relevant) |
| Novice interobserver masks (novice reader), n=30 | novice-reader folder, round 2 (key novice_masks_round2) | FINAL (round 2, one pyriform-fossa reconnect applied; round 1 kept for learning-curve only) |
| nnU-Net out-of-fold predictions, n=322 | `final_analysis/predicted_masks/` (+ `summary.json`) | FINAL. Evaluate as UNION of components (decision 2026-06-22), never largest-CC |
| Morphometric features, n=322 | `final_analysis/reference_full_morphology.csv` (2026-06-22) | CANONICAL feature table (corrected masks) |
| Features + clinical, n=322 | `final_analysis/airway_morphology_merged.csv` (2026-06-22) | CANONICAL analysis table. RPE presence = `retroinf` (complete, 163/322), NOT `rpe_present` |
| Clinical master | clinical master workbook (`paths_local.json`: clinical_master_xlsx) | Source of clinical variables (merged by `02_merge_clinical.py`) |
| Origin variables (`hammas`, `peritonsinf`) | `final_analysis/r/airway_features_with_outcomes.csv` | Clinical columns only. Its FEATURE columns are stale (2026-06-01) — features come from `reference_full_morphology.csv` |
| Staging / exclusion log | expert review workbook (keys expert_review_xlsx / expert_review_sheet) | 337 consecutive cases; inclusion status encoded as cell fill (green C6E0B4 included, gray A5A5A5 intubated, red FF0000 data deficient/missing; legend in column D rows 2–4). Source of Figure 1 counts |
| nnU-Net model (D005) | GPU host, Dataset005_NeckAirway, 3d_fullres, 5-fold | Trained pre-correction; small mask edits deemed immaterial for retraining (see `airway_nnunet_plan.md`) |

Known denominators: ICU 33/322; prolonged LOS 47/319 (3 missing `hospd`);
mediastinal edema 86/297; RPE volume n=299; origin subgroups 96 odontogenic /
120 pharyngotonsillar (definitions in `18_origin_recompute.py`).

## 2. Scripts (all in `final_analysis/` unless noted)

Core pipeline (run via `rerun_all.sh` after any mask change):
`01_run_full_morphology.py` (feature extraction, largest-CC on manual masks) →
`02_merge_clinical.py` → `03_analyze.py` → `05_prespecified_icu_table.py` →
`07_severity_figures.py` → `08_severity_cohort_compare.py`.

Separate steps, run after the core pipeline:
`09_seg_agreement.py` (DSC/HD95 raw + morphometric ICC, predictions as union);
`13_inter_rater.py` (expert, n=30); `14_novice_reader.py` (novice, both rounds);
`15_risk_stratification.py`; `15_reader_benchmark_table.py`;
`16_icc_compare_readers.py` (paired ΔICC bootstrap);
`17_table5_reproduction.py` (Table 5, added 2026-08-13);
`18_origin_recompute.py` (origin analysis on corrected data, added 2026-08-15);
`21_airway_vba_extract.py` → `22`–`26` (airway profile / VBA; cache
`21_airway_vba_cache.npz`).

Manuscript builds: `manuscript_drafts/build_tables_figures.py` (Tables 1–4
docx + Figures 2–6 + supp). R pipeline `r/analysis_pipeline.R` is retained for
the record; its report is superseded where noted below.

## 3. Results — manuscript element → canonical source

| Manuscript element | Canonical source |
|---|---|
| Table 1 clinical rows | `manuscript_drafts/tables_figures/Table1.docx` + `03_analysis_report.txt` §1. NOTE: manuscript RPE row must be 163 (50.6), stale 155 (51.8) in draft |
| Table 1 airway rows | `reference_full_morphology.csv` (verified 2026-08-13: volume 11.4 (8.2–15.8), mCSA 23.0 (10.6–41.4), mean CSA 125.6 (98.0–163.3), length 85.0, maxMLS 6.5 (4.3–9.5), LCSA50 14.2 (5.0–20.0)) |
| Table 2 (interobserver, n=30) | `13_inter_rater_report.txt` (expert), `14_novice_icc_round2.csv` (novice), `16_icc_compare.csv` (ΔICC/p columns) |
| Table 3 | `05_prespecified_icu_table_full.csv` + `03_analysis_report.txt` §4–5; generated `Table3.docx` verified identical to manuscript |
| Table 4 | `manuscript_drafts/tables_figures/Table4_data.csv` (z for four-feature model = +1.56) |
| Table 5 | `17_table5_reproduction.csv` — recomputed 2026-08-13, matches manuscript exactly |
| Origin analysis (Supp Table 3) | `18_origin_recompute_report.txt` (corrected data; maxMLS p=0.538→"0.54" as in manuscript). Supersedes `r/results_report.txt` §7 |
| Severity heatmap (5×6 grid, maxMLS 5/6) | `08_severity_compare_combined.csv` |
| Risk stratification (5.9/11.8/41.2%, RR 3.5, p=0.012) | `15_risk_strata.csv`, `15_joint_strata.csv` |
| Segmentation performance (Dice mean 0.911 / median 0.920, HD95 median 1.76 mm, 303/322 ≤5 mm, ICCs 0.853–0.973) | `09_seg_agreement_report.txt` + `09_seg_agreement_percase.csv` |
| Airway profile / VBA 1D clusters + abs+ sensitivity | `vba_rerun.log` (2026-06-22) |
| VBA 2D unfolded-map clusters | `23_2d_map_rerun_2026-08-16.log` — **ICU clusters p = 0.002 (right/anterolateral, 0–95%) and p = 0.016 (left posterolateral, 46–95%)**. CORRECTION 2026-08-16: the manuscript's "0.014 and 0.014" on the ICU sentence are actually the RPE clusters (radius increase, +1 sign); fixed in the condensed redline. RPE 2D clusters = 0.014/0.014 |
| **FIGURE LINEUP (FINAL v3, 2026-08-17: composite Fig 3 built by the senior author by hand)** | Main: **Fig 1** flowchart (`Fig1.jpg` after 17→63 + 332→322 fixes) · **Fig 2** feature definitions (`Figure_metrics3d`) · **Fig 3** COMPOSITE clinical associations: (a) full-sample severity heatmap (component file `Figure2.png`; sole display of RPE/ME/CRP/abscess-diameter effect sizes, NOT in any table) + (b) risk stratification (component `Figure5.png`) — composite assembled by the senior author by hand, image file NOT yet on disk in the repo (verify when it lands) · **Fig 4** airway-profile ICU 4-panel (`final_analysis/26_airway_vba_icu_main.png`, regen 2026-08-16) · **Fig 5** segmentation examples (`Figure_segmentation.png`) · **Fig 6** manual-vs-predicted scatter (`final_analysis/10_seg_scatter_manual_vs_predicted.png`, regen 2026-08-17, pingouin ICC matching text). Supplement: **S1** method schematic (24) · **S2** 2D wall maps (23). DROPPED: 1D grid, Bland–Altman, airway3d, boxplots, interobserver scatter, joint stratification. Docx callouts (3a/3b) + merged Fig 3 legend applied 2026-08-17 |
| Figure 1 (flowchart) | `manuscript_drafts/build_figure1_flowchart.py` → `tables_figures/Figure1_draft.png/.pdf` (2026-08-15 FINAL DRAFT, CONSORT style, SINGLE-STEP per the senior author). Flow: **535** (clinical master workbook, verified abs 373/162) → **213 excluded** (abscess on MRI without surgical confirmation 63; non-standard MRI protocol 26; insufficient or missing imaging data 119; intubation before imaging 5) → **322 analyzed** (222/100). The merged 119 = 213−63−26−5 folds together the parent chain's data bucket, unselected negatives, the 9 staging data exclusions, and the 1 uncompleted segmentation (one case); the merge makes the arm assignment of the 6 accession-unmapped cases irrelevant. Remaining assumption pending the annotator: the 63/26 buckets contributed no staged cases (the two staged cases outside the segmentation cohort taken from the data bucket; near-certain for the DICOM-only one). Also pending the annotator: one case intubation note. Lineage verified: seg-cohort 226 positives = 220 included + 6 excluded exactly |
| Reader benchmark (novice vs expert) | `15_reader_benchmark_table.csv`, deliverable: interobserver table (docx) |

Audit trail: full pre-submission numbers audit 2026-08-13/15 (10 errors, 9
checks, all documented in the condensation-redline handoff); Table 5 and
origin-analysis provenance gaps closed by scripts 17 and 18.

## 4. STALE / superseded — do not use for the manuscript

- `14_descriptive_prespecified.csv` (2026-06-01, pre-correction features)
- `16_dl_replicates_*` (2026-06-16, pre-correction AND largest-CC
  postprocessing; the old "DL loses fine-caliber" hinge lives here)
- `r/results_report.txt` §7 and the feature columns of
  `r/airway_features_with_outcomes.csv` (2026-06-01 features)
- `13_inter_rater_*_n20.csv` (interim 20-case, superseded by n=30)
- `14_novice_*_round1.csv` (learning-curve baseline only)
- `20_airway_profile_vba.*` (superseded by 21–26)
- `27_*`/`28_*` (postprocessing exploration; decision = union, kept for record)
- `legacy/`, `snapshot_*/`, `_pre_correction_backup_2026-06-22/`,
  the annotator's pilot folders, `multivol_exploration/`
- `manuscript_drafts/airway_paper1_*` / `airway_paper2_*` /
  `ZZ_COMBINED_DRAFT_*` (superseded by the combined ILM-morfometria doc)

## 5. Code repository

**Created 2026-08-17 (PRIVATE):**
https://github.com/jussihirvonen/mri-airway-morphometry-nnunet. Code only (38 files: final_analysis
py/sh, R pipeline, table/figure builders, this manifest, README); defensive
.gitignore blocks all data/output formats. At acceptance: flip public + mint
Zenodo DOI, per standing practice. Remember to re-sync scripts before the
public flip if the pipeline changes during revision.

**Re-synced 2026-09-02 (50 files):** all 38 originals verified byte-identical
to the project originals except this manifest (updated). Added the three manuscript figure
builders that were missing (`build_figure_metrics3d.py` = Fig 2,
`build_figure_segmentation.py` = Fig 5, `build_figure_airway3d.py` = 3D
lumen renderings) and `nnunet_setup/` (split generation, dataset staging,
single-GPU launcher used for the reported D005 model, two-GPU launcher variant,
`dataset.json`; `splits_final.json` was force-added at the time and removed again
in the 2026-10-27 scrub, see below). Excluded on purpose:
`case_manifest.csv` (per-case ICU label), `qc_report.txt`, `rename_masks.txt`,
`RERUN_README.md` (stale n=323 wording). The simplified Figure 1 variant
(coauthor request 2026-08-31) is NOT in the repo until adopted into
`build_figure1_flowchart.py`.

**Paths sanitized 2026-09-08 (applied to the project originals AND the
repo, so both stay byte-identical):** 35 Python scripts + 2 nnU-Net shell
scripts rewritten. Python roots now resolve from the script's own location
(`Path(__file__).resolve().parents[N]`), overridable via `AIRWAY_ROOT`,
two environment variables for the parent-study folder and the clinical workbook
(both retired in the 2026-10-27 scrub); shell scripts use `NNUNET_HOME` (default
`$HOME/nnunet`). Defaults resolve to exactly the previous absolute paths, so
nothing changes when run in place; every definition was verified against the
old literal and `17_table5_reproduction.py` regenerated byte-identical outputs.
Pre-sanitization originals backed up under the session scratchpad
`backup_2026-09-08/` (37 files). The exploratory server-side file-discovery
script was removed from the repo and stays in the project folder. Repo scan after
the pass: zero machine-specific absolute paths (user home or server data
directories); no personal identifiers, accession numbers, or dates.

**Identifier scrub 2026-10-27 (applied to the project originals AND the repo):**
repository policy is now that no server names, personal or study-specific folder
names, reader names, or case identifiers (even pseudonymous ones) appear anywhere
in the repository, including its history. Done: (1) the three nnU-Net launchers
renamed (`03_train_launch.sh`, `04_train_launch_2gpu.sh`, `05_stage_to_gpu_host.sh`)
and host names replaced by `$REMOTE` / "the GPU host"; (2) every person-named
folder, workbook and output file replaced by a generic key resolved from an
untracked `paths_local.json` at the project root (keys listed in the README), with
the analysis outputs renamed accordingly (`reference_full_morphology.csv`,
`airway_morphology_merged.csv`, `14_novice_*`) and their reader columns renamed
(`expert_`, `reference_`, `novice_`; values unchanged, verified by re-running
15/16/39); (3) representative case IDs for figures moved into the `examples`
block of `paths_local.json`; (4) `splits_final.json` dropped from the repo (fold
membership is regenerated deterministically by `01_make_splits_local.py`, seed 42);
(5) git history squashed to a single commit, the previous history kept locally as
a bundle outside the project. Pass 2 (same day): the clinical master workbook, the
accession map, the abscess clinical and shape-feature tables and the NIfTI source
folders of the parent study are likewise read from `paths_local.json` (keys
clinical_master_xlsx, clinical_master_xlsx_copy, accession_map_csv,
abscess_clinical_csv, shape_features_csv, nifti_pos_dir, nifti_neg_dir); the
parent-study and workbook environment variables are gone, `CLINICAL_MASTER_XLSX`
overrides the workbook entry.

## 6. Revision R1 (EJR-D-26-04240, decision 9 Oct 2026, due 29 Oct 2026)

Submitted files: the submission folder under `manuscript_drafts/` (ILM_maindocument.docx + ILM_Table1-5.docx;
Highlights and Supplementary Material were uploaded separately and are not on disk).
Reviewer 1 raised ten points; new analysis scripts (all in `final_analysis/`, outputs beside them):

| Script | Reviewer point | Output used in revision |
|---|---|---|
| `29_fdr_grid.py` | 3 (multiplicity) | FDR q (Benjamini–Hochberg) for the 30-test grid → Table 3 (q column), Supp Table S1; the Bonferroni column the script also computes is NOT reported (dropped 2026-10-26, see 6e) |
| `30_lcsa_threshold_sensitivity.py` | 6 (50 mm² threshold) | thresholds 25–100 mm² + relative → Supp Table S3 |
| `31_midline_positioning.py` | 10 | airway-axis detrending (NOT used in paper; over-corrects) |
| `32_penalized_icu_models.R` | 4 | Firth + optimism bootstrap + penalized LRT + LASSO → Table 4 (revised) |
| `33_risk_strat_validation.py` | 7 | 5-feature disclosure, continuous test, repeated CV, bootstrap, LOS → reported in the response letter only (analysis, Figure 3b and the former Supp Table S4 removed from the manuscript 2026-10-23) |
| `34_failure_analysis.py` | 9 | per-case mechanisms + montages `34_failure_montages/` → Supp Table S4 |
| `35_anatomical_midline.py` | 10 | symmetry-based midsagittal plane; maxMLS_anat → Supp Table S6; QC montages `35_midline_montages/` |
| `36_fixed_score_reference.R` | 5 | published Vierula 2025 score with fixed coefficients → Supp Table S2 |
| `37_expert_review_modifications.py` | 2a | 198/322 modified (119 endpoint-only, 79 in-plane) from the expert review workbook |
| `41_median_split_fdr.py` | 7 | FDR q across the five median-split Fisher tests of script 33 (compactness q = 0.060) → response letter and calculations handout only |

Supp Table S5 (union vs largest-CC bounds) is built from `27_pred_postproc_report.txt` (n=322 columns).
Builders: `manuscript_drafts/build_revision_tables.py` (Table3/4 revised + S1–S6) and
`manuscript_drafts/build_revision_response.py` (response letter + edit list); outputs in
the revision folder (`<submission>/revision_R1/`). Table 4 (revised) DeLong p values are computed on the Firth
predictions (0.073/0.118; abs+ 0.097/0.075) and supersede the ML-model values in `Table4_data.csv`.

Known code issue (no manuscript impact): in `01_run_full_morphology.py` the generic per-slice
`ap_*` and `lat_*` columns are swapped by name (NIfTI axes are L,P,S, so axis 0 is left–right);
`rl_dev_*` (midline shift) and the VBA scripts (21–26) use the correct axis. None of the mislabeled
columns appears in the manuscript. Rename before the public repo flip.

### 6b. Added 9 Oct 2026 (afternoon) after the external (LLM) second opinion on the review
| Script | Point | Output |
|---|---|---|
| `32_penalized_icu_models.R` (extended) | 4 | + bootstrap-validated calibration slopes, + age/sex-adjusted models (`32_penalized_icu_models.{csv,txt}`) |
| `38_gap_convention.py` | 8 | empty-slice (gap) convention for predicted caliber features; Part 2 = all five features under largest-CC / union / gap-as-zero vs CORRECTED manual masks (`38_conventions_icc.csv`) — supersedes the pre-correction numbers in `27_pred_postproc_report.txt` for Supp Table S5 |
| `39_reader_outliers.py` | 9 | human-reader Dice < 0.85 cases from 13_/14_ per-case files → Supp Table S4 part B |

Manuscript build (tracked changes applied to the SUBMITTED file, originals untouched):
`manuscript_drafts/trackchanges.py` (w:ins/w:del engine) + `manuscript_drafts/build_revised_manuscript.py` →
the revision folder: `ILM_maindocument_R1_TRACKED_09102026_JHi.docx`, `..._CLEAN_...docx`, `ILM_highlights_R1_09102026_JHi.docx`.
Reference changes: Lin 2026 (old 24) and Tuna 2025 (old 36) removed; Avrahami 1996 (old 34) retained (the swap to Avrahami & Englender 1995, AJNR 16:135–140, is applied only with STRICT = False, see 6c); added 35 Viertonen EurRadiol 2026, 36 Viertonen EJR 2026, 37 Benjamini-Hochberg 1995, 38 Heinze & Schemper 2002, 39 Harrell 1996, 40 Demler 2012. In-text citations renumbered programmatically (8 hyperlinks changed).

### 6c. Scope rule (senior author, 9 Oct 2026)
The revision contains only edits that answer a specific reviewer request. The three builders carry a
`STRICT = True` flag; the optional corrections suggested by the external second opinions (calibration
slopes, age/sex-adjusted result, bootstrap CIs for r, Avrahami 1996→1995 citation swap, exclusion counts and
voxel sizes in the text, ethics-legislation sentence, median/IQR wording for DSC, LV→AV, grammar, "nominal"
labelling of profile clusters, abstract Objectives/Methods trims, Yu-paragraph shortening) are applied only
with `STRICT = False`. Analysis scripts 32 and 40 still compute calibration and age/sex results; they are
simply not reported. Scripts 29–40 and `build_revision_tables.py` are in the GitHub repository; the
manuscript/letter builders (`build_revised_manuscript.py`, `build_revision_response.py`, `build_q10_text.py`,
`trackchanges.py`) contain manuscript prose and stay out of the public repository.


### 6d. 2026-10-23: compactness split removed (reviewer point 7)

The exploratory median split of compactness within large abscesses (Results paragraph, Figure 3 panel b, the former Supplementary Table S4, the Methods clause and the limitation sentence) was removed from the revised manuscript; the five-feature disclosure and selection-aware validation (`33_risk_strat_validation.py`, `40_selection_aware_validation.R`) are reported in the response letter only. Supplementary tables renumbered: failure analysis S5→S4, evaluation conventions S6→S5, head positioning S7→S6. Figure 3 is now the single heatmap panel; the composite image must be re-rendered without panel (b) by the first author. Results paragraph on the feature-by-marker grid now cites Figure 3 (not 3a).

### 6e. 2026-10-26: tight revision; Bonferroni dropped; FDR labels

- Word count: the first revised draft added a net 872 countable words (abstract to figure legends). Rebuilt so that the main text keeps one-sentence pointers and the detail moved to new Supplementary Methods and Supplementary Results sections, written by `build_revised_manuscript.py::write_supplement_text()` into `_supp_text.json` and read by `build_supplement.py`. Net addition now +256 words; the remaining overage is original text to be cut by hand.
- Bonferroni-adjusted p values are no longer reported anywhere (Table 3, Supp Table S1, Supplementary Methods, response letter, calculations handout). The reviewer asked for FDR control of a hypothesis-generating screen; a second, stricter family-wise column had no decision rule attached. Column labels read "q value (FDR)", not "BH"; the Benjamini–Hochberg procedure is named once in the Methods (ref 37) and in the Supp Table S1 note. Script 29 still computes `p_Bonf_30`; it is simply not used.
- The five-feature adjustment quoted in the reply for the removed compactness split (comment 7) is now the FDR q from `41_median_split_fdr.py` (0.060; identical to the former Bonferroni value for the smallest p).
- The profile analysis (Figures 4–5) keeps its pre-specified Bonferroni threshold of 0.05/16 = 0.0031; that is original submitted text and a separate family, left unchanged under the scope rule.
- All inserted text was Americanized to match the submitted manuscript (favors, analyzed, image-center).

### 6f. 2026-10-27: second-opinion correction pass (failure labels, citations, errata)

- `34_failure_analysis.py` flags `ZGAP` for any empty predicted slice inside the manual z-range, which also catches predictions that merely stop short of the manual extent (single-component cases such as the ones with 1, 2 or 7 missing cranial slices). Supp Table S4 therefore now derives the mechanism from the saved columns: interior interruption = `pred_zgap_slices − missing_slices_cranial − missing_slices_caudal` > 0 (10 of 32 cases; coinciding with a manually bridged pinpoint segment, `ZGAP@pinpoint`, in 1), "prediction shorter than the manual extent" = any missing end slice (17), plus the existing extra-end-slice, stray-component, over/under-segmentation and boundary labels. The earlier label "gap at obstruction (… bridges a total or near-total occlusion)" overclaimed and is withdrawn everywhere (S4, S5 note, Supplementary Results, response letter, handout). The 48 "Z_DISCONTINUITY" multi-component predictions of script 27 are now described as craniocaudal discontinuities, without the bridging claim; 25 predictions have interior empty slices per script 38, 2 of them at a manual pinpoint slice. The script itself is unchanged; the CSV columns suffice.
- Case identifiers removed from all supplementary tables (S4 part A numbered 1–32 by HD95; part B patients lettered A–…); supplementary-table notes shortened to 56–106 words.
- `build_revised_manuscript.py` gained `ERRATA = True` for four error-level fixes outside the strict scope rule: the second citation of Yu et al. (plain run, missed by the hyperlink renumbering pass) 26 → 25; "clinically established airway-obstruction threshold" → "this threshold" (second occurrence in the LCSA50 paragraph); segmentation metrics "mean and SD across five folds" → "per-case medians across the 322 cases" (what the Results report); "LV" → "AV" in the Discussion. A full audit of the 27 in-text citations against the submitted numbering found no other mismatch.
- Supplementary Results positioning sentence listed the Spearman correlations in the wrong order (offset, yaw, roll); now yaw −0.048, roll +0.101, offset +0.022 as in Supp Table S6 (source `_q10_response_short.txt`, hand-written).
- Abstract: Results shortened by four words and the conclusion narrowed to "in the prespecified comparison" (257 words including the four headings).
- Supplementary Methods now carry the MRI protocol table of the parent cohort (copied from the accepted segmentation paper's EJR supplement into `MRI_acquisition_parameters.docx`, picked up by `build_supplement.py`).
- Not changed (decision pending): selection-aware optimism-corrected AUCs as the principal Table 4 column; "nominal" labels for the profile clusters above the 0.0031 threshold (gated edits still available under STRICT=False); a trained-weights sentence in the availability statement; the original Methods claim that the network "leaves a gap at sites of complete luminal obstruction that the manual protocol bridges".

### 6g. 2026-10-27: approved second-opinion items

- Table 4: the optimism-corrected AUC of the four-feature model is now the selection-aware value from `40_selection_aware_validation.R` (univariate feature selection repeated within each bootstrap resample: 0.809 in the full cohort, 0.817 in the abscess-positive subset; corrected gains over the reference +0.021 and +0.028). The fixed-model values of script 32 (0.814, 0.823) are given in the table note, the Supplementary Results, the response letter and the calculations handout. The minimum-CSA model has no selection step and is unchanged; the Discussion now gives the corrected gain as +0.02 to +0.03.
- Profile analysis (Results, Figure 4 legend, Figure S2 legend): clusters above the pre-specified Bonferroni threshold of 0.0031 for ICU admission (CSA p = 0.020, midline shift p = 0.007, left posterolateral wall p = 0.016) are labeled nominal, as the RPE and prolonged-stay clusters already were; the threshold and the cluster p values are unchanged.
- New `42_slice_spacing.py`: header z-spacing of the 322 training volumes (4.4 to 5.0 mm in 314 of 322; range 3.3 to 5.0 mm; in-plane 0.33 to 0.59 mm). The Discussion's "slice thickness of 4.4-5.0 mm" is the reconstructed slice spacing including the interslice gap, whereas the protocol table in the Supplementary Methods gives the nominal 4 mm thickness; the Discussion now reads "slice spacing" and the table caption states both.
- Interrupted predictions: `38_gap_convention.py` shows interior gaps in 25 predictions, only 2 at a manually bridged pinpoint slice, with a manual minimum CSA median of 2.7 mm² in those cases. The Methods and Discussion now say the network leaves a gap "at near-occluded segments and at sites of complete luminal obstruction", and the Discussion calls the gap a marker of critical narrowing rather than of complete obstruction; the comment-8 sentence attributes the asymmetry to the continuity of the manual airway.
- Availability statement: trained network weights available from the corresponding author on request (code public with an archived DOI at acceptance; images not shareable).
- Scope rule: the 26 October compression of submitted sentences in the limitations paragraph and in the Results paragraph on feature associations was reverted; those paragraphs now keep the submitted sentences verbatim and carry the reviewer-driven sentences as tracked insertions only (local manuscript builder gained a pure-insertion helper). Main text 3,867 words (submitted 3,544; limit 4,000); abstract 257 words including headings.
- The MRI protocol table of the Supplementary Methods is regenerated by a local builder (not in this repository) from the accepted segmentation paper's supplement, with the slice-spacing sentence read from `42_slice_spacing.txt`.

### 6h. 27 Oct 2026, final package adjustments
- References renumbered to strict order of first appearance (the overlap sentence in the Methods cited the two new cohort papers before entries 31 to 34 first appeared). Applied post hoc to the package copies with `manuscript_drafts/renumber_references.py` (not in this repository; it rewrites citations inside the existing runs so the senior author's manual superscript edits are preserved). Mapping old to new: 35 to 31, 36 to 32, 31 to 33, 32 to 34, 33 to 35, 37 to 36, 38 to 37, 39 to 38, 40 to 39, 34 to 40. Verified: 40 entries consecutive, all cited, first appearance 1 to 40 in order, entry texts unchanged, tracked baseline unchanged.
- Table 4 footnote shortened from 181 to 136 words in `build_revision_tables.py` (same content: Firth/logistf, DeLong CI, 500-resample optimism correction, refitted reference model, penalized LR test vs DeLong, abbreviations, selection-aware note with the fixed-model values).
- Abstract Conclusion split into two sentences (senior author); abstract 249 words.

### 6i. 6 Nov 2026, public-release audit
Full tree and history re-scanned for people, hosts, paths, identifiers and secrets. Institution name removed from code labels and comments (model labels now "Reference model" / "published ICU-risk model"; `build_revision_tables.py` matches the reference row by prefix so older CSVs still load), author comment in `analysis_pipeline.R` genericized, and the Avrahami note in 6b corrected (1996 citation retained). No study-specific names, case identifiers, server names or folder names remain; the README citation is the only place with author and institution names.
