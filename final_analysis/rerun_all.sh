#!/bin/bash
# Rerun the full airway morphometry analysis pipeline.
#
# When to use:
#   - the annotator updates one or more reference masks.
#   - Cohort changes (case added or excluded).
#   - 01_run_full_morphology.py is modified (new features).
#
# What it does (sequentially):
#   1. Snapshot current outputs to snapshot_<DATE>/ for diffing.
#   2. Extract 70 morphometric features per case from masks.
#   3. Merge with the clinical master workbook + abscess radiomics.
#   4. Univariate ICU/LOS/RPE comparisons + LOO-CV nested logistic regression.
#
# What it does NOT do:
#   - nnU-Net retraining. The current Dataset005_NeckAirway training
#     reflects masks as of staging date. Small mask edits do not require
#     retraining for the morphometry paper; the DL paper would need a
#     fresh stage + retrain if any reference mask used in nnU-Net
#     splits is materially revised (see airway_nnunet_plan.md).
#
# Usage:
#   bash rerun_all.sh                   # snapshot + full rerun
#   bash rerun_all.sh nosnapshot        # rerun without taking a new snapshot
#
# Expected runtime: ~5 min total (morphometry ~30 s, analyze ~3 min for LOO-CV).

set -e

cd "$(dirname "$0")"

STAMP=$(date +%Y-%m-%d)
SNAPSHOT_DIR="snapshot_${STAMP}"

if [ "${1:-}" != "nosnapshot" ]; then
    # Find a non-colliding snapshot dir name (v1, v2, ...)
    n=1
    while [ -d "${SNAPSHOT_DIR}_v${n}" ]; do n=$((n+1)); done
    SNAPSHOT_DIR="${SNAPSHOT_DIR}_v${n}"
    mkdir -p "$SNAPSHOT_DIR"
    for f in reference_full_morphology.csv airway_morphology_merged.csv \
             03_analysis_report.txt 04_lasso_results.json \
             04_lasso_icu_logistic.json 05_prespecified_icu_table.csv \
             07_heatmap_severity.png 07_heatmap_severity.pdf \
             07_forest_severity.png 07_forest_severity.pdf \
             08_severity_compare_combined.csv \
             08_severity_compare_heatmap.png 08_severity_compare_heatmap.pdf; do
        [ -f "$f" ] && cp "$f" "$SNAPSHOT_DIR/"
    done
    echo "[snapshot] saved to $SNAPSHOT_DIR/"
fi

echo "[1/4] morphometry (01_run_full_morphology.py)"
python3 01_run_full_morphology.py

echo "[2/4] clinical merge (02_merge_clinical.py)"
python3 02_merge_clinical.py

echo "[3/4] analysis (03_analyze.py)  — LOO-CV is slow, ~3 min"
python3 03_analyze.py

echo "[4/6] prespecified 5-feature ICU table (05_prespecified_icu_table.py)"
python3 05_prespecified_icu_table.py

echo "[5/6] heatmap + forest figures (07_severity_figures.py)  — ~30 s bootstrap"
python3 07_severity_figures.py

echo "[6/6] cohort comparison (08_severity_cohort_compare.py)"
python3 08_severity_cohort_compare.py

echo
echo "Done. Compare new outputs against $SNAPSHOT_DIR/ to assess impact."
echo "Quick diff:"
echo "  diff $SNAPSHOT_DIR/05_prespecified_icu_table.csv 05_prespecified_icu_table.csv"
