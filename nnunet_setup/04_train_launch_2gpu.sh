#!/usr/bin/env bash
# 04_train_launch_2gpu.sh
#
# Two-GPU host launch for Dataset005_NeckAirway training (5-fold 3d_fullres,
# T1C single channel; n=321 at the time of this run, before the DICOM-only case was converted).
#
# Modeled on a colleague's two-GPU queue launcher.
# GPU 0 runs folds 0, 2, 4 (3 × ~6h = ~18h);
# GPU 1 runs folds 1, 3      (2 × ~6h = ~12h).
# Wall time: ~18h (GPU 0 bottleneck).
#
# Pre-requirements (already satisfied unless this is a fresh box):
#   - ~/ml-env venv with nnUNetv2 installed
#   - ~/nnunet_raw/Dataset005_NeckAirway/ rsynced from local
#   - ~/Dataset005_splits_final.json rsynced from local
#   - ~/.slack_webhook present
#
# Usage (on the GPU host, inside tmux):
#   bash ~/04_train_launch_2gpu.sh preprocess          # one-shot, ~15 min
#   bash ~/04_train_launch_2gpu.sh train_all           # main run, ~18 h wall
#   bash ~/04_train_launch_2gpu.sh all                 # preprocess + train_all

set -euo pipefail

ML_ENV=${ML_ENV:-$HOME/ml-env}
export nnUNet_raw=${nnUNet_raw:-$HOME/nnunet_raw}
export nnUNet_preprocessed=${nnUNet_preprocessed:-$HOME/nnunet_preprocessed}
export nnUNet_results=${nnUNet_results:-$HOME/nnunet_results}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
source "$ML_ENV/bin/activate"

LOG_DIR=$HOME/nnunet_logs
mkdir -p "$LOG_DIR"

DATASET=5
DATASET_DIR=Dataset005_NeckAirway
CONFIG=3d_fullres
TAG=D005_NeckAirway

slack() {
  [[ -s "$HOME/.slack_webhook" ]] || return 0
  URL=$(tr -d '[:space:]' < "$HOME/.slack_webhook")
  curl -s -o /dev/null -X POST -H 'Content-type: application/json' \
       --data "$(python3 -c 'import json,sys;print(json.dumps({"text": sys.argv[1]}))' "$1")" \
       "$URL" || true
}

preprocess() {
  slack ":hammer_and_wrench: ${TAG} preprocess starting (n=321, T1C, 3d_fullres)"
  echo "=== plan_and_preprocess D${DATASET} (3d_fullres) ==="
  nnUNetv2_plan_and_preprocess \
      -d $DATASET \
      -c $CONFIG \
      --verify_dataset_integrity \
      2>&1 | tee "$LOG_DIR/d${DATASET}_preprocess.log"

  # Install the stratified-by-abs splits AFTER preprocessing.
  TARGET="${nnUNet_preprocessed}/${DATASET_DIR}/splits_final.json"
  mkdir -p "${nnUNet_preprocessed}/${DATASET_DIR}"
  cp "$HOME/Dataset005_splits_final.json" "$TARGET"
  echo "  installed splits to $TARGET"
  slack ":white_check_mark: ${TAG} preprocess done — splits installed"
}

gpu_queue() {
  local gpu=$1; shift
  local folds=("$@")
  for f in "${folds[@]}"; do
    local log="$LOG_DIR/d${DATASET}_fold${f}_gpu${gpu}.log"
    echo "[$(date -Is)] gpu=$gpu d=${DATASET} fold=$f START" >> "$LOG_DIR/queue_d${DATASET}_gpu${gpu}.log"
    CUDA_VISIBLE_DEVICES=$gpu nnUNetv2_train $DATASET $CONFIG "$f" --npz \
      > "$log" 2>&1
    echo "[$(date -Is)] gpu=$gpu d=${DATASET} fold=$f DONE exit=$?" >> "$LOG_DIR/queue_d${DATASET}_gpu${gpu}.log"
  done
}

train_all() {
  slack ":arrows_counterclockwise: ${TAG} training queue launched — GPU0 folds 0,2,4 / GPU1 folds 1,3 (~18 h wall)"

  gpu_queue 0 0 2 4 &
  local pid0=$!
  gpu_queue 1 1 3 &
  local pid1=$!

  echo "[$(date -Is)] D${DATASET} queue launched: gpu0 pid=$pid0 (folds 0,2,4), gpu1 pid=$pid1 (folds 1,3)" \
    | tee "$LOG_DIR/queue_d${DATASET}_master.log"

  wait "$pid0" "$pid1"

  echo "[$(date -Is)] D${DATASET} all 5 folds finished" \
    | tee -a "$LOG_DIR/queue_d${DATASET}_master.log"

  # find_best across all 5 folds
  echo "=== find_best_configuration D${DATASET} ==="
  nnUNetv2_find_best_configuration $DATASET -c $CONFIG -f 0 1 2 3 4 \
    2>&1 | tee "$LOG_DIR/d${DATASET}_find_best.log" || true

  # Compose final Slack message with per-fold Dice
  RESULTS=$nnUNet_results/${DATASET_DIR}/nnUNetTrainer__nnUNetPlans__${CONFIG}
  CV=$RESULTS/crossval_results_folds_0_1_2_3_4/summary.json
  declare -a D
  for f in 0 1 2 3 4; do
    v=$(grep "Mean Validation Dice" "$RESULTS/fold_$f"/training_log_*.txt 2>/dev/null | tail -1 | grep -oE "0\.[0-9]+")
    D[$f]=$v
  done
  MEAN=$(python3 -c "
vals=[v for v in '${D[0]} ${D[1]} ${D[2]} ${D[3]} ${D[4]}'.split() if v]
print(round(sum(map(float,vals))/len(vals),4) if vals else 'NA')
")
  CVD=$(python3 -c "import json;print(round(json.load(open('$CV'))['foreground_mean']['Dice'],4))" 2>/dev/null || echo NA)

  slack ":checkered_flag: *${TAG} training complete* (5-fold, 3d_fullres, T1C, n=321)
Per-fold val Dice: ${D[0]}, ${D[1]}, ${D[2]}, ${D[3]}, ${D[4]}
5-fold mean Dice: *${MEAN}*  (find_best crossval Dice ${CVD})
Reference: Yu et al. 2026 healthy adults 0.927; manual inter-observer Dice 0.865 (n=20)
Next: rsync predicted_masks back to local; rerun 09_seg_agreement + 10/11/12 figures"
}

case "${1:-}" in
  preprocess) preprocess ;;
  train_all)  train_all ;;
  all)        preprocess; train_all ;;
  *) echo "usage: $0 {preprocess|train_all|all}"; exit 2 ;;
esac
