#!/bin/bash
# GPU-host launch script for Dataset005_NeckAirway training (single GPU; the reported model).
#
# Assumes:
#   - Dataset005_NeckAirway/ already rsynced to $NNUNET_HOME/nnunet_raw/
#   - splits_final.json already rsynced to ~/Dataset005_splits_final.json
#   - venv at $NNUNET_HOME/.venv exists with nnUNetv2 installed
#   - NNUNET_HOME = nnU-Net working directory on the GPU host (default: $HOME/nnunet)
#
# Usage on the GPU host:
#   export NNUNET_HOME=/path/to/nnunet-workdir
#   bash 03_train_launch.sh                 # preprocess + train all folds
#   bash 03_train_launch.sh preprocess      # preprocess only
#   bash 03_train_launch.sh train <fold>    # one fold
#   bash 03_train_launch.sh train all       # all five folds sequentially

set -e

NNUNET_HOME="${NNUNET_HOME:-$HOME/nnunet}"
cd "$NNUNET_HOME"
source .venv/bin/activate

export nnUNet_raw=$NNUNET_HOME/nnunet_raw
export nnUNet_preprocessed=$NNUNET_HOME/nnunet_preprocessed
export nnUNet_results=$NNUNET_HOME/nnunet_results

DATASET_ID=5
DATASET_DIR=Dataset005_NeckAirway
CONFIG=3d_fullres
LOG=$NNUNET_HOME/nnunet_d005_airway.log

mode="${1:-all}"
fold_arg="${2:-all}"

preprocess() {
    echo "=== Plan and preprocess Dataset $DATASET_ID — $(date) ===" | tee -a $LOG
    nnUNetv2_plan_and_preprocess -d $DATASET_ID --verify_dataset_integrity 2>&1 | tee -a $LOG

    # Install our stratified-by-abs splits AFTER preprocessing
    mkdir -p "${nnUNet_preprocessed}/${DATASET_DIR}"
    cp ~/Dataset005_splits_final.json "${nnUNet_preprocessed}/${DATASET_DIR}/splits_final.json"
    echo "  installed splits ($(wc -l < ${nnUNet_preprocessed}/${DATASET_DIR}/splits_final.json) lines)" | tee -a $LOG
}

train_fold() {
    local f=$1
    echo "=== Fold $f start: $(date) ===" | tee -a $LOG
    nnUNetv2_train $DATASET_ID $CONFIG $f --npz 2>&1 | tee -a $LOG
    echo "=== Fold $f done: $(date) ===" | tee -a $LOG
}

case "$mode" in
    preprocess)
        preprocess
        ;;
    train)
        if [ "$fold_arg" = "all" ]; then
            for f in 0 1 2 3 4; do train_fold $f; done
            echo "=== find best configuration ===" | tee -a $LOG
            nnUNetv2_find_best_configuration $DATASET_ID -c $CONFIG -f 0 1 2 3 4 2>&1 | tee -a $LOG
        else
            train_fold $fold_arg
        fi
        ;;
    all)
        preprocess
        for f in 0 1 2 3 4; do train_fold $f; done
        echo "=== find best configuration ===" | tee -a $LOG
        nnUNetv2_find_best_configuration $DATASET_ID -c $CONFIG -f 0 1 2 3 4 2>&1 | tee -a $LOG
        ;;
    *)
        echo "Unknown mode: $mode. Use preprocess | train [fold|all] | all"
        exit 1
        ;;
esac

echo "=== Script complete: $(date) ===" | tee -a $LOG
