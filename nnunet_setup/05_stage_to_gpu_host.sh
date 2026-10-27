#!/usr/bin/env bash
# 05_stage_to_gpu_host.sh — LOCAL-SIDE rsync to the two-GPU host (ssh alias in $REMOTE).
#
# What it does (idempotent):
#   1. rsync Dataset005_NeckAirway/ to $REMOTE:~/nnunet_raw/Dataset005_NeckAirway/
#   2. rsync splits_final.json    to $REMOTE:~/Dataset005_splits_final.json
#   3. rsync 04_train_launch_2gpu.sh to $REMOTE:~/04_train_launch_2gpu.sh
#
# It does NOT launch training. After running this, on the GPU host:
#   tmux new -s d5_train
#   bash ~/04_train_launch_2gpu.sh all          # preprocess + train_all
#   # detach with Ctrl-b d; reattach with `tmux a -t d5_train`
#
# Optional per-fold Slack watchers (in a second tmux session):
#   tmux new -s d5_watch
#   bash ~/fold_watcher.sh ~/nnunet_logs/d5_fold0_gpu0.log D5/f0   # window 0
#   # Ctrl-b c to open a new window, then repeat for folds 1..4

set -euo pipefail

cd "$(dirname "$0")"

REMOTE=${REMOTE:?set REMOTE to the ssh alias of the GPU host}
RAW_REMOTE="${REMOTE}:nnunet_raw/"
HOME_REMOTE="${REMOTE}:"

echo "== sanity =="
[[ -d Dataset005_NeckAirway ]] || { echo "Dataset005_NeckAirway/ not staged locally; run 01+02 first"; exit 1; }
[[ -f splits_final.json ]]     || { echo "splits_final.json missing; run 01_make_splits_local.py first"; exit 1; }
[[ -f 04_train_launch_2gpu.sh ]]  || { echo "04_train_launch_2gpu.sh missing"; exit 1; }

N_IMG=$(ls Dataset005_NeckAirway/imagesTr/ | wc -l | tr -d ' ')
N_LBL=$(ls Dataset005_NeckAirway/labelsTr/ | wc -l | tr -d ' ')
echo "  Dataset005_NeckAirway: imagesTr=${N_IMG} labelsTr=${N_LBL}"
[[ "$N_IMG" == "$N_LBL" ]] || { echo "image/label count mismatch"; exit 1; }

echo
echo "== rsync staging -> ${REMOTE} =="
rsync -aP --delete Dataset005_NeckAirway/ "${REMOTE}:nnunet_raw/Dataset005_NeckAirway/"

echo
echo "== rsync splits + launch script =="
rsync -aP splits_final.json     "${REMOTE}:Dataset005_splits_final.json"
rsync -aP 04_train_launch_2gpu.sh  "${REMOTE}:04_train_launch_2gpu.sh"
ssh "$REMOTE" 'chmod +x ~/04_train_launch_2gpu.sh'

echo
echo "== verify on ${REMOTE} =="
ssh "$REMOTE" '
  echo "  imagesTr count: $(ls ~/nnunet_raw/Dataset005_NeckAirway/imagesTr | wc -l)"
  echo "  labelsTr count: $(ls ~/nnunet_raw/Dataset005_NeckAirway/labelsTr | wc -l)"
  echo "  dataset.json  : $(grep numTraining ~/nnunet_raw/Dataset005_NeckAirway/dataset.json)"
  echo "  splits file   : $(wc -l < ~/Dataset005_splits_final.json) lines"
  echo "  launch script : $(ls -la ~/04_train_launch_2gpu.sh | awk "{print \$1, \$5, \$NF}")"
'

cat <<EOF

Staging complete.

Next steps (on the GPU host — wait until current jobs finish):

  ssh "$REMOTE"
  tmux new -s d5_train
  bash ~/04_train_launch_2gpu.sh all          # preprocess + 5-fold queue
  # Ctrl-b d to detach

  # Optional per-fold Slack watchers in a second tmux session:
  tmux new -s d5_watch
  bash ~/fold_watcher.sh ~/nnunet_logs/d5_fold0_gpu0.log D5/f0    # window 0
  # Ctrl-b c, then for the rest:
  #   bash ~/fold_watcher.sh ~/nnunet_logs/d5_fold1_gpu1.log D5/f1
  #   bash ~/fold_watcher.sh ~/nnunet_logs/d5_fold2_gpu0.log D5/f2
  #   bash ~/fold_watcher.sh ~/nnunet_logs/d5_fold3_gpu1.log D5/f3
  #   bash ~/fold_watcher.sh ~/nnunet_logs/d5_fold4_gpu0.log D5/f4

Wall time ~18h (GPU 0 bottleneck: folds 0,2,4).
EOF
