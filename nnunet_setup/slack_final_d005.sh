#!/bin/bash
# One-shot watcher: when find_best_configuration writes the 5-fold crossval
# summary for Dataset005_NeckAirway, post a final-results summary to Slack.
# Launch nohup'd on the GPU host so it fires independently of any SSH session.

NNUNET_HOME="${NNUNET_HOME:-$HOME/nnunet}"
RESULTS=$NNUNET_HOME/nnunet_results/Dataset005_NeckAirway/nnUNetTrainer__nnUNetPlans__3d_fullres
CV=$RESULTS/crossval_results_folds_0_1_2_3_4/summary.json
NOTIFY=$NNUNET_HOME/slack_notify.sh
LOG=$NNUNET_HOME/slack_final_d005.log

echo "$(date): final-results watcher armed" >> "$LOG"

# Wait up to ~3 h for the crossval summary (find_best output).
for i in $(seq 1 360); do
    [ -f "$CV" ] && break
    sleep 30
done

if [ ! -f "$CV" ]; then
    bash "$NOTIFY" ":warning: D005 NeckAirway: find_best crossval summary did not appear within the watch window. Check the GPU host."
    echo "$(date): timed out waiting for $CV" >> "$LOG"
    exit 1
fi

# Per-fold validation Dice from each fold's training log.
declare -a D
for f in 0 1 2 3 4; do
    v=$(grep "Mean Validation Dice" "$RESULTS/fold_$f"/training_log_*.txt 2>/dev/null | tail -1 | grep -oE "0\.[0-9]+")
    D[$f]=$v
done

MEAN=$(python3 -c "
vals=[v for v in '${D[0]} ${D[1]} ${D[2]} ${D[3]} ${D[4]}'.split() if v]
print(round(sum(map(float,vals))/len(vals),4) if vals else 'NA')
")

CVD=$(python3 -c "import json;print(round(json.load(open('$CV'))['foreground_mean']['Dice'],4))" 2>/dev/null)

MSG=":checkered_flag: *D005 NeckAirway training complete* (5-fold, default 3d_fullres, T1C, n=322)\nPer-fold validation Dice: ${D[0]}, ${D[1]}, ${D[2]}, ${D[3]}, ${D[4]}\n5-fold mean Dice: *${MEAN}*  (find_best crossval Dice ${CVD})\nReference: Yu et al. 2026 healthy-adult 0.927; manual inter-observer ceiling 0.873\nNext: HD95 + manual-vs-predicted morphometric ICC for the segmentation paper"

bash "$NOTIFY" "$MSG"
echo "$(date): posted final results — mean=$MEAN cvd=$CVD" >> "$LOG"
