#!/usr/bin/env python3
"""
LOCAL prep step for the airway nnU-Net dataset.

Outputs (written to this directory, then rsynced to the GPU host):

  case_manifest.csv   — case_id, reference_mask_path, abs_positive (for splitting)
  splits_final.json   — nnU-Net 5-fold splits stratified by abs_positive
  rename_masks.txt    — pairs of (source reference-mask file, destination labelsTr name)

No host-side actions yet. The host-side launch script consumes these files.
"""

import os
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

BASE = Path(os.environ.get("AIRWAY_ROOT", Path(__file__).resolve().parents[1]))
_CFG_FILE = BASE / "paths_local.json"  # untracked: maps generic keys to the local folder layout (README, "Local layout")
_CFG = json.loads(_CFG_FILE.read_text()) if _CFG_FILE.exists() else {}
REF_MASKS = BASE / _CFG.get("reference_masks", "reference_masks")
OUT = BASE / "nnunet_setup"

CLINICAL = BASE / "final_analysis" / "airway_morphology_merged.csv"

SEED = 42

# Cases with airway masks but no NIfTI T1C available locally (DICOM-only).
# Empty: the one DICOM-only case was converted to NIfTI, so the full n=322 is stageable.
DROP_CASES = set()


def main():
    OUT.mkdir(exist_ok=True)

    cases = []
    for f in sorted(REF_MASKS.glob("*a.nii.gz")):
        m = re.match(r"(\d+)a\.nii\.gz", f.name)
        if not m:
            continue
        num = int(m.group(1))
        cid = f"case{num:03d}"
        if cid in DROP_CASES:
            continue
        cases.append({
            "case_id": cid,
            "reference_filename": f.name,
            "labelsTr_filename": f"{cid}.nii.gz",
            "imagesTr_filename": f"{cid}_0000.nii.gz",
        })
    df = pd.DataFrame(cases)
    print(f"Found {len(df)} reference segmentations.")

    # Merge abs_positive from the analysis dataset to use as stratification variable
    merged = pd.read_csv(CLINICAL)[["case_id", "abs_positive", "icu_binary"]]
    df = df.merge(merged, on="case_id", how="left")
    print(f"  abs_positive coverage: {df.abs_positive.notna().sum()}/{len(df)}")
    print(f"    abs+: {(df.abs_positive == 1).sum()}, abs-: {(df.abs_positive == 0).sum()},"
          f" unknown: {df.abs_positive.isna().sum()}")

    # For cases with unknown abs_positive (shouldn't be any after the FINAL.xlsx merge),
    # treat as a third stratum to keep them in folds.
    df["strat"] = df.abs_positive.fillna(-1).astype(int)

    # Build 5-fold splits stratified by abs_positive
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    case_ids = df.case_id.values
    strat = df.strat.values

    splits = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(case_ids, strat)):
        s = {
            "train": sorted(case_ids[train_idx].tolist()),
            "val": sorted(case_ids[val_idx].tolist()),
        }
        splits.append(s)
        train_pos = int(((strat[train_idx] == 1)).sum())
        train_neg = int(((strat[train_idx] == 0)).sum())
        val_pos = int(((strat[val_idx] == 1)).sum())
        val_neg = int(((strat[val_idx] == 0)).sum())
        val_icu = int(df.iloc[val_idx].icu_binary.fillna(0).sum())
        print(f"  fold {fold}: train n={len(train_idx)} (abs+ {train_pos}, abs- {train_neg}); "
              f"val n={len(val_idx)} (abs+ {val_pos}, abs- {val_neg}, icu+ {val_icu})")

    (OUT / "splits_final.json").write_text(json.dumps(splits, indent=2))
    print(f"\nWrote {OUT/'splits_final.json'}")

    df[["case_id", "reference_filename", "labelsTr_filename", "imagesTr_filename",
        "abs_positive", "icu_binary"]].to_csv(OUT / "case_manifest.csv", index=False)
    print(f"Wrote {OUT/'case_manifest.csv'}")

    # rename_masks.txt: src basename -> dest basename, one per line, tab-separated.
    with open(OUT / "rename_masks.txt", "w") as fp:
        for _, row in df.iterrows():
            fp.write(f"{row.reference_filename}\t{row.labelsTr_filename}\n")
    print(f"Wrote {OUT/'rename_masks.txt'}")

    # dataset.json template
    ds_json = {
        "channel_names": {"0": "T1C"},
        "labels": {"background": 0, "airway": 1},
        "numTraining": int(len(df)),
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "SimpleITKIO",
    }
    (OUT / "dataset.json").write_text(json.dumps(ds_json, indent=2))
    print(f"Wrote {OUT/'dataset.json'}")


if __name__ == "__main__":
    main()
