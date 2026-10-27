#!/usr/bin/env python3
"""
Revision analysis (EJR-D-26-04240, reviewer point 9, second part): the
low-Dice cases of the HUMAN interobserver substudy (Table 2 ranges), analysed
separately from the network failures (34_failure_analysis.py).
Source: 13_inter_rater_percase.csv (expert vs reference, n=30) and
14_novice_percase_round2.csv (novice vs reference, n=30), which hold Dice,
HD95 and the full feature set per reader. For every case with Dice < 0.85 the
mechanism is read from the feature differences: craniocaudal extent (number of
slices), volume ratio, and the effect on minimum CSA, LCSA50, compactness and
maximum midline shift.
Output: 39_reader_outliers.csv / .txt
"""
import os
from pathlib import Path
import numpy as np, pandas as pd

BASE = Path(os.environ.get("AIRWAY_ROOT", Path(__file__).resolve().parents[1])); FA = BASE / "final_analysis"
rows = []; L = []
for reader, f, pre in [("expert", "13_inter_rater_percase.csv", "expert"), ("novice", "14_novice_percase_round2.csv", "novice")]:
    d = pd.read_csv(FA / f)
    L.append(f"{reader} vs reference: n={len(d)}, Dice mean {d.dice.mean():.3f}, min {d.dice.min():.3f}; cases with Dice < 0.85: {(d.dice<0.85).sum()}; < 0.80: {(d.dice<0.80).sum()}")
    d["dz"] = d[f"{pre}_n_slices"] - d["reference_n_slices"]; d["vol_ratio"] = d[f"{pre}_volume_ml"] / d["reference_volume_ml"]
    rho = d[["dice", "reference_volume_ml"]].corr(method="spearman").iloc[0, 1]
    L.append(f"  Spearman Dice ~ reference volume: {rho:+.2f}; Dice ~ |slice-count difference|: {d[['dice']].assign(a=d.dz.abs()).corr(method='spearman').iloc[0,1]:+.2f}")
    for _, r in d[d.dice < 0.85].sort_values("dice").iterrows():
        mech = []
        if abs(r.dz) >= 2: mech.append(f"craniocaudal extent differs by {int(abs(r.dz))} slices")
        elif abs(r.dz) == 1: mech.append("one-slice endpoint difference")
        if r.vol_ratio < 0.8: mech.append(f"under-segmentation (volume ratio {r.vol_ratio:.2f})")
        if r.vol_ratio > 1.25: mech.append(f"over-segmentation (volume ratio {r.vol_ratio:.2f})")
        if not mech: mech.append("in-plane boundary disagreement")
        rows.append(dict(reader=reader, case=r.case_id, dice=round(r.dice, 3), hd95_mm=round(r.hd95_mm, 1), ref_volume_ml=r.reference_volume_ml, reader_volume_ml=r[f"{pre}_volume_ml"],
                         slice_diff=int(r.dz), ref_minCSA=r.reference_min_area_mm2, reader_minCSA=r[f"{pre}_min_area_mm2"], ref_LCSA50=r.reference_len_below_50mm2_mm, reader_LCSA50=r[f"{pre}_len_below_50mm2_mm"],
                         ref_maxMLS=r.reference_rl_dev_max_abs_mm, reader_maxMLS=r[f"{pre}_rl_dev_max_abs_mm"], ref_compactness=r.reference_compactness, reader_compactness=r[f"{pre}_compactness"], mechanism="; ".join(mech)))
T = pd.DataFrame(rows); T.to_csv(FA / "39_reader_outliers.csv", index=False, float_format="%.3f")
pd.set_option("display.width", 250); L.append(""); L.append(T.to_string(index=False))
(FA / "39_reader_outliers.txt").write_text("\n".join(L)); print("\n".join(L))
