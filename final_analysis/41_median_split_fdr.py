#!/usr/bin/env python3
"""
Revision analysis (EJR-D-26-04240, reviewer point 7): false-discovery-rate
adjustment (Benjamini-Hochberg) across the five median-split Fisher tests of
33_risk_strat_validation.csv (large-abscess subset, n = 68). Supersedes the
Bonferroni column (p_bonf5) of script 33 for the response letter and the
calculations handout; Bonferroni-adjusted values are not reported in the revision.
Output: 41_median_split_fdr.csv / .txt
"""
import os
from pathlib import Path
import pandas as pd
from statsmodels.stats.multitest import multipletests

BASE = Path(os.environ.get("AIRWAY_ROOT", Path(__file__).resolve().parents[1]))
FA = BASE / "final_analysis"
v = pd.read_csv(FA / "33_risk_strat_validation.csv")
ms = v[v.analysis == "median_split"][["feature", "p"]].copy()
ms["q_fdr5"] = multipletests(ms.p, method="fdr_bh")[1]
ms.to_csv(FA / "41_median_split_fdr.csv", index=False, float_format="%.5f")
lines = ["Median-split Fisher tests of ICU admission within large abscesses (n = 68); q = Benjamini-Hochberg FDR across the five features:"]
lines += [f"  {r.feature}: p = {r.p:.4f}, q = {r.q_fdr5:.4f}" for r in ms.itertuples()]
(FA / "41_median_split_fdr.txt").write_text("\n".join(lines) + "\n"); print("\n".join(lines))
