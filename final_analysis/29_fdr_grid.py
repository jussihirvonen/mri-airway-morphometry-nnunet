#!/usr/bin/env python3
"""
Revision analysis (EJR-D-26-04240, reviewer point 3): multiplicity control for
the 5 features x 6 severity markers grid (30 tests, full cohort) and for the
five ICU tests of Table 3. Benjamini-Hochberg false-discovery-rate q values
(primary) and Bonferroni-adjusted p (for reference) are added to the canonical
effect/p table 08_severity_compare_combined.csv. Also the abscess-positive
subset grid (30 tests) as sensitivity.
Output: 29_fdr_grid.csv / .txt
"""
import os
from pathlib import Path
import pandas as pd, numpy as np
from statsmodels.stats.multitest import multipletests

BASE = Path(os.environ.get("AIRWAY_ROOT", Path(__file__).resolve().parents[1]))
FA = BASE / "final_analysis"
g = pd.read_csv(FA / "08_severity_compare_combined.csv")
out = []; L = []
for cohort in ["Full cohort", "abs+"]:
    sub = g[(g.Cohort == cohort)].dropna(subset=["p"]).copy()
    sub["q_BH_30"] = multipletests(sub.p, method="fdr_bh")[1]
    sub["p_Bonf_30"] = multipletests(sub.p, method="bonferroni")[1]
    icu = sub["Severity axis"] == "ICU"
    sub.loc[icu, "q_BH_5_ICU"] = multipletests(sub.loc[icu, "p"], method="fdr_bh")[1]
    out.append(sub)
    L.append(f"{cohort}: {len(sub)} tests; nominal p<0.05: {(sub.p<0.05).sum()}; BH q<0.05: {(sub.q_BH_30<0.05).sum()}; Bonferroni: {(sub.p_Bonf_30<0.05).sum()}")
    lost = sub[(sub.p < 0.05) & (sub.q_BH_30 >= 0.05)]
    L.append("  nominal-only (lost under BH): " + ("; ".join(f"{r['Airway feature']} x {r['Severity axis']} (p={r.p:.3f}, q={r.q_BH_30:.3f})" for _, r in lost.iterrows()) or "none"))
    lostb = sub[(sub.p < 0.05) & (sub.p_Bonf_30 >= 0.05)]
    L.append("  lost under Bonferroni: " + ("; ".join(f"{r['Airway feature']} x {r['Severity axis']} (p={r.p:.4f})" for _, r in lostb.iterrows()) or "none"))
    L.append("  ICU column (Table 3): " + "; ".join(f"{r['Airway feature']} p={r.p:.4f} q30={r.q_BH_30:.4f} q5={r.q_BH_5_ICU:.4f} pBonf={r.p_Bonf_30:.3f}" for _, r in sub[icu].iterrows()))
    L.append("")
res = pd.concat(out)
res.to_csv(FA / "29_fdr_grid.csv", index=False, float_format="%.6g")
(FA / "29_fdr_grid.txt").write_text("\n".join(L)); print("\n".join(L))
print(res[res.Cohort == "Full cohort"].sort_values("p")[["Airway feature", "Severity axis", "Effect", "p", "q_BH_30", "p_Bonf_30"]].to_string(index=False))
