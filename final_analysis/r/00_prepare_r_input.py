#!/usr/bin/env python3
"""
One-time prep: build the R input CSV for the airway morphometry analyses.

Merges:
  - airway_morphology_merged.csv (Python pipeline output, n=322)
  - hammas + peritonsinf from the clinical master workbook (via accession_number)
  - per-case inter-rater (expert vs reference morphometry) from
    13_inter_rater_percase_n20.csv

Output:
  r/airway_features_with_outcomes.csv  (n=322, all columns the R pipeline needs)
"""
import os
import json
from pathlib import Path
import pandas as pd

BASE = Path(os.environ.get("AIRWAY_ROOT", Path(__file__).resolve().parents[2]))
_CFG_FILE = BASE / "paths_local.json"  # untracked: maps generic keys to the local folder layout (README, "Local layout")
_CFG = json.loads(_CFG_FILE.read_text()) if _CFG_FILE.exists() else {}
FA = BASE / "final_analysis"
OUT_CSV = FA / "r" / "airway_features_with_outcomes.csv"
IR_CSV  = FA / "r" / "inter_rater_n20.csv"

merged = pd.read_csv(FA / "airway_morphology_merged.csv").drop(columns=["hammas"], errors="ignore")
mri = pd.read_excel(BASE / _CFG.get("clinical_master_xlsx_copy", "clinical_master.xlsx"),
                    sheet_name="data")
extras = mri[["ac", "hammas", "peritonsinf", "leuk"]].rename(columns={"ac": "accession_number",
                                                                       "leuk": "wbc_10e9_per_L"})
merged["accession_number"] = pd.to_numeric(merged["accession_number"], errors="coerce")
extras["accession_number"] = pd.to_numeric(extras["accession_number"], errors="coerce")
df = merged.merge(extras, on="accession_number", how="left")
df.to_csv(OUT_CSV, index=False)
print(f"Wrote {OUT_CSV}  ({len(df)} rows × {df.shape[1]} columns)")

# Inter-rater file copy / slim
ir = pd.read_csv(FA / "13_inter_rater_percase_n20.csv")
ir.to_csv(IR_CSV, index=False)
print(f"Wrote {IR_CSV}  ({len(ir)} rows)")
