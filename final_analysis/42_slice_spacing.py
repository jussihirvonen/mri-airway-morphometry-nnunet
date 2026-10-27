#!/usr/bin/env python3
"""
Revision analysis (EJR-D-26-04240, second-opinion pass): voxel geometry of the
322 training volumes read from the NIfTI headers. The z zoom is the reconstructed
slice spacing (nominal slice thickness plus interslice gap); the protocol table in
the Supplementary Methods gives the nominal thickness. Source of the statement
"slice spacing 4.4 to 5.0 mm in 314 of 322 examinations" (Supplementary Methods
caption; Discussion wording "slice spacing").
Output: 42_slice_spacing.csv (per case) / 42_slice_spacing.txt (summary)
"""
import os
from pathlib import Path
import pandas as pd
import nibabel as nib

BASE = Path(os.environ.get("AIRWAY_ROOT", Path(__file__).resolve().parents[1]))
FA = BASE / "final_analysis"
LABELS = BASE / "nnunet_setup" / "Dataset005_NeckAirway" / "labelsTr"
rows = []
for f in sorted(LABELS.glob("*.nii.gz")):
    z = nib.load(str(f)).header.get_zooms()[:3]
    rows.append({"case": f.name.split(".")[0], "dx_mm": round(float(z[0]), 3), "dy_mm": round(float(z[1]), 3), "dz_mm": round(float(z[2]), 3)})
d = pd.DataFrame(rows); d.to_csv(FA / "42_slice_spacing.csv", index=False)
cnt = d.dz_mm.round(2).value_counts().sort_index()
n_main = int(((d.dz_mm >= 4.4) & (d.dz_mm <= 5.0)).sum())
lines = [f"n = {len(d)} training volumes (NIfTI headers)",
         "slice spacing (z) counts: " + ", ".join(f"{k:.2f} mm: {v}" for k, v in cnt.items()),
         f"slice spacing 4.4 to 5.0 mm: {n_main} of {len(d)}; range {d.dz_mm.min():.2f} to {d.dz_mm.max():.2f} mm",
         f"in-plane pixel size: {d.dx_mm.min():.2f} to {d.dx_mm.max():.2f} mm"]
(FA / "42_slice_spacing.txt").write_text("\n".join(lines) + "\n"); print("\n".join(lines))
