#!/usr/bin/env python3
"""
Merge full-cohort airway morphology with the clinical master workbook + abscess clinical data.

Pipeline:
- Load reference_full_morphology.csv (323 cases)
- Map case_id -> accession via the accession map (paths_local.json: accession_map_csv)
- Pull clinical data from the clinical master workbook (primary). For cases
  missing there, fall back to the abscess clinical table.
- Compute 5-slice min_area_5sl from the NIfTI files.
- Save airway_morphology_merged.csv.
"""

import os
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import ndimage

BASE = Path(os.environ.get("AIRWAY_ROOT", Path(__file__).resolve().parents[1]))
_CFG_FILE = BASE / "paths_local.json"  # untracked: maps generic keys to the local folder layout (README, "Local layout")
_CFG = json.loads(_CFG_FILE.read_text()) if _CFG_FILE.exists() else {}
OUT_DIR = BASE / "final_analysis"
MORPH = OUT_DIR / "reference_full_morphology.csv"
OUT = OUT_DIR / "airway_morphology_merged.csv"
REF_DIR = BASE / _CFG.get("reference_masks", "reference_masks")

AC_MAP = BASE / _CFG.get("accession_map_csv", "accession_map.csv")
CLINICAL_XLSX = Path(os.environ.get("CLINICAL_MASTER_XLSX", BASE / _CFG.get("clinical_master_xlsx", "clinical_master.xlsx")))  # all 535 rows have valid accession (rpevol variant has 28 NaN-ac rows)
ABS_CLIN = BASE / _CFG.get("abscess_clinical_csv", "abscess_clinical.csv")
SHAPE = BASE / _CFG.get("shape_features_csv", "shape_features.csv")


def compute_5sl(mask_file):
    nii = nib.load(str(mask_file))
    data = nii.get_fdata()
    vx, vy, _ = [float(z) for z in nii.header.get_zooms()]
    binary = (data > 0).astype(int)
    labeled, n = ndimage.label(binary, structure=ndimage.generate_binary_structure(3, 3))
    if n > 1:
        sizes = ndimage.sum(binary, labeled, range(1, n + 1))
        binary = (labeled == (np.argmax(sizes) + 1)).astype(int)
    areas = []
    for z in range(binary.shape[2]):
        n_vox = binary[:, :, z].sum()
        if n_vox > 0:
            areas.append(n_vox * vx * vy)
    if not areas:
        return None
    arr = np.array(areas)
    idx = int(np.argmin(arr))
    win5 = arr[max(0, idx-2):min(len(arr), idx+3)]
    win3 = arr[max(0, idx-1):min(len(arr), idx+2)]
    return {
        'min_area_1sl': float(arr[idx]),
        'min_area_3sl': float(np.mean(win3)),
        'min_area_5sl': float(np.mean(win5)),
    }


def main():
    morph = pd.read_csv(MORPH)
    print(f"morphology: {len(morph)} cases")

    # Add 5-slice
    fives = {}
    for f in sorted(REF_DIR.glob("*a.nii.gz")):
        m = re.match(r'(\d+)a\.nii\.gz', f.name)
        if m:
            cid = f"case{int(m.group(1)):03d}"
            r = compute_5sl(f)
            if r:
                fives[cid] = r
    five_df = pd.DataFrame.from_dict(fives, orient='index').reset_index().rename(columns={'index': 'case_id'})
    df = morph.merge(five_df, on='case_id', how='left')

    # Accession map
    ac_map = pd.read_csv(AC_MAP)[['case_id', 'accession_number']]
    ac_map['accession_number'] = ac_map['accession_number'].astype(int)
    df = df.merge(ac_map, on='case_id', how='left')
    print(f"  with accession: {df.accession_number.notna().sum()}/{len(df)}")

    # clinical master workbook (primary clinical source)
    mri = pd.read_excel(CLINICAL_XLSX)
    mri['accession_number'] = pd.to_numeric(mri['ac'], errors='coerce').astype('Int64')
    mri_cols = {
        'icu': 'icu_binary',
        'hospd': 'hospd',
        'rpevol': 'rpe_volume_ml_raw',
        'absvol': 'abs_volume_mri',
        'absdiamax': 'abs_diamax_mri',
        'abs': 'abs_positive',
        'crp': 'crp',
        'leuk': 'leuk',
        'temp': 'temp',
        'age': 'age',
        'sex': 'sex',
        'medoed': 'medoed_binary',
        'retroinf': 'retroinf',
        'retropaks': 'retropaks',
        'oper': 'oper',
        'opkaula': 'opkaula',
        'opkompl': 'opkompl',
        'hammas': 'hammas',
        'absloc': 'absloc',
        'trismus': 'trismus',
    }
    keep = ['accession_number'] + list(mri_cols.keys())
    mri_sub = mri[keep].rename(columns=mri_cols)
    df = df.merge(mri_sub, on='accession_number', how='left')

    # Fall back to the abscess clinical table for cases missing icu_binary
    fb = pd.read_csv(ABS_CLIN)[['case_id', 'icu_binary', 'hospd', 'crp', 'leuk',
                                 'temp', 'age', 'sex', 'absloc',
                                 'trismus', 'hammas']]
    fb = fb.rename(columns={c: c + '_fb' for c in fb.columns if c != 'case_id'})
    df = df.merge(fb, on='case_id', how='left')

    for col in ['icu_binary', 'hospd', 'crp', 'leuk', 'temp', 'age', 'sex',
                'absloc', 'trismus', 'hammas']:
        if col in df.columns and col + '_fb' in df.columns:
            df[col] = df[col].combine_first(df[col + '_fb'])
            df.drop(columns=[col + '_fb'], inplace=True)

    # Derived: hospd_prolonged = hospd > 5
    df['hospd_prolonged'] = (df['hospd'] > 5).astype(float)
    df.loc[df['hospd'].isna(), 'hospd_prolonged'] = np.nan

    # Derived: RPE binary = rpevol > 0
    df['rpe_present'] = (df['rpe_volume_ml_raw'].fillna(0) > 0).astype(float)
    df.loc[df['rpe_volume_ml_raw'].isna(), 'rpe_present'] = np.nan
    df['rpe_volume_ml'] = df['rpe_volume_ml_raw'] / 1000.0  # mm³ → mL (same convention as n=100)

    # Shape features for abscess+ cases
    shp = pd.read_csv(SHAPE)
    shp = shp[['case_id', 'Maximum3DDiameter', 'Sphericity', 'compactness',
               'volume_ml', 'SurfaceArea']]
    shp = shp.rename(columns={
        'Maximum3DDiameter': 'abs_max3d_radiomics',
        'Sphericity': 'abs_sphericity_radiomics',
        'compactness': 'abs_compactness_radiomics',
        'volume_ml': 'abs_volume_radiomics_ml',
        'SurfaceArea': 'abs_surface_area_radiomics',
    })
    df = df.merge(shp, on='case_id', how='left')

    # Final cleanup: drop cases without ICU after both sources
    n_total = len(df)
    n_no_clin = df.icu_binary.isna().sum()
    print(f"  after merge: {n_total} cases, {n_no_clin} without ICU after fallback")
    print(f"  drop these: {df.loc[df.icu_binary.isna(), 'case_id'].tolist()}")

    # Save full table (keep all rows, even those without clinical)
    df.to_csv(OUT, index=False)
    print(f"\nSaved: {OUT}")
    print(f"  total rows: {len(df)}")
    print(f"  with ICU: {df.icu_binary.notna().sum()}")
    print(f"    ICU+: {(df.icu_binary == 1).sum()}")
    print(f"    ICU-: {(df.icu_binary == 0).sum()}")
    print(f"  with RPE volume: {df.rpe_volume_ml.notna().sum()}")
    print(f"  abscess positive (clinical master abs=1): {(df.abs_positive == 1).sum()}")
    print(f"  abscess negative (clinical master abs=0): {(df.abs_positive == 0).sum()}")
    print(f"  with shape (abscess radiomics): {df.abs_volume_radiomics_ml.notna().sum()}")
    print(f"  prolonged LOS: {(df.hospd_prolonged == 1).sum()} / {df.hospd_prolonged.notna().sum()}")


if __name__ == "__main__":
    main()
