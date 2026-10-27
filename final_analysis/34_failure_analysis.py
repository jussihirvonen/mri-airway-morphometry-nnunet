#!/usr/bin/env python3
"""
Revision analysis (EJR-D-26-04240, reviewer point 9): per-case failure
analysis of the nnU-Net out-of-fold predictions.

Failure set = HD95 > 5 mm (n=19) OR Dice < 0.85, from 09_seg_agreement_percase.csv.
For each case the script quantifies the mechanism automatically:
  - craniocaudal extent of manual vs predicted (slices above / below the manual range)
  - volume ratio predicted/manual
  - number of predicted components, voxels outside the largest component
  - fraction of HD95-contributing surface voxels that lie outside the manual
    z-range (endpoint error) vs inside (in-plane boundary error)
  - whether the manual mask bridges a total obstruction (any manual slice with
    area <= 2 voxels) and whether the prediction has a z-gap there
and assigns a provisional category:
  OVER (pred/manual volume > 1.25), UNDER (< 0.8), ENDPOINT (>=2 slices extent
  difference), FRAGMENTED (z-gap at an obstruction site), OUTLIER (stray
  component > 200 voxels far from the airway), else BOUNDARY.
Clinical context (ICU, abscess, diameter, RPE) is appended.
A review montage (sagittal MIP overlay + worst axial slice) is written per
case to 34_failure_montages/ for visual confirmation by the radiologists.

Output: 34_failure_analysis.csv / .txt, 34_failure_montages/*.png
"""
import os, re
import json
from pathlib import Path
import numpy as np, pandas as pd, nibabel as nib
from scipy import ndimage
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(os.environ.get("AIRWAY_ROOT", Path(__file__).resolve().parents[1]))
_CFG_FILE = BASE / "paths_local.json"  # untracked: maps generic keys to the local folder layout (README, "Local layout")
_CFG = json.loads(_CFG_FILE.read_text()) if _CFG_FILE.exists() else {}
FA = BASE / "final_analysis"
REF_MASKS = BASE / _CFG.get("reference_masks", "reference_masks"); PRED = FA / "predicted_masks"
IMG = BASE / "nnunet_setup" / "Dataset005_NeckAirway" / "imagesTr"
OUTD = FA / "34_failure_montages"; OUTD.mkdir(exist_ok=True)

pc = pd.read_csv(FA / "09_seg_agreement_percase.csv")
m = pd.read_csv(FA / "airway_morphology_merged.csv"); m["case_id"] = m.case_id.astype(str)
d = pc.merge(m[["case_id", "icu_binary", "abs_positive", "abs_diamax_mri", "retroinf", "hospd"]], on="case_id")
fail = d[(d.hd95_mm > 5) | (d.dice < 0.85)].copy().sort_values("hd95_mm", ascending=False)

def load(cid):
    num = int(cid.replace("case", ""))
    man = nib.load(str(REF_MASKS / f"{num:04d}a.nii.gz")); pre = nib.load(str(PRED / f"{cid}.nii.gz"))
    img = nib.load(str(IMG / f"{cid}_0000.nii.gz"))
    return (man.get_fdata() > 0), (pre.get_fdata() > 0), img.get_fdata(), [float(z) for z in man.header.get_zooms()]

def surface(b):
    return b & ~ndimage.binary_erosion(b)

rows = []; L = []
for _, r in fail.iterrows():
    cid = r.case_id
    man, pre, img, (vx, vy, vz) = load(cid)
    zm = np.where(man.any(axis=(0, 1)))[0]; zp = np.where(pre.any(axis=(0, 1)))[0]
    caud = int(max(0, zm.min() - zp.min())); cran = int(max(0, zp.max() - zm.max()))  # z increases cranially (S axis)
    short_caud = int(max(0, zp.min() - zm.min())); short_cran = int(max(0, zm.max() - zp.max()))
    vol_ratio = pre.sum() / man.sum()
    lab, n = ndimage.label(pre, structure=ndimage.generate_binary_structure(3, 3))
    sizes = np.sort(ndimage.sum(pre, lab, range(1, n + 1)))[::-1] if n else np.array([0])
    extra_vox = int(sizes[1:].sum()) if n > 1 else 0
    # manual bridged obstruction (area <= 2 voxels on a slice) and predicted z-gap inside manual range
    man_areas = man.sum(axis=(0, 1)); pre_areas = pre.sum(axis=(0, 1))
    pinpoint = [int(z) for z in zm if man_areas[z] <= 2]
    zgap = [int(z) for z in range(zm.min(), zm.max() + 1) if pre_areas[z] == 0]
    # HD95 attribution: distance from predicted surface voxels to manual mask; take those > 5mm
    dt_man = ndimage.distance_transform_edt(~man, sampling=(vx, vy, vz))
    sp = surface(pre); dists = dt_man[sp]; far = sp & (dt_man > 5)
    far_z = np.where(far.any(axis=(0, 1)))[0]
    far_outside = int(sum((far[:, :, z]).sum() for z in far_z if z < zm.min() or z > zm.max()))
    far_total = int(far.sum())
    # also manual surface far from prediction (missed tube)
    dt_pre = ndimage.distance_transform_edt(~pre, sampling=(vx, vy, vz)); sm = surface(man); missed_far = int((sm & (dt_pre > 5)).sum())
    # category
    cat = []
    if caud + cran >= 2: cat.append("ENDPOINT-extra")
    if short_caud + short_cran >= 2: cat.append("ENDPOINT-short")
    if vol_ratio > 1.25: cat.append("OVER")
    if vol_ratio < 0.8: cat.append("UNDER")
    if zgap: cat.append("ZGAP" + ("@pinpoint" if any(abs(g - p) <= 1 for g in zgap for p in pinpoint) else ""))
    if extra_vox > 200: cat.append("STRAY>200vox")
    if not cat: cat.append("BOUNDARY")
    rows.append(dict(case=cid, dice=round(r.dice, 3), hd95_mm=round(r.hd95_mm, 1), n_comp=int(n), extra_component_vox=extra_vox,
                     vol_ratio_pred_over_manual=round(vol_ratio, 2), extra_slices_caudal=caud, extra_slices_cranial=cran,
                     missing_slices_caudal=short_caud, missing_slices_cranial=short_cran, manual_pinpoint_slices=len(pinpoint), pred_zgap_slices=len(zgap),
                     far_pred_surface_vox=far_total, far_pred_surface_outside_manual_zrange=far_outside, manual_surface_missed_gt5mm=missed_far,
                     manual_slices=len(zm), category=";".join(cat), ICU=int(r.icu_binary), abscess=int(r.abs_positive), abs_diam_mm=r.abs_diamax_mri, RPE=int(r.retroinf),
                     manual_volume_ml=r.manual_volume_ml, pred_volume_ml=r.pred_volume_ml, manual_minCSA=r.manual_min_area_mm2, pred_minCSA=r.pred_min_area_mm2))
    # montage
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.5))
    mid = int(np.median(np.where(man | pre)[0]))
    sag = np.rot90(img[mid, :, :]); mm = np.rot90(man[mid, :, :]); pm = np.rot90(pre[mid, :, :])
    asp = vz / vy
    ax[0].imshow(sag, cmap="gray", aspect=asp); ax[0].contour(mm, levels=[.5], colors="lime", linewidths=1); ax[0].contour(pm, levels=[.5], colors="red", linewidths=1, linestyles="--"); ax[0].set_title(f"{cid} sagittal x={mid}")
    mipm = np.rot90(man.any(axis=0)); mipp = np.rot90(pre.any(axis=0))
    ax[1].imshow(np.zeros_like(mipm), cmap="gray", aspect=asp); ax[1].contour(mipm, levels=[.5], colors="lime"); ax[1].contour(mipp, levels=[.5], colors="red", linestyles="--"); ax[1].set_title("sagittal MIP manual(green)/pred(red)")
    # worst axial slice = max symmetric difference
    diff = (man ^ pre).sum(axis=(0, 1)); zw = int(np.argmax(diff))
    ax[2].imshow(np.rot90(img[:, :, zw]), cmap="gray"); ax[2].contour(np.rot90(man[:, :, zw]), levels=[.5], colors="lime"); ax[2].contour(np.rot90(pre[:, :, zw]), levels=[.5], colors="red", linestyles="--"); ax[2].set_title(f"worst axial z={zw}  Dice {r.dice:.2f} HD95 {r.hd95_mm:.1f}mm")
    for a in ax: a.axis("off")
    fig.suptitle(f"{cid}: {';'.join(cat)} | ICU={int(r.icu_binary)} abs={int(r.abs_positive)} ({r.abs_diamax_mri:.0f} mm) RPE={int(r.retroinf)}", fontsize=10)
    fig.tight_layout(); fig.savefig(OUTD / f"{cid}.png", dpi=110); plt.close(fig)

T = pd.DataFrame(rows)
T.to_csv(FA / "34_failure_analysis.csv", index=False)
pd.set_option("display.width", 250)
L.append(f"Failure set: {len(T)} cases (HD95>5 mm: {(fail.hd95_mm>5).sum()} incl. {((fail.hd95_mm>5)&(fail.hd95_mm<5.1)).sum()} at 5.0-5.1 mm, Dice<0.85: {(T.dice<0.85).sum()})")
L.append(T[["case", "dice", "hd95_mm", "n_comp", "vol_ratio_pred_over_manual", "extra_slices_caudal", "extra_slices_cranial", "missing_slices_caudal", "missing_slices_cranial", "pred_zgap_slices", "manual_pinpoint_slices", "far_pred_surface_outside_manual_zrange", "far_pred_surface_vox", "category", "ICU", "abscess", "abs_diam_mm"]].to_string(index=False))
L.append("")
L.append("Category counts (a case may carry several):")
from collections import Counter
c = Counter(x for s in T.category for x in s.split(";")); L += [f"   {k}: {v}" for k, v in c.most_common()]
L.append(f"\nICU in failure set {T.ICU.sum()}/{len(T)} ({100*T.ICU.mean():.0f}%) vs whole cohort 33/322 (10.2%)")
L.append(f"Median manual volume in failure set {T.manual_volume_ml.median():.1f} mL vs cohort 11.4 mL; median manual minCSA {T.manual_minCSA.median():.1f} vs 23.0 mm2")
(FA / "34_failure_analysis.txt").write_text("\n".join(L)); print("\n".join(L))
