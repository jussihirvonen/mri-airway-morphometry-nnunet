#!/usr/bin/env python3
"""
Stage Dataset005_Airway LOCALLY under nnunet_setup/Dataset005_Airway/.

Sources:
  - T1C images:  <nifti_pos_dir>/<case_id>/t1c.nii.gz  (abs+ original cohort)
                 <nifti_neg_dir>/<case_id>/t1c.nii.gz  (abs- expansion; keys in paths_local.json)
  - Airway masks: the reference-mask folder, <NNNNa.nii.gz> (paths_local.json key reference_masks)

Special-cases:
  - a case without a NIfTI image (DICOM only) is dropped unless converted first
  - a mask that covers only the neck is zero-padded to the image extent
             (image extends to the chest)

Output layout:
  Dataset005_Airway/
    dataset.json
    imagesTr/case<NNN>_0000.nii.gz ...   (T1C, copied or symlinked)
    labelsTr/case<NNN>.nii.gz ...        (airway mask, validated binary)

After this, rsync the folder to the GPU host's nnUNet_raw, copy splits_final.json to
the corresponding nnUNet_preprocessed dir after plan_and_preprocess.
"""

import os
import json
import re
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np

AIRWAY = Path(os.environ.get("AIRWAY_ROOT", Path(__file__).resolve().parents[1]))
_CFG_FILE = AIRWAY / "paths_local.json"  # untracked: maps generic keys to the local folder layout (README, "Local layout")
_CFG = json.loads(_CFG_FILE.read_text()) if _CFG_FILE.exists() else {}
REF_MASKS = AIRWAY / _CFG.get("reference_masks", "reference_masks")
POS_DIR = AIRWAY / _CFG.get("nifti_pos_dir", "images/positive")
NEG_DIR = AIRWAY / _CFG.get("nifti_neg_dir", "images/negative")
SETUP = AIRWAY / "nnunet_setup"
OUT = SETUP / "Dataset005_NeckAirway"

DROP_CASES = set()  # empty: the one DICOM-only case was converted to NIfTI


def find_t1c(case_id: str):
    for root in (POS_DIR, NEG_DIR):
        p = root / case_id / "t1c.nii.gz"
        if p.exists():
            return p
    return None


def pad_mask_to_image(mask_path: Path, image_path: Path, dest_path: Path):
    """Pad/crop mask z-extent to match image, preserving in-plane voxels."""
    mask = nib.load(str(mask_path))
    img = nib.load(str(image_path))
    if mask.shape == img.shape:
        # No padding needed: just ensure binary 0/1 and save with image's affine
        data = (mask.get_fdata() > 0).astype(np.uint8)
        nii = nib.Nifti1Image(data, img.affine, img.header)
        nib.save(nii, str(dest_path))
        return "ok"

    if mask.shape[:2] != img.shape[:2]:
        raise RuntimeError(
            f"in-plane shape mismatch for {mask_path.stem}: "
            f"mask {mask.shape}, img {img.shape}"
        )

    # Affine alignment: derive how mask z-slices map onto image z-slices
    if not np.allclose(mask.affine[:3, :3], img.affine[:3, :3], atol=1e-3):
        raise RuntimeError(f"affine rotation/scale mismatch for {mask_path.stem}")
    dz = img.affine[2, 2]
    z0_img = img.affine[2, 3]
    z0_msk = mask.affine[2, 3]
    offset = int(round((z0_msk - z0_img) / dz))

    out = np.zeros(img.shape, dtype=np.uint8)
    m = (mask.get_fdata() > 0).astype(np.uint8)
    z_lo = max(0, offset)
    z_hi = min(img.shape[2], offset + mask.shape[2])
    m_lo = z_lo - offset
    m_hi = m_lo + (z_hi - z_lo)
    out[:, :, z_lo:z_hi] = m[:, :, m_lo:m_hi]
    nii = nib.Nifti1Image(out, img.affine, img.header)
    nib.save(nii, str(dest_path))
    return f"padded ({mask.shape[2]}->{img.shape[2]})"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "imagesTr").mkdir(exist_ok=True)
    (OUT / "labelsTr").mkdir(exist_ok=True)

    qc = []
    for f in sorted(REF_MASKS.glob("*a.nii.gz")):
        m = re.match(r"(\d+)a\.nii\.gz", f.name)
        if not m:
            continue
        num = int(m.group(1))
        cid = f"case{num:03d}"

        if cid in DROP_CASES:
            qc.append((cid, "DROPPED", "no T1C NIfTI"))
            continue

        t1c = find_t1c(cid)
        if t1c is None:
            qc.append((cid, "MISSING_T1C", str(t1c)))
            continue

        # Copy T1C as imagesTr/<cid>_0000.nii.gz (force binary uint8 on labels later)
        dest_img = OUT / "imagesTr" / f"{cid}_0000.nii.gz"
        if not dest_img.exists():
            shutil.copy2(t1c, dest_img)

        dest_lbl = OUT / "labelsTr" / f"{cid}.nii.gz"
        try:
            status = pad_mask_to_image(f, t1c, dest_lbl)
        except Exception as e:
            qc.append((cid, "ERROR", str(e)))
            continue

        # Verify final state
        L = nib.load(str(dest_lbl)); I = nib.load(str(dest_img))
        d = L.get_fdata()
        vox = int((d > 0).sum())
        uniq = sorted(np.unique(d.astype(int)).tolist())
        ok = (L.shape == I.shape) and (vox > 0) and (set(uniq) <= {0, 1})
        qc.append((cid, "OK" if ok else "FAIL",
                   f"{status}; voxels={vox}; uniq={uniq}; shape={L.shape}"))

    # Write QC report
    rep = SETUP / "qc_report.txt"
    with open(rep, "w") as fp:
        n_ok = sum(1 for r in qc if r[1] == "OK")
        n_dropped = sum(1 for r in qc if r[1] == "DROPPED")
        n_err = len(qc) - n_ok - n_dropped
        fp.write(f"Dataset005_Airway staging QC report\n")
        fp.write(f"  total cases:   {len(qc)}\n")
        fp.write(f"  OK:            {n_ok}\n")
        fp.write(f"  DROPPED:       {n_dropped}\n")
        fp.write(f"  errors/fails:  {n_err}\n\n")
        for cid, status, note in qc:
            fp.write(f"  {cid}  {status:12s}  {note}\n")
    print(f"Wrote {rep}")
    n_ok = sum(1 for r in qc if r[1] == "OK")
    print(f"  OK = {n_ok}  DROPPED = {sum(1 for r in qc if r[1]=='DROPPED')}  "
          f"errors = {len(qc)-n_ok-sum(1 for r in qc if r[1]=='DROPPED')}")

    # Update dataset.json with numTraining = OK count
    ds = json.loads((SETUP / "dataset.json").read_text())
    ds["numTraining"] = n_ok
    (OUT / "dataset.json").write_text(json.dumps(ds, indent=2))
    print(f"Wrote {OUT/'dataset.json'} (numTraining={n_ok})")


if __name__ == "__main__":
    main()
