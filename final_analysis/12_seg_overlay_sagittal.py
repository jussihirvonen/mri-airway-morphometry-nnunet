#!/usr/bin/env python3
"""
Qualitative segmentation overlays, MID-SAGITTAL view (Paper 2 companion to
11_seg_overlay_examples.py). Shows the whole airway column in one plane.

Panels (1x3): Good / Intermediate / Poor Dice example cases.
Each panel = the mid-sagittal T1C slice through the airway, with manual
(red, solid) and predicted (blue, dashed) contours.

Raw predicted masks are used (islands NOT removed) so any stray components
are visible; the morphometry pipeline removes islands separately.

Output: 12_seg_overlay_sagittal.png / .pdf
"""

import os
import json
from pathlib import Path
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

AIR = Path(os.environ.get("AIRWAY_ROOT", Path(__file__).resolve().parents[1]))
_CFG_FILE = AIR / "paths_local.json"  # untracked: maps generic keys to the local folder layout (README, "Local layout")
_CFG = json.loads(_CFG_FILE.read_text()) if _CFG_FILE.exists() else {}
IMG = AIR / "nnunet_setup" / "Dataset005_NeckAirway" / "imagesTr"
LAB = AIR / "nnunet_setup" / "Dataset005_NeckAirway" / "labelsTr"
PRED = AIR / "final_analysis" / "predicted_masks"
PERCASE = AIR / "final_analysis" / "09_seg_agreement_percase.csv"
OUT = AIR / "final_analysis" / "12_seg_overlay_sagittal"

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial'],
    'font.size': 10, 'figure.dpi': 300,
})
MANUAL_COLOR = '#e74c3c'
PRED_COLOR = '#3498db'
OVERRIDE = _CFG.get("examples", {}).get("overlay")  # {'Good': case, 'Intermediate': case, 'Poor': case}; None = auto-select


def orient_sag(arr2d, aff):
    """arr2d is (Y, Z) for a fixed X. Return (Z, Y) display: superior up, anterior left."""
    disp = arr2d.T  # (Z, Y)
    if aff[2, 2] > 0:      # increasing voxel-z = superior -> put superior on top
        disp = np.flipud(disp)
    if aff[1, 1] > 0:      # increasing voxel-y = anterior -> put anterior on left
        disp = np.fliplr(disp)
    return disp


def main():
    df = pd.read_csv(PERCASE)
    cases = [(df[df.case_id == OVERRIDE[q]].iloc[0], q)
             for q in ['Good', 'Intermediate', 'Poor']]

    fig, axes = plt.subplots(1, 3, figsize=(9, 5.2))

    for ax, (row, quality) in zip(axes, cases):
        cid = row['case_id']
        dice = row['dice']
        t1c_nii = nib.load(str(IMG / f"{cid}_0000.nii.gz"))
        aff = t1c_nii.affine
        vx, vy, vz = [float(z) for z in t1c_nii.header.get_zooms()[:3]]
        t1c = t1c_nii.get_fdata()
        manual = nib.load(str(LAB / f"{cid}.nii.gz")).get_fdata() > 0.5
        pred = nib.load(str(PRED / f"{cid}.nii.gz")).get_fdata() > 0.5

        # mid-sagittal x = plane with most manual airway
        x = int(np.argmax([manual[xx, :, :].sum() for xx in range(manual.shape[0])]))

        sl = orient_sag(t1c[x, :, :].astype(float), aff)
        m = orient_sag(manual[x, :, :].astype(float), aff)
        p = orient_sag(pred[x, :, :].astype(float), aff)

        # crop to manual airway bbox + margin (rows=Z, cols=Y)
        rr = np.where(m.any(axis=1))[0]; cc = np.where(m.any(axis=0))[0]
        mz, my = 4, 45
        r0 = max(rr.min() - mz, 0); r1 = min(rr.max() + mz, m.shape[0])
        c0 = max(cc.min() - my, 0); c1 = min(cc.max() + my, m.shape[1])

        sub = sl[r0:r1, c0:c1]
        p2, p98 = np.percentile(sub, (2, 98))
        sub = np.clip((sub - p2) / (p98 - p2 + 1e-8), 0, 1)
        ax.imshow(sub, cmap='gray', vmin=0, vmax=1, aspect=vz / vy)

        if m[r0:r1, c0:c1].max() > 0:
            ax.contour(m[r0:r1, c0:c1], levels=[0.5], colors=[MANUAL_COLOR], linewidths=1.3)
        if p[r0:r1, c0:c1].max() > 0:
            ax.contour(p[r0:r1, c0:c1], levels=[0.5], colors=[PRED_COLOR],
                       linewidths=1.3, linestyles='dashed')
        ax.axis('off')
        ax.set_title(f'{quality}\nDice = {dice:.2f}', fontsize=10, fontweight='bold')

    legend_elements = [
        Line2D([0], [0], color=MANUAL_COLOR, lw=1.5, label='Manual'),
        Line2D([0], [0], color=PRED_COLOR, lw=1.5, ls='--', label='Predicted'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2,
               fontsize=10, frameon=True, bbox_to_anchor=(0.5, 0.0))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(OUT.with_suffix('.png'), dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(OUT.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved {OUT.with_suffix('.png').name} / .pdf")
    for row, q in cases:
        print(f"  {q}: {row['case_id']} (Dice={row['dice']:.3f})")


if __name__ == "__main__":
    main()
