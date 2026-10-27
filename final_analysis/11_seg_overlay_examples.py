#!/usr/bin/env python3
"""
Qualitative segmentation overlays for the airway paper (Paper 2),
replicating the abscess segmentation paper figure
(28_segmentation_paper_figures.py :: fig4_qualitative_examples).

Rows    = Good / Intermediate / Poor Dice example cases.
Columns = three axial levels through the airway (superior / middle / inferior),
          replacing the abscess paper's T1C/T2FS/ADC columns since airway
          segmentation uses T1C only.
Overlay = manual (red, solid) and predicted (blue, dashed) contours on T1C.

Inputs:
  T1C       : nnunet_setup/Dataset005_NeckAirway/imagesTr/caseXXX_0000.nii.gz
  manual    : nnunet_setup/Dataset005_NeckAirway/labelsTr/caseXXX.nii.gz
  predicted : final_analysis/predicted_masks/caseXXX.nii.gz
  per-case  : final_analysis/09_seg_agreement_percase.csv (Dice for selection)

Output: 11_seg_overlay_examples.png / .pdf
"""

import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

AIR = Path(os.environ.get("AIRWAY_ROOT", Path(__file__).resolve().parents[1]))
_CFG_FILE = AIR / "paths_local.json"  # untracked: maps generic keys to the local folder layout (README, "Local layout")
_CFG = json.loads(_CFG_FILE.read_text()) if _CFG_FILE.exists() else {}
IMG = AIR / "nnunet_setup" / "Dataset005_NeckAirway" / "imagesTr"
LAB = AIR / "nnunet_setup" / "Dataset005_NeckAirway" / "labelsTr"
PRED = AIR / "final_analysis" / "predicted_masks"
PERCASE = AIR / "final_analysis" / "09_seg_agreement_percase.csv"
OUT = AIR / "final_analysis" / "11_seg_overlay_examples"

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'font.size': 10,
    'figure.dpi': 300,
})

MANUAL_COLOR = '#e74c3c'
PRED_COLOR = '#3498db'

# Optional manual override of selected cases; leave None to auto-select.
OVERRIDE = _CFG.get("examples", {}).get("overlay")  # {'Good': case, 'Intermediate': case, 'Poor': case}; None = auto-select


def select_cases(df):
    if OVERRIDE:
        out = []
        for q in ['Good', 'Intermediate', 'Poor']:
            row = df[df.case_id == OVERRIDE[q]].iloc[0]
            out.append((row, q))
        return out
    med_vol = df.manual_volume_ml.median()
    good = df[df.manual_volume_ml > med_vol].nlargest(1, 'dice').iloc[0]
    med_d = df.dice.median()
    inter = df.iloc[(df.dice - med_d).abs().argsort().iloc[0]]
    poor = df[df.manual_volume_ml > 5].nsmallest(1, 'dice').iloc[0]
    return [(good, 'Good'), (inter, 'Intermediate'), (poor, 'Poor')]


def superior_first(zs, affine):
    """Order voxel-z indices so the most superior (largest world z) comes first."""
    world_z = [affine[2, 2] * z + affine[2, 3] for z in zs]
    order = np.argsort(world_z)[::-1]  # descending world-z = superior first
    return [zs[i] for i in order]


def crop_box(mask_rot_mip, margin=55):
    rows = np.where(mask_rot_mip.any(axis=1))[0]
    cols = np.where(mask_rot_mip.any(axis=0))[0]
    r0, r1 = rows.min(), rows.max()
    c0, c1 = cols.min(), cols.max()
    cr, cc = (r0 + r1) // 2, (c0 + c1) // 2
    half = int(max(r1 - r0, c1 - c0) / 2 + margin)
    H, W = mask_rot_mip.shape
    r0 = max(cr - half, 0); r1 = min(cr + half, H)
    c0 = max(cc - half, 0); c1 = min(cc + half, W)
    return r0, r1, c0, c1


def rot(a):
    return np.rot90(a, k=-1)


def main():
    df = pd.read_csv(PERCASE)
    cases = select_cases(df)

    fig, axes = plt.subplots(3, 3, figsize=(7.8, 8.2))

    for row_idx, (case_row, quality) in enumerate(cases):
        cid = case_row['case_id']
        dice = case_row['dice']

        t1c = nib.load(str(IMG / f"{cid}_0000.nii.gz"))
        aff = t1c.affine
        t1c = t1c.get_fdata()
        manual = nib.load(str(LAB / f"{cid}.nii.gz")).get_fdata() > 0.5
        pred = nib.load(str(PRED / f"{cid}.nii.gz")).get_fdata() > 0.5

        # airway-bearing z slices, pick 20/50/80th percentile, order superior-first
        zs_air = [z for z in range(manual.shape[2]) if manual[:, :, z].sum() > 0]
        picks = [zs_air[int(p * (len(zs_air) - 1))] for p in (0.2, 0.5, 0.8)]
        picks = superior_first(picks, aff)

        # consistent crop from rotated manual MIP over the airway extent
        man_mip = rot(manual[:, :, zs_air].max(axis=2).astype(float)) > 0.5
        r0, r1, c0, c1 = crop_box(man_mip)

        col_titles = ['Superior', 'Middle', 'Inferior']
        for col_idx, z in enumerate(picks):
            ax = axes[row_idx, col_idx]
            sl = rot(t1c[:, :, z]).astype(float)
            p2, p98 = np.percentile(sl, (2, 98))
            sl = np.clip((sl - p2) / (p98 - p2 + 1e-8), 0, 1)
            ax.imshow(sl[r0:r1, c0:c1], cmap='gray', vmin=0, vmax=1)

            m_sl = rot(manual[:, :, z].astype(float))[r0:r1, c0:c1]
            if m_sl.max() > 0:
                ax.contour(m_sl, levels=[0.5], colors=[MANUAL_COLOR], linewidths=1.2)
            p_sl = rot(pred[:, :, z].astype(float))[r0:r1, c0:c1]
            if p_sl.max() > 0:
                ax.contour(p_sl, levels=[0.5], colors=[PRED_COLOR], linewidths=1.2,
                           linestyles='dashed')
            ax.axis('off')
            if row_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=11, fontweight='bold')
            if col_idx == 0:
                ax.text(-0.12, 0.5, f'{quality}\nDice = {dice:.2f}',
                        transform=ax.transAxes, fontsize=9, va='center', ha='right',
                        fontweight='bold')

    legend_elements = [
        Line2D([0], [0], color=MANUAL_COLOR, lw=1.5, label='Manual'),
        Line2D([0], [0], color=PRED_COLOR, lw=1.5, ls='--', label='Predicted'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2,
               fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout(rect=[0.06, 0.03, 1, 1])
    fig.savefig(OUT.with_suffix('.png'), dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(OUT.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved {OUT.with_suffix('.png').name} / .pdf")
    for case_row, q in cases:
        print(f"  {q}: {case_row['case_id']} (Dice={case_row['dice']:.3f})")


if __name__ == "__main__":
    main()
