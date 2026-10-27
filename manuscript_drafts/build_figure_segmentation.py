#!/usr/bin/env python3
"""
Figure (segmentation illustration) for the airway morphometry paper.

Manual vs nnU-Net (out-of-fold) upper-airway segmentation on T1C, two cases:
a representative airway (close agreement at all levels) and a focal stenosis
(excellent where wide, but the network overestimates the minimum cross-section).

Edit the CONFIG block below and re-run:
    python3 build_figure_segmentation.py

Inputs:
    nnunet_setup/Dataset005_NeckAirway/imagesTr/<case>_0000.nii.gz   (T1C)
    nnunet_setup/Dataset005_NeckAirway/labelsTr/<case>.nii.gz        (manual)
    final_analysis/predicted_masks/<case>.nii.gz                     (OOF prediction)
    final_analysis/09_seg_agreement_percase.csv                      (per-case DSC)
Output:
    manuscript_drafts/tables_figures/Figure_segmentation.png / .pdf
"""
import os
import json
import string
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import ndimage as ndi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ----------------------------------------------------------------------
# CONFIG  (edit here)
# ----------------------------------------------------------------------
BASE = Path(os.environ.get("AIRWAY_ROOT", Path(__file__).resolve().parents[1]))
_CFG_FILE = BASE / "paths_local.json"  # untracked: maps generic keys to the local folder layout (README, "Local layout")
_CFG = json.loads(_CFG_FILE.read_text()) if _CFG_FILE.exists() else {}


def _example(key):
    """Representative case for a figure, supplied by the untracked paths_local.json ("examples" block)."""
    try:
        return _CFG["examples"][key]
    except KeyError:
        raise SystemExit(f"paths_local.json: examples.{key} is required by this script")
DSET = BASE / "nnunet_setup" / "Dataset005_NeckAirway"
PRED = BASE / "final_analysis" / "predicted_masks"
DICE_CSV = BASE / "final_analysis" / "09_seg_agreement_percase.csv"
OUT = BASE / "manuscript_drafts" / "tables_figures" / "Figure_segmentation"

SPACING_MM = 0.39                 # in-plane pixel size (mm); PX area = SPACING_MM**2
# (case_id, row title [\n allowed], [axial slice indices, left->right])
CASES = [
    (_example("segmentation_representative"), "Representative\nairway", [21, 27, 30]),
    (_example("segmentation_stenosis"), "Focal\nstenosis", [25, 28, 30]),
]
CROP_HALF_PX   = 85               # half crop size in px (170 px ~ 66 mm field)
RING_DILATE_IT = 28               # peri-airway ring for windowing (~11 mm)
WIN_BLACK_PCT  = 60               # black point = this pct of lumen-void intensities
WIN_WHITE_PCT  = 80               # white point = this pct of peri-airway intensities
COL_MANUAL     = "#2ee64d"        # manual reference contour
COL_PRED       = "#ff2d2d"        # nnU-Net prediction contour
LW_MANUAL, LW_PRED = 1.5, 1.1
LABEL_COLOR    = "#222222"        # all CSA labels (single colour)
FIGSIZE        = (8.2, 6.1)
LEGEND_XY      = (0.575, 0.02)    # legend anchor (figure coords); raise y to lift
MARGINS = dict(left=0.155, right=0.995, top=0.96, bottom=0.135, wspace=0.04, hspace=0.16)

# ----------------------------------------------------------------------
PX = SPACING_MM * SPACING_MM
dd = pd.read_csv(DICE_CSV).set_index("case_id")


def largest_cc(mask):
    """3D largest connected component (matches the morphometry pipeline)."""
    lab, n = ndi.label(mask)
    if n <= 1:
        return mask
    sizes = ndi.sum(mask, lab, range(1, n + 1))
    return lab == (1 + int(np.argmax(sizes)))


def load(cid):
    f = lambda p: nib.as_closest_canonical(nib.load(str(p))).get_fdata()
    img = f(DSET / "imagesTr" / f"{cid}_0000.nii.gz")
    man = largest_cc(f(DSET / "labelsTr" / f"{cid}.nii.gz") > 0.5)
    prd = largest_cc(f(PRED / f"{cid}.nii.gz") > 0.5)
    return img, man, prd


def main():
    letters = iter(string.ascii_lowercase)
    fig, axes = plt.subplots(2, 3, figsize=FIGSIZE)
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42})

    for ri, (cid, desc, zs) in enumerate(CASES):
        img, man, prd = load(cid)
        ys, xs = np.where(man.any(axis=2))
        cy, cx = int(ys.mean()), int(xs.mean())
        y0, y1 = cy - CROP_HALF_PX, cy + CROP_HALF_PX
        x0, x1 = cx - CROP_HALF_PX, cx + CROP_HALF_PX
        for ci, z in enumerate(zs):
            ax = axes[ri, ci]
            sl = img[y0:y1, x0:x1, z]
            ms = man[y0:y1, x0:x1, z]
            ps = prd[y0:y1, x0:x1, z]
            # window: black point near the lumen void, white point in soft-tissue band
            seed = ms if ms.any() else ps
            ring = ndi.binary_dilation(seed, iterations=RING_DILATE_IT)
            lum = sl[ms] if ms.any() else sl[ps]
            blk = np.percentile(lum, WIN_BLACK_PCT)
            wht = np.percentile(sl[ring], WIN_WHITE_PCT)
            ax.imshow(sl.T, cmap="gray", vmin=blk, vmax=wht, origin="lower",
                      aspect="equal", interpolation="bilinear")
            if ms.any():
                ax.contour(ms.T.astype(float), [0.5], colors=COL_MANUAL, linewidths=LW_MANUAL, origin="lower")
            if ps.any():
                ax.contour(ps.T.astype(float), [0.5], colors=COL_PRED, linewidths=LW_PRED, origin="lower")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            ma, pa = ms.sum() * PX, ps.sum() * PX
            ax.text(0.5, -0.075, f"CSA  manual {ma:.0f}  ·  predicted {pa:.0f} mm²",
                    transform=ax.transAxes, ha="center", va="top", fontsize=7.4, color=LABEL_COLOR)
            ax.text(0.035, 0.965, next(letters), transform=ax.transAxes, ha="left", va="top",
                    fontsize=10, fontweight="bold", color="w")

    fig.subplots_adjust(**MARGINS)
    # horizontal row titles in the left margin (descriptor + DSC, no case number)
    for ri, (cid, desc, zs) in enumerate(CASES):
        pos = axes[ri, 0].get_position()
        fig.text(0.058, pos.y0 + pos.height / 2, f"{desc}\nDSC {dd.loc[cid, 'dice']:.2f}",
                 ha="center", va="center", rotation=0, fontsize=9.5)

    leg = [Line2D([0], [0], color=COL_MANUAL, lw=2.2, label="Manual reference"),
           Line2D([0], [0], color=COL_PRED, lw=2.2, label="nnU-Net prediction")]
    fig.legend(handles=leg, loc="lower center", ncol=2, frameon=False, fontsize=10,
               bbox_to_anchor=LEGEND_XY)

    fig.savefig(str(OUT) + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(str(OUT) + ".pdf", bbox_inches="tight")
    print("wrote", OUT.with_suffix(".png"), "and .pdf")


if __name__ == "__main__":
    main()
