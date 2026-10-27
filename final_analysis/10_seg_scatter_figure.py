#!/usr/bin/env python3
"""
Manual-vs-predicted scatter figure for the airway segmentation paper (Paper 2),
replicating the style of the abscess segmentation paper figure
(28_segmentation_paper_figures.py :: fig5, top "morphometric correlation" row).
The rim-to-center (RTC ratio) row is omitted (not applicable to the airway).

Input : 09_seg_agreement_percase.csv  (manual_*/pred_* per case, 322)
Output: 10_seg_scatter_manual_vs_predicted.png / .pdf

Each panel: identity line, scatter, and a wheat box with ICC(2,1) + 95% CI,
Pearson r, and Spearman rho — identical to the source figure.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import f as fdist

BASE = Path(os.environ.get("AIRWAY_ROOT", Path(__file__).resolve().parents[1])) / "final_analysis"
PERCASE = BASE / "09_seg_agreement_percase.csv"
OUT = BASE / "10_seg_scatter_manual_vs_predicted"

# ── Style (identical to the abscess segmentation paper figures) ──────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'font.size': 10,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'figure.dpi': 300,
})

# Pre-specified airway features (manual_/pred_ column stems), with display label + unit
FEATURES = [
    ('volume_ml', 'Volume', 'mL'),
    ('min_area_mm2', 'Minimum CSA', 'mm²'),
    ('compactness', 'Compactness', ''),
    ('rl_dev_max_abs_mm', 'Maximum midline shift', 'mm'),
    ('len_below_50mm2_mm', 'Length <50 mm²', 'mm'),
]
PANEL_LABELS = ['a', 'b', 'c', 'd', 'e', 'f']


def icc_2_1(rater1, rater2):
    """ICC(2,1) absolute agreement via pingouin, matching 09_seg_agreement.py
    (previous hand-rolled F-based CI gave slightly different bounds)."""
    import pandas as pd
    import pingouin as pg
    n = len(rater1)
    d = pd.DataFrame({
        "targets": list(range(n)) * 2,
        "raters": ["a"] * n + ["b"] * n,
        "score": list(rater1) + list(rater2),
    })
    row = pg.intraclass_corr(d, targets="targets", raters="raters",
                             ratings="score").set_index("Type").loc["ICC2"]
    lo, hi = row["CI95%"]
    return float(row["ICC"]), float(lo), float(hi)


def main():
    df = pd.read_csv(PERCASE)

    fig, axes = plt.subplots(2, 3, figsize=(10, 6.5))
    flat = axes.flatten()

    for ax, (feat, label, unit), panel in zip(flat, FEATURES, PANEL_LABELS):
        col_m = f'manual_{feat}'
        col_p = f'pred_{feat}'
        valid = df.dropna(subset=[col_m, col_p])
        x = valid[col_m].values
        y = valid[col_p].values

        ax.scatter(x, y, alpha=0.6, s=25, edgecolors='k', linewidths=0.3)

        # Identity line
        lims = [min(x.min(), y.min()), max(x.max(), y.max())]
        margin = (lims[1] - lims[0]) * 0.05
        lims = [lims[0] - margin, lims[1] + margin]
        ax.plot(lims, lims, 'k--', lw=1, alpha=0.5)

        # Statistics
        r_p, _ = stats.pearsonr(x, y)
        r_s, _ = stats.spearmanr(x, y)
        icc_val, ci_lo, ci_hi = icc_2_1(x, y)

        textstr = (f'ICC={icc_val:.3f} ({ci_lo:.3f}–{ci_hi:.3f})\n'
                   f'r={r_p:.3f}\n'
                   f'ρ={r_s:.3f}')
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=8,
                va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        unit_str = f' ({unit})' if unit else ''
        ax.set_xlabel(f'Manual{unit_str}', fontsize=9)
        ax.set_ylabel(f'Predicted{unit_str}', fontsize=9)
        ax.set_title(f'({panel}) {label}', fontsize=10, fontweight='bold', loc='left')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.3)

    # Hide the unused 6th panel (5 features)
    for ax in flat[len(FEATURES):]:
        ax.axis('off')

    plt.tight_layout()
    fig.savefig(OUT.with_suffix('.png'), dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(OUT.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved {OUT.with_suffix('.png').name} / .pdf")


if __name__ == "__main__":
    main()
