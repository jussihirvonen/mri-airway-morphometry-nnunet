#!/usr/bin/env python3
"""
Bland-Altman figure for the airway segmentation paper (Paper 2),
replicating the abscess segmentation paper figure style
(Data/scripts/22_segmentation_analysis.py :: bland_altman_plot).

Input : 09_seg_agreement_percase.csv  (manual_*/pred_* per case, 322)
Output: 10b_seg_bland_altman.png / .pdf

Each panel: scatter (mean vs Δ), blue mean line, red dashed LoA (±1.96 SD),
grey dotted zero line, identical to the source figure.
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path(os.environ.get("AIRWAY_ROOT", Path(__file__).resolve().parents[1])) / "final_analysis"
PERCASE = BASE / "09_seg_agreement_percase.csv"
OUT = BASE / "10b_seg_bland_altman"

# Style — identical to the abscess segmentation paper figures
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'font.size': 10,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
})

FEATURES = [
    ('volume_ml', 'Volume', 'mL'),
    ('min_area_mm2', 'Minimum CSA', 'mm²'),
    ('compactness', 'Compactness', ''),
    ('rl_dev_max_abs_mm', 'Maximum midline shift', 'mm'),
    ('len_below_50mm2_mm', 'Length <50 mm²', 'mm'),
]
PANEL_LABELS = ['a', 'b', 'c', 'd', 'e', 'f']


def bland_altman(ax, manual, pred, title, unit=''):
    """Same style as the abscess segmentation analysis Bland-Altman plot."""
    manual = np.asarray(manual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mean_vals = (manual + pred) / 2
    diff_vals = pred - manual
    mean_diff = float(np.mean(diff_vals))
    std_diff = float(np.std(diff_vals, ddof=1))
    loa_upper = mean_diff + 1.96 * std_diff
    loa_lower = mean_diff - 1.96 * std_diff

    ax.scatter(mean_vals, diff_vals, alpha=0.6, s=25, edgecolors='k', linewidths=0.3)
    ax.axhline(mean_diff, color='blue', linestyle='-', lw=1.5,
               label=f'Mean: {mean_diff:.3f}')
    ax.axhline(loa_upper, color='red', linestyle='--', lw=1,
               label=f'+1.96 SD: {loa_upper:.3f}')
    ax.axhline(loa_lower, color='red', linestyle='--', lw=1,
               label=f'-1.96 SD: {loa_lower:.3f}')
    ax.axhline(0, color='grey', linestyle=':', lw=0.5)
    unit_str = f' ({unit})' if unit else ''
    ax.set_xlabel(f'Mean of manual and predicted{unit_str}', fontsize=9)
    ax.set_ylabel(f'Predicted − Manual{unit_str}', fontsize=9)
    ax.set_title(title, fontsize=10, fontweight='bold', loc='left')
    ax.legend(fontsize=7, loc='best')
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=8)


def main():
    df = pd.read_csv(PERCASE)
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    flat = axes.flatten()
    for ax, (feat, label, unit), panel in zip(flat, FEATURES, PANEL_LABELS):
        col_m = f'manual_{feat}'
        col_p = f'pred_{feat}'
        valid = df.dropna(subset=[col_m, col_p])
        bland_altman(ax, valid[col_m].values, valid[col_p].values,
                     f'({panel}) {label}', unit)
    for ax in flat[len(FEATURES):]:
        ax.axis('off')
    plt.tight_layout()
    fig.savefig(OUT.with_suffix('.png'), dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(OUT.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved {OUT.with_suffix('.png').name} / .pdf  (n={len(df)})")


if __name__ == "__main__":
    main()
