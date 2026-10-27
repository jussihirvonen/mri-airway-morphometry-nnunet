#!/usr/bin/env python3
"""
Severity-axes figures: heatmap + forest plots.

Inputs: airway_morphology_merged.csv  (abs+ subset, n=223)

Outputs:
  07_heatmap_severity.png/pdf      5 features × 7 severity axes (effect-size heatmap)
  07_forest_severity.png/pdf       5-panel forest, one per airway feature

Effect sizes:
  Binary outcomes     -> rank-biserial r (Mann-Whitney U)
  Continuous outcomes -> Spearman rho

95 % CIs: case-resample bootstrap, 1000 replicates.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from scipy.stats import mannwhitneyu, spearmanr

BASE = Path(__file__).parent
MERGED = BASE / "airway_morphology_merged.csv"

AIRWAY = [
    ("volume_ml", "Volume (mL)"),
    ("min_area_mm2", "Min CSA (mm²)"),
    ("compactness", "Compactness"),
    ("rl_dev_max_abs_mm", "Max midline shift"),
    ("len_below_50mm2_mm", "Length <50 mm²"),
]

BINARY = [
    ("icu_binary", "ICU"),
    ("hospd_prolonged", "Prolonged LOS"),
    ("rpe_present", "RPE+"),
    ("medoed_binary", "ME+"),
]

CONTINUOUS = [
    ("crp", "CRP"),
    # RPE volume dropped 2026-06-01 — redundant with binary rpe_present.
    ("abs_diamax_mri", "Abscess diam"),
]

N_BOOT = 1000
SEED = 42


def rb_from_u(u, n1, n2):
    return 2 * u / (n1 * n2) - 1


def effect_binary(x_pos, x_neg):
    if len(x_pos) == 0 or len(x_neg) == 0:
        return np.nan, np.nan
    u, p = mannwhitneyu(x_pos, x_neg, alternative='two-sided')
    return rb_from_u(u, len(x_pos), len(x_neg)), p


def effect_continuous(x, y):
    if len(x) < 5:
        return np.nan, np.nan
    return spearmanr(x, y)


def boot_ci_binary(x_pos, x_neg, rng):
    if len(x_pos) < 3 or len(x_neg) < 3:
        return (np.nan, np.nan)
    rs = []
    for _ in range(N_BOOT):
        bp = rng.choice(x_pos, size=len(x_pos), replace=True)
        bn = rng.choice(x_neg, size=len(x_neg), replace=True)
        u, _ = mannwhitneyu(bp, bn, alternative='two-sided')
        rs.append(rb_from_u(u, len(bp), len(bn)))
    return tuple(np.percentile(rs, [2.5, 97.5]))


def boot_ci_continuous(x, y, rng):
    n = len(x)
    if n < 5:
        return (np.nan, np.nan)
    idx = np.arange(n)
    rs = []
    for _ in range(N_BOOT):
        bi = rng.choice(idx, size=n, replace=True)
        if len(np.unique(bi)) < 3:
            continue
        rho, _ = spearmanr(x[bi], y[bi])
        if not np.isnan(rho):
            rs.append(rho)
    if not rs:
        return (np.nan, np.nan)
    return tuple(np.percentile(rs, [2.5, 97.5]))


def collect(df):
    rng = np.random.default_rng(SEED)
    eff = np.zeros((len(AIRWAY), len(BINARY) + len(CONTINUOUS)))
    pmat = np.zeros_like(eff)
    lo = np.zeros_like(eff)
    hi = np.zeros_like(eff)
    eff[:] = np.nan; pmat[:] = np.nan; lo[:] = np.nan; hi[:] = np.nan

    for i, (ac, _) in enumerate(AIRWAY):
        for j, (sc, _) in enumerate(BINARY):
            sub = df[[ac, sc]].dropna()
            xp = sub[sub[sc] == 1][ac].values
            xn = sub[sub[sc] == 0][ac].values
            e, p = effect_binary(xp, xn)
            eff[i, j] = e
            pmat[i, j] = p
            lo[i, j], hi[i, j] = boot_ci_binary(xp, xn, rng)
        for j, (sc, _) in enumerate(CONTINUOUS):
            jj = len(BINARY) + j
            sub = df[[ac, sc]].dropna()
            x = sub[ac].values
            y = sub[sc].values
            e, p = effect_continuous(x, y)
            eff[i, jj] = e
            pmat[i, jj] = p
            lo[i, jj], hi[i, jj] = boot_ci_continuous(x, y, rng)
    return eff, pmat, lo, hi


def p_marker(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def draw_heatmap(eff, pmat, outpath):
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    vmax = max(0.45, np.nanmax(np.abs(eff)))
    cmap = plt.cm.RdBu_r
    im = ax.imshow(eff, cmap=cmap, vmin=-vmax, vmax=vmax, aspect='auto')

    col_labels = [s[1] for s in BINARY] + [s[1] for s in CONTINUOUS]
    row_labels = [a[1] for a in AIRWAY]

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=35, ha='right')
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)

    # group separator between binary and continuous severity axes
    sep_x = len(BINARY) - 0.5
    ax.axvline(sep_x, color='black', linewidth=1.2)
    ax.text(len(BINARY) / 2 - 0.5, -1.0, "Binary (rank-biserial r)",
            ha='center', va='bottom', fontsize=9)
    ax.text(len(BINARY) + len(CONTINUOUS) / 2 - 0.5, -1.0,
            "Continuous (Spearman ρ)", ha='center', va='bottom', fontsize=9)

    for i in range(eff.shape[0]):
        for j in range(eff.shape[1]):
            v = eff[i, j]; p = pmat[i, j]
            if np.isnan(v):
                continue
            txt = f"{v:+.2f}{p_marker(p)}"
            color = "white" if abs(v) > 0.25 else "black"
            ax.text(j, i, txt, ha='center', va='center',
                    color=color, fontsize=9)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Effect size", rotation=90, va="bottom")
    ax.set_xlabel("")
    ax.set_ylabel("Airway feature")
    fig.tight_layout()
    fig.savefig(outpath.with_suffix(".png"), dpi=300, bbox_inches='tight')
    fig.savefig(outpath.with_suffix(".pdf"), bbox_inches='tight')
    plt.close(fig)
    print(f"Wrote {outpath.with_suffix('.png').name} / .pdf")


def draw_forest(eff, pmat, lo, hi, outpath):
    fig, axes = plt.subplots(1, len(AIRWAY), figsize=(15, 4.2), sharex=True)
    col_labels = [s[1] for s in BINARY] + [s[1] for s in CONTINUOUS]
    n_axes = len(col_labels)
    y_pos = np.arange(n_axes)[::-1]

    for k, (ax, (_, alabel)) in enumerate(zip(axes, AIRWAY)):
        e_row = eff[k]; p_row = pmat[k]
        lo_row = lo[k]; hi_row = hi[k]

        ax.axvline(0, color='gray', lw=0.8, ls='--')
        for yi, idx in enumerate(range(n_axes)):
            v = e_row[idx]; p = p_row[idx]
            if np.isnan(v):
                continue
            yc = y_pos[idx]
            is_bin = idx < len(BINARY)
            color = "#1f6feb" if is_bin else "#cc4c02"
            sig = (p < 0.05) if not np.isnan(p) else False
            mfc = color if sig else "white"
            xerr_lo = v - lo_row[idx] if not np.isnan(lo_row[idx]) else 0
            xerr_hi = hi_row[idx] - v if not np.isnan(hi_row[idx]) else 0
            ax.errorbar(v, yc, xerr=[[xerr_lo], [xerr_hi]],
                        fmt='o', color=color, markerfacecolor=mfc,
                        markeredgewidth=1.2, capsize=3, lw=1.2, markersize=7)

        # group separator between binary and continuous outcomes
        ax.axhline(y_pos[len(BINARY)] + 0.5, color='black', lw=0.6, alpha=0.4)

        ax.set_yticks(y_pos)
        if k == 0:
            ax.set_yticklabels(col_labels, fontsize=9)
        else:
            ax.set_yticklabels([])
        ax.set_xlim(-0.7, 0.7)
        ax.set_xticks([-0.5, -0.25, 0, 0.25, 0.5])
        ax.set_xlabel(alabel, fontsize=10)
        ax.grid(axis='x', alpha=0.2)
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)

    fig.text(0.5, -0.02, "Effect size  (rank-biserial r — binary;  Spearman ρ — continuous)",
             ha='center', fontsize=10)
    fig.tight_layout()
    fig.savefig(outpath.with_suffix(".png"), dpi=300, bbox_inches='tight')
    fig.savefig(outpath.with_suffix(".pdf"), bbox_inches='tight')
    plt.close(fig)
    print(f"Wrote {outpath.with_suffix('.png').name} / .pdf")


def main():
    mpl.rcParams['font.family'] = 'DejaVu Sans'
    mpl.rcParams['pdf.fonttype'] = 42
    df = pd.read_csv(MERGED)
    df = df[df.abs_positive == 1].copy()
    print(f"abs+ cohort: n={len(df)}")

    eff, pmat, lo, hi = collect(df)
    draw_heatmap(eff, pmat, BASE / "07_heatmap_severity")
    draw_forest(eff, pmat, lo, hi, BASE / "07_forest_severity")


if __name__ == "__main__":
    main()
