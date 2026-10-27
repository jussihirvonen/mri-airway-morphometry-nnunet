#!/usr/bin/env python3
"""
Cohort-comparison version of the severity-axes panel.

Runs the same 5-feature × 7-axis association panel under three cohort
restrictions, so the impact of including/excluding abs-negative DNI
patients can be inspected directly.

  Full cohort  (n=323)   primary candidate: broader generalizability
  abs+ only    (n=223)   matches established Turku ICU model scope
  abs- only    (n=100)   stratified sensitivity; sparse for ICU & Mediast

Axes that are undefined for a cohort are auto-skipped:
  abs- only: abscess max diameter (all zero by definition)

Outputs:
  08_severity_compare_combined.csv  long-form table (feature × axis × cohort)
  08_severity_compare_heatmap.png/pdf  3-panel heatmap, one per cohort
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
    # RPE volume dropped 2026-06-01 — redundant with binary rpe_present;
    # keep abs_diam (Turku model component, continuous dose-response on abscess size).
    ("abs_diamax_mri", "Abscess diam"),
]

COHORTS = [
    ("all", "Full cohort", None),
    ("absp", "abs+", lambda d: d[d.abs_positive == 1]),
    ("absn", "abs−", lambda d: d[d.abs_positive == 0]),
]


def rb_from_u(u, n1, n2):
    return 2 * u / (n1 * n2) - 1


def effect_binary(x_pos, x_neg):
    if len(x_pos) < 3 or len(x_neg) < 3:
        return np.nan, np.nan
    u, p = mannwhitneyu(x_pos, x_neg, alternative='two-sided')
    return rb_from_u(u, len(x_pos), len(x_neg)), p


def effect_continuous(x, y):
    if len(x) < 5:
        return np.nan, np.nan
    return spearmanr(x, y)


def p_marker(p):
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def build_panel(df, drop_axes=()):
    """Return eff and p matrices: rows=AIRWAY, cols=BINARY+CONTINUOUS."""
    n_cols = len(BINARY) + len(CONTINUOUS)
    eff = np.full((len(AIRWAY), n_cols), np.nan)
    pmat = np.full((len(AIRWAY), n_cols), np.nan)
    counts = np.zeros((len(AIRWAY), n_cols), dtype=int)

    for i, (ac, _) in enumerate(AIRWAY):
        for j, (sc, slabel) in enumerate(BINARY):
            if slabel in drop_axes:
                continue
            sub = df[[ac, sc]].dropna()
            xp = sub[sub[sc] == 1][ac].values
            xn = sub[sub[sc] == 0][ac].values
            e, p = effect_binary(xp, xn)
            eff[i, j] = e
            pmat[i, j] = p
            counts[i, j] = len(sub)
        for j, (sc, slabel) in enumerate(CONTINUOUS):
            jj = len(BINARY) + j
            if slabel in drop_axes:
                continue
            sub = df[[ac, sc]].dropna()
            e, p = effect_continuous(sub[ac].values, sub[sc].values)
            eff[i, jj] = e
            pmat[i, jj] = p
            counts[i, jj] = len(sub)
    return eff, pmat, counts


def long_form(eff, pmat, counts, cohort_label):
    cols = [s[1] for s in BINARY] + [s[1] for s in CONTINUOUS]
    rows = []
    for i, (_, alabel) in enumerate(AIRWAY):
        for j, clabel in enumerate(cols):
            rows.append({
                "Cohort": cohort_label,
                "Airway feature": alabel,
                "Severity axis": clabel,
                "Effect": eff[i, j] if not pd.isna(eff[i, j]) else "",
                "p": pmat[i, j] if not pd.isna(pmat[i, j]) else "",
                "n": int(counts[i, j]) if counts[i, j] else "",
            })
    return rows


def draw_heatmap_grid(panels, outpath):
    n_p = len(panels)
    fig, axes = plt.subplots(1, n_p, figsize=(7.2 * n_p, 4.0), sharey=True)
    if n_p == 1:
        axes = [axes]

    col_labels = [s[1] for s in BINARY] + [s[1] for s in CONTINUOUS]
    row_labels = [a[1] for a in AIRWAY]

    # Use a shared vmax across panels so colors are comparable
    all_eff = np.concatenate([np.abs(p["eff"][~np.isnan(p["eff"])]) for p in panels])
    vmax = max(0.45, np.nanmax(all_eff))
    cmap = plt.cm.RdBu_r

    last_im = None
    for ax, panel in zip(axes, panels):
        eff = panel["eff"]; pmat = panel["pmat"]
        im = ax.imshow(eff, cmap=cmap, vmin=-vmax, vmax=vmax, aspect='auto')
        last_im = im
        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=35, ha='right')
        if ax is axes[0]:
            ax.set_yticks(np.arange(len(row_labels)))
            ax.set_yticklabels(row_labels)
            ax.set_ylabel("Airway feature")
        else:
            ax.set_yticks(np.arange(len(row_labels)))
            ax.set_yticklabels([])

        # binary/continuous separator
        ax.axvline(len(BINARY) - 0.5, color='black', linewidth=1.2)

        for i in range(eff.shape[0]):
            for j in range(eff.shape[1]):
                v = eff[i, j]; p = pmat[i, j]
                if np.isnan(v):
                    ax.text(j, i, "—", ha='center', va='center',
                            color='black', fontsize=10)
                    continue
                txt = f"{v:+.2f}{p_marker(p)}"
                color = "white" if abs(v) > 0.25 else "black"
                ax.text(j, i, txt, ha='center', va='center',
                        color=color, fontsize=8.5)

        ax.set_xlabel(f"{panel['label']} (n={panel['n']})")

    cbar = fig.colorbar(last_im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("Effect size", rotation=90, va="bottom")
    fig.savefig(outpath.with_suffix(".png"), dpi=300, bbox_inches='tight')
    fig.savefig(outpath.with_suffix(".pdf"), bbox_inches='tight')
    plt.close(fig)
    print(f"Wrote {outpath.with_suffix('.png').name} / .pdf")


def main():
    mpl.rcParams['font.family'] = 'DejaVu Sans'
    mpl.rcParams['pdf.fonttype'] = 42
    df_full = pd.read_csv(MERGED)
    # RPE presence: use the complete clinical flag (retroinf, 0/1 for all 322),
    # not the volumetric derivation rpe_present (NA for 23 cases without a
    # volumetric RPE segmentation). The two agree perfectly where both exist.
    assert df_full["retroinf"].notna().all(), "retroinf must be complete"
    df_full["rpe_present"] = df_full["retroinf"]
    print(f"Full cohort: n={len(df_full)}")

    panels = []
    rows_all = []
    for key, label, filt in COHORTS:
        d = filt(df_full) if filt else df_full
        drop = ()
        if key == "absn":
            drop = ("Abscess diam",)  # all zero by definition
        eff, pmat, counts = build_panel(d, drop_axes=drop)
        panels.append({"key": key, "label": label, "n": len(d),
                       "eff": eff, "pmat": pmat, "counts": counts})
        rows_all.extend(long_form(eff, pmat, counts, label))
        print(f"  {label:12s} n={len(d)}")

    pd.DataFrame(rows_all).to_csv(BASE / "08_severity_compare_combined.csv", index=False)
    print(f"Wrote 08_severity_compare_combined.csv")
    draw_heatmap_grid(panels, BASE / "08_severity_compare_heatmap")

    # Compact concordance summary across cohorts
    print("\nConcordance (sig p<0.05 AND in expected severity direction):")
    expected = {"Volume (mL)": -1, "Min CSA (mm²)": -1, "Compactness": -1,
                "Max midline shift": +1, "Length <50 mm²": +1}
    for panel in panels:
        eff = panel["eff"]; pmat = panel["pmat"]
        print(f"\n  {panel['label']:12s} (n={panel['n']})")
        n_axes_used = (~np.isnan(eff[0])).sum()
        for i, (_, alabel) in enumerate(AIRWAY):
            sig_concord = 0
            for j in range(eff.shape[1]):
                if np.isnan(eff[i, j]) or np.isnan(pmat[i, j]):
                    continue
                if pmat[i, j] < 0.05 and np.sign(eff[i, j]) == expected[alabel]:
                    sig_concord += 1
            print(f"    {alabel:22s} {sig_concord}/{n_axes_used} axes")


if __name__ == "__main__":
    main()
