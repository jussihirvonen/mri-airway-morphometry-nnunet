#!/usr/bin/env python3
"""
Airway VBA — 1D SPM grid: per-slice metric x outcome, patient-level cluster
permutation, Bonferroni across the grid, plus abscess-positive ICU sensitivity.
Reads 21_airway_vba_cache.npz.  Outputs 22_airway_vba_1d_grid.png + stdout table.
"""
import os
from pathlib import Path
import numpy as np
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(os.environ.get("AIRWAY_ROOT", Path(__file__).resolve().parents[1]))
CACHE = BASE / "final_analysis" / "21_airway_vba_cache.npz"
FIG = BASE / "final_analysis" / "22_airway_vba_1d_grid.png"
NPERM, SEED = 2000, 42
METRICS = [("csa", "CSA"), ("ap", "AP diam"), ("lat", "Lateral diam"),
           ("shift", "Midline shift")]
OUTCOMES = [("icu", "ICU"), ("prolonged", "Prolonged stay"),
            ("rpe", "RPE"), ("abscess", "Abscess")]


def clusters(tvec, tcrit):
    sig = np.abs(tvec) > tcrit
    out, i = [], 0
    while i < len(sig):
        if sig[i]:
            j = i
            while j < len(sig) and sig[j] and np.sign(tvec[j]) == np.sign(tvec[i]):
                j += 1
            out.append((i, j, float(np.abs(tvec[i:j]).sum()), int(np.sign(tvec[i]))))
            i = j
        else:
            i += 1
    return out


def cluster_perm(M, grp, nperm=NPERM, seed=SEED):
    rng = np.random.default_rng(seed)
    ok = ~np.isnan(grp)
    M, grp = M[ok], grp[ok].astype(int)
    keep = ~np.isnan(M).any(0)            # positions defined for all
    tcrit = stats.t.ppf(0.975, len(grp) - 2)
    def tvec(lbl):
        a, b = M[lbl == 1], M[lbl == 0]
        t, _ = stats.ttest_ind(a, b, equal_var=False)
        t = np.nan_to_num(t); t[~keep] = 0
        return t
    tobs = tvec(grp)
    obs = clusters(tobs, tcrit)
    n1 = int(grp.sum())
    null = np.empty(nperm)
    for k in range(nperm):
        p = rng.permutation(len(grp))
        lbl = np.zeros(len(grp), int); lbl[p[:n1]] = 1
        cl = clusters(tvec(lbl), tcrit)
        null[k] = max([c[2] for c in cl], default=0.0)
    res = [dict(s=i, e=j, mass=m, sign=sg, p=(1 + np.sum(null >= m)) / (nperm + 1))
           for (i, j, m, sg) in obs]
    return tobs, res


def main():
    z = np.load(CACHE, allow_pickle=True)
    grid = z["grid"] * 100
    groups = {k: z[k] for k in ["icu", "prolonged", "rpe", "abscess", "abscess"]}
    nmap = len(METRICS) * len(OUTCOMES)
    bonf = 0.05 / nmap
    print(f"1D SPM grid: {len(METRICS)} metrics x {len(OUTCOMES)} outcomes = {nmap} maps; "
          f"Bonferroni alpha = {bonf:.4f}\n")
    print(f"{'metric':14}{'outcome':16}{'cluster pos%':18}{'sign':6}{'p_clust':10}{'Bonf':6}")
    results = {}
    fig, axes = plt.subplots(len(METRICS), len(OUTCOMES), figsize=(13.5, 13), sharex=True)
    for mi, (mk, ml) in enumerate(METRICS):
        M = z[mk]
        for oi, (ok, ol) in enumerate(OUTCOMES):
            grp = z[ok]
            tobs, res = cluster_perm(M, grp)
            results[(mk, ok)] = res
            ax = axes[mi, oi]
            valid = ~np.isnan(grp)
            g = grp[valid].astype(int); Mv = M[valid]
            for lab, sel, col in [("+", g == 1, "#d62828"), ("−", g == 0, "#1f6fb2")]:
                mu = np.nanmean(Mv[sel], 0); se = np.nanstd(Mv[sel], 0) / np.sqrt(sel.sum())
                ax.plot(grid, mu, color=col, lw=1); ax.fill_between(grid, mu - se, mu + se, color=col, alpha=0.18)
            for c in res:
                sig_b = c["p"] < bonf
                if c["p"] < 0.05:
                    ax.axvspan(grid[c["s"]], grid[c["e"] - 1],
                               color=("#444" if sig_b else "#bbb"), alpha=0.28, zorder=0)
                seg = f"{grid[c['s']]:.0f}-{grid[c['e']-1]:.0f}"
                flag = "**" if sig_b else ("*" if c["p"] < 0.05 else "")
                print(f"{ml:14}{ol:16}{seg:18}{c['sign']:+d}    {c['p']:.4f}   {flag}")
            if not res:
                print(f"{ml:14}{ol:16}{'(none)':18}{'-':6}{'-':10}")
            if mi == 0: ax.set_title(ol, fontsize=10)
            if oi == 0: ax.set_ylabel(ml, fontsize=10)
            if mi == len(METRICS) - 1: ax.set_xlabel("position % (palate→cords)", fontsize=8)
    fig.suptitle("Airway 1D SPM grid — group+ (red) vs group− (blue); shaded = cluster p<0.05 "
                 "(dark = Bonferroni-significant), patient-level permutation", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(FIG, dpi=170)
    print("\nwrote", FIG)

    # abscess-positive subset: ICU sensitivity
    print("\n=== Abscess-positive subset: ICU sensitivity ===")
    absid = z["abscess"] == 1
    for mk, ml in METRICS:
        M = z[mk][absid]; grp = z["icu"][absid]
        _, res = cluster_perm(M, grp)
        sig = [c for c in res if c["p"] < 0.05]
        if sig:
            for c in sig:
                print(f"  {ml:14} pos {grid[c['s']]:.0f}-{grid[c['e']-1]:.0f}%  sign {c['sign']:+d}  p={c['p']:.4f}")
        else:
            print(f"  {ml:14} (no cluster p<0.05)")


if __name__ == "__main__":
    main()
