#!/usr/bin/env python3
"""
Consolidated ICU result figure (main-figure candidate): CSA, lateral-diameter and
midline-shift cranio-caudal profiles (ICU vs non-ICU, significant clusters shaded)
plus the 2D unfolded ICU t-map with the significant wall cluster outlined.
"""
import os
from pathlib import Path
import numpy as np
from scipy import stats, ndimage as ndi
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(os.environ.get("AIRWAY_ROOT", Path(__file__).resolve().parents[1]))
CACHE = BASE / "final_analysis" / "21_airway_vba_cache.npz"
OUT = BASE / "final_analysis" / "26_airway_vba_icu_main.png"
NPERM1, NPERM2, SEED = 2000, 1000, 42


def clusters1(t, tc):
    sig = np.abs(t) > tc; out, i = [], 0
    while i < len(sig):
        if sig[i]:
            j = i
            while j < len(sig) and sig[j] and np.sign(t[j]) == np.sign(t[i]): j += 1
            out.append((i, j, float(np.abs(t[i:j]).sum()), int(np.sign(t[i]))))
            i = j
        else: i += 1
    return out


def perm1(M, g, nperm=NPERM1):
    rng = np.random.default_rng(SEED); keep = ~np.isnan(M).any(0)
    tc = stats.t.ppf(0.975, len(g) - 2)
    def tv(l):
        t, _ = stats.ttest_ind(M[l == 1], M[l == 0], equal_var=False)
        t = np.nan_to_num(t); t[~keep] = 0; return t
    obs = clusters1(tv(g), tc); n1 = int(g.sum()); null = np.empty(nperm)
    for k in range(nperm):
        p = rng.permutation(len(g)); l = np.zeros(len(g), int); l[p[:n1]] = 1
        null[k] = max([c[2] for c in clusters1(tv(l), tc)], default=0.0)
    return [dict(s=i, e=j, sign=sg, p=(1 + np.sum(null >= m)) / (nperm + 1)) for (i, j, m, sg) in obs]


def wrap_label(B):
    lab, n = ndi.label(B)
    if n == 0: return lab, 0
    par = list(range(n + 1))
    def f(a):
        while par[a] != a: par[a] = par[par[a]]; a = par[a]
        return a
    for i in range(B.shape[0]):
        a, b = lab[i, 0], lab[i, -1]
        if a and b:
            ra, rb = f(a), f(b); par[max(ra, rb)] = min(ra, rb)
    ids = {}; out = np.zeros_like(lab)
    for x in range(1, n + 1):
        r = f(x); ids.setdefault(r, len(ids) + 1)
    for x in range(1, n + 1): out[lab == x] = ids[f(x)]
    return out, len(ids)


def perm2(maps, g, nperm=NPERM2):
    rng = np.random.default_rng(SEED); N = len(g); n1 = int(g.sum())
    tc = stats.t.ppf(0.975, N - 2)
    def tm(l):
        t, _ = stats.ttest_ind(maps[l == 1], maps[l == 0], axis=0, equal_var=False)
        return np.nan_to_num(t)
    def cl(t):
        o = []
        for s in (+1, -1):
            lab, n = wrap_label(t * s > tc)
            for c in range(1, n + 1):
                mk = lab == c; o.append((float(np.abs(t[mk]).sum()), mk))
        return o
    tobs = tm(g); obs = cl(tobs); null = np.empty(nperm)
    for k in range(nperm):
        p = rng.permutation(N); l = np.zeros(N, int); l[p[:n1]] = 1
        null[k] = max([c[0] for c in cl(tm(l))], default=0.0)
    sig = [(mk, (1 + np.sum(null >= m)) / (nperm + 1)) for (m, mk) in obs
           if (1 + np.sum(null >= m)) / (nperm + 1) < 0.05]
    return tobs, sig


def main():
    z = np.load(CACHE, allow_pickle=True)
    grid = z["grid"] * 100; nang = int(z["nang"]); ngrid = z["maps"].shape[1]
    icu = z["icu"].astype(int)
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2))

    for ax, (mk, ml) in zip([axes[0, 0], axes[0, 1], axes[1, 0]],
                            [("csa", "Cross-sectional area (mm²)"),
                             ("lat", "Lateral diameter (mm)"),
                             ("shift", "Midline shift (mm)")]):
        M = z[mk]; res = perm1(M, icu)
        for sel, col, lab in [(icu == 1, "#d62828", "ICU"), (icu == 0, "#1f6fb2", "No ICU")]:
            mu = np.nanmean(M[sel], 0); se = np.nanstd(M[sel], 0) / np.sqrt(sel.sum())
            ax.plot(grid, mu, color=col, label=lab); ax.fill_between(grid, mu - se, mu + se, color=col, alpha=0.2)
        for c in res:
            if c["p"] < 0.05:
                ax.axvspan(grid[c["s"]], grid[c["e"] - 1], color="#999", alpha=0.28, zorder=0)
                plab = "p<0.001" if c["p"] < 0.001 else f"p={c['p']:.3f}"
                ax.text((grid[c["s"]] + grid[c["e"] - 1]) / 2, ax.get_ylim()[1],
                        plab, ha="center", va="top", fontsize=8)
        ax.set_xlabel("position % (palate→cords)"); ax.set_ylabel(ml); ax.legend(frameon=False, fontsize=9)
    axes[0, 0].set_title("Airway caliber", fontsize=10)
    axes[0, 1].set_title("Lateral diameter (Bonferroni-significant)", fontsize=10)
    axes[1, 0].set_title("Midline shift", fontsize=10)

    axm = axes[1, 1]
    tobs, sig = perm2(z["maps"], icu)
    tmax = np.nanpercentile(np.abs(tobs), 99)
    im = axm.imshow(tobs, aspect="auto", cmap="RdBu_r", vmin=-tmax, vmax=tmax, extent=[0, nang, 100, 0])
    plt.colorbar(im, ax=axm, fraction=0.046, label="t (ICU − non-ICU)")
    ypos = np.linspace(0, 100, ngrid)          # row 0 = palate (top), matches imshow extent
    for mk, pval in sig:
        axm.contour(np.arange(nang) + 0.5, ypos, mk.astype(float),
                    levels=[0.5], colors="k", linewidths=1.4)
        ys, xs = np.where(mk)
        axm.text(xs.mean(), ypos[int(ys.mean())], f"p={pval:.3f}", ha="center", va="center",
                 fontsize=8, fontweight="bold", color="k",
                 bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))
    axm.set_xticks([0, nang//4, nang//2, 3*nang//4]); axm.set_xticklabels(["Ant", "Right", "Post", "Left"], fontsize=8)
    axm.set_ylabel("position % (palate→cords)")
    axm.set_title("2D unfolded wall map: ICU narrower (blue), cluster outlined", fontsize=10)

    fig.tight_layout(); fig.savefig(OUT, dpi=200); print("wrote", OUT)


if __name__ == "__main__":
    main()
