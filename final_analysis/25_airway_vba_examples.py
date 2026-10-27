#!/usr/bin/env python3
"""
VBA explanatory figures from the cache:
  25a single-patient unfolded maps: a normal and an abnormal example (paths_local.json, examples block)
  25b individual-profile "spaghetti" (CSA, lateral diam, midline shift) by ICU
  25c metric x outcome cluster-significance summary heatmap
"""
import os
import json
from pathlib import Path
import numpy as np
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(os.environ.get("AIRWAY_ROOT", Path(__file__).resolve().parents[1]))
_CFG_FILE = BASE / "paths_local.json"  # untracked: maps generic keys to the local folder layout (README, "Local layout")
_CFG = json.loads(_CFG_FILE.read_text()) if _CFG_FILE.exists() else {}


def _example(key):
    """Representative case for a figure, supplied by the untracked paths_local.json ("examples" block)."""
    try:
        return _CFG["examples"][key]
    except KeyError:
        raise SystemExit(f"paths_local.json: examples.{key} is required by this script")

NORMAL, ABNORMAL = _example("profile_normal"), _example("profile_abnormal")
CACHE = BASE / "final_analysis" / "21_airway_vba_cache.npz"
OUTDIR = BASE / "final_analysis"
NPERM, SEED = 2000, 42
METRICS = [("csa", "CSA"), ("ap", "AP diam"), ("lat", "Lateral diam"),
           ("shift", "Midline shift")]
OUTCOMES = [("icu", "ICU"), ("prolonged", "Prolonged"), ("rpe", "RPE"), ("abscess", "Abscess")]


def clusters(t, tc):
    sig = np.abs(t) > tc; out, i = [], 0
    while i < len(sig):
        if sig[i]:
            j = i
            while j < len(sig) and sig[j] and np.sign(t[j]) == np.sign(t[i]):
                j += 1
            out.append((i, j, float(np.abs(t[i:j]).sum()), int(np.sign(t[i]))))
            i = j
        else:
            i += 1
    return out


def best_cluster(M, grp):
    rng = np.random.default_rng(SEED)
    ok = ~np.isnan(grp); M, g = M[ok], grp[ok].astype(int)
    keep = ~np.isnan(M).any(0); tc = stats.t.ppf(0.975, len(g) - 2)
    def tv(l):
        t, _ = stats.ttest_ind(M[l == 1], M[l == 0], equal_var=False)
        t = np.nan_to_num(t); t[~keep] = 0; return t
    obs = clusters(tv(g), tc)
    if not obs:
        return None
    n1 = int(g.sum()); null = np.empty(NPERM)
    for k in range(NPERM):
        p = rng.permutation(len(g)); l = np.zeros(len(g), int); l[p[:n1]] = 1
        cl = clusters(tv(l), tc); null[k] = max([c[2] for c in cl], default=0.0)
    best = max(obs, key=lambda c: c[2])
    p = (1 + np.sum(null >= best[2])) / (NPERM + 1)
    return dict(s=best[0], e=best[1], sign=best[3], p=p)


def main():
    z = np.load(CACHE, allow_pickle=True)
    grid = z["grid"] * 100; nang = int(z["nang"]); ids = list(z["ids"])
    xt = [0, nang//4, nang//2, 3*nang//4]; xl = ["Ant", "Right", "Post", "Left"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})

    # 25a single-patient unfolded maps
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.4))
    vmax = np.nanpercentile(z["maps"][[ids.index(NORMAL), ids.index(ABNORMAL)]], 98)
    for ax, cid, ttl in [(axes[0], NORMAL, "Normal airway"), (axes[1], ABNORMAL, "Abnormal airway")]:
        im = ax.imshow(z["maps"][ids.index(cid)], aspect="auto", cmap="viridis", vmin=0, vmax=vmax,
                       extent=[0, nang, 100, 0])
        plt.colorbar(im, ax=ax, fraction=0.046, label="radius (mm)")
        ax.set_xticks(xt); ax.set_xticklabels(xl, fontsize=8); ax.set_ylabel("position % (palate→cords)")
        ax.set_title(f"{ttl} ({cid})", fontsize=10)
    fig.suptitle("Single-patient unfolded wall maps", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(OUTDIR / "25a_unfold_examples.png", dpi=200)
    print("wrote 25a")

    # 25b spaghetti
    icu = z["icu"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, (mk, ml) in zip(axes, [("csa", "CSA (mm²)"), ("lat", "Lateral diam (mm)"), ("shift", "Midline shift (mm)")]):
        M = z[mk]
        for sel, col in [(icu == 0, "#1f6fb2"), (icu == 1, "#d62828")]:
            for row in M[sel]:
                ax.plot(grid, row, color=col, alpha=0.06, lw=0.6)
        for sel, col, lab in [(icu == 0, "#1f6fb2", "No ICU"), (icu == 1, "#d62828", "ICU")]:
            ax.plot(grid, np.nanmean(M[sel], 0), color=col, lw=2.4, label=lab)
        ax.set_xlabel("position % (palate→cords)"); ax.set_ylabel(ml); ax.legend(frameon=False, fontsize=9)
    fig.suptitle("Individual airway profiles (thin) with group means (bold) — ICU vs non-ICU", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(OUTDIR / "25b_spaghetti.png", dpi=200)
    print("wrote 25b")

    # 25c summary heatmap (signed -log10 p of best cluster)
    H = np.zeros((len(METRICS), len(OUTCOMES))); ann = [["" for _ in OUTCOMES] for _ in METRICS]
    for mi, (mk, ml) in enumerate(METRICS):
        for oi, (ok, ol) in enumerate(OUTCOMES):
            bc = best_cluster(z[mk], z[ok])
            if bc:
                H[mi, oi] = -np.log10(bc["p"]) * bc["sign"]
                ann[mi][oi] = f"{bc['sign']:+d}\np={bc['p']:.3f}\n{grid[bc['s']]:.0f}-{grid[bc['e']-1]:.0f}%"
    fig, ax = plt.subplots(figsize=(8.5, 6))
    lim = np.nanmax(np.abs(H)) or 1
    im = ax.imshow(H, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(OUTCOMES))); ax.set_xticklabels([o[1] for o in OUTCOMES])
    ax.set_yticks(range(len(METRICS))); ax.set_yticklabels([m[1] for m in METRICS])
    for mi in range(len(METRICS)):
        for oi in range(len(OUTCOMES)):
            ax.text(oi, mi, ann[mi][oi], ha="center", va="center", fontsize=7.5)
    plt.colorbar(im, ax=ax, label="signed −log₁₀(cluster p)  (sign = group+ higher/lower)")
    ax.set_title("Airway VBA summary — best cluster per metric × outcome\n(red = group+ higher, blue = group+ lower)", fontsize=10)
    fig.tight_layout(); fig.savefig(OUTDIR / "25c_summary_heatmap.png", dpi=200)
    print("wrote 25c")


if __name__ == "__main__":
    main()
