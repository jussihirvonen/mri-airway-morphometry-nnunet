#!/usr/bin/env python3
"""
Airway VBA — 2D "unfolded" wall map: radius r(cranio-caudal position, angle),
ICU and RPE vs rest, with angle-wrap-aware 2D cluster permutation (patient-level).
Shows WHICH WALL (anterior/right/posterior/left) and WHERE along the axis the
lumen is narrower. Reads 21_airway_vba_cache.npz. Output: 23_airway_vba_2d_map.png
"""
import os
from pathlib import Path
import numpy as np
from scipy import stats, ndimage as ndi
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(os.environ.get("AIRWAY_ROOT", Path(__file__).resolve().parents[1]))
CACHE = BASE / "final_analysis" / "21_airway_vba_cache.npz"
FIG = BASE / "final_analysis" / "23_airway_vba_2d_map.png"
NPERM, SEED = 1000, 42


def wrap_label(B):
    """4-connectivity labels with wrap on axis 1 (angle)."""
    lab, n = ndi.label(B)
    if n == 0:
        return lab, 0
    parent = list(range(n + 1))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[max(ra, rb)] = min(ra, rb)
    for i in range(B.shape[0]):
        a, b = lab[i, 0], lab[i, -1]
        if a and b: union(a, b)
    remap = {}
    for x in range(1, n + 1):
        remap[x] = find(x)
    out = np.zeros_like(lab)
    ids = {}
    for x in range(1, n + 1):
        r = remap[x]
        if r not in ids: ids[r] = len(ids) + 1
    for x in range(1, n + 1):
        out[lab == x] = ids[remap[x]]
    return out, len(ids)


def clusters_2d(tmap, tcrit):
    out = []
    for sgn in (+1, -1):
        B = (tmap * sgn > tcrit)
        lab, n = wrap_label(B)
        for c in range(1, n + 1):
            mask = lab == c
            out.append((float(np.abs(tmap[mask]).sum()), sgn, mask))
    return out


def cluster_perm_2d(maps, grp, nperm=NPERM, seed=SEED):
    rng = np.random.default_rng(seed)
    ok = ~np.isnan(grp)
    M, g = maps[ok], grp[ok].astype(int)
    N = len(g); n1 = int(g.sum())
    tcrit = stats.t.ppf(0.975, N - 2)
    def tmap(lbl):
        a, b = M[lbl == 1], M[lbl == 0]
        t, _ = stats.ttest_ind(a, b, axis=0, equal_var=False)
        return np.nan_to_num(t)
    tobs = tmap(g)
    obs = clusters_2d(tobs, tcrit)
    null = np.empty(nperm)
    for k in range(nperm):
        p = rng.permutation(N); lbl = np.zeros(N, int); lbl[p[:n1]] = 1
        cl = clusters_2d(tmap(lbl), tcrit)
        null[k] = max([c[0] for c in cl], default=0.0)
    res = [dict(mass=m, sign=s, mask=mk, p=(1 + np.sum(null >= m)) / (nperm + 1))
           for (m, s, mk) in obs]
    return tobs, res


def main():
    z = np.load(CACHE, allow_pickle=True)
    maps = z["maps"]            # [n, NGRID, NANG]
    nang = int(z["nang"]); ngrid = maps.shape[1]
    xticks = [0, nang // 4, nang // 2, 3 * nang // 4]
    xlabs = ["Ant", "Right", "Post", "Left"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ri, (ok, ol) in enumerate([("icu", "ICU"), ("rpe", "RPE")]):
        grp = z[ok]; ok_m = ~np.isnan(grp)
        M = maps[ok_m]; g = grp[ok_m].astype(int)
        tobs, res = cluster_perm_2d(maps, grp)
        sig = [c for c in res if c["p"] < 0.05]
        print(f"{ol}: {len(sig)} significant 2D cluster(s):")
        for c in sig:
            ys, xs = np.where(c["mask"])
            print(f"   sign {c['sign']:+d}  mass {c['mass']:.0f}  p={c['p']:.4f}  "
                  f"pos {ys.min()}-{ys.max()}%  angle-bins {sorted(set(xs))[:6]}...")
        mu1, mu0 = M[g == 1].mean(0), M[g == 0].mean(0)
        vmax = np.nanpercentile(np.r_[mu1, mu0], 98)
        for ax, data, ttl in [(axes[ri, 0], mu1, f"{ol}+ mean radius (mm)"),
                              (axes[ri, 1], mu0, f"{ol}− mean radius (mm)")]:
            im = ax.imshow(data, aspect="auto", cmap="viridis", vmin=0, vmax=vmax,
                           extent=[0, nang, 100, 0])
            plt.colorbar(im, ax=ax, fraction=0.046)
            ax.set_title(ttl, fontsize=10)
        axt = axes[ri, 2]
        tmax = np.nanpercentile(np.abs(tobs), 99)
        im = axt.imshow(tobs, aspect="auto", cmap="RdBu_r", vmin=-tmax, vmax=tmax,
                        extent=[0, nang, 100, 0])
        plt.colorbar(im, ax=axt, fraction=0.046)
        for c in sig:
            axt.contour(np.arange(nang) + 0.5, np.linspace(0, 100, ngrid), c["mask"].astype(float),
                        levels=[0.5], colors="k", linewidths=1.4)
        axt.set_title(f"{ol} t-map (outline: cluster p<0.05)", fontsize=10)
        for ax in axes[ri]:
            ax.set_xticks(xticks); ax.set_xticklabels(xlabs, fontsize=8)
            ax.set_ylabel("position % (palate→cords)", fontsize=8)
    fig.suptitle("Airway 2D unfolded wall map — radius by cranio-caudal position × circumferential angle",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG, dpi=170)
    print("wrote", FIG)


if __name__ == "__main__":
    main()
