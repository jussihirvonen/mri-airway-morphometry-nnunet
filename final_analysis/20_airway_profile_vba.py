#!/usr/bin/env python3
"""
PROOF OF CONCEPT — airway cranio-caudal profile analysis ("1D voxel-based").

Instead of unreliable 3D image co-registration (the abscess problem), we use the
airway's intrinsic axis: for every patient we walk the lumen from hard palate
(superior) to vocal cords (inferior), record per-slice cross-sectional area and
midline shift, normalise the axis to [0, 1], and resample to a common grid.
We then run a position-wise ICU vs non-ICU comparison with a PATIENT-LEVEL
cluster-permutation test (same t-statistic in observed and null; cluster mass =
sum|t|; FWER via the max-cluster-mass null) — the statistically clean version of
what was attempted for abscesses.

Outputs:
  20_airway_profile_vba.npz   cached profiles
  20_airway_profile_vba.png   figure (CSA + shift profiles, significant clusters)
  stdout                      cluster p-values
"""
import os
from pathlib import Path
import numpy as np, pandas as pd, nibabel as nib
from scipy import ndimage as ndi
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(os.environ.get("AIRWAY_ROOT", Path(__file__).resolve().parents[1]))
LABELS = BASE / "nnunet_setup" / "Dataset005_NeckAirway" / "labelsTr"
OUTC = BASE / "final_analysis" / "r" / "airway_features_with_outcomes.csv"
CACHE = BASE / "final_analysis" / "20_airway_profile_vba.npz"
FIG = BASE / "final_analysis" / "20_airway_profile_vba.png"
NGRID = 100          # resampled positions palate(0) -> cords(1)
NPERM = 2000
SEED = 42


def lcc(b):
    lab, n = ndi.label(b)
    return b if n <= 1 else lab == (1 + int(np.argmax(ndi.sum(b, lab, range(1, n + 1)))))


def extract_profiles():
    df = pd.read_csv(OUTC)
    grid = np.linspace(0, 1, NGRID)
    csa_M, dev_M, ids, icu = [], [], [], []
    for _, row in df.iterrows():
        cid = row["case_id"]
        f = LABELS / f"{cid}.nii.gz"
        if not f.exists():
            continue
        img = nib.as_closest_canonical(nib.load(str(f)))
        sp = img.header.get_zooms()
        m = lcc(img.get_fdata() > 0.5)
        nx = m.shape[0]; px = sp[0] * sp[1]; midx = (nx - 1) / 2 * sp[0]
        zc = np.where([m[:, :, z].any() for z in range(m.shape[2])])[0]
        if len(zc) < 4:
            continue
        order = zc[::-1]                       # superior (palate) -> inferior (cords)
        csa = np.array([m[:, :, z].sum() * px for z in order])
        dev = np.array([abs(np.argwhere(m[:, :, z]).mean(axis=0)[0] * sp[0] - midx) for z in order])
        pos = np.linspace(0, 1, len(order))
        csa_M.append(np.interp(grid, pos, csa))
        dev_M.append(np.interp(grid, pos, dev))
        ids.append(cid); icu.append(int(row["icu_binary"]))
    return (np.array(csa_M), np.array(dev_M), np.array(icu), np.array(ids), grid)


def cluster_perm(M, icu, direction, nperm=NPERM, seed=SEED):
    """Position-wise Welch t (group1=ICU vs group0); cluster mass = sum|t| over
    contiguous positions with p<0.05; FWER via max-cluster-mass label permutation.
    direction: +1 expect ICU higher, -1 expect ICU lower (for one-sided cluster sign)."""
    rng = np.random.default_rng(seed)
    g1, g0 = M[icu == 1], M[icu == 0]
    tcrit = stats.t.ppf(0.975, len(icu) - 2)

    def tstat(a, b):
        t, _ = stats.ttest_ind(a, b, equal_var=False)
        return np.nan_to_num(t)

    def clusters(tvec):
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

    tobs = tstat(g1, g0)
    obs = clusters(tobs)
    # null distribution of max cluster mass
    null = np.zeros(nperm)
    n1 = int(icu.sum())
    for k in range(nperm):
        perm = rng.permutation(len(icu))
        a, b = M[perm[:n1]], M[perm[n1:]]
        cl = clusters(tstat(a, b))
        null[k] = max([c[2] for c in cl], default=0.0)
    res = []
    for (i, j, mass, sgn) in obs:
        p = (1 + np.sum(null >= mass)) / (nperm + 1)
        res.append(dict(start=i, end=j, mass=mass, sign=sgn, p=p))
    return tobs, res


def main():
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        csa_M, dev_M, icu, ids, grid = z["csa"], z["dev"], z["icu"], z["ids"], z["grid"]
        print(f"loaded cache: {csa_M.shape[0]} airways")
    else:
        csa_M, dev_M, icu, ids, grid = extract_profiles()
        np.savez(CACHE, csa=csa_M, dev=dev_M, icu=icu, ids=ids, grid=grid)
        print(f"extracted {csa_M.shape[0]} airways (ICU={int(icu.sum())})")

    print("\n=== CSA(position) ICU vs non-ICU — cluster permutation ===")
    tc, rc = cluster_perm(csa_M, icu, -1)
    for c in rc:
        print(f"  cluster pos {c['start']}-{c['end']} ({grid[c['start']]:.2f}-{grid[c['end']-1]:.2f}), "
              f"sign {c['sign']:+d}, mass {c['mass']:.1f}, p={c['p']:.4f}")
    print("\n=== midline-shift(position) ICU vs non-ICU — cluster permutation ===")
    td, rd = cluster_perm(dev_M, icu, +1)
    for c in rd:
        print(f"  cluster pos {c['start']}-{c['end']} ({grid[c['start']]:.2f}-{grid[c['end']-1]:.2f}), "
              f"sign {c['sign']:+d}, mass {c['mass']:.1f}, p={c['p']:.4f}")

    # figure
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, M, res, ylab, ttl in [
        (axes[0], csa_M, rc, "Cross-sectional area (mm²)", "Airway caliber"),
        (axes[1], dev_M, rd, "Midline shift (mm)", "Midline shift")]:
        x = grid * 100
        for lab, mask, col in [("ICU", icu == 1, "#d62828"), ("No ICU", icu == 0, "#1f6fb2")]:
            mu = M[mask].mean(0); se = M[mask].std(0) / np.sqrt(mask.sum())
            ax.plot(x, mu, color=col, label=lab)
            ax.fill_between(x, mu - se, mu + se, color=col, alpha=0.2)
        for c in res:
            if c["p"] < 0.05:
                ax.axvspan(x[c["start"]], x[c["end"] - 1], color="#999999", alpha=0.25, zorder=0)
        ax.set_xlabel("Cranio-caudal position (%)  palate → cords")
        ax.set_ylabel(ylab); ax.set_title(ttl); ax.legend(frameon=False, fontsize=9)
    fig.suptitle("Airway profile SPM: ICU vs non-ICU (shaded = significant cluster, patient-level permutation)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG, dpi=200)
    print("\nwrote", FIG)


if __name__ == "__main__":
    main()
