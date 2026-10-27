#!/usr/bin/env python3
"""
Airway VBA — extraction pass (cached).

For every patient, walk the lumen palate(superior)->cords(inferior) and record,
per axial slice:
  csa     cross-sectional area (mm^2)
  ap      antero-posterior extent (mm)
  lat     lateral extent (mm)
  circ    circularity 4*pi*A/P^2
  shift   |slice centroid - image midline| (mm)
and a 2D "unfolded" radius map r(position, angle): max centroid->boundary
distance per angular bin (0=anterior, 90=right, 180=posterior, 270=left).

The cranio-caudal axis is normalised to [0,1] and resampled to NGRID points
(intrinsic parameterisation -> no image co-registration needed).

Output: 21_airway_vba_cache.npz
"""
import os
from pathlib import Path
import numpy as np, pandas as pd, nibabel as nib
from scipy import ndimage as ndi
from skimage import measure

BASE = Path(os.environ.get("AIRWAY_ROOT", Path(__file__).resolve().parents[1]))
LABELS = BASE / "nnunet_setup" / "Dataset005_NeckAirway" / "labelsTr"
OUTC = BASE / "final_analysis" / "r" / "airway_features_with_outcomes.csv"
CACHE = BASE / "final_analysis" / "21_airway_vba_cache.npz"
NGRID, NANG = 100, 36


def lcc(b):
    lab, n = ndi.label(b)
    return b if n <= 1 else lab == (1 + int(np.argmax(ndi.sum(b, lab, range(1, n + 1)))))


def slice_metrics(sl, sp, midx):
    """sl: 2D bool (axis0=R, axis1=A). Returns csa, ap, lat, circ, cen_x_world, radius[NANG]."""
    pts = np.argwhere(sl)
    px = sp[0] * sp[1]
    csa = sl.sum() * px
    xs, ys = pts[:, 0], pts[:, 1]
    lat = (xs.max() - xs.min() + 1) * sp[0]
    ap = (ys.max() - ys.min() + 1) * sp[1]
    # circularity (in-plane isotropic)
    try:
        rp = max(measure.regionprops(measure.label(sl)), key=lambda r: r.area)
        per = rp.perimeter if rp.perimeter > 0 else np.nan
        circ = 4 * np.pi * rp.area / (per ** 2) if per and not np.isnan(per) else np.nan
        circ = min(circ, 1.0)
    except Exception:
        circ = np.nan
    cx, cy = xs.mean(), ys.mean()
    cen_x_world = cx * sp[0]
    # radius map per angle bin (relative to slice centroid)
    dx = (xs - cx) * sp[0]   # R
    dy = (ys - cy) * sp[1]   # A
    r = np.hypot(dx, dy)
    th = (np.degrees(np.arctan2(dx, dy)) % 360)   # 0=anterior, 90=right, 180=post, 270=left
    rad = np.zeros(NANG)
    bins = (th / (360 / NANG)).astype(int) % NANG
    for b in range(NANG):
        sel = bins == b
        rad[b] = r[sel].max() if sel.any() else 0.0
    return csa, ap, lat, circ, cen_x_world, rad


def main():
    df = pd.read_csv(OUTC)
    grid = np.linspace(0, 1, NGRID)
    keys = ["csa", "ap", "lat", "circ", "shift"]
    prof = {k: [] for k in keys}
    maps, ids = [], []
    rows = []
    for _, row in df.iterrows():
        cid = row["case_id"]
        f = LABELS / f"{cid}.nii.gz"
        if not f.exists():
            continue
        img = nib.as_closest_canonical(nib.load(str(f)))
        sp = tuple(float(z) for z in img.header.get_zooms()[:3])
        m = lcc(img.get_fdata() > 0.5)
        nx = m.shape[0]; midx = (nx - 1) / 2 * sp[0]
        zc = np.where([m[:, :, z].any() for z in range(m.shape[2])])[0]
        if len(zc) < 4:
            continue
        order = zc[::-1]                                  # palate(superior)->cords(inferior)
        per = {k: [] for k in keys}; rmap = []
        for z in order:
            csa, ap, lat, circ, cxw, rad = slice_metrics(m[:, :, z], sp, midx)
            per["csa"].append(csa); per["ap"].append(ap); per["lat"].append(lat)
            per["circ"].append(circ); per["shift"].append(abs(cxw - midx))
            rmap.append(rad)
        pos = np.linspace(0, 1, len(order))
        for k in keys:
            v = np.array(per[k], float)
            ok = ~np.isnan(v)
            prof[k].append(np.interp(grid, pos[ok], v[ok]) if ok.sum() > 1 else np.full(NGRID, np.nan))
        rmap = np.array(rmap)                              # [nslice, NANG]
        m2 = np.vstack([np.interp(grid, pos, rmap[:, a]) for a in range(NANG)]).T  # [NGRID, NANG]
        maps.append(m2); ids.append(cid)
        rows.append(row)
    out = pd.DataFrame(rows)
    np.savez(CACHE,
             grid=grid, ids=np.array(ids),
             csa=np.array(prof["csa"]), ap=np.array(prof["ap"]), lat=np.array(prof["lat"]),
             circ=np.array(prof["circ"]), shift=np.array(prof["shift"]),
             maps=np.array(maps),
             icu=out["icu_binary"].values.astype(float),
             prolonged=out["hospd_prolonged"].values.astype(float),
             rpe=out["retroinf"].values.astype(float),          # complete RPE
             me=out["medoed_binary"].values.astype(float),
             abscess=out["abs_positive"].values.astype(float),
             nang=NANG)
    print(f"cached {len(ids)} airways, maps {np.array(maps).shape}  -> {CACHE.name}")


if __name__ == "__main__":
    main()
