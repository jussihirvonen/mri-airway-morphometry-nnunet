#!/usr/bin/env python3
"""
VBA method schematic: how one airway -> 1D cranio-caudal profile + 2D unfolded
wall map. (A) 3D lumen with the palate->cords axis and three sample axial levels;
(B) the cross-sectional-area profile with those levels marked; (C) the unfolded
radius map r(position, angle). Reads cache for the profile/map of CASE.
"""
import os
import json
from pathlib import Path
import numpy as np, nibabel as nib
from scipy import ndimage as ndi
import pyvista as pv
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
LABELS = BASE / "nnunet_setup" / "Dataset005_NeckAirway" / "labelsTr"
CACHE = BASE / "final_analysis" / "21_airway_vba_cache.npz"
OUT = BASE / "final_analysis" / "24_airway_vba_method.png"
CASE = _example("profile_method")
LEVEL_POS = [0.20, 0.50, 0.80]        # sample levels (fraction palate->cords)
LEVEL_COL = ["#e69f00", "#009e73", "#cc3311"]
pv.OFF_SCREEN = True


def lcc(b):
    lab, n = ndi.label(b)
    return b if n <= 1 else lab == (1 + int(np.argmax(ndi.sum(b, lab, range(1, n + 1)))))


def surf(arr, sp):
    g = pv.ImageData(dimensions=arr.shape, spacing=sp)
    g.point_data["m"] = arr.astype(float).flatten(order="F")
    s = g.contour([0.5], scalars="m")
    return s.smooth_taubin(n_iter=40, pass_band=0.08) if s.n_points else s


def render3d(m, sp):
    zc = np.where([m[:, :, z].any() for z in range(m.shape[2])])[0]
    order = zc[::-1]                                   # palate(sup)->cords(inf)
    lumen = surf(m, sp)
    pl = pv.Plotter(off_screen=True, window_size=(620, 1000)); pl.set_background("white")
    pl.add_mesh(lumen, color="#9db4d4", opacity=0.28, smooth_shading=True)
    for pos, col in zip(LEVEL_POS, LEVEL_COL):
        z = int(order[int(round(pos * (len(order) - 1)))])
        sl = np.zeros_like(m); sl[:, :, z] = m[:, :, z]
        pl.add_mesh(surf(sl, sp), color=col, opacity=0.95, smooth_shading=True)
    # palate->cords axis arrow alongside
    b = lumen.bounds
    x = b[0] - 8; y = (b[2] + b[3]) / 2
    pl.add_mesh(pv.Arrow(start=(x, y, b[5]), direction=(0, 0, -1),
                         scale=(b[5] - b[4]) * 1.05, tip_length=0.08, tip_radius=0.02,
                         shaft_radius=0.006), color="#333333")
    pl.enable_parallel_projection()
    c = np.array(lumen.center); pl.camera.focal_point = c
    pl.camera.position = c + np.array([0.35, -1, 0.18]) / np.linalg.norm([0.35, -1, 0.18]) * 400
    pl.camera.up = (0, 0, 1)
    pl.camera.parallel_scale = (zc.max() - zc.min() + 1) * sp[2] / 2 * 1.18
    img = pl.screenshot(transparent_background=True, return_img=True); pl.close()
    return img


def main():
    img = nib.as_closest_canonical(nib.load(str(LABELS / f"{CASE}.nii.gz")))
    sp = tuple(float(z) for z in img.header.get_zooms()[:3])
    m = lcc(img.get_fdata() > 0.5)
    ren = render3d(m, sp)

    z = np.load(CACHE, allow_pickle=True)
    i = list(z["ids"]).index(CASE); grid = z["grid"] * 100
    csa = z["csa"][i]; rmap = z["maps"][i]; nang = int(z["nang"])

    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    fig = plt.figure(figsize=(12.5, 5.2))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 1.5, 1.5], height_ratios=[1, 1])
    ax3d = fig.add_subplot(gs[:, 0]); ax3d.imshow(ren); ax3d.axis("off")
    ax3d.set_title("A. Airway lumen + palate→cords axis\n(3 sample levels)", fontsize=10)

    axb = fig.add_subplot(gs[:, 1])
    axb.plot(grid, csa, color="#333"); axb.set_xlabel("position % (palate→cords)")
    axb.set_ylabel("cross-sectional area (mm²)")
    for pos, col in zip(LEVEL_POS, LEVEL_COL):
        axb.axvline(pos * 100, color=col, ls="--", lw=1.5)
    axb.set_title("B. 1D profile (per metric)", fontsize=10)

    axc = fig.add_subplot(gs[:, 2])
    im = axc.imshow(rmap, aspect="auto", cmap="viridis", extent=[0, nang, 100, 0])
    plt.colorbar(im, ax=axc, fraction=0.046, label="radius (mm)")
    axc.set_xticks([0, nang//4, nang//2, 3*nang//4]); axc.set_xticklabels(["Ant", "Right", "Post", "Left"], fontsize=8)
    axc.set_ylabel("position % (palate→cords)")
    for pos, col in zip(LEVEL_POS, LEVEL_COL):
        axc.axhline(pos * 100, color=col, ls="--", lw=1.5)
    axc.set_title("C. 2D unfolded wall map r(position, angle)", fontsize=10)

    fig.suptitle(f"Airway VBA — intrinsic parameterization (no image co-registration), example {CASE}",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT, dpi=200); print("wrote", OUT)


if __name__ == "__main__":
    main()
