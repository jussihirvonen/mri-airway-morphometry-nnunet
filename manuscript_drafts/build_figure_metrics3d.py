#!/usr/bin/env python3
"""
Feature-definition figure: the five pre-specified airway morphometric features
illustrated on ONE representative airway (same lumen, same camera, one highlight
per panel).

  LV         whole semi-transparent lumen (the volume)
  mCSA       narrowest axial slice highlighted (red slab)
  Compactness lumen inside its axis-aligned bounding box (wireframe)
  maxMLS     image midline (line) + arrow to the most laterally shifted centroid
  LCSA50     longest contiguous run of slices < 50 mm^2 (red segment)

PyVista renders each panel (transparent bg, common parallel scale); panels are
composed + labelled in matplotlib.  Edit CONFIG and re-run:
    python3 build_figure_metrics3d.py
"""
import os
import json
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy import ndimage as ndi
import pyvista as pv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
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
OUT = BASE / "manuscript_drafts" / "tables_figures" / "Figure_metrics3d"
CASE = _example("metrics3d_case")          # representative (abnormal) airway, all 5 features visible
CASE_NORMAL = _example("metrics3d_normal") # normal reference airway (patent, LCSA50 = 0)
CSA_THRESHOLD = 50.0
COL_LUMEN, COL_HILITE, COL_BOX, COL_MID = "#9db4d4", "#d62828", "#555555", "#1f6fb2"
OP_GHOST, OP_FILL, OP_HILITE = 0.22, 0.55, 0.95
SMOOTH_ITERS = 40
VIEW_VECTOR, VIEW_UP = (0.35, -1.0, 0.18), (0, 0, 1)
WINDOW = (640, 950)
SCALE_MARGIN = 1.15
pv.OFF_SCREEN = True


def lcc(mask):
    lab, n = ndi.label(mask)
    return mask if n <= 1 else lab == (1 + int(np.argmax(ndi.sum(mask, lab, range(1, n + 1)))))


def load_case(cid):
    img = nib.as_closest_canonical(nib.load(str(LABELS / f"{cid}.nii.gz")))
    sp = tuple(float(z) for z in img.header.get_zooms()[:3])
    m = lcc(img.get_fdata() > 0.5)
    g = geometry(m, sp)
    return {"m": m, "sp": sp, "g": g, "lumen": surf(m, sp)}


def height_mm(cd):
    zc = np.where([cd["m"][:, :, z].any() for z in range(cd["m"].shape[2])])[0]
    return (zc.max() - zc.min() + 1) * cd["sp"][2]


def geometry(m, sp):
    nx, ny, nz = m.shape
    px = sp[0] * sp[1]
    pres = np.array([m[:, :, z].any() for z in range(nz)])
    csa = np.array([m[:, :, z].sum() * px for z in range(nz)])
    zc = np.where(pres)[0]
    g = {}
    g["LV"] = m.sum() * px * sp[2] / 1000.0
    g["zmin"] = int(zc[np.argmin(csa[zc])]); g["mCSA"] = float(csa[zc].min())
    # compactness = volume / axis-aligned bbox volume
    xs, ys, zs = np.where(m)
    bbox = ((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1) * (zs.max() - zs.min() + 1))
    g["compactness"] = m.sum() / bbox
    g["bounds"] = (xs.min()*sp[0], xs.max()*sp[0], ys.min()*sp[1], ys.max()*sp[1], zs.min()*sp[2], zs.max()*sp[2])
    # maxMLS: slice centroid (axis0 = L-R) vs image midline
    mid_x = (nx - 1) / 2 * sp[0]
    cen_x = np.array([np.argwhere(m[:, :, z]).mean(axis=0)[0] * sp[0] if pres[z] else np.nan for z in range(nz)])
    cen_y = np.array([np.argwhere(m[:, :, z]).mean(axis=0)[1] * sp[1] if pres[z] else np.nan for z in range(nz)])
    dev = cen_x - mid_x
    zdev = int(zc[np.nanargmax(np.abs(dev[zc]))])
    g["maxMLS"] = float(np.nanmax(np.abs(dev[zc])))
    g["mid_x"] = mid_x; g["zdev"] = zdev
    g["cen_x_dev"] = float(cen_x[zdev]); g["yc"] = float(np.nanmean(cen_y))
    g["z_lo"], g["z_hi"] = float(zc.min()*sp[2]), float(zc.max()*sp[2])
    # LCSA50 longest run
    nar = (csa < CSA_THRESHOLD) & pres
    best = (0, 0, 0); i = 0
    while i < len(nar):
        if nar[i]:
            j = i
            while j < len(nar) and nar[j]:
                j += 1
            if j - i > best[0]:
                best = (j - i, i, j)
            i = j
        else:
            i += 1
    g["lcsa_run"] = (best[1], best[2]); g["LCSA50"] = best[0] * sp[2]
    return g


def surf(arr, sp):
    grid = pv.ImageData(dimensions=arr.shape, spacing=sp)
    grid.point_data["m"] = arr.astype(float).flatten(order="F")
    s = grid.contour([0.5], scalars="m")
    return s.smooth_taubin(n_iter=SMOOTH_ITERS, pass_band=0.08) if s.n_points else s


def render(panel, m, sp, g, lumen, pscale):
    pl = pv.Plotter(off_screen=True, window_size=WINDOW)
    pl.set_background("white")
    ghost = OP_FILL if panel in ("LV", "normal") else OP_GHOST
    pl.add_mesh(lumen, color=COL_LUMEN, opacity=ghost, smooth_shading=True, specular=0.3)

    if panel == "mCSA":
        sl = np.zeros_like(m); sl[:, :, g["zmin"]] = m[:, :, g["zmin"]]
        pl.add_mesh(surf(sl, sp), color=COL_HILITE, opacity=OP_HILITE, smooth_shading=True)
    elif panel == "compactness":
        x0, x1, y0, y1, z0, z1 = g["bounds"]
        box = pv.Box(bounds=(x0, x1, y0, y1, z0, z1)).outline()
        pl.add_mesh(box, color=COL_BOX, line_width=3)
    elif panel == "maxMLS":
        midline = pv.Line((g["mid_x"], g["yc"], g["z_lo"]), (g["mid_x"], g["yc"], g["z_hi"]))
        pl.add_mesh(midline.tube(radius=0.6), color=COL_MID)
        zw = g["zdev"] * sp[2]
        start = np.array([g["mid_x"], g["yc"], zw]); end = np.array([g["cen_x_dev"], g["yc"], zw])
        d = end - start
        arrow = pv.Arrow(start=start, direction=d, scale=float(np.linalg.norm(d)),
                         tip_length=0.18, tip_radius=0.06, shaft_radius=0.025)
        pl.add_mesh(arrow, color=COL_HILITE)
        pl.add_mesh(pv.Sphere(radius=1.6, center=end), color=COL_HILITE)
    elif panel == "LCSA50":
        za, zb = g["lcsa_run"]; seg = np.zeros_like(m); seg[:, :, za:zb] = m[:, :, za:zb]
        pl.add_mesh(surf(seg, sp), color=COL_HILITE, opacity=OP_HILITE, smooth_shading=True)
    elif panel == "normal":
        # orientation key on the reference airway: bounding box + S-I/R-L/A-P labels
        b = lumen.bounds; off = 6.0
        pl.add_mesh(pv.Box(bounds=b).outline(), color="#888888", line_width=2)
        cx, cy, cz = (b[0]+b[1])/2, (b[2]+b[3])/2, (b[4]+b[5])/2
        pts = [[cx, cy, b[5]+off], [cx, cy, b[4]-off], [b[1]+off, cy, cz],
               [b[0]-off, cy, cz], [cx, b[3]+off, cz], [cx, b[2]-off, cz]]
        pl.add_point_labels(pts, ["S", "I", "R", "L", "A", "P"], font_size=18,
                            text_color="black", bold=True, shape=None,
                            show_points=False, always_visible=True)

    pl.enable_parallel_projection()
    c = np.array(lumen.center)
    pl.camera.focal_point = c
    pl.camera.position = c + np.array(VIEW_VECTOR) / np.linalg.norm(VIEW_VECTOR) * 400
    pl.camera.up = VIEW_UP
    pl.camera.parallel_scale = pscale
    img = pl.screenshot(transparent_background=True, return_img=True)
    pl.close()
    return img


def main():
    normal = load_case(CASE_NORMAL)
    main_cd = load_case(CASE)
    g = main_cd["g"]
    pscale = max(height_mm(normal), height_mm(main_cd)) / 2 * SCALE_MARGIN

    # (panel key, title, case data)
    PANELS = [
        ("normal",      "Normal airway\n(reference)",               normal),
        ("LV",          f"Luminal volume (LV)\n{g['LV']:.0f} mL",   main_cd),
        ("mCSA",        f"Minimum CSA (mCSA)\n{g['mCSA']:.0f} mm²",  main_cd),
        ("compactness", f"Compactness\n{g['compactness']:.2f}",     main_cd),
        ("maxMLS",      f"Maximum midline shift\n{g['maxMLS']:.0f} mm", main_cd),
        ("LCSA50",      f"Length < 50 mm² (LCSA50)\n{g['LCSA50']:.0f} mm", main_cd),
    ]
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, len(PANELS), figsize=(15.5, 4.6))
    for ax, (panel, title, cd) in zip(axes, PANELS):
        ax.imshow(render(panel, cd["m"], cd["sp"], cd["g"], cd["lumen"], pscale))
        ax.set_title(title, fontsize=10.5)
        ax.axis("off")
    leg = [Patch(facecolor=COL_LUMEN, alpha=0.5, label="Airway lumen"),
           Patch(facecolor=COL_HILITE, label="Highlighted feature"),
           Line2D([0], [0], color=COL_BOX, lw=3, label="Bounding box"),
           Line2D([0], [0], color=COL_MID, lw=3, label="Image midline")]
    fig.legend(handles=leg, loc="lower center", ncol=4, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, 0.005))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.90, bottom=0.10, wspace=0.01)
    # divider between the normal reference and the abnormal-airway metric panels (after layout)
    sep_x = (axes[0].get_position().x1 + axes[1].get_position().x0) / 2
    fig.add_artist(Line2D([sep_x, sep_x], [0.12, 0.88], color="#cccccc", lw=1.2))
    fig.savefig(str(OUT) + ".png", dpi=300)
    fig.savefig(str(OUT) + ".pdf")
    print("wrote", OUT.with_suffix(".png"),
          "| LV %.1f mCSA %.0f comp %.2f maxMLS %.1f LCSA50 %.0f" %
          (g["LV"], g["mCSA"], g["compactness"], g["maxMLS"], g["LCSA50"]))


if __name__ == "__main__":
    main()
