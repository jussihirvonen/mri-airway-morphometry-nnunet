#!/usr/bin/env python3
"""
3D semi-transparent renderings of the airway lumen: a normal airway versus one
with a long sub-50 mm² segment (LCSA50). The longest contiguous run of axial
slices with cross-sectional area < 50 mm² (the LCSA50 segment) is highlighted.

PyVista renders each lumen (transparent background, common parallel scale); the
two panels are composed + labelled in matplotlib for crisp text and PDF output.

Edit CONFIG and re-run:
    python3 build_figure_airway3d.py
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
import matplotlib.image as mpimg

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
OUT = BASE / "manuscript_drafts" / "tables_figures" / "Figure_airway3d"

# (case_id, panel title)
CASES = [
    (_example("airway3d_normal"), "Normal airway"),
    (_example("airway3d_stenotic"), "Long stenotic segment"),
]
CSA_THRESHOLD = 50.0          # mm^2 (LCSA50 definition)
COL_LUMEN = "#9db4d4"         # semi-transparent lumen
COL_NARROW = "#d62828"        # highlighted sub-50 mm^2 segment
OP_LUMEN, OP_NARROW = 0.30, 0.95
SMOOTH_ITERS = 40
VIEW_VECTOR = (0.35, -1.0, 0.18)   # slight oblique anterior view
VIEW_UP = (0, 0, 1)
WINDOW = (760, 1080)               # per-panel render size (px)
SCALE_MARGIN = 1.18                # parallel-scale padding
pv.OFF_SCREEN = True


def largest_cc(mask):
    lab, n = ndi.label(mask)
    if n <= 1:
        return mask
    return lab == (1 + int(np.argmax(ndi.sum(mask, lab, range(1, n + 1)))))


def load_case(cid):
    img = nib.as_closest_canonical(nib.load(str(LABELS / f"{cid}.nii.gz")))
    sp = tuple(float(z) for z in img.header.get_zooms()[:3])
    m = largest_cc(img.get_fdata() > 0.5)
    px = sp[0] * sp[1]
    present = np.array([m[:, :, z].any() for z in range(m.shape[2])])
    csa = np.array([m[:, :, z].sum() * px for z in range(m.shape[2])])
    narrow_slices = (csa < CSA_THRESHOLD) & present
    # longest contiguous run = LCSA50 segment
    best, i = (0, 0, 0), 0
    while i < len(narrow_slices):
        if narrow_slices[i]:
            j = i
            while j < len(narrow_slices) and narrow_slices[j]:
                j += 1
            if j - i > best[0]:
                best = (j - i, i, j)
            i = j
        else:
            i += 1
    nrun, za, zb = best
    narrow = np.zeros_like(m)
    if nrun > 0:
        narrow[:, :, za:zb] = m[:, :, za:zb]
    lcsa50_mm = nrun * sp[2]
    return m, narrow, sp, lcsa50_mm


def to_surface(arr, sp):
    g = pv.ImageData(dimensions=arr.shape, spacing=sp)
    g.point_data["m"] = arr.astype(float).flatten(order="F")
    s = g.contour([0.5], scalars="m")
    if s.n_points:
        s = s.smooth_taubin(n_iter=SMOOTH_ITERS, pass_band=0.08)
    return s


def add_orientation(pl, mesh, off=6.0):
    """Bounding box + anatomical axis labels (S-I, R-L, A-P) for the reference airway.
    RAS canonical: +x=R, +y=A, +z=S."""
    b = mesh.bounds
    pl.add_mesh(pv.Box(bounds=b).outline(), color="#888888", line_width=2)
    cx, cy, cz = (b[0]+b[1])/2, (b[2]+b[3])/2, (b[4]+b[5])/2
    pts = [[cx, cy, b[5]+off], [cx, cy, b[4]-off],     # S, I
           [b[1]+off, cy, cz], [b[0]-off, cy, cz],     # R, L
           [cx, b[3]+off, cz], [cx, b[2]-off, cz]]     # A, P
    pl.add_point_labels(pts, ["S", "I", "R", "L", "A", "P"], font_size=20,
                        text_color="black", bold=True, shape=None,
                        show_points=False, always_visible=True)


def render_case(cid, pscale, orient=False):
    m, narrow, sp, lcsa = load_case(cid)
    lumen = to_surface(m, sp)
    pl = pv.Plotter(off_screen=True, window_size=WINDOW)
    pl.set_background("white")
    pl.add_mesh(lumen, color=COL_LUMEN, opacity=OP_LUMEN, smooth_shading=True, specular=0.3)
    if narrow.any():
        pl.add_mesh(to_surface(narrow, sp), color=COL_NARROW, opacity=OP_NARROW,
                    smooth_shading=True, specular=0.3)
    if orient:
        add_orientation(pl, lumen)
    pl.enable_parallel_projection()
    c = np.array(lumen.center)
    pl.camera.focal_point = c
    pl.camera.position = c + np.array(VIEW_VECTOR) / np.linalg.norm(VIEW_VECTOR) * 400
    pl.camera.up = VIEW_UP
    pl.camera.parallel_scale = pscale
    img = pl.screenshot(transparent_background=True, return_img=True)
    pl.close()
    return img, lcsa


def main():
    # common parallel scale from the taller airway (so both panels share scale)
    half = []
    for cid, _ in CASES:
        m, _, sp, _ = load_case(cid)
        zext = m.any(axis=(0, 1))
        zc = np.where(zext)[0]
        half.append((zc.max() - zc.min() + 1) * sp[2] / 2)
    pscale = max(half) * SCALE_MARGIN

    fig, axes = plt.subplots(1, len(CASES), figsize=(7.0, 5.6))
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    lcsa_report = {}
    for i, (ax, (cid, title)) in enumerate(zip(np.atleast_1d(axes), CASES)):
        img, lcsa = render_case(cid, pscale, orient=(i == 0))   # orientation box on the normal airway only
        lcsa_report[cid] = round(lcsa)
        ax.imshow(img)
        ax.set_title(f"{title}\nLCSA50 = {lcsa:.0f} mm", fontsize=11)
        ax.axis("off")
    # legend
    from matplotlib.patches import Patch
    leg = [Patch(facecolor=COL_LUMEN, alpha=0.5, label="Airway lumen"),
           Patch(facecolor=COL_NARROW, label="Segment < 50 mm² (LCSA50)")]
    fig.legend(handles=leg, loc="lower center", ncol=2, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, 0.01))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.09, wspace=0.02)
    fig.savefig(str(OUT) + ".png", dpi=300)
    fig.savefig(str(OUT) + ".pdf")
    print("wrote", OUT.with_suffix(".png"), "and .pdf  | LCSA50:", lcsa_report)


if __name__ == "__main__":
    main()
