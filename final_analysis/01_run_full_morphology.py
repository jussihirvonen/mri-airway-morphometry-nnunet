#!/usr/bin/env python3
"""
Run the full 62-feature airway morphology analysis on the final reference
segmentations (folder given by paths_local.json, key reference_masks).
"""

import os
import json
import csv
import re
from pathlib import Path
import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_erosion
from scipy.stats import skew, kurtosis

BASE = Path(os.environ.get("AIRWAY_ROOT", Path(__file__).resolve().parents[1]))
_CFG_FILE = BASE / "paths_local.json"  # untracked: maps generic keys to the local folder layout (README, "Local layout")
_CFG = json.loads(_CFG_FILE.read_text()) if _CFG_FILE.exists() else {}
REF_DIR = BASE / _CFG.get("reference_masks", "reference_masks")
OUT_DIR = BASE / "final_analysis"
CSV_FILE = OUT_DIR / "reference_full_morphology.csv"

CSV_COLUMNS = [
    'case_id',
    'volume_ml', 'cc_length_mm', 'n_slices',
    'min_area_mm2', 'max_area_mm2', 'mean_area_mm2',
    'stenosis_ratio', 'stenosis_pct',
    'ap_min_mm', 'ap_max_mm', 'ap_mean_mm',
    'lat_min_mm', 'lat_max_mm', 'lat_mean_mm',
    'ap_lat_ratio_mean',
    'ap_at_narrowest_mm', 'lat_at_narrowest_mm', 'ap_lat_ratio_at_narrowest',
    'rl_dev_mean_mm', 'rl_dev_max_mm', 'rl_dev_range_mm', 'rl_dev_at_narrowest_mm',
    'sphericity', 'compactness', 'elongation', 'equiv_diameter_mean_mm',
    'gradient_max_narrowing_mm2_per_mm', 'gradient_max_widening_mm2_per_mm',
    'gradient_mean_abs_mm2_per_mm',
    'area_skewness', 'area_kurtosis', 'area_cv',
    'area_q10_mm2', 'area_q25_mm2', 'area_median_mm2', 'area_q75_mm2',
    'narrowest_rel_pos', 'ref_area_mm2',
    'norm_min_area', 'norm_mean_area', 'norm_area_q10', 'norm_area_q25',
    'norm_area_median', 'norm_stenosis_pct', 'norm_volume_mm',
    'norm_mean_upper', 'norm_mean_lower', 'upper_lower_ratio', 'norm_elongation',
    'circularity_mean', 'circularity_min', 'circularity_at_narrowest',
    'hydraulic_diam_mean_mm', 'hydraulic_diam_min_mm', 'hydraulic_diam_at_narrowest_mm',
    'narrowed_length_50pct_mm', 'narrowed_length_25pct_mm', 'narrowed_fraction_50pct',
    # New absolute-threshold length features (added 2026-05-19 — chosen as headline)
    'len_below_25mm2_mm', 'len_below_50mm2_mm', 'len_below_75mm2_mm', 'len_below_100mm2_mm',
    # New relative-to-max-CSA length features (sensitivity / RPE mechanism)
    'len_below_25pct_max_mm', 'len_below_33pct_max_mm', 'len_below_50pct_max_mm',
    # Absolute midline shift (signed value cancels across patients; abs() is the clinical metric)
    'rl_dev_max_abs_mm', 'rl_dev_mean_abs_mm', 'rl_dev_at_narrowest_abs_mm',
    'tortuosity', 'flatness_pca', 'sa_vol_ratio',
    'islands_removed',
]


def analyze_case(case_id, mask_file, keep_largest=True):
    """keep_largest=True (default): retain only the largest 26-connected component
    (correct for the obstruction-bridged MANUAL masks, which are single-component).
    keep_largest=False: evaluate the whole mask as the UNION of all components
    (used for nnU-Net PREDICTIONS, which leave Z-gaps at total obstructions; largest-CC
    would delete real tube there — see 27_/28_ analyses, decision 2026-06-22)."""
    if not mask_file.exists():
        return None

    mask_nii = nib.load(str(mask_file))
    mask = mask_nii.get_fdata()
    vx, vy, vz = [float(z) for z in mask_nii.header.get_zooms()]

    binary = (mask > 0).astype(int)

    labeled, n_comp = ndimage.label(binary, structure=ndimage.generate_binary_structure(3, 3))
    islands_removed = 0
    if keep_largest and n_comp > 1:
        sizes = ndimage.sum(binary, labeled, range(1, n_comp + 1))
        largest = np.argmax(sizes) + 1
        islands_removed = n_comp - 1
        binary = (labeled == largest).astype(int)

    slice_data = []
    for z in range(binary.shape[2]):
        sl = binary[:, :, z]
        n_vox = sl.sum()
        if n_vox > 0:
            area_mm2 = n_vox * vx * vy
            coords = np.where(sl > 0)
            ap = (coords[0].max() - coords[0].min() + 1) * vx
            lat = (coords[1].max() - coords[1].min() + 1) * vy
            cx = np.mean(coords[0])
            cy = np.mean(coords[1])
            padded = np.pad(sl, 1, mode='constant', constant_values=0)
            row_trans = np.abs(np.diff(padded.astype(int), axis=0)).sum()
            col_trans = np.abs(np.diff(padded.astype(int), axis=1)).sum()
            perimeter = row_trans * vy + col_trans * vx
            circ = (4 * np.pi * area_mm2) / (perimeter ** 2) if perimeter > 0 else 0
            hyd = (4 * area_mm2) / perimeter if perimeter > 0 else 0
            slice_data.append({'z': z, 'area': area_mm2, 'ap': ap, 'lat': lat,
                               'cx': cx, 'cy': cy, 'perimeter': perimeter,
                               'circularity': circ, 'hydraulic_diam': hyd})

    if not slice_data:
        return None

    areas = [s['area'] for s in slice_data]
    ap_diams = [s['ap'] for s in slice_data]
    lat_diams = [s['lat'] for s in slice_data]
    ap_lat_ratios = [s['ap'] / s['lat'] if s['lat'] > 0 else 0 for s in slice_data]

    voxel_vol = vx * vy * vz
    volume_mm3 = binary.sum() * voxel_vol
    volume_ml = volume_mm3 / 1000

    min_idx = int(np.argmin(areas))
    min_area = areas[min_idx]
    max_area = max(areas)
    stenosis_ratio = min_area / max_area
    stenosis_pct = (1 - stenosis_ratio) * 100

    z_min = slice_data[0]['z']
    z_max = slice_data[-1]['z']
    cc_length = (z_max - z_min) * vz

    cx_img = binary.shape[0] / 2
    rl_devs = [(s['cx'] - cx_img) * vx for s in slice_data]

    surface = binary.astype(bool) & ~binary_erosion(binary.astype(bool))
    avg_face = (vx*vy + vx*vz + vy*vz) / 3
    surface_area = surface.sum() * avg_face
    sphericity = (np.pi**(1/3) * (6*volume_mm3)**(2/3)) / surface_area if surface_area > 0 else 0

    coords_3d = np.where(binary > 0)
    bbox_vol = ((coords_3d[0].max() - coords_3d[0].min() + 1) * vx *
                (coords_3d[1].max() - coords_3d[1].min() + 1) * vy *
                (coords_3d[2].max() - coords_3d[2].min() + 1) * vz)
    compactness = volume_mm3 / bbox_vol if bbox_vol > 0 else 0

    mean_area = np.mean(areas)
    equiv_diam = 2 * np.sqrt(mean_area / np.pi)
    elongation = cc_length / equiv_diam if equiv_diam > 0 else 0

    gradients = []
    for i in range(1, len(slice_data)):
        dz = (slice_data[i]['z'] - slice_data[i-1]['z']) * vz
        if dz > 0:
            gradients.append((areas[i] - areas[i-1]) / dz)

    areas_arr = np.array(areas)
    area_skewness = float(skew(areas_arr))
    area_kurt = float(kurtosis(areas_arr))
    area_cv = float(np.std(areas_arr) / mean_area) if mean_area > 0 else 0
    area_q10, area_q25, area_median, area_q75 = np.percentile(areas_arr, [10, 25, 50, 75])
    narrowest_rel_pos = min_idx / (len(areas) - 1) if len(areas) > 1 else 0.5

    ref_area = areas[-1]
    norm_areas = areas_arr / ref_area if ref_area > 0 else areas_arr
    norm_min = float(np.min(norm_areas))
    norm_mean = float(np.mean(norm_areas))
    norm_q10, norm_q25, norm_median = np.percentile(norm_areas, [10, 25, 50])
    norm_stenosis_pct = (1 - norm_min) * 100
    norm_volume_mm = volume_mm3 / ref_area if ref_area > 0 else 0

    mid = len(norm_areas) // 2
    norm_lower = norm_areas[:mid]
    norm_upper = norm_areas[mid:]
    norm_mean_upper = float(np.mean(norm_upper))
    norm_mean_lower = float(np.mean(norm_lower))
    upper_lower_ratio = norm_mean_upper / norm_mean_lower if norm_mean_lower > 0 else 0

    ref_equiv_diam = 2 * np.sqrt(ref_area / np.pi) if ref_area > 0 else 0
    norm_elongation = cc_length / ref_equiv_diam if ref_equiv_diam > 0 else 0

    circularities = [s['circularity'] for s in slice_data]
    hydraulic_diams = [s['hydraulic_diam'] for s in slice_data]
    circularity_mean = np.mean(circularities)
    circularity_min = min(circularities)
    circularity_at_narrowest = circularities[min_idx]
    hydraulic_diam_mean = np.mean(hydraulic_diams)
    hydraulic_diam_min = min(hydraulic_diams)
    hydraulic_diam_at_narrowest = hydraulic_diams[min_idx]

    def _longest_run(bool_list):
        mx = cur = 0
        for v in bool_list:
            if v:
                cur += 1
                mx = max(mx, cur)
            else:
                cur = 0
        return mx

    narrowed_50 = [a < 0.5 * ref_area for a in areas]
    narrowed_25 = [a < 0.25 * ref_area for a in areas]
    narrowed_len_50 = _longest_run(narrowed_50) * vz
    narrowed_len_25 = _longest_run(narrowed_25) * vz
    narrowed_frac_50 = sum(narrowed_50) / len(narrowed_50) if areas else 0

    len_below_25mm2 = _longest_run([a < 25 for a in areas]) * vz
    len_below_50mm2 = _longest_run([a < 50 for a in areas]) * vz
    len_below_75mm2 = _longest_run([a < 75 for a in areas]) * vz
    len_below_100mm2 = _longest_run([a < 100 for a in areas]) * vz

    len_below_25pct_max = _longest_run([a <= 0.25 * max_area for a in areas]) * vz
    len_below_33pct_max = _longest_run([a <= 0.33 * max_area for a in areas]) * vz
    len_below_50pct_max = _longest_run([a <= 0.50 * max_area for a in areas]) * vz

    rl_dev_max_abs = max(abs(d) for d in rl_devs)
    rl_dev_mean_abs = float(np.mean([abs(d) for d in rl_devs]))
    rl_dev_at_narrowest_abs = abs(rl_devs[min_idx])

    centroids_3d = [(s['cx'] * vx, s['cy'] * vy, s['z'] * vz) for s in slice_data]
    if len(centroids_3d) > 1:
        path_len = sum(np.sqrt(sum((c2[k] - c1[k])**2 for k in range(3)))
                       for c1, c2 in zip(centroids_3d[:-1], centroids_3d[1:]))
        endpt_dist = np.sqrt(sum((centroids_3d[-1][k] - centroids_3d[0][k])**2 for k in range(3)))
        tortuosity_val = path_len / endpt_dist if endpt_dist > 0 else 1.0
    else:
        tortuosity_val = 1.0

    coords_scaled = np.column_stack([coords_3d[0] * vx, coords_3d[1] * vy, coords_3d[2] * vz])
    cov_matrix = np.cov(coords_scaled.T)
    eigenvalues = np.sort(np.linalg.eigvalsh(cov_matrix))[::-1]
    flatness_pca = eigenvalues[2] / eigenvalues[1] if eigenvalues[1] > 0 else 0

    sa_vol_ratio = surface_area / volume_mm3 if volume_mm3 > 0 else 0

    return {
        'case_id': case_id,
        'volume_ml': round(volume_ml, 2),
        'cc_length_mm': round(cc_length, 1),
        'n_slices': len(slice_data),
        'min_area_mm2': round(min_area, 1),
        'max_area_mm2': round(max_area, 1),
        'mean_area_mm2': round(mean_area, 1),
        'stenosis_ratio': round(stenosis_ratio, 3),
        'stenosis_pct': round(stenosis_pct, 1),
        'ap_min_mm': round(min(ap_diams), 1),
        'ap_max_mm': round(max(ap_diams), 1),
        'ap_mean_mm': round(np.mean(ap_diams), 1),
        'lat_min_mm': round(min(lat_diams), 1),
        'lat_max_mm': round(max(lat_diams), 1),
        'lat_mean_mm': round(np.mean(lat_diams), 1),
        'ap_lat_ratio_mean': round(np.mean(ap_lat_ratios), 2),
        'ap_at_narrowest_mm': round(ap_diams[min_idx], 1),
        'lat_at_narrowest_mm': round(lat_diams[min_idx], 1),
        'ap_lat_ratio_at_narrowest': round(ap_lat_ratios[min_idx], 2),
        'rl_dev_mean_mm': round(np.mean(rl_devs), 1),
        'rl_dev_max_mm': round(max(rl_devs, key=abs), 1),
        'rl_dev_range_mm': round(max(rl_devs) - min(rl_devs), 1),
        'rl_dev_at_narrowest_mm': round(rl_devs[min_idx], 1),
        'sphericity': round(sphericity, 4),
        'compactness': round(compactness, 4),
        'elongation': round(elongation, 2),
        'equiv_diameter_mean_mm': round(equiv_diam, 1),
        'gradient_max_narrowing_mm2_per_mm': round(min(gradients), 1) if gradients else 0,
        'gradient_max_widening_mm2_per_mm': round(max(gradients), 1) if gradients else 0,
        'gradient_mean_abs_mm2_per_mm': round(np.mean(np.abs(gradients)), 1) if gradients else 0,
        'area_skewness': round(area_skewness, 2),
        'area_kurtosis': round(area_kurt, 2),
        'area_cv': round(area_cv, 2),
        'area_q10_mm2': round(area_q10, 1),
        'area_q25_mm2': round(area_q25, 1),
        'area_median_mm2': round(float(area_median), 1),
        'area_q75_mm2': round(area_q75, 1),
        'narrowest_rel_pos': round(narrowest_rel_pos, 2),
        'ref_area_mm2': round(ref_area, 1),
        'norm_min_area': round(norm_min, 3),
        'norm_mean_area': round(norm_mean, 2),
        'norm_area_q10': round(float(norm_q10), 3),
        'norm_area_q25': round(float(norm_q25), 3),
        'norm_area_median': round(float(norm_median), 3),
        'norm_stenosis_pct': round(norm_stenosis_pct, 1),
        'norm_volume_mm': round(norm_volume_mm, 1),
        'norm_mean_upper': round(norm_mean_upper, 3),
        'norm_mean_lower': round(norm_mean_lower, 3),
        'upper_lower_ratio': round(upper_lower_ratio, 2),
        'norm_elongation': round(norm_elongation, 2),
        'circularity_mean': round(circularity_mean, 4),
        'circularity_min': round(circularity_min, 4),
        'circularity_at_narrowest': round(circularity_at_narrowest, 4),
        'hydraulic_diam_mean_mm': round(hydraulic_diam_mean, 2),
        'hydraulic_diam_min_mm': round(hydraulic_diam_min, 2),
        'hydraulic_diam_at_narrowest_mm': round(hydraulic_diam_at_narrowest, 2),
        'narrowed_length_50pct_mm': round(narrowed_len_50, 1),
        'narrowed_length_25pct_mm': round(narrowed_len_25, 1),
        'narrowed_fraction_50pct': round(narrowed_frac_50, 3),
        'len_below_25mm2_mm': round(len_below_25mm2, 1),
        'len_below_50mm2_mm': round(len_below_50mm2, 1),
        'len_below_75mm2_mm': round(len_below_75mm2, 1),
        'len_below_100mm2_mm': round(len_below_100mm2, 1),
        'len_below_25pct_max_mm': round(len_below_25pct_max, 1),
        'len_below_33pct_max_mm': round(len_below_33pct_max, 1),
        'len_below_50pct_max_mm': round(len_below_50pct_max, 1),
        'rl_dev_max_abs_mm': round(rl_dev_max_abs, 1),
        'rl_dev_mean_abs_mm': round(rl_dev_mean_abs, 1),
        'rl_dev_at_narrowest_abs_mm': round(rl_dev_at_narrowest_abs, 1),
        'tortuosity': round(tortuosity_val, 4),
        'flatness_pca': round(flatness_pca, 4),
        'sa_vol_ratio': round(sa_vol_ratio, 4),
        'islands_removed': islands_removed,
    }


def main():
    nii_files = sorted(REF_DIR.glob("*a.nii.gz"))
    cases = []
    for f in nii_files:
        m = re.match(r'(\d+)a\.nii\.gz', f.name)
        if m:
            num = int(m.group(1))
            cases.append((f"case{num:03d}", f))

    print(f"Found {len(cases)} reference segmentation files")

    results, failed = [], []
    for i, (cid, mf) in enumerate(cases, 1):
        try:
            r = analyze_case(cid, mf)
            if r:
                results.append(r)
                if i % 25 == 0 or i == len(cases):
                    print(f"  [{i}/{len(cases)}] {cid}: Vol={r['volume_ml']:.1f} mL, "
                          f"min CSA={r['min_area_mm2']:.0f} mm²")
            else:
                failed.append(cid)
        except Exception as e:
            print(f"  ERROR {cid}: {e}")
            failed.append(cid)

    OUT_DIR.mkdir(exist_ok=True)
    with open(CSV_FILE, 'w', newline='') as fp:
        w = csv.DictWriter(fp, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for row in sorted(results, key=lambda r: r['case_id']):
            w.writerow(row)

    print(f"\n{'='*60}")
    print(f"Saved: {CSV_FILE}")
    print(f"Analyzed: {len(results)}  Failed: {len(failed)}")
    if failed:
        print(f"Failed: {failed}")


if __name__ == "__main__":
    main()
