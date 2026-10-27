#!/usr/bin/env python3
"""Figure 1 - CONSORT-style patient flow for the airway morphometry + DL paper.

SINGLE-STEP integrated chain (senior author, 2026-08-15): 535 -> 213 excluded -> 322,
concrete criteria, no companion-study references.

Verified at build time:
  535  clinical master workbook; abs 373 positive / 162 negative
  337  staged = rows of the expert review workbook (internal
       bookkeeping step, not shown in the figure)
  15   staging exclusions from cell fills: intubated 5 (gray), insufficient/
       missing data 9 (red fill plus one unfilled row), not completed 1 (staged
       but no final mask)
  322  analyzed (222/100), cross-checked against airway_morphology_merged.csv

Exclusion box: 213 = no surgical confirmation 63 + non-standard protocol 26
+ insufficient/missing imaging data 119 + intubation 5. The merged 119
(= 213 - 63 - 26 - 5) folds together the parent chain's imaging-data bucket,
the unselected abscess-negatives, the 9 staging data exclusions, and the one
uncompleted segmentation. NOTE: merging made the arm assignment of the 6
accession-unmapped cases irrelevant (it cancels between sub-buckets); the
only remaining assumption, pending the annotator, is that the 63 and 26 buckets
contributed no cases to the airway sample (the two staged cases outside the
segmentation cohort are taken to come from the imaging-data bucket; near-certain
for the DICOM-only one). Also pending the annotator: one intubation note ambiguous.

Output: tables_figures/Figure1_draft.png / .pdf
"""
import os
import json
from pathlib import Path
import re
import pandas as pd
from openpyxl import load_workbook
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

AIRWAY = str(Path(os.environ.get("AIRWAY_ROOT", Path(__file__).resolve().parents[1])))
_CFG_FILE = Path(AIRWAY) / "paths_local.json"  # untracked: maps generic keys to the local folder layout (README, "Local layout")
_CFG = json.loads(_CFG_FILE.read_text()) if _CFG_FILE.exists() else {}
CLINICAL_XLSX = os.environ.get("CLINICAL_MASTER_XLSX", str(Path(AIRWAY) / _CFG.get("clinical_master_xlsx", "clinical_master.xlsx")))
OUT = f"{AIRWAY}/manuscript_drafts/tables_figures/Figure1_draft"

# ---------- derive and assert counts ----------
mri = pd.read_excel(CLINICAL_XLSX)
n_parent = len(mri)
abs_counts = mri["abs"].value_counts()
assert n_parent == 535 and abs_counts[1] == 373 and abs_counts[0] == 162, (
    n_parent, dict(abs_counts))

ws = load_workbook(Path(AIRWAY) / _CFG.get("expert_review_xlsx", "expert_review_log.xlsx"))[_CFG.get("expert_review_sheet", "review")]

def fill_of(cell):
    f = cell.fill
    if f and f.fill_type == "solid" and isinstance(f.start_color.rgb, str):
        return f.start_color.rgb[-6:]
    return None

final_ids = set(pd.read_csv(
    f"{AIRWAY}/final_analysis/airway_morphology_merged.csv").case_id)

def to3(cid):
    num = int(re.match(r"case0*(\d+)$", cid).group(1))
    return f"case{num:03d}"

intubated, data_missing, not_completed, included = [], [], [], []
for row in ws.iter_rows(min_row=2):
    if not row[0].value:
        continue
    cid = str(row[0].value).strip()
    color = fill_of(row[0])
    note = str(row[2].value or "")
    if color == "A5A5A5":
        intubated.append(cid)
    elif color == "FF0000" or (color is None and "vajavainen" in note):
        data_missing.append(cid)
    elif to3(cid) in final_ids:
        included.append(cid)
    else:
        not_completed.append(cid)  # staged ok but never analyzed (one case)

n_staged = len(intubated) + len(data_missing) + len(not_completed) + len(included)
n_not_staged = n_parent - n_staged
assert n_staged == 337 and n_not_staged == 198, (n_staged, n_not_staged)
assert len(included) == 322 == len(final_ids), len(included)
assert (len(intubated), len(data_missing), len(not_completed)) == (5, 9, 1), (
    intubated, data_missing, not_completed)

# exclusion breakdown: segmentation-chain buckets, then single-step merge
N_NO_SURG = 63    # MRI abscess without surgical confirmation
N_PROTOCOL = 26   # non-standard MRI protocol
# merged imaging-data bucket = everything else (parent-chain data bucket,
# unselected negatives, staging data exclusions, uncompleted segmentation);
# invariant to the arm assignment of the 6 accession-unmapped cases
n_excluded = n_parent - len(included)
n_data_merged = n_excluded - N_NO_SURG - N_PROTOCOL - len(intubated)
assert n_excluded == 213 and n_data_merged == 119, (n_excluded, n_data_merged)
print(f"derived: {n_parent} -> excl {n_excluded} "
      f"({N_NO_SURG}+{N_PROTOCOL}+{n_data_merged}+{len(intubated)}) -> "
      f"{len(included)}")

# ---------- draw ----------
FS = 11.5
LH = 2.55

fig, ax = plt.subplots(figsize=(9.5, 12.5))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

def bold(s):
    return r"$\bf{" + s.replace(" ", r"\ ") + r"}$"

def draw_box(x0, y0, x1, y1, lines):
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="white",
                           edgecolor="black", linewidth=1.8, zorder=2))
    y = y1 - 1.6 - LH / 2
    for line in lines:
        ax.text(x0 + 1.8, y, line, ha="left", va="center",
                fontsize=FS, zorder=3)
        y -= LH

def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                 mutation_scale=22, linewidth=2.4, color="black", zorder=1))

draw_box(2, 83, 98, 98, [
    bold(f"{n_parent} patients") + " with",
    "* Emergency neck MRI between April 1, 2013, and August 30, 2021",
    "* Final clinical diagnosis of an acute neck infection by an",
    "  otorhinolaryngologist or an oral and maxillofacial surgeon",
    "* Sufficient demographic, clinical, and laboratory data",
])

draw_box(44, 55, 98, 75.5, [
    bold(f"{n_excluded} patients") + " excluded",
    "* Abscess on MRI without surgical",
    f"  confirmation (N = {N_NO_SURG})",
    f"* Non-standard MRI protocol (N = {N_PROTOCOL})",
    "* Insufficient or missing imaging",
    f"  data (N = {n_data_merged})",
    f"* Intubation before imaging (N = {len(intubated)})",
])

draw_box(2, 20, 98, 46, [
    bold(f"{len(included)} patients") + " analyzed for upper airway morphometry",
    "(222 abscess-positive, 100 abscess-negative)",
    "* Manual upper airway segmentation on post-contrast T1-weighted MRI",
    "* Five prespecified morphometric features and airway profile analysis",
    "* Interobserver agreement in 30 cases (expert and novice readers)",
    "* Deep learning-based automated segmentation using nnU-Net",
    "* Segmentation performance: DSC, HD95; feature agreement: ICC",
    "* Clinical associations: ICU admission, prolonged stay, severity markers",
])

arrow(30, 83, 30, 46.6)      # parent -> analyzed
arrow(30, 65.3, 43.4, 65.3)  # -> excluded

fig.savefig(OUT + ".png", dpi=300, bbox_inches="tight")
fig.savefig(OUT + ".pdf", bbox_inches="tight")
print("wrote", OUT + ".png/.pdf")
