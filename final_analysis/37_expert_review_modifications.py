#!/usr/bin/env python3
"""
Revision analysis (EJR-D-26-04240, reviewer point 2a): what fraction of the
322 reference segmentations was modified at neuroradiologist review.
Source: the expert review workbook (paths_local.json) — column B = reviewer check
(1 = reviewed), column C = reviewer's free-text correction request per case
(Finnish). Rows with exclusion notes (intubated / insufficient data) are not
analysed cases. A request is classified as ENDPOINT (add/remove one or more
slices at the cranial or caudal boundary: 'alhaalta/ylhäältä ... pois',
'alas/ylös ... lisää', 'leike ... pois/mukaan' with directional words) and/or
IN-PLANE (edits within a slice: strands 'juoste/rihma', pockets 'tasku',
'pienemmäksi', 'etuosa/takaosa/reuna/sivu ... pois', epiglottis/uvula).
Output: 37_expert_review_modifications.csv / .txt
"""
import os, re
import json
from pathlib import Path
import pandas as pd
from openpyxl import load_workbook

BASE = Path(os.environ.get("AIRWAY_ROOT", Path(__file__).resolve().parents[1]))
_CFG_FILE = BASE / "paths_local.json"  # untracked: maps generic keys to the local folder layout (README, "Local layout")
_CFG = json.loads(_CFG_FILE.read_text()) if _CFG_FILE.exists() else {}
FA = BASE / "final_analysis"
ws = load_workbook(BASE / _CFG.get("expert_review_xlsx", "expert_review_log.xlsx"), data_only=True)[_CFG.get("expert_review_sheet", "review")]
EXCL = ("intuboitu", "vajavainen", "puuttuu", "excl")
# endpoint phrases: direction word (+ optional count/slice word) + add/remove verb, or explicit boundary wording
END_RE = re.compile(r"(alhaalta|ylhäältä|alas|ylös|alhaalla|ylhäällä)[\w\s]{0,25}?(pois|lisää|lisätä|lisästä|mukaan|yhden|yksi|1|2)|"
                    r"(ylin|alin|ylimmän|alimmainen|alimmaisen)\s*leik|yläraja|alaraja|alkaen leike|ylärajan taso|"
                    r"(leike|leikettä)\s*(mukaan|lisää)|(yksi|yhden)\s*leik\w*\s*(pois|lisää|mukaan)|ylhäältä yksi|alhaalta vielä")
# in-plane edit vocabulary (strands, pockets, size, anatomical parts, slice-number + edit)
INP_WORDS = ("juoste", "juonte", "rihma", "tasku", "pienemm", "pienent", "pieneksi", "pienemmiksi", "etuosa", "etuosi", "takaosa", "takimmai", "reuna", "sivu ",
             "sivujuo", "osio", "epiglott", "uvula", "fossa", "laajent", "laajastikin", "lavea", "pensseli", "minimaal", "vajaa", "epätark",
             "tiplu", "rähm", "trimm", "tarkemmin", "kolonen", "suulake", "isompi", "keskustaa", "yhtenäistä", "epäloog", "ilmatask", "ilmattomat")
INP_RE = re.compile(r"\b\d{1,2}\s*(ja|–|-|,)?\s*(\d{1,2})?\s*(oik|vas|etu|tak|reun|sivu|juost|pienem|epiglott|uvula|fossa|anterior|posterior|keskell|segment|tarkemmin|minimaal|laajent|isompi|täytyy|jättäisin|on |voi |pois|ihan|selv|vähän|ei )")
rows = []
for r in range(2, ws.max_row + 1):
    cid = ws.cell(row=r, column=1).value
    if not cid: continue
    chk = ws.cell(row=r, column=2).value; com = ws.cell(row=r, column=3).value
    if chk not in (1, 1.0): continue
    c = (str(com) if com not in (None, "") else "").strip()
    if any(w in c.lower() for w in EXCL): c = ""
    low = c.lower()
    endpoint = bool(END_RE.search(low))
    low_wo_end = END_RE.sub(" ", low)
    inplane = any(w in low_wo_end for w in INP_WORDS) or bool(INP_RE.search(low_wo_end))
    rows.append(dict(case_id=str(cid), modified=bool(c), endpoint=endpoint and bool(c), inplane=inplane and bool(c), comment=c))
T = pd.DataFrame(rows)
T.to_csv(FA / "37_expert_review_modifications.csv", index=False)
n = len(T); mod = T.modified.sum(); eo = (T.endpoint & ~T.inplane).sum(); ip = T.inplane.sum(); both = (T.endpoint & T.inplane).sum()
L = [f"Reviewed cases: {n}", f"Modified after review: {mod} ({100*mod/n:.1f}%)", f"  endpoint-only (craniocaudal extent, typically one slice): {eo} ({100*eo/n:.1f}%)",
     f"  any in-plane boundary edit: {ip} ({100*ip/n:.1f}%)  [of which also endpoint: {both}]", f"Unchanged: {n-mod} ({100*(n-mod)/n:.1f}%)",
     "", "Endpoint-only requests, distinct wording (for audit):"] + [f"  {k} x{v}" for k, v in T[T.endpoint & ~T.inplane].comment.value_counts().items()] + ["", "In-plane edit requests (for audit):"] + [f"  {r.case_id}: {r.comment}" for r in T[T.inplane].itertuples()] + ["", "Modified but unclassified:"] + [f"  {r.case_id}: {r.comment}" for r in T[T.modified & ~T.endpoint & ~T.inplane].itertuples()]
(FA / "37_expert_review_modifications.txt").write_text("\n".join(L)); print("\n".join(L[:6])); print(len(T[T.inplane]), "in-plane listed")
