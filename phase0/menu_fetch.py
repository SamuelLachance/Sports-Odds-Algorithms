"""Fetch every archived daily 'Menu' workbook (Google Sheets) as xlsx.

Index: scratchpad/menu_archive_index.json (505 date->sheet-id entries,
2025-03-01 .. 2026-07-23). Raw xlsx files land in scratchpad/menu_raw/
(NOT the repo — market data stays out of the model pipeline; only parsed
research artifacts get committed). Resumable: skips files already present.
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
import urllib.request

SP = (r"C:/Users/Admin/AppData/Local/Temp/claude/"
      r"C--Users-Admin-Projects-Sports-Odds-Algorithms/"
      r"c82ef32b-d2a5-4470-9d81-a522f1e0284f/scratchpad")
RAW = SP + "/menu_raw"
os.makedirs(RAW, exist_ok=True)

idx = json.load(open(SP + "/menu_archive_index.json"))
seen = set()
todo = []
for e in idx:
    if e["id"] not in seen:
        seen.add(e["id"])
        todo.append(e)
print(f"{len(todo)} unique daily sheets", flush=True)

fails = []
t0 = time.time()
for k, e in enumerate(todo):
    out = f"{RAW}/{e['id']}.xlsx"
    if os.path.exists(out) and os.path.getsize(out) > 2000:
        continue
    url = f"https://docs.google.com/spreadsheets/d/{e['id']}/export?format=xlsx"
    ok = False
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            data = urllib.request.urlopen(req, timeout=45).read()
            if len(data) > 2000 and data[:2] == b"PK":
                open(out, "wb").write(data)
                ok = True
                break
            raise ValueError(f"bad payload {len(data)}b")
        except Exception as ex:  # noqa: BLE001
            if attempt == 2:
                fails.append({"date": str(e["date"]), "id": e["id"], "err": str(ex)[:120]})
            else:
                time.sleep(5 * (attempt + 1))
    time.sleep(0.7)
    if (k + 1) % 25 == 0:
        print(f"  {k+1}/{len(todo)}  [{time.time()-t0:.0f}s]  fails={len(fails)}", flush=True)

json.dump(fails, open(SP + "/menu_fetch_fails.json", "w"), indent=1)
n_have = sum(1 for e in todo if os.path.exists(f"{RAW}/{e['id']}.xlsx"))
print(f"done: {n_have}/{len(todo)} fetched, {len(fails)} failed "
      f"[{time.time()-t0:.0f}s]", flush=True)
