"""Salvage sweep: try to fetch every pre-2025-03-01 Menu daily sheet.

Owner purges old dailies (HTTP 410); this rescues whatever still responds.
Appends to scratchpad/menu_raw/, logs liveness by month.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections import Counter

SP = (r"C:/Users/Admin/AppData/Local/Temp/claude/"
      r"C--Users-Admin-Projects-Sports-Odds-Algorithms/"
      r"c82ef32b-d2a5-4470-9d81-a522f1e0284f/scratchpad")
RAW = SP + "/menu_raw"

idx = json.load(open(SP + "/menu_archive_index_full.json"))
old = [e for e in idx if e["date"] < "2025-03-01"]
print(f"{len(old)} pre-2025-03-01 sheets to probe", flush=True)

alive, dead = [], Counter()
t0 = time.time()
for k, e in enumerate(old):
    out = f"{RAW}/{e['id']}.xlsx"
    if os.path.exists(out) and os.path.getsize(out) > 2000:
        alive.append(e["date"])
        continue
    url = f"https://docs.google.com/spreadsheets/d/{e['id']}/export?format=xlsx"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        data = urllib.request.urlopen(req, timeout=45).read()
        if len(data) > 2000 and data[:2] == b"PK":
            open(out, "wb").write(data)
            alive.append(e["date"])
        else:
            dead[e["date"][:7]] += 1
    except urllib.error.HTTPError as ex:
        dead[e["date"][:7]] += 1
        if ex.code not in (404, 410, 403):
            print(f"  {e['date']} HTTP {ex.code}", flush=True)
            time.sleep(5)
    except Exception:  # noqa: BLE001
        dead[e["date"][:7]] += 1
    time.sleep(0.8)
    if (k + 1) % 50 == 0:
        print(f"  {k+1}/{len(old)}  alive={len(alive)}  [{time.time()-t0:.0f}s]", flush=True)

json.dump({"alive_dates": sorted(alive), "dead_by_month": dict(sorted(dead.items()))},
          open(SP + "/menu_sweep_result.json", "w"), indent=1)
print(f"\nalive: {len(alive)}  dead: {sum(dead.values())}")
print("dead by month:", dict(sorted(dead.items())))
print("alive dates:", sorted(alive)[:10], "..." if len(alive) > 10 else "")
