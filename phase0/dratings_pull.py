"""Reconstruct DRatings MLB predictions from Wayback-archived game pages.

CDX inventory: scratchpad/drat_cdx_full.json (unique UUID game pages).
Fetches the EARLIEST capture per page (minimizes post-game recompute risk),
parses title 'Away at Home - MM/DD/YY' + the two win percentages.
Output: data/dratings_mlb.csv (date, away, home, p_away, p_home, captured).
Resumable via scratchpad/drat_done.json.
"""
from __future__ import annotations

import csv
import json
import os
import re
import time
import urllib.request

SP = (r"C:/Users/Admin/AppData/Local/Temp/claude/"
      r"C--Users-Admin-Projects-Sports-Odds-Algorithms/"
      r"c82ef32b-d2a5-4470-9d81-a522f1e0284f/scratchpad")

d = json.load(open(SP + "/drat_cdx_full.json"))
rows = [r for r in d[1:] if len(r[2].rstrip("/").split("/")[-1]) > 30]
earliest = {}
for k, ts, url in rows:
    if k not in earliest or ts < earliest[k][0]:
        earliest[k] = (ts, url)
todo = sorted(earliest.values())
print(f"{len(todo)} unique game pages", flush=True)

done = {}
if os.path.exists(SP + "/drat_done.json"):
    done = json.load(open(SP + "/drat_done.json"))

TITLE = re.compile(r"<title>\s*(.*?)\s+at\s+(.*?)\s*-\s*(\d\d)/(\d\d)/(\d\d)\s*-", re.S)
PCT = re.compile(r"(\d\d?(?:\.\d)?)%")

out_rows = list(done.values())
fails = 0
t0 = time.time()
for k, (ts, url) in enumerate(todo):
    key = url.rstrip("/").split("/")[-1]
    if key in done:
        continue
    wb = f"http://web.archive.org/web/{ts}id_/{url}"
    try:
        req = urllib.request.Request(wb, headers={"User-Agent": "Mozilla/5.0"})
        h = urllib.request.urlopen(req, timeout=60).read().decode("utf-8", "replace")
        m = TITLE.search(h)
        pcts = PCT.findall(h)
        if m and len(pcts) >= 2:
            away, home = m.group(1).strip(), m.group(2).strip()
            yy = 2000 + int(m.group(5))
            rec = {"date": f"{yy}-{m.group(3)}-{m.group(4)}",
                   "away_nick": re.sub(r"<[^>]+>", "", away)[:30],
                   "home_nick": re.sub(r"<[^>]+>", "", home)[:30],
                   "p_away": float(pcts[0]) / 100.0, "p_home": float(pcts[1]) / 100.0,
                   "captured": ts}
            done[key] = rec
            out_rows.append(rec)
        else:
            fails += 1
    except Exception:  # noqa: BLE001
        fails += 1
        time.sleep(3)
    time.sleep(1.2)
    if (k + 1) % 50 == 0:
        json.dump(done, open(SP + "/drat_done.json", "w"))
        print(f"  {k+1}/{len(todo)}  parsed={len(done)}  fails={fails}  "
              f"[{time.time()-t0:.0f}s]", flush=True)

json.dump(done, open(SP + "/drat_done.json", "w"))
with open("data/dratings_mlb.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=["date", "away_nick", "home_nick",
                                       "p_away", "p_home", "captured"])
    w.writeheader()
    w.writerows(sorted(done.values(), key=lambda r: r["date"]))
from collections import Counter
yr = Counter(r["date"][:4] for r in done.values())
print(f"\n{len(done)} predictions parsed, {fails} failed -> data/dratings_mlb.csv")
print("by year:", dict(sorted(yr.items())))
