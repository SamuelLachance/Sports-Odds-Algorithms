"""DRatings MLB predictions v2: UUID discovery via Wayback, content via LIVE site.

Game pages stay live forever at dratings.com/predictor/mlb-baseball-predictions/<uuid5>
(Samuel's find). Wayback is used only as a UUID directory:
  (a) every archived MAIN predictor-page capture (~15 game links each),
  (b) every game-page URL in the CDX index.
All unique UUIDs are then fetched from the live site (title + win probs).
Resumable via scratchpad/drat_done.json (shares format with v1).
Output: data/dratings_mlb.csv
"""
from __future__ import annotations

import csv
import gzip
import json
import os
import re
import time
import urllib.request

SP = (r"C:/Users/Admin/AppData/Local/Temp/claude/"
      r"C--Users-Admin-Projects-Sports-Odds-Algorithms/"
      r"c82ef32b-d2a5-4470-9d81-a522f1e0284f/scratchpad")
UUID_RE = re.compile(r"mlb-baseball-predictions/([0-9a-f]{8}-[0-9a-f]{4}-5[0-9a-f]{3}-[0-9a-f]{4}-[0-9a-f]{12})")
TITLE = re.compile(r"<title>\s*(.*?)\s+at\s+(.*?)\s*-\s*(\d\d)/(\d\d)/(\d\d)\s*-", re.S)
PCT = re.compile(r"(\d\d?(?:\.\d)?)%")

def get(url, timeout=40):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0",
                                               "Accept-Encoding": "gzip"})
    raw = urllib.request.urlopen(req, timeout=timeout).read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return raw.decode("utf-8", "replace")

# ---- 1. UUIDs from game-page CDX index ----
uuids = set()
d = json.load(open(SP + "/drat_cdx_full.json"))
for r in d[1:]:
    m = UUID_RE.search(r[2])
    if m:
        uuids.add(m.group(1))
print(f"game-page CDX uuids: {len(uuids)}", flush=True)

# ---- 2. UUIDs from all main-page wayback captures ----
cdx = (SP + "/drat_main_cdx.json")
if not os.path.exists(cdx):
    u = ("http://web.archive.org/cdx/search/cdx?url=dratings.com/predictor/"
         "mlb-baseball-predictions/&output=json&filter=statuscode:200&limit=5000")
    open(cdx, "w").write(get(u, timeout=60))
caps = json.load(open(cdx))[1:]
print(f"main-page captures: {len(caps)}", flush=True)
seen_caps = set()
mp_new = 0
for r in caps:
    ts = r[1]
    if ts[:8] in seen_caps:              # one capture per day is plenty
        continue
    seen_caps.add(ts[:8])
    try:
        h = get(f"http://web.archive.org/web/{ts}id_/{r[2]}", timeout=60)
        found = set(UUID_RE.findall(h))
        mp_new += len(found - uuids)
        uuids |= found
    except Exception:  # noqa: BLE001
        pass
    time.sleep(0.8)
print(f"after main-page harvest: {len(uuids)} uuids (+{mp_new} new)", flush=True)
json.dump(sorted(uuids), open(SP + "/drat_uuids.json", "w"))

# ---- 3. live-site fetch ----
done = {}
if os.path.exists(SP + "/drat_done.json"):
    done = json.load(open(SP + "/drat_done.json"))
print(f"already parsed: {len(done)}", flush=True)
fails = 0
t0 = time.time()
todo = sorted(uuids - set(done))
for k, uid in enumerate(todo):
    try:
        h = get(f"https://www.dratings.com/predictor/mlb-baseball-predictions/{uid}",
                timeout=30)
        m = TITLE.search(h)
        pcts = PCT.findall(h)
        if m and len(pcts) >= 2:
            yy = 2000 + int(m.group(5))
            done[uid] = {"date": f"{yy}-{m.group(3)}-{m.group(4)}",
                         "away_nick": re.sub(r"<[^>]+>", "", m.group(1)).strip()[:30],
                         "home_nick": re.sub(r"<[^>]+>", "", m.group(2)).strip()[:30],
                         "p_away": float(pcts[0]) / 100.0,
                         "p_home": float(pcts[1]) / 100.0, "captured": "live"}
        else:
            fails += 1
    except Exception:  # noqa: BLE001
        fails += 1
        time.sleep(2)
    time.sleep(0.5)
    if (k + 1) % 100 == 0:
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
print(f"\n{len(done)} predictions, {fails} failed -> data/dratings_mlb.csv")
print("by year:", dict(sorted(yr.items())))
