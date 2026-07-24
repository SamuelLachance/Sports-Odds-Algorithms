"""Pull Covers contest-consensus history (overall / expert / top-10%) for MLB.

contests.covers.com serves historical dates directly. One row per game per
variant: matchup, consensus %, pick counts. Output data/covers_mlb.csv.
Research artifact — crowd-pick data, evaluation-layer only (constitution).
"""
from __future__ import annotations

import csv
import datetime as dt
import gzip
import html as H
import re
import time
import urllib.request

VARIANTS = ["overall", "expert", "top10pct"]
SPANS = [("2021-03-15", "2021-11-03"), ("2022-04-01", "2022-11-06"),
         ("2023-03-25", "2023-11-02"), ("2024-03-15", "2024-11-01"),
         ("2025-03-15", "2025-11-02"), ("2026-03-20", "2026-07-23")]

def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0",
                                               "Accept-Encoding": "gzip"})
    raw = urllib.request.urlopen(req, timeout=30).read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return raw.decode("utf-8", "replace")

ROW = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S)
CELL = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.S)

def parse(h, date, variant, out):
    tb = re.search(r"<table[^>]*>(.*?)</table>", h, re.S)
    if not tb:
        return 0
    n = 0
    for tr in ROW.findall(tb.group(1)):
        tds = [" ".join(H.unescape(re.sub(r"<[^>]+>", " ", td)).split())
               for td in CELL.findall(tr)]
        if len(tds) < 5 or not tds[0].startswith("MLB"):
            continue
        teams = tds[0].split()[1:]
        pcts = re.findall(r"(\d{1,3})%", tds[2])
        picks = re.findall(r"\d+", tds[4])
        if len(teams) < 2 or len(pcts) < 2:
            continue
        out.append({"date": date, "variant": variant,
                    "away": teams[0], "home": teams[1],
                    "away_pct": int(pcts[0]), "home_pct": int(pcts[1]),
                    "away_picks": int(picks[0]) if picks else None,
                    "home_picks": int(picks[1]) if len(picks) > 1 else None,
                    "time": tds[1][:28]})
        n += 1
    return n

rows, fails = [], 0
t0 = time.time()
for lo, hi in SPANS:
    d = dt.date.fromisoformat(lo)
    end = dt.date.fromisoformat(hi)
    while d <= end:
        ds = d.isoformat()
        for v in VARIANTS:
            try:
                parse(get(f"https://contests.covers.com/consensus/topconsensus/mlb/{v}/{ds}"),
                      ds, v, rows)
            except Exception:  # noqa: BLE001
                fails += 1
                time.sleep(2)
            time.sleep(0.5)
        d += dt.timedelta(days=1)
    print(f"  span {lo[:4]} done: {len(rows)} rows, {fails} failed fetches "
          f"[{time.time()-t0:.0f}s]", flush=True)

with open("data/covers_mlb.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=["date", "variant", "away", "home", "away_pct",
                                       "home_pct", "away_picks", "home_picks", "time"])
    w.writeheader()
    w.writerows(rows)
from collections import Counter
print(f"\n{len(rows)} rows -> data/covers_mlb.csv")
print("by year:", dict(sorted(Counter(r['date'][:4] for r in rows).items())))
print("by variant:", dict(Counter(r['variant'] for r in rows)))
