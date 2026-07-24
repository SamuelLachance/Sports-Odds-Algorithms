"""Pull OddsShark MLB daily scores API for 2023-2026: odds, votes, roof, teams.

History floor ~2023 (earlier dates return HTTP 500). Output
data/oddsshark_mlb.csv, one row per game. Research artifact for the
market-signal predictivity study — market fields are evaluation-layer only,
NEVER model inputs (constitution).
"""
from __future__ import annotations

import csv
import datetime as dt
import gzip
import json
import time
import urllib.request

HEADERS = {"Accept": "application/json", "Accept-Encoding": "gzip",
           "Referer": "https://www.oddsshark.com/mlb/scores",
           "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/118.0.0.0 Safari/537.36")}

def fetch(date):
    url = f"https://www.oddsshark.com/api/scores/mlb/{date}?_format=json"
    raw = urllib.request.urlopen(
        urllib.request.Request(url, headers=HEADERS), timeout=30).read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw)

def safe(d, *ks):
    for k in ks:
        d = d.get(k) if isinstance(d, dict) else None
    return d

SPANS = [("2023-03-30", "2023-11-02"), ("2024-03-20", "2024-11-01"),
         ("2025-03-18", "2025-11-02"), ("2026-03-25", "2026-07-23")]

rows, fails = [], []
t0 = time.time()
for lo, hi in SPANS:
    d = dt.date.fromisoformat(lo)
    end = dt.date.fromisoformat(hi)
    while d <= end:
        ds = d.isoformat()
        try:
            sc = fetch(ds).get("scores") or []
            for g in sc:
                h, a = safe(g, "teams", "home") or {}, safe(g, "teams", "away") or {}
                rows.append({
                    "date": ds,
                    "home": safe(h, "names", "abbreviation") or "",
                    "away": safe(a, "names", "abbreviation") or "",
                    "home_name": safe(h, "names", "name") or "",
                    "away_name": safe(a, "names", "name") or "",
                    "status": g.get("eventStatus") or "",
                    "hs": h.get("score"), "as": a.get("score"),
                    "ml_h": h.get("moneyLine"), "ml_a": a.get("moneyLine"),
                    "spread_h": h.get("spread"), "spread_h_price": h.get("spreadPrice"),
                    "total": g.get("total"),
                    "over_price": g.get("overPrice"), "under_price": g.get("underPrice"),
                    "votes_h": h.get("votes"), "votes_a": a.get("votes"),
                    "over_votes": g.get("overVotes"), "under_votes": g.get("underVotes"),
                    "roof": safe(g, "stadium", "roof") or "",
                    "stadium": safe(g, "stadium", "stadium") or "",
                    "rec_h": h.get("record") or "", "rec_a": a.get("record") or "",
                })
        except Exception as ex:  # noqa: BLE001
            fails.append((ds, str(ex)[:60]))
        time.sleep(0.6)
        d += dt.timedelta(days=1)
    print(f"  span {lo[:4]} done: {len(rows)} rows, {len(fails)} failed days "
          f"[{time.time()-t0:.0f}s]", flush=True)

with open("data/oddsshark_mlb.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"\n{len(rows)} game rows, {len(fails)} failed days -> data/oddsshark_mlb.csv")
if fails:
    print("failed days sample:", fails[:8])
