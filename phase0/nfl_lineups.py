"""Dynamic smart projected lineups (MLB-parity): each build refreshes the live
nflverse depth charts and resolves every team's projected starting 11s.

Resolution per depth-chart slot (LT/LG/C/RG/RT, QB/RB/WR/TE, LDE/NT/WLB/CBs/...):
  1. take the latest snapshot's rank order for the slot
  2. drop players not on the current 2026 roster (cuts/trades leave stale rows)
  3. drop players ruled Out/Doubtful/IR when an injury report exists (in-season)
  4. SMART flip: if the next man has BOTH much more real 2025 usage
     (snap share +0.35) AND a better per-play TrueSkill rating (+5), the depth
     chart is presumed stale and he is promoted (marked src="usage")
Everything is attached to site/data/nfl.json: teams[].lineup {off, def, dt},
with snap share + rating riding along for the game-page display. QB1 fields are
re-derived from the same resolution so all surfaces agree.
"""
from __future__ import annotations

import csv
import json
import os
import urllib.request
from collections import defaultdict

DC = "data/depth_charts_2026.csv"
DC_URL = ("https://github.com/nflverse/nflverse-data/releases/download/"
          "depth_charts/depth_charts_2026.csv")
ST_SLOTS = {"KR", "PR", "PK", "P", "LS", "H", "KO", "K"}

# ---- refresh the depth charts (dynamic: newest snapshot every build) ----
try:
    tmp = DC + ".tmp"
    urllib.request.urlretrieve(DC_URL, tmp)
    if os.path.getsize(tmp) > 1_000_000:
        os.replace(tmp, DC)
        print("depth charts refreshed from nflverse")
except Exception as e:  # noqa: BLE001 — build degrades to the local copy
    print(f"depth chart refresh skipped ({type(e).__name__}); using local copy")

payload = json.load(open("site/data/nfl.json"))
players = payload["players"]

# current rosters (drop cut/traded players from stale depth rows)
FR = {"JAC": "JAX", "WSH": "WAS", "AZ": "ARI", "LAR": "LA"}
on_roster = defaultdict(set)
for r in csv.DictReader(open("data/roster_2026.csv", encoding="utf-8")):
    if r.get("gsis_id") and r.get("team"):
        on_roster[FR.get(r["team"], r["team"])].add(r["gsis_id"])

# injury report (in-season only; preseason file doesn't exist yet)
ruled_out = set()
try:
    for r in csv.DictReader(open("data/inj_2026.csv", encoding="utf-8")):
        if (r.get("report_status") or "") in ("Out", "Doubtful"):
            ruled_out.add(r.get("gsis_id"))
    print(f"injury report: {len(ruled_out)} ruled out/doubtful")
except FileNotFoundError:
    pass

# 2025 usage (snap share) + ratings from the payload's player table
share_of = {pid: (p.get("snap_share") or 0.0) for pid, p in players.items()}
rating_of = {pid: (p["rating"]["r"] if p.get("rating") else None)
             for pid, p in players.items()}

# ---- latest snapshot per team ----
latest = defaultdict(str)
for r in csv.DictReader(open(DC, encoding="utf-8")):
    if r["dt"] > latest[r["team"]]:
        latest[r["team"]] = r["dt"]
slots = defaultdict(list)          # (team, slot_key) -> [(rank, gsis, name, pos)]
for r in csv.DictReader(open(DC, encoding="utf-8")):
    t = FR.get(r["team"], r["team"])
    if r["dt"] != latest[r["team"]] or not r["gsis_id"]:
        continue
    if r["pos_abb"] in ST_SLOTS or r["pos_grp"] == "Special Teams":
        continue
    side = "off" if r["pos_grp"] not in ("Base 3-4 D", "Base 4-3 D") else "def"
    key = (t, side, r["pos_grp"], r["pos_slot"], r["pos_abb"])
    slots[key].append((int(r["pos_rank"]), r["gsis_id"], r["player_name"]))

n_flip = 0
lineups = defaultdict(lambda: {"off": [], "def": []})
for (t, side, grp, slot, pos), cand in sorted(slots.items(),
                                              key=lambda kv: (kv[0][0], kv[0][1], int(kv[0][3]))):
    cand.sort()
    ok = [c for c in cand if c[1] in on_roster.get(t, ()) and c[1] not in ruled_out]
    if not ok:
        ok = [c for c in cand if c[1] not in ruled_out] or cand
    top = ok[0]
    src = "depth"
    if len(ok) > 1:                             # smart flip: usage + quality contradict
        nxt = ok[1]
        s0, s1 = share_of.get(top[1], 0.0), share_of.get(nxt[1], 0.0)
        r0, r1 = rating_of.get(top[1]), rating_of.get(nxt[1])
        if s1 - s0 > 0.35 and r0 is not None and r1 is not None and r1 > r0 + 5:
            top, src = nxt, "usage"
            n_flip += 1
    lineups[t][side].append({
        "slot": pos, "id": top[1], "name": top[2], "src": src,
        "share": round(share_of.get(top[1], 0.0), 2),
        "r": rating_of.get(top[1]),
    })

# serve-time ratings power: projected lineup aggregated through the v6 engine
# states (2026 boundary applied: mu shrunk 1/3, sigma widened)
ts_state = {}
sal26 = {}
try:
    ts_state = json.load(open("data/nfl_ts_state.json"))
    sal26 = json.load(open("data/nfl_sal2026.json"))
except FileNotFoundError:
    pass
def rpow_of(lu):
    tot = 0.0
    for e in lu["off"] + lu["def"]:
        st = ts_state.get(e["id"])
        if not st:
            continue
        tgt = 25.0 + sal26.get(e["id"], 0.0)
        mu_ = tgt + (st[0] - tgt) * (2.0 / 3.0)
        s2_ = min(st[1] + 1.5 ** 2, (25.0 / 3.0) ** 2)
        v = max(-3.0, min(3.0, mu_ - 25.0)) * (1.0 / (1.0 + s2_))
        tot += (e.get("share") or 0.05) * v
    return tot

QBN = {}
rp = {}
for t, lu in lineups.items():
    if t not in payload["teams"]:
        continue
    payload["teams"][t]["lineup"] = {**lu, "dt": latest.get(t, "")[:10]}
    if ts_state:
        rp[t] = rpow_of(lu)
    qb = next((e for e in lu["off"] if e["slot"] == "QB"), None)
    if qb:
        payload["teams"][t]["qb1"] = {"id": qb["id"], "name": qb["name"]}
        QBN[t] = qb["name"]
if rp:
    order = sorted(rp, key=lambda t: -rp[t])
    for rk, t in enumerate(order, 1):
        payload["teams"][t]["rpow"] = round(rp[t], 2)
        payload["teams"][t]["rpow_rank"] = rk
    print("ratings power top-5:", [(t, round(rp[t], 2)) for t in order[:5]])
sizes = {t: (len(l["off"]), len(l["def"])) for t, l in lineups.items()}
assert len(lineups) == 32 and all(9 <= o <= 13 and 10 <= d <= 13 for o, d in sizes.values()), sizes

# ---- regenerate data/nfl_qb2026.json from the SAME live QB1 resolution ----
# (formerly a frozen 2026-07-23 snapshot with no generator; the serve reads it
# at nfl_season_serve.py for the QB-Elo 2026 ids + payload name map, so it now
# tracks every depth-chart refresh. If the download fails this run degrades to
# the local depth-chart copy; if the derivation ever comes back incomplete the
# previously written file persists on disk as the serve's fallback.)
qb26 = {}
for t in sorted(lineups):
    qb = next((e for e in lineups[t]["off"] if e["slot"] == "QB"), None)
    if qb:
        qb26[t] = {"gsis": qb["id"], "name": qb["name"],
                   "how": "depth_chart" if qb["src"] == "depth" else "usage_flip"}
if len(qb26) == 32:
    dt_max = max(latest.values())
    obj = {"source": "depth_charts_2026",
           "qb": qb26,
           "notes": [f"regenerated by phase0/nfl_lineups.py from depth_charts_2026.csv "
                     f"latest snapshot per team (max dt {dt_max}); roster/injury "
                     f"filtered + usage smart-flip, identical to the served lineup QB1s."]}
    json.dump(obj, open("data/nfl_qb2026.json.tmp", "w"), indent=1)
    os.replace("data/nfl_qb2026.json.tmp", "data/nfl_qb2026.json")
    print(f"nfl_qb2026.json regenerated (32 QB1s, snapshot max dt {dt_max[:10]})")
else:
    print(f"nfl_qb2026.json NOT regenerated ({len(qb26)}/32 QB1s); keeping prior file")

json.dump(payload, open("site/data/nfl.json", "w"))
print(f"lineups attached for {len(lineups)} teams ({n_flip} usage flips); "
      f"sample SEA off: {[(e['slot'], e['name']) for e in lineups['SEA']['off']][:6]}")
