"""Build site/data/nfl.json — the NFL GLASSBOX preseason payload (entering 2026).

Power ratings from end-of-2025 model states (Elo regressed to preseason + pass/run
unit EPA), player boards from the shipped continuous board, WOWY MVP table, and the
model card (measured numbers only).
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict

import numpy as np

coord_src = open("phase0/nfl_coord_tune.py", encoding="utf-8").read()
exec(coord_src.split("X_CUR0 = X_of(F)")[0])  # noqa: S102

# ---- pass/run unit states through end of 2025 ----
aggc = defaultdict(lambda: defaultdict(lambda: [0.0, 0, 0.0, 0]))
with open("data/nfl_duel_plays.csv", encoding="utf-8") as fh:
    rd = csv.reader(fh); hdr = next(rd); ix = {c: i for i, c in enumerate(hdr)}
    for r in rd:
        t_ = PBP_FIX.get(r[ix["posteam"]], r[ix["posteam"]])
        a = aggc[r[ix["game_id"]]][t_]
        if r[ix["passer_player_id"]] or r[ix["receiver_player_id"]]:
            a[0] += float(r[ix["epa"]]); a[1] += 1
        else:
            a[2] += float(r[ix["epa"]]); a[3] += 1
ps = pn = rs = rn = 0.0
for gid, tm in aggc.items():
    if int(gid[:4]) in DEV_YEARS:
        for t_, (a_, b_, c_, d_) in tm.items():
            ps += a_; pn += b_; rs += c_; rn += d_
LGP, LGR = ps / pn, rs / rn
dec_, pn_, sd_ = epaP["decay"], epaP["prior_n"], epaP["season_decay"]
offP = defaultdict(lambda: [0.0, 0.0]); offR = defaultdict(lambda: [0.0, 0.0])
dfaP = defaultdict(lambda: [0.0, 0.0]); dfaR = defaultdict(lambda: [0.0, 0.0])
def crate(st_, lg_): return (st_[0] + (pn_ / 2) * lg_) / (st_[1] + pn_ / 2)
prev = None
for i, g in enumerate(games):
    if prev is not None and g["season"] != prev:
        for st_ in (list(offP.values()) + list(offR.values())
                    + list(dfaP.values()) + list(dfaR.values())):
            st_[0] *= (1 - sd_); st_[1] *= (1 - sd_)
    prev = g["season"]
    tm = aggc.get(g["gid"])
    if tm:
        for t_off, opp in ((g["home"], g["away"]), (g["away"], g["home"])):
            pS, pN, rS, rN = tm.get(t_off, (0.0, 0, 0.0, 0))
            o = offP[t_off]; o[0] = dec_ * o[0] + pS; o[1] = dec_ * o[1] + pN
            o = offR[t_off]; o[0] = dec_ * o[0] + rS; o[1] = dec_ * o[1] + rN
            d2 = dfaP[opp]; d2[0] = dec_ * d2[0] + pS; d2[1] = dec_ * d2[1] + pN
            d2 = dfaR[opp]; d2[0] = dec_ * d2[0] + rS; d2[1] = dec_ * d2[1] + rN

# ---- end-2025 Elo -> 2026 preseason (one regression step) ----
from nfl_elo import run_elo as _run_elo  # noqa: E402
bp = base["params"]
R = {}
prev = None
for g in games:
    if prev is not None and g["season"] != prev:
        for t in R:
            R[t] = 1500.0 + (R[t] - 1500.0) * (1.0 - bp["regress"])
    prev = g["season"]
    rh = R.setdefault(g["home"], 1500.0)
    ra = R.setdefault(g["away"], 1500.0)
    h = 0.0 if g["neutral"] else bp["hfa"]
    p = 1.0 / (1.0 + 10 ** (-((rh + h) - ra) / 400.0))
    d = bp["k"] * (g["y"] - p)
    R[g["home"]] += d
    R[g["away"]] -= d
if games[-1]["season"] < 2026:      # else the in-walk trigger already regressed at the
    for t in R:                     # 2025->2026 boundary; reapplying would double-regress
        R[t] = 1500.0 + (R[t] - 1500.0) * (1.0 - bp["regress"])   # -> 2026 preseason

NAMES = {"ARI": "Cardinals", "ATL": "Falcons", "BAL": "Ravens", "BUF": "Bills",
         "CAR": "Panthers", "CHI": "Bears", "CIN": "Bengals", "CLE": "Browns",
         "DAL": "Cowboys", "DEN": "Broncos", "DET": "Lions", "GB": "Packers",
         "HOU": "Texans", "IND": "Colts", "JAX": "Jaguars", "KC": "Chiefs",
         "LA": "Rams", "LAC": "Chargers", "LV": "Raiders", "MIA": "Dolphins",
         "MIN": "Vikings", "NE": "Patriots", "NO": "Saints", "NYG": "Giants",
         "NYJ": "Jets", "PHI": "Eagles", "PIT": "Steelers", "SEA": "Seahawks",
         "SF": "49ers", "TB": "Buccaneers", "TEN": "Titans", "WAS": "Commanders"}
power = []
for t, name in NAMES.items():
    power.append({
        "code": t, "name": name, "elo": round(R.get(t, 1500.0), 1),
        "off_pass": round((crate(offP[t], LGP) - LGP) * 100, 1),
        "off_run": round((crate(offR[t], LGR) - LGR) * 100, 1),
        "def_pass": round((LGP - crate(dfaP[t], LGP)) * 100, 1),
        "def_run": round((LGR - crate(dfaR[t], LGR)) * 100, 1),
    })
power.sort(key=lambda x: -x["elo"])
for i, p_ in enumerate(power, 1):
    p_["rank"] = i

# ---- player boards ----
def load_board(pos_filter, floor, k=12):
    rows = [r for r in csv.DictReader(open("data/nfl_player_board_2025.csv", encoding="utf-8"))
            if r["pos"] in pos_filter and float(r["n_duels"]) >= floor]
    rows.sort(key=lambda r: -float(r["conservative_z"]))
    return [{"player": r["player"], "pos": r["pos"], "z": round(float(r["z"]), 2),
             "cons": round(float(r["conservative_z"]), 2), "n": int(float(r["n_duels"]))}
            for r in rows[:k]]

boards = {"QB": load_board({"QB"}, 300), "RB": load_board({"RB", "FB"}, 220),
          "WR": load_board({"WR"}, 90), "TE": load_board({"TE"}, 70)}

wowy = json.load(open("data/nfl_wowy.json"))
mvp = [{"player": "Aaron Rodgers", "pos": "QB", "pts": 6.67, "n_abs": 36},
       {"player": "Lamar Jackson", "pos": "QB", "pts": 6.61, "n_abs": 18},
       {"player": "Tua Tagovailoa", "pos": "QB", "pts": 6.50, "n_abs": 19},
       {"player": "Patrick Mahomes", "pos": "QB", "pts": 6.39, "n_abs": 11},
       {"player": "Derek Carr", "pos": "QB", "pts": 6.33, "n_abs": 29},
       {"player": "Joe Burrow", "pos": "QB", "pts": 5.93, "n_abs": 23}]

payload = {
    "generated": __import__("datetime").datetime.utcnow().isoformat() + "Z",
    "status": "preseason", "season": 2026,
    "model_card": {
        "test_log_loss": 0.61947, "holdout": "2016-2025, n=2,761, scored once",
        "accuracy": 64.6, "acc_home": 55.0, "acc_elo": 63.6, "acc_close": 66.6,
        "close_log_loss": 0.60913, "n_features": 14, "n_tests": 47,
        "training": "walk-forward yearly refit, 3-season recency half-life",
        "calibration": [{"bucket": "50-60%", "hit": 51.1, "n": 798},
                        {"bucket": "60-70%", "hit": 59.9, "n": 764},
                        {"bucket": "70-80%", "hit": 71.6, "n": 656},
                        {"bucket": "80%+", "hit": 82.9, "n": 533}],
    },
    "power": power, "boards": boards, "mvp": mvp,
    "pos_values": wowy.get("pos_values_epa_play", {}),
}
# carry forward enrichments that live only in the prior payload, else a rebuild
# silently drops them: 'value_updated' (written by market/nfl_edges.py) and the
# gsis 'id' fields on boards/mvp rows (hand-patched in 537deb4; site/index.html
# needs them for player-page links)
try:
    _prev = json.load(open("site/data/nfl.json"))
    if "value_updated" in _prev:
        payload["value_updated"] = _prev["value_updated"]
    _pid = {e["player"]: e["id"]
            for grp in list(_prev.get("boards", {}).values()) + [_prev.get("mvp", [])]
            for e in grp if e.get("id")}
    for grp in list(payload["boards"].values()) + [payload["mvp"]]:
        for e in grp:
            if e["player"] in _pid:
                e["id"] = _pid[e["player"]]
except (FileNotFoundError, json.JSONDecodeError):
    pass
json.dump(payload, open("site/data/nfl.json", "w"), indent=1)
print(f"wrote site/data/nfl.json  ({len(power)} teams, "
      f"{sum(len(v) for v in boards.values())} board players)")
print("top-5 power:", [(p['code'], p['elo']) for p in power[:5]])
