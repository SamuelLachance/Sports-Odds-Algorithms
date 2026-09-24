"""Build the NFL GLASSBOX payload skeleton (site/data/nfl.json) — step 1 of the chain.

Power ratings from the model states walked through the latest final (Elo, plus
pass/run unit EPA; regressed to a preseason prior only while no game of the
serve season has been played), player boards from the shipped continuous board,
the WOWY MVP table, and the model card (measured numbers only).

Writes the payload the chain is building: under phase0/nfl_weekly.py that is a
STAGING copy (NFL_PAYLOAD), swapped onto the live file only after the whole
chain validated, so the site never sees this schedule-less intermediate state.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "phase0")
import nfl_payload as NP  # noqa: E402

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
PRESEASON = games[-1]["season"] < 2026
if PRESEASON:                       # else the in-walk trigger already regressed at the
    for t in R:                     # 2025->2026 boundary; reapplying would double-regress
        R[t] = 1500.0 + (R[t] - 1500.0) * (1.0 - bp["regress"])   # -> 2026 preseason

power = []
for t in NP.TEAMS:
    power.append({
        "code": t, **NP.team_names(t), "elo": round(R.get(t, 1500.0), 1),
        "off_pass": round((crate(offP[t], LGP) - LGP) * 100, 1),
        "off_run": round((crate(offR[t], LGR) - LGR) * 100, 1),
        "def_pass": round((LGP - crate(dfaP[t], LGP)) * 100, 1),
        "def_run": round((LGR - crate(dfaR[t], LGR)) * 100, 1),
    })
power.sort(key=lambda x: -x["elo"])
for i, p_ in enumerate(power, 1):
    p_["rank"] = i

# ---- player boards ----
BOARD_SEASON = 2025                  # the per-play board is a full-season table
BOARD_CSV = f"data/nfl_player_board_{BOARD_SEASON}.csv"


def load_board(pos_filter, floor, k=12):
    rows = [r for r in csv.DictReader(open(BOARD_CSV, encoding="utf-8"))
            if r["pos"] in pos_filter and float(r["n_duels"]) >= floor]
    rows.sort(key=lambda r: -float(r["conservative_z"]))
    # the gsis id is in the board table itself: every row links to its player
    return [{"player": r["player"], "id": r.get("gsis_id") or None, "pos": r["pos"],
             "z": round(float(r["z"]), 2),
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
# The with-vs-without study is a one-off (data/nfl_wowy.json keeps only the
# position values, not per-player splits), so these rows are a fixed table:
# resolve their ids by exact, unique display name so they link like the boards.
_by_name = defaultdict(list)
for r in csv.DictReader(open("data/nfl_players.csv", encoding="utf-8")):
    if r.get("gsis_id") and r.get("position") == "QB":
        _by_name[r.get("display_name", "")].append(r["gsis_id"])
for m in mvp:
    ids = _by_name.get(m["player"], [])
    m["id"] = ids[0] if len(ids) == 1 else None

# ---- model card ----
# The LIVE payload: carries forward what only the published file holds.
try:
    _prev = json.load(open(NP.LIVE, encoding="utf-8"))
except (FileNotFoundError, json.JSONDecodeError):
    _prev = {}
_prev_mc = _prev.get("model_card") or {}
MODEL = json.load(open("data/nfl_model.json", encoding="utf-8"))
# The headline TEST numbers belong to the serve: nfl_season_serve.py re-measures
# test_log_loss / accuracy / n_tests on every run and overwrites these three.
# Until it runs, the card keeps the last served values (else the adopted
# model's own file) - never a literal typed once and left behind (it read
# 0.61947 / 64.6% / 47 tests long after the model had moved on).
with open("data/nfl_test_ledger.csv", "rb") as _fh:      # cp1252 bytes in old rows
    N_TESTS = _fh.read().rstrip(b"\n").count(b"\n")
# Measured ONCE, when the model then served (TEST log loss 0.61947, ledger row
# 47) was adopted and scored on the locked 2016-2025 holdout. The locked-split
# rule forbids re-measuring anything on TEST here, and the serve does not
# re-measure these, so they ship WITH their provenance and the page labels
# them. The calibration buckets are that model's; the home-always / Elo /
# closing-line accuracies and the closing line's log loss are properties of
# the holdout games and of those baselines, not of the served blend.
MEASURED = {
    "asof": "2026-07-23", "ledger_row": 47, "model_test_ll": 0.61947,
    "holdout": "2016-2025", "n": 2761,
    "keys": ["calibration", "acc_home", "acc_elo", "acc_close", "close_log_loss"],
    "note": "measured once at adoption of the 2026-07-23 model (TEST log loss "
            "0.61947); not re-measured for later models",
}
# The ratings-only model the page cites: engine v7 (ledger row 66, TEST
# 2022-2025, scored once), with team Elo on the same games. The serve still
# writes the v6 row-62 numbers under ratings_model; this block is the current
# engine's and carries its source.
RATINGS_ONLY = {"test_ll": 0.62973, "acc": 65.8, "elo_ll": 0.63824,
                "seasons": "2022-2025", "engine": "v7", "ledger_row": 66}
payload = {
    "generated": NP.now_utc_iso(),
    "status": "preseason" if PRESEASON else "season", "season": 2026,
    # what the power table is: states walked through this final (a preseason
    # prior only while no 2026 game has been played)
    "power_asof": games[-1]["date"], "power_preseason": PRESEASON,
    "boards_season": BOARD_SEASON,
    "mvp_source": "one-off with-vs-without study, 2016-2025 absences (fixed table)",
    "model_card": {
        "test_log_loss": _prev_mc.get("test_log_loss", MODEL["test_ll"]),
        "accuracy": _prev_mc.get("accuracy"),
        "n_tests": N_TESTS,
        "holdout": "2016-2025, n=2,761, scored once",
        "acc_home": 55.0, "acc_elo": 63.6, "acc_close": 66.6,
        "close_log_loss": 0.60913, "n_features": len(MODEL["features"]),
        "training": "walk-forward yearly refit, 3-season recency half-life",
        "calibration": [{"bucket": "50-60%", "hit": 51.1, "n": 798},
                        {"bucket": "60-70%", "hit": 59.9, "n": 764},
                        {"bucket": "70-80%", "hit": 71.6, "n": 656},
                        {"bucket": "80%+", "hit": 82.9, "n": 533}],
        "measured": MEASURED,
        "ratings_only": RATINGS_ONLY,
    },
    "power": power, "boards": boards, "mvp": mvp,
    "pos_values": wowy.get("pos_values_epa_play", {}),
    # what pos_values is (phase0/nfl_wowy_eval.py): the unit's EPA/play drop per
    # full-time absent starter at the position, a with-vs-without fit on
    # 2013-2025. Display only - the WOWY feature lost its one TEST look.
    "pos_values_source": {"what": "unit EPA per play lost per full-time absent starter",
                          "seasons": "2013-2025", "test_ll": wowy.get("test_wowy"),
                          "model_ll": wowy.get("test_current"), "adopted": False},
}
# carry forward the one enrichment that lives only in the published payload,
# else a rebuild silently drops it: 'value_updated' (written by
# market/nfl_edges.py). Read from the LIVE file, whatever this run writes to.
if "value_updated" in _prev:
    payload["value_updated"] = _prev["value_updated"]
_out = NP.payload_path()
NP.dump_atomic(payload, _out, indent=1)
print(f"wrote {_out}  ({len(power)} teams, "
      f"{sum(len(v) for v in boards.values())} board players, "
      f"{sum(1 for m in mvp if m['id'])}/{len(mvp)} MVP ids)")
print("top-5 power:", [(p['code'], p['elo']) for p in power[:5]])
