"""How much of tonight's starting goalie is predictable BEFORE it is confirmed?

Serving-feasibility task of the breakthrough program. A goalie-aware NHL model
can be served at two information tiers (documents/pick_policy.md):

  CONFIRMED  the official starter is posted -> the model uses the actual starter
             (what every round-1 screen used: first-shot goalie == official
             starter, data/bt_nhl_starter_backfill.json)
  PROJECTED  no confirmation yet -> the model must use a PREDICTED starter
             (or a mixture over the team's goalies)

This script measures, on DEV only (scored 2011-12..2017-18, 2010-11 warm-up;
regular season; game ids < 2018000000), how often simple walk-forward rules,
fixed a priori and never searched, name the starter correctly from information
available the day before — i.e. how much of the goalie channel a PROJECTED serve
could carry without any confirmation feed. Outcomes and odds are not read; no
model metric is computed.

Rules (per team, state read before the game, updated after it):
  prev    the team's previous starter starts again
  modal   modal starter of the team's last 20 games (ties -> most recent)
  b2b     modal, except on the 2nd night of a back-to-back when the modal
          goalie started the 1st night -> the most frequent OTHER goalie of the
          last 20 (if the team has one), else modal

    python -X utf8 phase0/bt_nhl_starter_predictability.py
    -> data/bt_nhl_starter_predictability.json
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from collections import Counter, defaultdict, deque

import pandas as pd

sys.path.insert(0, "phase0")
from bt_nhl_anatomy_build import MAX_GID, norm  # noqa: E402

OUT = "data/bt_nhl_starter_predictability.json"
WARM = 20102011
SCORED = (20112012, 20122013, 20132014, 20142015, 20152016, 20162017, 20172018)
WIN = 20   # fixed a priori (same window as the anatomy's WIN_GOALIE)


def main():
    g = pd.read_csv("data/nhl_games.csv", usecols=["game_id", "date", "season", "type",
                                                    "away", "home"])
    g = g[(g.type == 2) & (g.game_id < MAX_GID) & (g.season <= 20172018)]
    g = g.sort_values(["date", "game_id"])
    sh = pd.read_csv("data/nhl_shots.csv",
                     usecols=["nhl_game_id", "period", "per_sec", "is_home", "goalie_id"])
    sh = sh[sh.nhl_game_id < MAX_GID].dropna(subset=["goalie_id"])
    sh = sh.sort_values(["nhl_game_id", "period", "per_sec"])
    sh["def_home"] = 1 - sh["is_home"]
    first = sh.groupby(["nhl_game_id", "def_home"], sort=False).first().reset_index()
    st = {(int(r.nhl_game_id), int(r.def_home)): int(r.goalie_id)
          for r in first.itertuples(index=False)}

    hist = defaultdict(lambda: deque(maxlen=WIN))   # team -> recent starters (oldest first)
    last_date = {}                                   # team -> date of previous game
    last_start = {}                                  # team -> previous starter
    tally = defaultdict(Counter)                     # (season|all, rule) -> counts
    for r in g.itertuples(index=False):
        d = dt.date.fromisoformat(r.date)
        for side, team in ((1, norm(r.home)), (0, norm(r.away))):
            actual = st.get((int(r.game_id), side))
            h = list(hist[team])
            b2b = team in last_date and (d - last_date[team]).days == 1
            if actual is not None and h and r.season in SCORED:
                cnt = Counter(h)
                top = max(cnt.values())
                modal = next(x for x in reversed(h) if cnt[x] == top)
                prev = last_start[team]
                others = [x for x in cnt if x != modal]
                backup = (max(others, key=lambda x: (cnt[x], max(i for i, y in enumerate(h) if y == x)))
                          if others else None)
                b2b_pick = backup if (b2b and prev == modal and backup is not None) else modal
                non1 = actual != modal
                for key in ("all", str(r.season)):
                    t = tally[key]
                    t["n"] += 1
                    t["non1_nights"] += non1
                    t["b2b_second_nights"] += b2b
                    for name, pick in (("prev", prev), ("modal", modal), ("b2b", b2b_pick)):
                        t[f"{name}_correct"] += pick == actual
                        t[f"{name}_non1_correct"] += non1 and pick == actual
                        t[f"{name}_pred_non1"] += pick != modal
                        t[f"{name}_pred_non1_correct"] += pick != modal and pick == actual
                    if b2b:
                        t["b2b_n"] += 1
                        t["b2b_non1"] += non1
                        t["b2b_rule_correct_on_b2b"] += b2b_pick == actual
            if actual is not None:
                hist[team].append(actual)
                last_start[team] = actual
            last_date[team] = d

    def summarise(t):
        n = max(t["n"], 1)
        n1 = max(t["non1_nights"], 1)
        out = {"team_games": t["n"], "non1_share": round(t["non1_nights"] / n, 4),
               "b2b_second_night_share": round(t["b2b_second_nights"] / n, 4)}
        for name in ("prev", "modal", "b2b"):
            out[name] = {
                "accuracy": round(t[f"{name}_correct"] / n, 4),
                "non1_recall": round(t[f"{name}_non1_correct"] / n1, 4),
                "non1_precision": (round(t[f"{name}_pred_non1_correct"] / t[f"{name}_pred_non1"], 4)
                                   if t[f"{name}_pred_non1"] else None)}
        if t["b2b_n"]:
            out["on_b2b_second_nights"] = {
                "n": t["b2b_n"], "non1_share": round(t["b2b_non1"] / t["b2b_n"], 4),
                "b2b_rule_accuracy": round(t["b2b_rule_correct_on_b2b"] / t["b2b_n"], 4)}
        return out

    res = {"generated": dt.datetime.now().isoformat(timespec="seconds"),
           "protocol": {"dev_only": True, "test_touched": False, "outcomes_read": False,
                        "market_use": "none", "warm": WARM, "scored": list(SCORED),
                        "starter_source": "first-shot goalie (== official starter on "
                                          "300/300 sampled DEV sides)",
                        "rules_fixed_a_priori": True, "window": WIN},
           "all": summarise(tally["all"]),
           "by_season": {s: summarise(tally[s]) for s in sorted(k for k in tally if k != "all")}}
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)
    print(json.dumps(res["all"], indent=1))
    for s, v in res["by_season"].items():
        print(s, v["team_games"], v["non1_share"], v["modal"]["accuracy"], v["b2b"]["accuracy"],
              v["b2b"]["non1_recall"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
