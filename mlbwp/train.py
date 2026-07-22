"""Fit the model on all available history and freeze it to mlbwp/artifacts/.

Produces:
  artifacts/ratings.json  — team Elo, per-pitcher FIP adjustment, a name index
  artifacts/metrics.json  — the locked-holdout backtest numbers the site publishes

Ratings are fit through the last complete Retrosheet season, then regressed once
to the following season's preseason state (the model's own between-season step),
so the frozen ratings are the correct prior for predicting the current season.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from mlbwp import MODEL_NAME, MODEL_VERSION
from mlbwp import ingest
from mlbwp.ingest import league_fip_core, load_games, pitcher_names
from mlbwp.rating import FipPitcherElo

ART = Path(__file__).resolve().parent / "artifacts"

# Tuned on DEV 2000-2015 only; see data/mlb_fip_elo.json.
PARAMS = dict(
    k_team=3.6487, hfa=26.5952, team_regress=0.5398,
    beta=37.1196, decay=0.9531, prior_ip=74.3335, season_decay=0.0087,
)


def norm_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9 ]+", " ", s.lower())
    return " ".join(s.split())


def main():
    games = load_games()
    lg_fip = league_fip_core(games)
    last_season = games[-1]["season"]
    print(f"training on {len(games):,} games, {games[0]['season']}-{last_season}, "
          f"league FIP-core={lg_fip:.3f}")

    m = FipPitcherElo(lg_fip=lg_fip, **PARAMS)
    m.fit(games)
    # Advance ratings to next season's preseason prior (the model's own step),
    # since we predict the season AFTER the last one with results.
    m.new_season()
    serve_season = last_season + 1

    names = pitcher_names()
    pitchers = {}
    name_index = {}
    for pid in m.pf:
        ip = m.pitcher_ip(pid)
        if ip < 3:                       # skip trivial samples
            continue
        nm = names.get(pid, pid)
        s = m.pf[pid]
        # Freeze the full decayed FIP EWMA state so serving can RECONSTRUCT the
        # engine and replay this season's starts through FipPitcherElo.update —
        # i.e. continue the exact training walk rather than freeze a stale value.
        pitchers[pid] = {
            "name": nm, "adj": round(m.pitcher_adj(pid), 2), "ip": round(ip, 1),
            "pf": [round(s["HR"], 4), round(s["BB"], 4), round(s["HBP"], 4),
                   round(s["SO"], 4), round(s["outs"], 4)],
        }
        key = norm_name(nm)
        # keep the pitcher with more innings when names collide
        if key not in name_index or ip > pitchers.get(name_index[key], {}).get("ip", 0):
            name_index[key] = pid

    teams = {t: round(r, 1) for t, r in m.R.items()}
    ratings = {
        "model": MODEL_NAME, "version": MODEL_VERSION,
        "serve_season": serve_season,
        "trained_through_season": last_season,
        "params": PARAMS, "league_fip": round(lg_fip, 4),
        "league_hrfb": round(ingest.LEAGUE_HRFB, 5) if ingest.LEAGUE_HRFB else None,
        "teams": dict(sorted(teams.items())),
        "pitchers": pitchers,
        "name_index": name_index,
    }
    ART.mkdir(exist_ok=True)
    (ART / "ratings.json").write_text(json.dumps(ratings, indent=1), encoding="utf-8")
    print(f"froze {len(teams)} team ratings, {len(pitchers)} pitcher ratings "
          f"(serve season {serve_season})")

    # metrics from the locked backtest (already computed, provably out-of-sample)
    src = Path(__file__).resolve().parents[1] / "data" / "mlb_fip_elo.json"
    bt = json.loads(src.read_text())
    metrics = {
        "holdout": "2016-2024 (2020 excluded), n=%d" % bt["test_n"],
        "model_log_loss": round(bt["fip_elo"]["test_ll"], 5),
        "plain_elo_log_loss": round(bt["plain_elo"]["test_ll"], 5),
        "improvement_nats": round(bt["test_delta"], 5),
        "improvement_ci": [round(bt["ci"][0], 5), round(bt["ci"][1], 5)],
        "significant": bt["ci"][0] > 0,
        "constant_log_loss": 0.69315,
    }
    (ART / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    print("froze metrics:", metrics["model_log_loss"], "vs elo", metrics["plain_elo_log_loss"])

    top = sorted(pitchers.items(), key=lambda kv: -kv[1]["adj"])[:6]
    print("top pitchers:", ", ".join(f"{v['name']} {v['adj']:+.0f}" for _, v in top))
    strong = sorted(teams.items(), key=lambda kv: -kv[1])[:5]
    print("top teams:", ", ".join(f"{t} {r:.0f}" for t, r in strong))


if __name__ == "__main__":
    main()
