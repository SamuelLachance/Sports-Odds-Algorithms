"""Serve predictions from the frozen model, brought current to the live season.

The frozen ratings are the model as of the last complete Retrosheet season.
Team form for the in-progress season is applied here from StatsAPI final scores
(a plain team-Elo walk, using the frozen pitcher adjustments in the expectation
so the update is consistent with how the model was trained). Pitcher ratings stay
at their career-through-last-season value — pitcher true talent is stable, and
in-season pitcher lines require a box-score + id-crosswalk backfill that is the
next increment, not this one. Every prediction states this limitation on its card.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

ART = Path(__file__).resolve().parent / "artifacts"


def norm_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode()
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", s.lower()).split())


class Predictor:
    def __init__(self, ratings_path=ART / "ratings.json"):
        d = json.loads(Path(ratings_path).read_text(encoding="utf-8"))
        self.model = d["model"]
        self.version = d["version"]
        self.serve_season = d["serve_season"]
        self.trained_through = d["trained_through_season"]
        self.params = d["params"]
        self.hfa = d["params"]["hfa"]
        self.k_team = d["params"]["k_team"]
        self.R = dict(d["teams"])
        self.pitchers = d["pitchers"]
        self.name_index = d["name_index"]
        self.current_through = None      # set by bring_current

    # --- pitcher resolution ---------------------------------------------
    def resolve_pitcher(self, name: str):
        pid = self.name_index.get(norm_name(name))
        if pid and pid in self.pitchers:
            return pid, self.pitchers[pid]
        return None, None

    def pitcher_adj(self, name: str) -> tuple[float, dict | None]:
        pid, rec = self.resolve_pitcher(name)
        return (rec["adj"], rec) if rec else (0.0, None)

    # --- bring team ratings current -------------------------------------
    def bring_current(self, finals: list[dict]) -> int:
        """Apply this season's completed games to team Elo (in date order)."""
        n = 0
        for g in sorted(finals, key=lambda x: x["date"]):
            h, a = g["home"], g["away"]
            if h not in self.R or a not in self.R:
                continue
            adj_h = self.pitcher_adj(g.get("home_sp") or "")[0]
            adj_a = self.pitcher_adj(g.get("away_sp") or "")[0]
            diff = (self.R[h] + self.hfa + adj_h) - (self.R[a] + adj_a)
            e = 1.0 / (1.0 + 10 ** (-diff / 400.0))
            dt = self.k_team * (g["home_win"] - e)
            self.R[h] += dt
            self.R[a] -= dt
            n += 1
            self.current_through = g["date"]
        return n

    # --- prediction with a factor decomposition -------------------------
    def predict(self, home: str, away: str, home_sp: str, away_sp: str) -> dict:
        rh = self.R.get(home)
        ra = self.R.get(away)
        if rh is None or ra is None:
            return {"error": f"unknown team {home if rh is None else away}"}
        adj_h, rec_h = self.pitcher_adj(home_sp or "")
        adj_a, rec_a = self.pitcher_adj(away_sp or "")
        diff = (rh + self.hfa + adj_h) - (ra + adj_a)
        p = 1.0 / (1.0 + 10 ** (-diff / 400.0))
        # decompose each rating-point contribution into probability points
        slope = p * (1 - p) * (2.302585 / 400.0)   # d/dElo of the logistic
        contrib = {
            "team": round((rh - ra) * slope * 100, 1),
            "home_field": round(self.hfa * slope * 100, 1),
            "home_pitcher": round(adj_h * slope * 100, 1),
            "away_pitcher": round(-adj_a * slope * 100, 1),
        }
        return {
            "home_win_prob": round(p, 4),
            "home_rating": round(rh, 1), "away_rating": round(ra, 1),
            "home_pitcher_adj": adj_h, "away_pitcher_adj": adj_a,
            "home_pitcher_matched": rec_h is not None,
            "away_pitcher_matched": rec_a is not None,
            "home_pitcher_ip": rec_h["ip"] if rec_h else 0.0,
            "away_pitcher_ip": rec_a["ip"] if rec_a else 0.0,
            "contributions_pp": contrib,
        }
