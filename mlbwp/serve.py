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
import math
import re
import unicodedata
from pathlib import Path

ART = Path(__file__).resolve().parent / "artifacts"


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


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

        # Blend layer: recalibration (always) + TrueSkill lineup feature (when
        # lineups are posted). Absent artifacts degrade to the raw FIP-Elo model.
        self.blend = _load(ART / "blend.json")
        self.ts = _load(ART / "ts_ratings.json") or {}
        self.mlbam_to_retro = _load(ART / "mlbam_to_retro.json") or {}
        self.TS_MU0 = 25.0

    # --- blend helpers ---------------------------------------------------
    def _recal(self, fip_p: float) -> float:
        if not self.blend:
            return fip_p
        b = self.blend["recal"]
        lg = math.log(min(max(fip_p, 1e-9), 1 - 1e-9) / (1 - min(max(fip_p, 1e-9), 1 - 1e-9)))
        return 1.0 / (1.0 + math.exp(-(b["b0"] + b["b1"] * lg)))

    def _ts_mu(self, retro: str | None) -> float:
        return self.ts.get(retro or "", self.TS_MU0)

    def ts_edge(self, home_lineup, away_lineup, home_sp, away_sp) -> float | None:
        """Lineup-vs-starter TrueSkill offensive differential (home advantage).

        Lineups are mlbam ids; starters are names. Returns None if a lineup is
        missing or too short to be meaningful.
        """
        if not home_lineup or not away_lineup or len(home_lineup) < 5 or len(away_lineup) < 5:
            return None
        h_off = [self._ts_mu(self.mlbam_to_retro.get(str(b))) for b in home_lineup]
        a_off = [self._ts_mu(self.mlbam_to_retro.get(str(b))) for b in away_lineup]
        home_off = sum(h_off) / len(h_off)
        away_off = sum(a_off) / len(a_off)
        home_def = self._ts_mu(self.name_index.get(norm_name(home_sp)))
        away_def = self._ts_mu(self.name_index.get(norm_name(away_sp)))
        return (home_off - away_off) + (home_def - away_def)

    def _full(self, fip_p: float, edge: float) -> float:
        b = self.blend["full"]
        z = (edge - self.blend["ts_mu"]) / self.blend["ts_sd"]
        lg = math.log(min(max(fip_p, 1e-9), 1 - 1e-9) / (1 - min(max(fip_p, 1e-9), 1 - 1e-9)))
        return 1.0 / (1.0 + math.exp(-(b["b0"] + b["b1"] * lg + b["b2"] * z)))

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
    def predict(self, home: str, away: str, home_sp: str, away_sp: str,
                home_lineup=None, away_lineup=None) -> dict:
        rh = self.R.get(home)
        ra = self.R.get(away)
        if rh is None or ra is None:
            return {"error": f"unknown team {home if rh is None else away}"}
        adj_h, rec_h = self.pitcher_adj(home_sp or "")
        adj_a, rec_a = self.pitcher_adj(away_sp or "")
        diff = (rh + self.hfa + adj_h) - (ra + adj_a)
        fip_p = 1.0 / (1.0 + 10 ** (-diff / 400.0))

        # Recalibrated FIP-Elo is the always-available number; the full model
        # adds the TrueSkill lineup feature once lineups are posted.
        recal_p = self._recal(fip_p)
        served, tier, edge, lineup_pp = recal_p, "recal", None, 0.0
        if self.blend and home_lineup and away_lineup:
            edge = self.ts_edge(home_lineup, away_lineup, home_sp, away_sp)
            if edge is not None:
                served = self._full(fip_p, edge)
                tier = "lineup"
                lineup_pp = round((served - recal_p) * 100, 1)

        # decompose the FIP-Elo rating points into probability points (around recal_p)
        slope = recal_p * (1 - recal_p) * (2.302585 / 400.0) * self.blend["recal"]["b1"] \
            if self.blend else recal_p * (1 - recal_p) * (2.302585 / 400.0)
        contrib = {
            "team": round((rh - ra) * slope * 100, 1),
            "home_field": round(self.hfa * slope * 100, 1),
            "home_pitcher": round(adj_h * slope * 100, 1),
            "away_pitcher": round(-adj_a * slope * 100, 1),
            "lineup": lineup_pp,
        }
        return {
            "home_win_prob": round(served, 4),
            "fip_prob": round(fip_p, 4), "recal_prob": round(recal_p, 4),
            "tier": tier, "ts_edge": round(edge, 3) if edge is not None else None,
            "home_rating": round(rh, 1), "away_rating": round(ra, 1),
            "home_pitcher_adj": adj_h, "away_pitcher_adj": adj_a,
            "home_pitcher_matched": rec_h is not None,
            "away_pitcher_matched": rec_a is not None,
            "home_pitcher_ip": rec_h["ip"] if rec_h else 0.0,
            "away_pitcher_ip": rec_a["ip"] if rec_a else 0.0,
            "contributions_pp": contrib,
        }
