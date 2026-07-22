"""Serve predictions from the frozen model, brought current to the live season.

The frozen ratings are the model as of the last complete Retrosheet season.
Team form for the in-progress season is applied here from StatsAPI final scores
(a plain team-Elo walk, using the frozen pitcher adjustments in the expectation
so the update is consistent with how the model was trained). Bullpen quality is
also brought current: each team's season-to-date reliever FIP-core is loaded from
this season's box scores (set_bullpen) and fed to the blend. Starting-pitcher
ratings stay at their career-through-last-season value — starter true talent is
stable, and an in-season starter backfill is a further increment. Every prediction
states which inputs are current on its card.
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

        # Blend layer: recalibration (always) + a season-to-date bullpen-quality
        # term (whenever this season's reliever lines are loaded) + the TrueSkill
        # lineup feature (when lineups are posted). Absent artifacts degrade to
        # the raw FIP-Elo model.
        self.blend = _load(ART / "blend.json")
        self.ts = _load(ART / "ts_ratings.json") or {}
        self.mlbam_to_retro = _load(ART / "mlbam_to_retro.json") or {}
        self.TS_MU0 = 25.0

        # Bullpen quality: each team's season-to-date reliever FIP-core, filled
        # by set_bullpen() from this season's box scores. bp_fip() shrinks to the
        # league bullpen core, matching phase0/freeze_trueskill.reliever_aggs.
        self.bp_num: dict[str, float] = {}
        self.bp_out: dict[str, float] = {}
        self.bp_lg_core = (self.blend or {}).get("bp_lg_core", 0.976)
        self.bp_prior_outs = (self.blend or {}).get("bp_prior_outs", 200.0)
        self.bullpen_through = None       # set by set_bullpen

        # Batter power: per-batter career isolated power (extra bases per PA),
        # frozen counts {retro: [PA,H,TB]}. A lineup feature like ts_edge (needs
        # the posted lineup); the orthogonal power dimension the on-base rating
        # misses. bat_iso() shrinks to league ISO, matching the frozen feature.
        self.power = _load(ART / "power_ratings.json") or {}
        self.pwr_lg_iso = (self.blend or {}).get("pwr_lg_iso", 0.140)
        self.pwr_prior_pa = (self.blend or {}).get("pwr_prior_pa", 150.0)

    # --- blend helpers ---------------------------------------------------
    @staticmethod
    def _logit(p: float) -> float:
        p = min(max(p, 1e-9), 1 - 1e-9)
        return math.log(p / (1 - p))

    def _recal(self, fip_p: float) -> float:
        if not self.blend:
            return fip_p
        b = self.blend["recal"]
        return 1.0 / (1.0 + math.exp(-(b["b0"] + b["b1"] * self._logit(fip_p))))

    def _recbp(self, fip_p: float, bp_z: float) -> float:
        b = self.blend["recbp"]
        return 1.0 / (1.0 + math.exp(-(b["b0"] + b["b1"] * self._logit(fip_p) + b["b3"] * bp_z)))

    # --- bullpen quality -------------------------------------------------
    def bp_fip(self, team: str) -> float:
        """Season-to-date reliever FIP-core, empirical-Bayes shrunk to the league
        bullpen core. Identical formula to the frozen training feature."""
        num = self.bp_num.get(team, 0.0)
        outs = self.bp_out.get(team, 0.0)
        prior = self.bp_prior_outs * self.bp_lg_core / 3.0
        return (num + prior) / (outs + self.bp_prior_outs) * 3.0

    def bp_edge(self, home: str, away: str) -> float:
        """Home-minus-away bullpen differential; sign matches the frozen feature
        (away FIP - home FIP, so positive favours the home bullpen: lower = better)."""
        return self.bp_fip(away) - self.bp_fip(home)

    def _bp_z(self, home: str, away: str):
        if not self.blend or "bp_mu" not in self.blend:
            return None, 0.0
        edge = self.bp_edge(home, away)
        return (edge - self.blend["bp_mu"]) / self.blend["bp_sd"], edge

    def set_bullpen(self, team_aggs: dict, through: str | None = None) -> int:
        """Load this season's reliever aggregates: {team: (fip_num, outs)}."""
        self.bp_num = {t: float(v[0]) for t, v in team_aggs.items()}
        self.bp_out = {t: float(v[1]) for t, v in team_aggs.items()}
        self.bullpen_through = through
        return sum(1 for v in team_aggs.values() if v[1] > 0)

    # --- batter power ----------------------------------------------------
    def bat_iso(self, retro: str | None) -> float:
        """A batter's isolated power (extra bases per PA), empirical-Bayes shrunk
        to league ISO. Identical formula to the frozen training feature."""
        c = self.power.get(retro or "")
        prior = self.pwr_prior_pa * self.pwr_lg_iso
        if not c:
            return self.pwr_lg_iso
        pa, h, tb = c
        return (tb - h + prior) / (pa + self.pwr_prior_pa)

    def power_edge(self, home_lineup, away_lineup) -> float | None:
        """Home-minus-away lineup isolated-power differential; sign matches the
        frozen feature (home lineup ISO - away lineup ISO). Lineups are mlbam ids."""
        if not self.power or not home_lineup or not away_lineup \
                or len(home_lineup) < 5 or len(away_lineup) < 5:
            return None
        h = [self.bat_iso(self.mlbam_to_retro.get(str(b))) for b in home_lineup]
        a = [self.bat_iso(self.mlbam_to_retro.get(str(b))) for b in away_lineup]
        return sum(h) / len(h) - sum(a) / len(a)

    def _pw_z(self, home_lineup, away_lineup):
        if not self.blend or "pw_mu" not in self.blend:
            return None, None
        e = self.power_edge(home_lineup, away_lineup)
        if e is None:
            return None, None
        return (e - self.blend["pw_mu"]) / self.blend["pw_sd"], e

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

    def _full(self, fip_p: float, edge: float, bp_z: float, pw_z: float = 0.0) -> float:
        b = self.blend["full"]
        z = (edge - self.blend["ts_mu"]) / self.blend["ts_sd"]
        return 1.0 / (1.0 + math.exp(
            -(b["b0"] + b["b1"] * self._logit(fip_p) + b["b2"] * z
              + b["b3"] * bp_z + b.get("b4", 0.0) * pw_z)))

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

        # Tiers, cheapest first: recal (FIP-Elo, always) -> bullpen (adds this
        # season's reliever-quality term) -> lineup (adds TrueSkill once lineups
        # are posted). Each layer's marginal effect is reported as a contribution.
        recal_p = self._recal(fip_p)
        bp_z, bp_edge = self._bp_z(home, away)
        if bp_z is not None:
            served, tier = self._recbp(fip_p, bp_z), "bullpen"
        else:
            served, tier = recal_p, "recal"
        bullpen_pp = round((served - recal_p) * 100, 1)
        base_p = served                          # the pre-lineup served number

        edge, pw_edge, lineup_pp, power_pp = None, None, 0.0, 0.0
        if self.blend and home_lineup and away_lineup:
            edge = self.ts_edge(home_lineup, away_lineup, home_sp, away_sp)
            if edge is not None:
                pw_z, pw_edge = self._pw_z(home_lineup, away_lineup)
                # p_ts adds only the on-base (TrueSkill) lineup term; served then
                # adds the orthogonal power term, so the two contributions split
                # the lineup effect additively.
                p_ts = self._full(fip_p, edge, bp_z or 0.0, 0.0)
                served = self._full(fip_p, edge, bp_z or 0.0, pw_z or 0.0)
                tier = "lineup"
                lineup_pp = round((p_ts - base_p) * 100, 1)
                power_pp = round((served - p_ts) * 100, 1)

        # decompose the FIP-Elo rating points into probability points (around recal_p)
        slope = recal_p * (1 - recal_p) * (2.302585 / 400.0) * self.blend["recal"]["b1"] \
            if self.blend else recal_p * (1 - recal_p) * (2.302585 / 400.0)
        contrib = {
            "team": round((rh - ra) * slope * 100, 1),
            "home_field": round(self.hfa * slope * 100, 1),
            "home_pitcher": round(adj_h * slope * 100, 1),
            "away_pitcher": round(-adj_a * slope * 100, 1),
            "bullpen": bullpen_pp,
            "lineup": lineup_pp,
            "power": power_pp,
        }
        return {
            "home_win_prob": round(served, 4),
            "fip_prob": round(fip_p, 4), "recal_prob": round(recal_p, 4),
            "tier": tier, "ts_edge": round(edge, 3) if edge is not None else None,
            "bullpen_edge": round(bp_edge, 3) if bp_z is not None else None,
            "home_bp_fip": round(self.bp_fip(home), 3) if bp_z is not None else None,
            "away_bp_fip": round(self.bp_fip(away), 3) if bp_z is not None else None,
            "power_edge": round(pw_edge, 4) if pw_edge is not None else None,
            "home_rating": round(rh, 1), "away_rating": round(ra, 1),
            "home_pitcher_adj": adj_h, "away_pitcher_adj": adj_a,
            "home_pitcher_matched": rec_h is not None,
            "away_pitcher_matched": rec_a is not None,
            "home_pitcher_ip": rec_h["ip"] if rec_h else 0.0,
            "away_pitcher_ip": rec_a["ip"] if rec_a else 0.0,
            "contributions_pp": contrib,
        }
