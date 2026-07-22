"""Serve predictions by CONTINUING the training walk to today — not by freezing.

The artifacts are a lossless checkpoint of the training walk at the last complete
season: the FIP-Elo engine state (team Elo + each pitcher's decayed FIP EWMA) and
the TrueSkill engine state (each player's mu, var). At serve time we reconstruct
the SAME engine objects the offline build used, seed them from the checkpoint, and
replay this season's games and plate appearances through the IDENTICAL update
methods (FipPitcherElo.update, TrueSkill1v1.update). Bullpen quality (season-to-date
reliever FIP) and batter power (career + this season's counts) are accumulated the
same way training accumulates them. So the served rating state equals a full
retrain through today — train and serve are the same computation.

The only things fixed at serve are the TRAINED parameters (blend coefficients, the
Elo/FIP/TrueSkill hyperparameters, the DEV standardizations) — i.e. the model
itself. No data-derived rating is frozen: every one is walked to the latest
completed game. Market-free throughout.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

from mlbwp.rating import FipPitcherElo
from mlbwp.trueskill import TrueSkill1v1
from mlbwp.siera import SieraRater

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
        self.lg_fip = d.get("league_fip", 1.2)
        self.lg_hrfb = d.get("league_hrfb")     # xFIP: league HR per fly ball
        self.pitchers = d["pitchers"]
        self.name_index = d["name_index"]
        self.current_through = None      # set by replay_season

        self.blend = _load(ART / "blend.json")
        self.mlbam_to_retro = _load(ART / "mlbam_to_retro.json") or {}

        # --- reconstruct the FIP-Elo engine from the checkpoint ---
        self.fe = FipPitcherElo(lg_fip=self.lg_fip, **self.params)
        for t, r in d["teams"].items():
            self.fe.R[t] = r
        for pid, rec in self.pitchers.items():
            pf = rec.get("pf")
            if pf:
                self.fe.pf[pid] = {"HR": pf[0], "BB": pf[1], "HBP": pf[2],
                                   "SO": pf[3], "outs": pf[4]}

        # --- reconstruct the TrueSkill engine from the checkpoint ---
        tsp = (self.blend or {}).get("ts_params", {})
        self.tse = TrueSkill1v1(beta=tsp.get("beta", 25.0 / 6.0), tau=tsp.get("tau", 25.0 / 300.0),
                                mu0=tsp.get("mu0", 25.0), sig0=tsp.get("sig0", 25.0 / 3.0))
        self.tse.load_state(_load(ART / "ts_ratings.json") or {})

        # Bullpen quality: season-to-date reliever FIP-core, accumulated from this
        # season's box scores (set_bullpen); shrinks to the league bullpen core.
        self.bp_num: dict[str, float] = {}
        self.bp_out: dict[str, float] = {}
        self.bp_lg_core = (self.blend or {}).get("bp_lg_core", 0.976)
        self.bp_prior_outs = (self.blend or {}).get("bp_prior_outs", 200.0)
        self.bullpen_through = None

        # Batter power: frozen career counts {retro: [PA,H,TB]} plus this season's
        # counts (set_power_season); bat_iso shrinks to league ISO.
        self.power = _load(ART / "power_ratings.json") or {}
        self.pwr_lg_iso = (self.blend or {}).get("pwr_lg_iso", 0.140)
        self.pwr_prior_pa = (self.blend or {}).get("pwr_prior_pa", 150.0)
        self.pw_season: dict[str, list] = {}
        self.pitchers_through = None

        # Batter baserunning: frozen career counts {retro: [PA,SB,CS,XB,OOB,GIDP]}
        # plus this season's counts (set_baserun_season). bat_brv run-values them
        # and shrinks to 0 (a league-average baserunner adds nothing). The third leg
        # of offense (with on-base + power); needs the posted lineup.
        self.baserun = _load(ART / "baserun_ratings.json") or {}
        self.br_rv = tuple((self.blend or {}).get("br_rv", (0.20, -0.41, 0.19, -0.41, -0.25)))
        self.br_prior_pa = (self.blend or {}).get("br_prior_pa", 200.0)
        self.br_season: dict[str, list] = {}

        # SIERA: per-starter EWMA of SO/BB/GB/FB/PU/PA, reconstructed from the
        # checkpoint and walked forward this season. A pitcher-quality signal
        # (ground balls + non-linearity) that xFIP lacks; needs no lineup.
        b = self.blend or {}
        self.siera = SieraRater(decay=self.params["decay"], prior=b.get("siera_prior", 120.0),
                                lg=b.get("siera_lg", {"so": .186, "bb": .078, "gb": .310, "fb": .195, "pu": .031}))
        self.siera.load_state(_load(ART / "siera_ratings.json") or {})

    # --- replay this season through the engines (== continue the training walk) ---
    def replay_season(self, finals: list[dict], pas: list, bullpen=None, power=None,
                      baserun=None) -> int:
        """Continue the training walk with this season's data.
          finals: game dicts {home,away,home_sp,away_sp(retro ids),y,home_line,away_line,date}
                  replayed through FipPitcherElo.update -> team Elo + starter FIP, together, in date order.
          pas:    (batter_retro, pitcher_retro, on_base) in date order -> TrueSkill.update.
          bullpen {team:(fip_num,outs)}, power {retro:[PA,H,TB]},
          baserun {retro:[PA,SB,CS,XB,OOB,GIDP]}: season-to-date accumulations."""
        n = 0
        for g in sorted(finals, key=lambda x: x["date"]):
            self.fe.update(g)
            for sp_key, sl_key in (("home_sp", "home_sline"), ("away_sp", "away_sline")):
                sl = g.get(sl_key)
                if sl:
                    self.siera.update(g[sp_key], *sl)      # so,bb,gb,fb,pu,pa
            self.current_through = g["date"]
            n += 1
        for batter, pitcher, on_base in pas:
            self.tse.apply_pa(batter, pitcher, on_base)
        if bullpen is not None:
            self.set_bullpen(bullpen, through=self.current_through)
        if power is not None:
            self.set_power_season(power)
        if baserun is not None:
            self.set_baserun_season(baserun)
        self.pitchers_through = self.current_through
        return n

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

    def _recbp(self, fip_p: float, bp_z: float, si_z: float = 0.0) -> float:
        b = self.blend["recbp"]
        return 1.0 / (1.0 + math.exp(
            -(b["b0"] + b["b1"] * self._logit(fip_p) + b["b3"] * bp_z + b.get("b5", 0.0) * si_z)))

    def _full(self, fip_p: float, edge: float, bp_z: float, pw_z: float = 0.0,
              si_z: float = 0.0, br_z: float = 0.0) -> float:
        b = self.blend["full"]
        z = (edge - self.blend["ts_mu"]) / self.blend["ts_sd"]
        return 1.0 / (1.0 + math.exp(
            -(b["b0"] + b["b1"] * self._logit(fip_p) + b["b2"] * z
              + b["b3"] * bp_z + b.get("b4", 0.0) * pw_z + b.get("b5", 0.0) * si_z
              + b.get("b6", 0.0) * br_z)))

    def _si_z(self, home_sp, away_sp):
        """Standardised home-minus-away SIERA differential; starters by name."""
        if not self.blend or "si_mu" not in self.blend:
            return None, None
        hp = self.name_index.get(norm_name(home_sp))
        ap = self.name_index.get(norm_name(away_sp))
        edge = self.siera.value(hp) - self.siera.value(ap)
        return (edge - self.blend["si_mu"]) / self.blend["si_sd"], edge

    # --- bullpen quality -------------------------------------------------
    def bp_fip(self, team: str) -> float:
        num = self.bp_num.get(team, 0.0)
        outs = self.bp_out.get(team, 0.0)
        prior = self.bp_prior_outs * self.bp_lg_core / 3.0
        return (num + prior) / (outs + self.bp_prior_outs) * 3.0

    def bp_edge(self, home: str, away: str) -> float:
        return self.bp_fip(away) - self.bp_fip(home)

    def _bp_z(self, home: str, away: str):
        if not self.blend or "bp_mu" not in self.blend:
            return None, 0.0
        edge = self.bp_edge(home, away)
        return (edge - self.blend["bp_mu"]) / self.blend["bp_sd"], edge

    def set_bullpen(self, team_aggs: dict, through: str | None = None) -> int:
        self.bp_num = {t: float(v[0]) for t, v in team_aggs.items()}
        self.bp_out = {t: float(v[1]) for t, v in team_aggs.items()}
        self.bullpen_through = through
        return sum(1 for v in team_aggs.values() if v[1] > 0)

    # --- batter power ----------------------------------------------------
    def bat_iso(self, retro: str | None) -> float:
        c = self.power.get(retro or "") or (0, 0, 0)
        s = self.pw_season.get(retro or "") or (0, 0, 0)
        pa, h, tb = c[0] + s[0], c[1] + s[1], c[2] + s[2]
        if pa == 0:
            return self.pwr_lg_iso
        prior = self.pwr_prior_pa * self.pwr_lg_iso
        return (tb - h + prior) / (pa + self.pwr_prior_pa)

    def set_power_season(self, counts: dict) -> int:
        self.pw_season = {r: list(v) for r, v in counts.items()}
        return sum(1 for v in counts.values() if v[0] > 0)

    def power_edge(self, home_lineup, away_lineup) -> float | None:
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

    # --- batter baserunning ---------------------------------------------
    def bat_brv(self, retro: str | None) -> float:
        """Run-valued baserunning rate = (SB,CS,XB,OOB,GIDP . run_values) / (PA+prior).
        Career (frozen) + this season's counts; 0 for an unknown/league-average runner."""
        c = self.baserun.get(retro or "") or (0, 0, 0, 0, 0, 0)
        s = self.br_season.get(retro or "") or (0, 0, 0, 0, 0, 0)
        pa = c[0] + s[0]
        brv = sum(self.br_rv[k] * (c[1 + k] + s[1 + k]) for k in range(5))
        return brv / (pa + self.br_prior_pa)

    def set_baserun_season(self, counts: dict) -> int:
        self.br_season = {r: list(v) for r, v in counts.items()}
        return sum(1 for v in counts.values() if v[0] > 0)

    def baserun_edge(self, home_lineup, away_lineup) -> float | None:
        if not self.baserun or not home_lineup or not away_lineup \
                or len(home_lineup) < 5 or len(away_lineup) < 5:
            return None
        h = [self.bat_brv(self.mlbam_to_retro.get(str(b))) for b in home_lineup]
        a = [self.bat_brv(self.mlbam_to_retro.get(str(b))) for b in away_lineup]
        return sum(h) / len(h) - sum(a) / len(a)

    def _br_z(self, home_lineup, away_lineup):
        if not self.blend or "br_mu" not in self.blend:
            return None, None
        e = self.baserun_edge(home_lineup, away_lineup)
        if e is None:
            return None, None
        return (e - self.blend["br_mu"]) / self.blend["br_sd"], e

    # --- on-base TrueSkill ----------------------------------------------
    def _ts_mu(self, retro: str | None) -> float:
        return self.tse.mu.get(retro or "", self.tse.mu0)

    def ts_edge(self, home_lineup, away_lineup, home_sp, away_sp) -> float | None:
        if not home_lineup or not away_lineup or len(home_lineup) < 5 or len(away_lineup) < 5:
            return None
        h_off = [self._ts_mu(self.mlbam_to_retro.get(str(b))) for b in home_lineup]
        a_off = [self._ts_mu(self.mlbam_to_retro.get(str(b))) for b in away_lineup]
        home_off = sum(h_off) / len(h_off)
        away_off = sum(a_off) / len(a_off)
        home_def = self._ts_mu(self.name_index.get(norm_name(home_sp)))
        away_def = self._ts_mu(self.name_index.get(norm_name(away_sp)))
        return (home_off - away_off) + (home_def - away_def)

    # --- pitcher resolution (name -> engine adjustment) ------------------
    def resolve_pitcher(self, name: str):
        pid = self.name_index.get(norm_name(name))
        if pid and pid in self.pitchers:
            return pid, self.pitchers[pid]
        return None, None

    def pitcher_adj(self, name: str) -> tuple[float, dict | None]:
        pid, rec = self.resolve_pitcher(name)
        if not rec:
            return 0.0, None
        return self.fe.pitcher_adj(pid), rec

    # --- prediction with a factor decomposition -------------------------
    def predict(self, home: str, away: str, home_sp: str, away_sp: str,
                home_lineup=None, away_lineup=None) -> dict:
        rh = self.fe.R.get(home)
        ra = self.fe.R.get(away)
        if rh is None or ra is None:
            return {"error": f"unknown team {home if rh is None else away}"}
        adj_h, rec_h = self.pitcher_adj(home_sp or "")
        adj_a, rec_a = self.pitcher_adj(away_sp or "")
        diff = (rh + self.hfa + adj_h) - (ra + adj_a)
        fip_p = 1.0 / (1.0 + 10 ** (-diff / 400.0))

        recal_p = self._recal(fip_p)
        bp_z, bp_edge = self._bp_z(home, away)
        si_z, si_edge = self._si_z(home_sp, away_sp)
        siz = si_z or 0.0
        if bp_z is not None or si_z is not None:
            p_bp = self._recbp(fip_p, bp_z or 0.0, 0.0)          # bullpen only
            served = self._recbp(fip_p, bp_z or 0.0, siz)        # + SIERA
            tier = "bullpen"
        else:
            p_bp = served = recal_p
            tier = "recal"
        bullpen_pp = round((p_bp - recal_p) * 100, 1)
        siera_pp = round((served - p_bp) * 100, 1)
        base_p = served

        edge, pw_edge, br_edge = None, None, None
        lineup_pp, power_pp, baserun_pp = 0.0, 0.0, 0.0
        if self.blend and home_lineup and away_lineup:
            edge = self.ts_edge(home_lineup, away_lineup, home_sp, away_sp)
            if edge is not None:
                pw_z, pw_edge = self._pw_z(home_lineup, away_lineup)
                br_z, br_edge = self._br_z(home_lineup, away_lineup)
                p_ts = self._full(fip_p, edge, bp_z or 0.0, 0.0, siz, 0.0)          # lineup on-base
                p_pw = self._full(fip_p, edge, bp_z or 0.0, pw_z or 0.0, siz, 0.0)  # + power
                served = self._full(fip_p, edge, bp_z or 0.0, pw_z or 0.0, siz, br_z or 0.0)  # + baserunning
                tier = "lineup"
                lineup_pp = round((p_ts - base_p) * 100, 1)
                power_pp = round((p_pw - p_ts) * 100, 1)
                baserun_pp = round((served - p_pw) * 100, 1)

        slope = recal_p * (1 - recal_p) * (2.302585 / 400.0) * self.blend["recal"]["b1"] \
            if self.blend else recal_p * (1 - recal_p) * (2.302585 / 400.0)
        contrib = {
            "team": round((rh - ra) * slope * 100, 1),
            "home_field": round(self.hfa * slope * 100, 1),
            "home_pitcher": round(adj_h * slope * 100, 1),
            "away_pitcher": round(-adj_a * slope * 100, 1),
            "bullpen": bullpen_pp,
            "siera": siera_pp,
            "lineup": lineup_pp,
            "power": power_pp,
            "baserun": baserun_pp,
        }
        return {
            "home_win_prob": round(served, 4),
            "fip_prob": round(fip_p, 4), "recal_prob": round(recal_p, 4),
            "tier": tier, "ts_edge": round(edge, 3) if edge is not None else None,
            "bullpen_edge": round(bp_edge, 3) if bp_z is not None else None,
            "home_bp_fip": round(self.bp_fip(home), 3) if bp_z is not None else None,
            "away_bp_fip": round(self.bp_fip(away), 3) if bp_z is not None else None,
            "power_edge": round(pw_edge, 4) if pw_edge is not None else None,
            "baserun_edge": round(br_edge, 4) if br_edge is not None else None,
            "siera_edge": round(si_edge, 3) if si_edge is not None else None,
            "home_rating": round(rh, 1), "away_rating": round(ra, 1),
            "home_pitcher_adj": adj_h, "away_pitcher_adj": adj_a,
            "home_pitcher_matched": rec_h is not None,
            "away_pitcher_matched": rec_a is not None,
            "home_pitcher_ip": rec_h["ip"] if rec_h else 0.0,
            "away_pitcher_ip": rec_a["ip"] if rec_a else 0.0,
            "contributions_pp": contrib,
        }
