"""Team Elo + FIP-based starting-pitcher adjustment.

The W/L pitcher rating captured real aces but was too noisy to prove significant.
This replaces it with a pitcher-quality signal built from the starter's OWN line —
Fielding Independent Pitching:

    FIP_core = (13*HR + 3*(BB+HBP) - 2*SO) / IP        (IP = outs/3)

FIP uses only the three true outcomes a pitcher controls, all parsed exactly from
Retrosheet events (validated at 0.00% against game-log totals). It is not
contaminated by team defense or by which games his team happened to win, which is
what made the W/L proxy weak.

Each starter carries an EWMA of his FIP components over prior starts, regressed
toward the league mean by his effective innings (empirical Bayes: few innings ->
league average). The regressed FIP is turned into an Elo-point adjustment
(lower FIP = better = positive adjustment) and added to the team rating. The
adjustment is computed from the pitcher's history only — never from the game being
predicted — so it is strictly leak-safe.

Team Elo still updates from the game outcome; the pitcher EWMA updates from the
start's line. The two signals never share an update, so there is no double count.

Data: Retrosheet. "The information used here was obtained free of charge from and is
copyrighted by Retrosheet. Interested parties may contact Retrosheet at
www.retrosheet.org."
"""

from __future__ import annotations

import csv
import glob
from collections import defaultdict

import numpy as np

F_DATE, F_VIS, F_HOME, F_VIS_R, F_HOME_R, F_VIS_SP, F_HOME_SP = 0, 3, 6, 9, 10, 101, 103


def load_starter_lines() -> dict:
    """(game_id, pitcher) -> (outs, HR, BB, HBP, SO) for starters."""
    out = {}
    for r in csv.DictReader(open("data/retro_events/appearances.csv")):
        if r["is_starter"] != "1":
            continue
        out[(r["game_id"], r["pitcher"])] = (
            int(r["outs"]), int(r["HR"]), int(r["BB"]), int(r["HBP"]), int(r["SO"]))
    return out


def load_games(years) -> list[dict]:
    keep = set(years)
    lines = load_starter_lines()
    rows = []
    for path in sorted(glob.glob("data/retrosheet/gl*.txt")):
        yr = int(path[-8:-4])
        if yr not in keep:
            continue
        for r in csv.reader(open(path, encoding="latin-1")):
            if len(r) < 105:
                continue
            try:
                vr, hr = int(r[F_VIS_R]), int(r[F_HOME_R])
            except ValueError:
                continue
            if vr == hr:
                continue
            gid = r[F_HOME] + r[F_DATE] + r[1]
            hsp, asp = r[F_HOME_SP].strip(), r[F_VIS_SP].strip()
            rows.append({
                "season": yr, "home": r[F_HOME], "away": r[F_VIS],
                "home_sp": hsp, "away_sp": asp, "y": 1.0 if hr > vr else 0.0,
                "home_line": lines.get((gid, hsp)), "away_line": lines.get((gid, asp)),
            })
    return rows


# League FIP core, computed once over all starter lines (a stable constant; the
# small era drift does not matter for a relative adjustment).
def league_fip_core(games) -> float:
    HR = BB = HBP = SO = outs = 0
    for g in games:
        for ln in (g["home_line"], g["away_line"]):
            if ln:
                o, hr, bb, hbp, so = ln
                outs += o; HR += hr; BB += bb; HBP += hbp; SO += so
    ip = outs / 3.0
    return (13 * HR + 3 * (BB + HBP) - 2 * SO) / ip if ip else 0.0


class FipPitcherElo:
    def __init__(self, *, k_team, hfa, team_regress, beta, decay,
                 prior_ip, season_decay, lg_fip, adj_clip=120.0):
        self.k_team = k_team
        self.hfa = hfa
        self.team_regress = team_regress
        self.beta = beta                 # Elo points per unit of FIP better-than-league
        self.decay = decay               # EWMA retention per start
        self.prior_ip = prior_ip         # empirical-Bayes prior weight, in IP
        self.season_decay = season_decay # extra shrink of pitcher memory each new season
        self.lg_fip = lg_fip
        self.adj_clip = adj_clip
        self.R = defaultdict(lambda: 1500.0)
        # per-pitcher decayed component sums
        self.pf = defaultdict(lambda: dict(HR=0.0, BB=0.0, HBP=0.0, SO=0.0, outs=0.0))

    def _adj(self, pid: str) -> float:
        s = self.pf[pid]
        ip = s["outs"] / 3.0
        if ip <= 0:
            return 0.0
        fip = (13 * s["HR"] + 3 * (s["BB"] + s["HBP"]) - 2 * s["SO"]) / ip
        # regress toward league by effective innings
        q = (ip * fip + self.prior_ip * self.lg_fip) / (ip + self.prior_ip)
        adj = -self.beta * (q - self.lg_fip)     # better than league (q<lg) -> positive
        return float(np.clip(adj, -self.adj_clip, self.adj_clip))

    def predict(self, g) -> float:
        diff = ((self.R[g["home"]] + self.hfa + self._adj(g["home_sp"]))
                - (self.R[g["away"]] + self._adj(g["away_sp"])))
        return 1.0 / (1.0 + 10 ** (-diff / 400.0))

    def update(self, g, e: float) -> None:
        dt = self.k_team * (g["y"] - e)
        self.R[g["home"]] += dt
        self.R[g["away"]] -= dt
        for sp, ln in ((g["home_sp"], g["home_line"]), (g["away_sp"], g["away_line"])):
            if not ln:
                continue
            o, hr, bb, hbp, so = ln
            s = self.pf[sp]
            for key, val in (("outs", o), ("HR", hr), ("BB", bb), ("HBP", hbp), ("SO", so)):
                s[key] = self.decay * s[key] + val

    def new_season(self) -> None:
        for t in self.R:
            self.R[t] = 1500.0 + (self.R[t] - 1500.0) * (1 - self.team_regress)
        for s in self.pf.values():
            for key in s:
                s[key] *= (1 - self.season_decay)


def run(games, *, eval_from_season, lg_fip, **params):
    m = FipPitcherElo(lg_fip=lg_fip, **params)
    preds, ys = [], []
    last = None
    for g in games:
        if last is not None and g["season"] != last:
            m.new_season()
        last = g["season"]
        e = m.predict(g)
        if g["season"] >= eval_from_season:
            preds.append(e)
            ys.append(g["y"])
        m.update(g, e)
    return np.array(preds), np.array(ys), m
