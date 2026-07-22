"""Team Elo + starting-pitcher rating — a two-factor rating for MLB.

The premise, established this session: team Elo is blind to who starts, and the
starting pitcher is the one large game-to-game swing that is orthogonal to team
strength. So each game's expected result is driven by the TEAM ratings plus a
STARTING-PITCHER adjustment, and both are learned online from the same residual.

The separation that makes the pitcher term meaningful:

    E = sigmoid( (R_home + hfa + P_home_sp) - (R_away + P_away_sp) )
    residual = y - E
    R_home += k_team   * residual ;  R_away -= k_team   * residual
    P_home += k_pitch  * residual ;  P_away -= k_pitch  * residual

E already contains the team ratings, so the residual is net of team quality. A
starter on a strong team whose games go exactly as team-Elo expects gets no
pitcher update; only SYSTEMATIC over/under-performance when HE starts moves his
number. Because a starter appears ~every fifth game while the team plays daily,
the shared residual is partitioned by frequency — the pitcher rating converges
to that pitcher's repeatable marginal effect.

Honest limitation, stated up front: the pitcher signal here comes from binary
game outcomes, not from the pitcher's own line (K/BB/HR/IP). FiveThirtyEight
rated pitchers from performance stats, which is less noisy. If this W/L-only
version shows a real but small gain, the next step is a Game-Score-based pitcher
rating from box scores. This is the fast, self-contained test of the thesis on
data already in hand.

Data: Retrosheet game logs. "The information used here was obtained free of charge
from and is copyrighted by Retrosheet. Interested parties may contact Retrosheet at
www.retrosheet.org."
"""

from __future__ import annotations

import csv
import glob
from collections import defaultdict

import numpy as np

F_DATE, F_VIS, F_HOME, F_VIS_R, F_HOME_R, F_VIS_SP, F_HOME_SP = 0, 3, 6, 9, 10, 101, 103


def load_games(years) -> list[dict]:
    keep = set(years)
    rows = []
    for path in sorted(glob.glob("data/retrosheet/gl*.txt")):
        yr = int(path[-8:-4])
        if yr not in keep:
            continue
        with open(path, newline="", encoding="latin-1") as fh:
            for r in csv.reader(fh):
                if len(r) < 105:
                    continue
                try:
                    vr, hr = int(r[F_VIS_R]), int(r[F_HOME_R])
                except ValueError:
                    continue
                if vr == hr:
                    continue
                rows.append({
                    "season": yr, "home": r[F_HOME], "away": r[F_VIS],
                    "home_sp": r[F_HOME_SP].strip(), "away_sp": r[F_VIS_SP].strip(),
                    "y": 1.0 if hr > vr else 0.0,
                })
    return rows  # already chronological within & across files


class PitcherElo:
    def __init__(self, *, k_team, hfa, team_regress,
                 k_pitch, pitch_regress, pitch_clip=150.0):
        self.k_team = k_team
        self.hfa = hfa
        self.team_regress = team_regress
        self.k_pitch = k_pitch
        self.pitch_regress = pitch_regress
        self.pitch_clip = pitch_clip
        self.R = defaultdict(lambda: 1500.0)
        self.P = defaultdict(float)   # pitcher adjustment in Elo points; 0 = league-avg starter

    def predict(self, g) -> float:
        diff = ((self.R[g["home"]] + self.hfa + self.P[g["home_sp"]])
                - (self.R[g["away"]] + self.P[g["away_sp"]]))
        return 1.0 / (1.0 + 10 ** (-diff / 400.0))

    def update(self, g, e: float) -> None:
        resid = g["y"] - e
        dt = self.k_team * resid
        self.R[g["home"]] += dt
        self.R[g["away"]] -= dt
        dp = self.k_pitch * resid
        self.P[g["home_sp"]] = float(np.clip(self.P[g["home_sp"]] + dp,
                                             -self.pitch_clip, self.pitch_clip))
        self.P[g["away_sp"]] = float(np.clip(self.P[g["away_sp"]] - dp,
                                             -self.pitch_clip, self.pitch_clip))

    def new_season(self) -> None:
        for t in self.R:
            self.R[t] = 1500.0 + (self.R[t] - 1500.0) * (1 - self.team_regress)
        for p in self.P:
            self.P[p] *= (1 - self.pitch_regress)


def run(games, *, eval_from_season, **params):
    m = PitcherElo(**params)
    preds, ys = [], []
    last_season = None
    for g in games:
        if last_season is not None and g["season"] != last_season:
            m.new_season()
        last_season = g["season"]
        e = m.predict(g)
        if g["season"] >= eval_from_season:
            preds.append(e)
            ys.append(g["y"])
        m.update(g, e)
    return np.array(preds), np.array(ys), m
