"""Glicko-2 for MLB.

Glickman's Glicko-2 with the three adaptations a baseball league needs:

* **Home advantage** applied at prediction time only, never folded into the
  rating — otherwise a team's number drifts with its home/road schedule balance.
* **Rating periods** rather than per-game updates. The variance estimator assumes
  strength is constant within a period and that all of the period's results are
  applied at once; updating game-by-game is a different, mis-specified estimator.
* **Season carryover** — rosters turn over, so ratings regress toward the mean and
  rating deviation inflates over the winter. The system should become *less*
  certain across an offseason, not more.

Modern MLB has no draws, so the draw branch is omitted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

SCALE = 173.7178
BASE_RATING = 1500.0
BASE_RD = 350.0
BASE_VOL = 0.06

_EPS = 1e-6
_MAX_ITER = 100


@dataclass
class Team:
    rating: float = BASE_RATING
    rd: float = BASE_RD
    vol: float = BASE_VOL
    games: int = 0

    @property
    def mu(self) -> float:
        return (self.rating - BASE_RATING) / SCALE

    @property
    def phi(self) -> float:
        return self.rd / SCALE


@dataclass
class Glicko2:
    tau: float = 0.5
    home_adv: float = 24.0
    season_regress: float = 0.25
    season_rd_bump: float = 45.0
    max_rd: float = BASE_RD
    min_rd: float = 30.0
    teams: dict[str, Team] = field(default_factory=dict)

    def team(self, name: str) -> Team:
        return self.teams.setdefault(str(name), Team())

    @staticmethod
    def _g(phi: float) -> float:
        return 1.0 / math.sqrt(1.0 + 3.0 * phi * phi / (math.pi ** 2))

    def expect_home(self, home: str, away: str) -> float:
        """P(home wins), widening for BOTH sides' uncertainty.

        Using only the opponent's RD (the raw Glicko form) overstates confidence
        when the home side is itself poorly estimated — exactly the situation in
        April and after a roster overhaul.
        """
        h, a = self.team(home), self.team(away)
        mu_h = h.mu + self.home_adv / SCALE
        phi = math.sqrt(h.phi ** 2 + a.phi ** 2)
        return 1.0 / (1.0 + math.exp(-self._g(phi) * (mu_h - a.mu)))

    def _new_vol(self, phi: float, v: float, delta: float, sigma: float) -> float:
        """Illinois-variant regula falsi for the volatility update."""
        a = math.log(sigma ** 2)
        phi2, d2 = phi ** 2, delta ** 2

        def f(x: float) -> float:
            ex = math.exp(x)
            return (ex * (d2 - phi2 - v - ex)) / (2.0 * (phi2 + v + ex) ** 2) - (x - a) / (self.tau ** 2)

        A = a
        if d2 > phi2 + v:
            B = math.log(d2 - phi2 - v)
        else:
            k = 1
            while f(a - k * self.tau) < 0 and k < _MAX_ITER:
                k += 1
            B = a - k * self.tau

        fa, fb = f(A), f(B)
        for _ in range(_MAX_ITER):
            if abs(B - A) <= _EPS:
                break
            C = A + (A - B) * fa / (fb - fa) if (fb - fa) != 0 else (A + B) / 2.0
            fc = f(C)
            if fc * fb <= 0:
                A, fa = B, fb
            else:
                fa /= 2.0
            B, fb = C, fc
        return math.exp(A / 2.0)

    def update_period(self, results: list[tuple[str, str, float]]) -> None:
        """One rating period. ``results`` = [(home, away, home_score in [0,1])]."""
        # Snapshot pre-period state so every team in the period is updated
        # against the ratings as they stood at the START of it.
        opp: dict[str, list[tuple[Team, float]]] = {}
        for home, away, s in results:
            h, a = self.team(home), self.team(away)
            opp.setdefault(str(home), []).append((Team(a.rating, a.rd, a.vol), s))
            opp.setdefault(str(away), []).append((Team(h.rating, h.rd, h.vol), 1.0 - s))

        new: dict[str, tuple[float, float, float]] = {}
        for name, games in opp.items():
            t = self.team(name)
            mu, phi = t.mu, t.phi
            v_inv = 0.0
            dsum = 0.0
            for o, score in games:
                g_j = self._g(o.phi)
                e_j = 1.0 / (1.0 + math.exp(-g_j * (mu - o.mu)))
                v_inv += g_j ** 2 * e_j * (1.0 - e_j)
                dsum += g_j * (score - e_j)
            if v_inv <= 0:
                continue
            v = 1.0 / v_inv
            delta = v * dsum
            sig = self._new_vol(phi, v, delta, t.vol)
            phi_star = math.sqrt(phi ** 2 + sig ** 2)
            phi_p = 1.0 / math.sqrt(1.0 / phi_star ** 2 + 1.0 / v)
            mu_p = mu + phi_p ** 2 * dsum
            new[name] = (BASE_RATING + SCALE * mu_p,
                         min(max(SCALE * phi_p, self.min_rd), self.max_rd),
                         sig)

        for name, (r, rd, vol) in new.items():
            t = self.team(name)
            t.rating, t.rd, t.vol = r, rd, vol
            t.games += 1

        for name, t in self.teams.items():
            if name not in new:      # idle teams grow less certain
                t.rd = min(math.sqrt(t.rd ** 2 + (SCALE * t.vol) ** 2), self.max_rd)

    def new_season(self) -> None:
        for t in self.teams.values():
            t.rating = BASE_RATING + (t.rating - BASE_RATING) * (1.0 - self.season_regress)
            t.rd = min(math.sqrt(t.rd ** 2 + self.season_rd_bump ** 2), self.max_rd)
