"""The per-plate-appearance TrueSkill engine, shared by training and serving.

Every plate appearance is a 1v1 match — batter (offense) vs pitcher (run
prevention), one binary outcome (reached base = batter wins, retired = pitcher
wins). Both live on one latent scale. The engine is the single source of truth:
the offline build (phase0/trueskill_pa.py) walks it over all historical PAs and
the live server (mlbwp/serve.py) reconstructs it from the frozen (mu, var)
checkpoint and replays this season's PAs through the identical update, so the
served ratings equal a full retrain through today.

No market input of any kind — only who batted, who pitched, and whether the
batter reached base.
"""

from __future__ import annotations

import math
from collections import defaultdict

MU0, SIG0 = 25.0, 25.0 / 3.0
BETA = SIG0 / 2.0
TAU = SIG0 / 100.0
_SQRT2PI = math.sqrt(2 * math.pi)


def _pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / _SQRT2PI


def _cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


class TrueSkill1v1:
    """Closed-form two-player TrueSkill. Winner/loser passed each update."""

    def __init__(self, beta=BETA, tau=TAU, mu0=MU0, sig0=SIG0):
        self.beta, self.tau, self.mu0, self.sig0 = beta, tau, mu0, sig0
        self.mu = defaultdict(lambda: mu0)
        self.var = defaultdict(lambda: sig0 * sig0)
        self.n = defaultdict(int)

    def update(self, winner, loser) -> None:
        vw = self.var[winner] + self.tau ** 2
        vl = self.var[loser] + self.tau ** 2
        c2 = 2 * self.beta ** 2 + vw + vl
        c = math.sqrt(c2)
        t = (self.mu[winner] - self.mu[loser]) / c
        denom = _cdf(t)
        v = _pdf(t) / denom if denom > 1e-12 else -t
        w = min(max(v * (v + t), 0.0), 1.0)
        self.mu[winner] += vw / c * v
        self.mu[loser] -= vl / c * v
        self.var[winner] = vw * (1 - vw / c2 * w)
        self.var[loser] = vl * (1 - vl / c2 * w)
        self.n[winner] += 1
        self.n[loser] += 1

    def apply_pa(self, batter, pitcher, on_base) -> None:
        """One plate appearance: batter reached (on_base) => batter wins."""
        if on_base:
            self.update(batter, pitcher)
        else:
            self.update(pitcher, batter)

    def load_state(self, state: dict) -> "TrueSkill1v1":
        """Seed mu/var from a frozen checkpoint {id: [mu, var]}."""
        for pid, (mu, var) in state.items():
            self.mu[pid] = mu
            self.var[pid] = var
            self.n[pid] = 1
        return self
