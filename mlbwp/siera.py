"""SIERA pitcher rating engine, shared by the offline build and the live server.

SIERA estimates a pitcher from strikeout rate, walk rate and net ground-ball rate
with squares and interactions -- it captures ground balls + non-linearity that FIP
and xFIP do not. Like the FIP core it is a per-pitcher EWMA of the raw components
(SO, BB, GB, FB, PU, PA) decayed per start; SIERA() reads the current, EB-shrunk
rates and applies the published FanGraphs formula. The offline build walks it over
history and the live server reconstructs it from the frozen checkpoint and replays
this season's starts through the identical update -- so the served value equals a
full retrain through today. Market-free: only the pitcher's own outcomes.
"""

from __future__ import annotations

from collections import defaultdict

# component order in the state vector / checkpoint
KEYS = ("SO", "BB", "GB", "FB", "PU", "PA")


def siera_value(so: float, bb: float, netgb: float) -> float:
    """FanGraphs SIERA from per-PA rates (the constant cancels in a differential)."""
    return (6.145 - 16.986 * so + 11.434 * bb - 1.858 * netgb
            + 7.653 * so * so - 6.664 * netgb * netgb
            + 10.130 * so * netgb - 5.195 * bb * netgb)


class SieraRater:
    def __init__(self, decay: float, prior: float, lg: dict):
        self.decay = decay
        self.prior = prior            # PA-equivalent shrink of each rate to league
        self.lg = lg                  # league per-PA rates {so,bb,gb,fb,pu}
        self.lg_netgb = lg["gb"] - lg["fb"] - lg["pu"]
        self.E = defaultdict(lambda: [0.0] * 6)   # pid -> EWMA [SO,BB,GB,FB,PU,PA]

    def update(self, pid, so, bb, gb, fb, pu, pa) -> None:
        e = self.E[pid]
        for i, v in enumerate((so, bb, gb, fb, pu, pa)):
            e[i] = self.decay * e[i] + v

    def value(self, pid) -> float:
        e = self.E[pid]
        pa = e[5] + self.prior
        so = (e[0] + self.prior * self.lg["so"]) / pa
        bb = (e[1] + self.prior * self.lg["bb"]) / pa
        netgb = ((e[2] - e[3] - e[4]) + self.prior * self.lg_netgb) / pa
        return siera_value(so, bb, netgb)

    def edge(self, home_sp, away_sp) -> float:
        """Home-minus-away SIERA differential (lower SIERA = better pitcher)."""
        return self.value(home_sp) - self.value(away_sp)

    def load_state(self, state: dict) -> "SieraRater":
        for pid, vec in state.items():
            self.E[pid] = list(vec)
        return self
