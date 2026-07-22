"""Faithful backtest of the JamesQuintero MLB algorithm (algo.py, calculate_V2).

Source: https://github.com/JamesQuintero/Sports-Odds-Algorithms (no license stated,
last push 2020-01-28). Reimplemented here rather than executed, because the original
depends on a 2016-era ESPN scraper and its own cached team_data. The arithmetic below
is transcribed line-for-line from `algo.py::calculate_V2` and `calculate_points`, MLB
branch, including the truncation, the floor-at-50 and the sign restoration.

Sign convention matches the original: `returned1` is the AWAY team, `returned2` the
HOME team, so a positive score favours the away side.

Every feature is computed from games strictly BEFORE the game being predicted, so this
is a genuine walk-forward backtest. Retrosheet game logs are the only input; the model
uses no odds, consistent with the original (its oddsportal data drove betting-strategy
backtests, not the model).

Data: Retrosheet game logs. "The information used here was obtained free of charge from
and is copyrighted by Retrosheet. Interested parties may contact Retrosheet at
www.retrosheet.org."
"""

from __future__ import annotations

import csv
import glob
from collections import defaultdict, deque

import numpy as np

EPS = 1e-15

# --- transcribed constants, MLB branch -------------------------------------
DIVIDERS = [6.0, 6.0, 3.0, 3.0, 0.3, 1.5]
MAX_POINTS = [10.0, 10.0, 7.0, 7.0, 10.0, 8.0]


def _trunc2(v: float) -> float:
    """universal_functions.convert_number: int(number*100)/100 (truncates)."""
    return int(v * 100) / 100


def _poly(i: int, x: float) -> float:
    if i == 0:
        return -0.0378 * x ** 2 + 1.5474 * x + 50.776
    if i == 1:
        return -0.2226 * x ** 2 + 3.8472 * x + 47.282
    if i == 2:
        return 0.3025 * x ** 2 + 1.4568 * x + 49.518
    if i == 3:
        return 0.3039 * x ** 2 + 0.1154 * x + 51.6
    if i == 4:
        return -0.1938 * x ** 2 + 3.1638 * x + 49.105
    return 0.0301 * x ** 3 + 0.5611 * x ** 2 - 0.6103 * x + 51.278


def quintero_prob_away(algo_vars: list[float]) -> float:
    """Return P(away win) exactly as calculate_V2 computes it."""
    v = list(algo_vars)
    for i in range(6):
        v[i] /= DIVIDERS[i]
        v[i] = max(-MAX_POINTS[i], min(MAX_POINTS[i], v[i]))

    odds = []
    for i in range(6):
        y = _trunc2(_poly(i, abs(v[i])))
        if y < 50:          # original floors sub-50 outputs at 50
            y = 50.0
        y -= 50.0
        if v[i] < 0:        # restore direction
            y = -y
        odds.append(y)

    average = sum(odds) / len(odds)
    # Original then does average+=50 / average-=50 purely for display; the
    # signed magnitude is the away-side edge either way.
    return (50.0 + average) / 100.0


# --- Retrosheet game-log parsing -------------------------------------------
# Field positions per Retrosheet's game log spec.
F_DATE, F_VIS, F_HOME, F_VIS_R, F_HOME_R = 0, 3, 6, 9, 10


def load_games() -> list[dict]:
    rows = []
    for path in sorted(glob.glob("data/retrosheet/gl*.txt")):
        with open(path, newline="", encoding="latin-1") as fh:
            for r in csv.reader(fh):
                if len(r) < 11:
                    continue
                try:
                    vr, hr = int(r[F_VIS_R]), int(r[F_HOME_R])
                except ValueError:
                    continue
                if vr == hr:
                    continue  # no ties in modern MLB
                rows.append({
                    "date": r[F_DATE], "season": int(r[F_DATE][:4]),
                    "away": r[F_VIS], "home": r[F_HOME],
                    "away_r": vr, "home_r": hr,
                    "home_win": 1.0 if hr > vr else 0.0,
                })
    rows.sort(key=lambda g: (g["date"], g["home"]))
    return rows


class TeamState:
    """Running, strictly-prior-only record and scoring state for one team."""

    def __init__(self):
        self.w = self.l = 0
        self.home_w = self.home_l = 0
        self.away_w = self.away_l = 0
        self.rs = self.ra = 0
        self.n = 0
        self.last10 = deque(maxlen=10)          # 1/0 win flags
        self.last10_rs = deque(maxlen=10)
        self.last10_ra = deque(maxlen=10)
        self.home10 = deque(maxlen=10)          # win flags, home games only
        self.away10 = deque(maxlen=10)

    def update(self, won: bool, rs: int, ra: int, is_home: bool):
        self.w += won
        self.l += not won
        self.rs += rs
        self.ra += ra
        self.n += 1
        self.last10.append(1 if won else 0)
        self.last10_rs.append(rs)
        self.last10_ra.append(ra)
        if is_home:
            self.home_w += won
            self.home_l += not won
            self.home10.append(1 if won else 0)
        else:
            self.away_w += won
            self.away_l += not won
            self.away10.append(1 if won else 0)

    # feature helpers
    def diff_record(self) -> float:
        return self.w - self.l

    def avg_diff(self) -> float:
        return (self.rs - self.ra) / self.n if self.n else 0.0

    def avg_diff10(self) -> float:
        if not self.last10_rs:
            return 0.0
        return (sum(self.last10_rs) - sum(self.last10_ra)) / len(self.last10_rs)


def build_algo_vars(a: TeamState, h: TeamState) -> list[float]:
    """The six calculate_points outputs, away-minus-home."""
    return [
        a.diff_record() - h.diff_record(),                                   # seasonal_records
        (a.away_w - a.away_l) - (h.home_w - h.home_l),                       # home_away_records
        (sum(a.away10) - (len(a.away10) - sum(a.away10)))
        - (sum(h.home10) - (len(h.home10) - sum(h.home10))),                 # home_away_10_game
        sum(a.last10) - sum(h.last10),                                       # last_10_games
        a.avg_diff() - h.avg_diff(),                                         # avg_points
        a.avg_diff10() - h.avg_diff10(),                                     # avg_points_10_games
    ]


# --- baselines --------------------------------------------------------------
class Elo:
    def __init__(self, k=4.0, hfa=24.0, revert=0.25):
        self.k, self.hfa, self.revert = k, hfa, revert
        self.r = defaultdict(lambda: 1500.0)

    def p_home(self, home, away):
        return 1.0 / (1.0 + 10 ** (-((self.r[home] + self.hfa - self.r[away]) / 400.0)))

    def update(self, home, away, home_win):
        e = self.p_home(home, away)
        d = self.k * (home_win - e)
        self.r[home] += d
        self.r[away] -= d

    def new_season(self):
        for t in self.r:
            self.r[t] = 1500.0 + (self.r[t] - 1500.0) * (1 - self.revert)


def ll_vec(p, y):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def boot(a, b, n=10000, seed=7):
    d = a - b
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def main():
    games = load_games()
    print(f"games loaded: {len(games):,}  ({games[0]['date']} .. {games[-1]['date']})")

    st: dict[str, TeamState] = defaultdict(TeamState)
    elo = Elo()
    season = None
    MIN_N = 20  # both teams need a real sample before the model means anything

    p_q, p_e, ys, skipped = [], [], [], 0

    for g in games:
        if season is not None and g["season"] != season:
            st = defaultdict(TeamState)   # records reset each season
            elo.new_season()
        season = g["season"]

        a, h = st[g["away"]], st[g["home"]]
        if a.n >= MIN_N and h.n >= MIN_N:
            p_away = quintero_prob_away(build_algo_vars(a, h))
            p_q.append(1.0 - p_away)               # convert to P(home win)
            p_e.append(elo.p_home(g["home"], g["away"]))
            ys.append(g["home_win"])
        else:
            skipped += 1

        hw = g["home_win"] == 1.0
        h.update(hw, g["home_r"], g["away_r"], True)
        a.update(not hw, g["away_r"], g["home_r"], False)
        elo.update(g["home"], g["away"], g["home_win"])

    p_q, p_e, ys = np.array(p_q), np.array(p_e), np.array(ys)
    print(f"scored: {len(ys):,}   skipped (warm-up, <{MIN_N} games): {skipped:,}")
    print(f"home win rate: {ys.mean():.4f}")

    def ll(p):
        return float(ll_vec(p, ys).mean())

    home_const = np.full_like(ys, ys.mean())
    print()
    print(f"{'model':<38} {'log loss':>10} {'Brier':>9} {'acc':>7}")
    print("-" * 68)
    for name, p in [
        ("Quintero MLB algo (calculate_V2)", p_q),
        ("Elo (k=4, hfa=24, 25% revert)", p_e),
        ("constant home rate", home_const),
        ("constant 0.5", np.full_like(ys, 0.5)),
    ]:
        print(f"{name:<38} {ll(p):>10.5f} {float(np.mean((p-ys)**2)):>9.5f} "
              f"{float(np.mean((p>0.5)==(ys>0.5))):>7.4f}")

    d, lo, hi = boot(ll_vec(p_q, ys), ll_vec(p_e, ys))
    print()
    print(f"Quintero minus Elo: {d:+.5f} nats  95% CI [{lo:+.5f}, {hi:+.5f}]")
    print("  (positive = Elo better)" if d > 0 else "  (negative = Quintero better)")
    print(f"prediction spread: sd={p_q.std():.4f} min={p_q.min():.3f} max={p_q.max():.3f}"
          f"   (Elo sd={p_e.std():.4f})")


if __name__ == "__main__":
    main()
