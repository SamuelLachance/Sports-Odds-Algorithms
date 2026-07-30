"""Does schedule (rest / density / travel / time-zone) add over the 0.67509 model?

Locked-split test. All computed leak-safe by walking games in date order and
reading each team's state as it stood BEFORE the game:
  rest      days since the team's last game
  density   games in the last 7 days
  travel    haversine km from the team's last venue to this game's venue
  tz        time-zone hours crossed since the last venue
Home-minus-away differentials are added to the DEV blend and scored once on TEST.
"""

from __future__ import annotations

import csv
import math
import os
from collections import defaultdict, deque
from datetime import datetime

_T = os.environ.get("NHL_EVAL_THREADS", "4")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, _T)

import sys  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

sys.path.insert(0, ".")
from mlbwp.ingest import league_fip_core, load_games  # noqa: E402
from mlbwp.rating import FipPitcherElo  # noqa: E402
from mlbwp.train import PARAMS  # noqa: E402

EPS = 1e-15
DEV = range(2002, 2016)
TEST = [y for y in range(2016, 2025) if y != 2020]

# retro code -> (lat, lon, tz_hours)
PARK = {
    "ANA": (33.80, -117.88, -8), "ARI": (33.45, -112.07, -7), "ATL": (33.89, -84.47, -5),
    "BAL": (39.28, -76.62, -5), "BOS": (42.35, -71.10, -5), "CHN": (41.95, -87.66, -6),
    "CHA": (41.83, -87.63, -6), "CIN": (39.10, -84.51, -5), "CLE": (41.50, -81.69, -5),
    "COL": (39.76, -104.99, -7), "DET": (42.34, -83.05, -5), "HOU": (29.76, -95.36, -6),
    "KCA": (39.05, -94.48, -6), "LAN": (34.07, -118.24, -8), "MIA": (25.78, -80.22, -5),
    "MIL": (43.03, -87.97, -6), "MIN": (44.98, -93.28, -6), "NYN": (40.76, -73.85, -5),
    "NYA": (40.83, -73.93, -5), "OAK": (37.75, -122.20, -8), "ATH": (37.75, -122.20, -8),
    "PHI": (39.91, -75.17, -5), "PIT": (40.45, -80.01, -5), "SDN": (32.71, -117.16, -8),
    "SEA": (47.59, -122.33, -8), "SFN": (37.78, -122.39, -8), "SLN": (38.62, -90.19, -6),
    "TBA": (27.77, -82.65, -5), "TEX": (32.75, -97.08, -6), "TOR": (43.64, -79.39, -5),
    "WAS": (38.87, -77.01, -5), "FLO": (25.94, -80.24, -5), "MON": (45.50, -73.57, -5),
}


def haversine(a, b):
    if a not in PARK or b not in PARK:
        return 0.0
    (la1, lo1, _), (la2, lo2, _) = PARK[a], PARK[b]
    r = 6371.0
    p1, p2 = math.radians(la1), math.radians(la2)
    dp, dl = math.radians(la2 - la1), math.radians(lo2 - lo1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def days(d1, d2):
    return (datetime.strptime(d1, "%Y%m%d") - datetime.strptime(d2, "%Y%m%d")).days


def ll_vec(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def ll(p, y):
    return float(ll_vec(p, y).mean())


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def boot(a, b, n=10000, seed=7):
    d = a - b
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def main():
    games = load_games()
    m = FipPitcherElo(lg_fip=league_fip_core(games), **PARAMS)
    ts = {r[0]: float(r[3]) for r in csv.reader(open("data/retro_events/ts_feature.csv")) if r[0] != "game_id"}

    last_date = {}          # team -> date str
    last_venue = {}         # team -> park code
    recent = defaultdict(lambda: deque(maxlen=15))   # team -> recent game dates
    rows = []
    prev_season = None
    for g in games:
        if prev_season is not None and g["season"] != prev_season:
            m.new_season()
            last_date.clear(); last_venue.clear(); recent.clear()
        prev_season = g["season"]
        venue = g["home"]

        def state(team):
            rest = min(days(g["date"], last_date[team]), 10) if team in last_date else 3
            dens = sum(1 for d in recent[team] if 0 <= days(g["date"], d) <= 7)
            trav = haversine(last_venue.get(team, venue), venue)
            tz = abs(PARK.get(last_venue.get(team, venue), (0, 0, 0))[2] - PARK.get(venue, (0, 0, 0))[2])
            return rest, dens, trav, tz
        hr = state(g["home"]); ar = state(g["away"])
        p = m.predict(g["home"], g["away"], g["home_sp"], g["away_sp"])
        if g["game_id"] in ts:
            rows.append((g["season"], p, g["y"], ts[g["game_id"]],
                         hr[0] - ar[0], hr[1] - ar[1], ar[2] - hr[2], ar[3] - hr[3]))
        # advance state
        m.update(g)
        for t in (g["home"], g["away"]):
            last_date[t] = g["date"]; last_venue[t] = venue; recent[t].append(g["date"])

    arr = np.array([r[:] for r in rows], dtype=float)
    seasons = arr[:, 0]
    lf = logit(arr[:, 1]); y = arr[:, 2]; tse = arr[:, 3]
    rest_d, dens_d, trav_d, tz_d = arr[:, 4], arr[:, 5], arr[:, 6], arr[:, 7]
    dmask = np.isin(seasons, list(DEV)); tmask = np.isin(seasons, TEST)

    def z(v):
        return (v - v[dmask].mean()) / (v[dmask].std() or 1.0)
    tsz = z(tse)
    F = {"rest": z(rest_d), "density": z(dens_d), "travel": z(trav_d), "tz": z(tz_d)}

    base = LogisticRegression(C=1e6, max_iter=1000).fit(
        np.column_stack([lf[dmask], tsz[dmask]]), y[dmask])
    Xd = np.column_stack([lf[dmask], tsz[dmask]] + [f[dmask] for f in F.values()])
    Xt = np.column_stack([lf[tmask], tsz[tmask]] + [f[tmask] for f in F.values()])
    plus = LogisticRegression(C=1e6, max_iter=1000).fit(Xd, y[dmask])
    print(f"games: {len(y):,}")
    print("DEV schedule coeffs:", {k: round(plus.coef_[0][2 + i], 4) for i, k in enumerate(F)})

    yt = y[tmask]
    p_base = base.predict_proba(np.column_stack([lf[tmask], tsz[tmask]]))[:, 1]
    p_plus = plus.predict_proba(Xt)[:, 1]
    print(f"\n=== LOCKED TEST  n={len(yt):,} ===")
    print(f"  FIP-Elo + TrueSkill      LL={ll(p_base, yt):.5f}   <- current model")
    print(f"  + schedule               LL={ll(p_plus, yt):.5f}")
    d, lo, hi = boot(ll_vec(p_base, yt), ll_vec(p_plus, yt))
    print(f"\ncurrent  minus  +schedule: {d:+.5f} nats (positive = schedule helps)")
    print(f"  95% paired bootstrap CI [{lo:+.5f}, {hi:+.5f}]")
    print(f"  {'SIGNIFICANT - schedule helps' if lo > 0 else ('SIGNIFICANT - it HURTS' if hi < 0 else 'NOT significant - no measurable gain')}")


if __name__ == "__main__":
    main()
