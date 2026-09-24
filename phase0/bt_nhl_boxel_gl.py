"""Breakthrough pick nhl_boxel -- shared STARTING-GOALIE LEVEL module (GL).

The pick spec says: "import phase0/bt_nhl_gmar_gl.py, built per pick nhl_gmar if
it does not exist yet". At build time (2026-09-24) that module did NOT exist, and
writing a file into another pick's namespace would collide with the nhl_gmar
agent, so this is boxel's own build of the same pre-declared construction:

    K = 400 xGA of prior weight, per-appearance decay 0.998, 0.85 season carry,
    walk-forward entrant prior, G expressed in goals/game.

Construction (fixed a priori, never searched):
  per goalie  A = decayed sum (xGA - GA)      X = decayed sum xGA
              decay 0.998 per appearance of THAT goalie; x0.85 at each season
              boundary (both A and X).
  rate        r = (A + K * mu) / (X + K)       K = 400 xGA
  level       G = r * xbar                    xbar = walk-forward mean team-game
                                              xGA (all strictly earlier games)
  prior mu    established goalies (first appearance before 2011-07-01, i.e.
              seen in the 2010-11 warm season): walk-forward league-mean rate
              sum(xGA-GA)/sum(xGA) over all strictly earlier goalie-games.
              entrants (first appearance on/after 2011-07-01) with < 82
              appearances: walk-forward mean rate of entrants' first 40
              appearances, pooled over COMPLETED seasons only. Fallback while
              that pool holds < 200 goalie-games: league-mean rate - 0.03
              (a-priori rookie discount, ~0.08 goals/game).
  tonight     the STARTER is the first-shot goalie from
              bt_nhl_anatomy_build.load_starters (public at warm-ups). A missing
              starter uses the league-mean rate (counted).

Walk-forward: one ordered pass, read-then-update; GA/xGA of tonight's game are
folded in only after tonight's level has been read. Per-goalie state changes
right after his game (a goalie plays at most once per date). LEAGUE-level means
(league rate, xbar) are DATE-BATCHED: a date's games are folded in only when the
next date starts, so two games on the same date never see each other. Every
walk-forward mean records the last date it absorbed and is asserted < tonight.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

K_XGA = 400.0
DECAY = 0.998
CARRY = 0.85
ENTRANT_DATE = "2011-07-01"
ENTRANT_MAXN = 82
ENTRANT_FIRST = 40
ENTRANT_MIN_ROWS = 200
FALLBACK_DELTA = -0.03


def goalie_levels(games, starters, ggames):
    """Per game: (G_home, G_away) in goals/game, plus bookkeeping counts.

    games    : chronological DEV game dicts (nhl_depth_eval.dev_games()).
    starters : gid -> {1: home starter id, 0: away starter id}.
    ggames   : gid -> [(goalie_id, team, xga, ga)] (every goalie who played).
    """
    n = len(games)
    Gh = np.zeros(n)
    Ga = np.zeros(n)
    A = defaultdict(float)
    X = defaultdict(float)
    napp = defaultdict(int)
    first = {}
    # walk-forward league sums (date-batched: day_* folded in at the next date)
    lg_num = lg_den = 0.0
    tx_sum = 0.0
    tx_cnt = 0
    lg_last = ""                 # last date absorbed into the league sums
    day = [0.0, 0.0, 0.0, 0]     # lg_num, lg_den, tx_sum, tx_cnt of the current date
    cur_date = None
    # entrant pool (completed seasons) + current-season buffer
    pool_num = pool_den = 0.0
    pool_rows = 0
    pool_max_season = None
    buf_num = buf_den = 0.0
    buf_rows = 0
    missing = 0
    prev = None
    for i, g in enumerate(games):
        s = g["season"]
        if g["date"] != cur_date:
            if cur_date is not None and day[3]:
                lg_num += day[0]
                lg_den += day[1]
                tx_sum += day[2]
                tx_cnt += day[3]
                lg_last = cur_date
            day = [0.0, 0.0, 0.0, 0]
            cur_date = g["date"]
        if prev is not None and s != prev:
            for k in A:
                A[k] *= CARRY
                X[k] *= CARRY
            pool_num += buf_num
            pool_den += buf_den
            pool_rows += buf_rows
            pool_max_season = prev
            buf_num = buf_den = 0.0
            buf_rows = 0
        prev = s
        # ---------------- read (strictly earlier data only) ----------------
        assert lg_last < g["date"]
        assert pool_max_season is None or pool_max_season < s
        lm = lg_num / lg_den if lg_den > 0 else 0.0
        xbar = tx_sum / tx_cnt if tx_cnt else 2.6
        if pool_rows >= ENTRANT_MIN_ROWS and pool_den > 0:
            ent_mu = pool_num / pool_den
        else:
            ent_mu = lm + FALLBACK_DELTA
        st = starters.get(g["game_id"], {})
        for flag, arr in ((1, Gh), (0, Ga)):
            gk = st.get(flag)
            if gk is None:
                missing += 1
                arr[i] = lm * xbar
                continue
            f = first.get(gk, g["date"])
            is_ent = f >= ENTRANT_DATE and napp[gk] < ENTRANT_MAXN
            mu = ent_mu if is_ent else lm
            arr[i] = (A[gk] + K_XGA * mu) / (X[gk] + K_XGA) * xbar
        # ---------------- update AFTER the read ----------------
        team_x = defaultdict(float)
        for gk, team, xga, ga in ggames.get(g["game_id"], []):
            if gk not in first:
                first[gk] = g["date"]
            A[gk] = DECAY * A[gk] + (xga - ga)
            X[gk] = DECAY * X[gk] + xga
            napp[gk] += 1
            if first[gk] >= ENTRANT_DATE and napp[gk] <= ENTRANT_FIRST:
                buf_num += xga - ga
                buf_den += xga
                buf_rows += 1
            day[0] += xga - ga
            day[1] += xga
            team_x[team] += xga
        for v in team_x.values():
            day[2] += v
            day[3] += 1
    info = {"missing_starter_sides": missing, "entrant_pool_rows_final": pool_rows,
            "construction": "boxel build (bt_nhl_gmar_gl.py absent at build time)",
            "K_xga": K_XGA, "decay": DECAY, "carry": CARRY}
    return Gh, Ga, info
