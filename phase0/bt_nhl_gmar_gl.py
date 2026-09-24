"""Shared NHL goalie module -- starting-goalie LEVEL *inside* the W/L Elo.

Breakthrough program (documents/breakthrough_program_prereg_2026_09_24.md),
screen nhl_gmar (picks nhl_rapmel / nhl_boxel import this module).

MLB's biggest single win was the starter's own process stat (FIP), EWMA-decayed,
empirical-Bayes regressed, converted to Elo points and added to the team rating
for that game, so the team rating learns strength NET of its pitchers. This is
the hockey transplant: tonight's starting goalie's GSAx per xGA faced, regressed
to a reliability-calibrated prior, converted to goals/game and to Elo points.

    gl_elo(games, starters, gg, gamma, mode='new'|'probe', starter_override=None)

Inputs (all DEV-filtered by the caller; this module never reads a file):
  games     harness game list (nhl_depth_eval.dev_games(), order (date, home))
  starters  gid -> {1: home starter id, 0: away starter id}  (first-shot goalie,
            bt_nhl_anatomy_build.load_starters; public at warmups)
  gg        gid -> [(goalie_id, team, xga, ga)] every appearance incl. relief
            (bt_nhl_anatomy_build.load_goalie_games) -- POST-game, update only

mode='new'  (construction G of the nhl_gmar spec)
  goalie state carried across teams and seasons:
    N_g = sum_j d^j (xGA_j - GA_j),  X_g = sum_j d^j xGA_j,  d = 0.998 / appearance,
    both x 0.85 at every season boundary.
  entrant = first appearance in the data on/after 2011-07-01 (2010-11 goalies are
  left-censored veterans).  prior mean m_g = m_E(s) for an entrant with < 100
  appearances, else 0;  m_E(s) = sum(xGA-GA)/sum(xGA) over entrants' first 100
  appearances in COMPLETED seasons < s (fallback -0.004/xGA below 1,000 xGA of
  evidence -- declared DEV constant, never searched).
  r_g = (N_g + K m_g) / (X_g + K), K = 400 xGA (fixed from DEV split-half
  reliability r = 0.131 of half-season GSAx/xGA; never searched).
  G = r_g * xbar_t, xbar_t = running league mean goalie-faced xGA per team-game
  over games on dates strictly before t.
  E = (R_h + gamma G_h + 30) - (R_a + gamma G_a);  p = 1/(1+10^(-E/400));
  R_h += 8 (y-p), R_a -= 8 (y-p);  R <- 1500 + 0.7 (R-1500) at season boundaries
  (shipped k/ha/regress, not re-tuned).  Optional A5 diagnostic: E also gets
  -delta*D_h + delta*D_a (absent-regular minutes above replacement).
  gamma = 0 reproduces nhl_depth_eval.run_elo_arr(games, 8, 30, 0.30).

mode='probe'  exactly bt_nhl_anatomy_probe.goalie_elo (gamma plays c_pts):
  per-appearance GSAx numerator / decayed appearance count, decay 0.98, prior
  weight 20 appearances, role prior -0.10 (<30 starts) / -0.05 (<100 starts).

Per-game order (W1): read G_h, G_a from state -> predict -> update R -> fold ALL
of this game's goalie rows into N, X -> (xbar committed only when the date
changes, i.e. strictly-earlier dates).  Every state read is guarded by an
assertion that it was last written by a strictly earlier game.

Market-blind: nothing here touches odds.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

# shipped Elo core (data/nhl_model.json / nhl_depth_eval.SHIPPED_ELO)
K_ELO, HA_ELO, REG_ELO = 8, 30, 0.30

# construction G constants -- fixed a priori, never searched
D_APP = 0.998            # per-appearance decay of goalie evidence
SEASON_DECAY = 0.85      # goalie evidence multiplier at each season boundary
K_PRIOR = 400.0          # EB prior weight in xGA (from DEV split-half r = 0.131)
ENTRANT_FROM = "2011-07-01"
ENTRANT_N = 100          # appearances during which an entrant gets the entrant prior
ME_MIN_EVID = 1000.0     # xGA of entrant evidence needed before m_E is data-driven
ME_FALLBACK = -0.004     # GSAx per xGA, declared DEV constant
XBAR_INIT = 2.4          # league xGA/team-game before ANY data (2010-10-07 only; warm-up)

# probe constants (bt_nhl_anatomy_probe)
P_DECAY, P_PRIOR = 0.98, 20.0


class _Guard:
    """W1 instrumentation: remembers the index of the last game that wrote each key
    and asserts every read at game i sees only writes from games < i."""

    def __init__(self):
        self.last = {}

    def wrote(self, key, i):
        self.last[key] = i

    def read(self, key, i):
        j = self.last.get(key, -1)
        assert j < i, f"W1 violated: state {key!r} written by game {j} read at game {i}"


def gl_elo(games, starters, gg, gamma, mode="new", starter_override=None,
           role_prior=True, d_home=None, d_away=None, delta=0.0, snap=None,
           guard=True):
    """Return dict(p, G_h, G_a, R_h, R_a, fb_h, fb_a, gk_h, gk_a, me, xbar_at, snap).

    p        home-win probability (pre-game)
    G_h/G_a  goalie term for tonight's starter: goals saved / game above average
             ('new') or regressed GSAx / appearance ('probe')
    R_h/R_a  pre-game team ratings (net of goalie)
    fb_h/a   1 where no shot row identified the starter and the team's previous
             starter was used ('new' only)
    snap     optional (date_str, [goalie ids]) -> G of those goalies at the start
             of that date (face validity)
    """
    n = len(games)
    out = np.empty(n)
    G_h = np.zeros(n)
    G_a = np.zeros(n)
    R_h = np.empty(n)
    R_a = np.empty(n)
    fb_h = np.zeros(n, dtype=bool)
    fb_a = np.zeros(n, dtype=bool)
    gk_h = np.full(n, -1, dtype=np.int64)
    gk_a = np.full(n, -1, dtype=np.int64)
    use_d = d_home is not None and delta != 0.0
    gd = _Guard() if guard else None

    R = {}
    if mode == "probe":
        num, den, starts = defaultdict(float), defaultdict(float), defaultdict(int)
        prev = None
        for i, g in enumerate(games):
            if prev is not None and g["season"] != prev:
                for t in R:
                    R[t] = 1500 + (R[t] - 1500) * (1 - REG_ELO)
            prev = g["season"]
            st = starters.get(g["game_id"], {})
            if starter_override is not None:
                st = starter_override.get(g["game_id"], {})
            adj = {}
            for side, flag, garr, karr in (("home", 1, G_h, gk_h), ("away", 0, G_a, gk_a)):
                gk = st.get(flag)
                if gk is None:
                    adj[side] = 0.0
                    continue
                if gd:
                    gd.read(("g", gk), i)
                m0 = 0.0
                if role_prior:
                    nst = starts[gk]
                    m0 = -0.10 if nst < 30 else (-0.05 if nst < 100 else 0.0)
                rt = (num[gk] + m0 * P_PRIOR) / (den[gk] + P_PRIOR)
                garr[i] = rt
                karr[i] = gk
                adj[side] = gamma * rt
            R_h[i] = R.setdefault(g["home"], 1500.0)
            R_a[i] = R.setdefault(g["away"], 1500.0)
            rh = R_h[i] + adj["home"]
            ra = R_a[i] + adj["away"]
            p = 1.0 / (1.0 + 10 ** (-((rh + HA_ELO) - ra) / 400.0))
            out[i] = p
            R[g["home"]] += K_ELO * (g["y"] - p)
            R[g["away"]] -= K_ELO * (g["y"] - p)
            for side, flag in (("home", 1), ("away", 0)):
                gk = st.get(flag)
                if gk is not None:
                    starts[gk] += 1
            for gk, _team, xga, ga in gg.get(g["game_id"], []):
                num[gk] = P_DECAY * num[gk] + (xga - ga)
                den[gk] = P_DECAY * den[gk] + 1.0
                if gd:
                    gd.wrote(("g", gk), i)
        return {"p": out, "G_h": G_h, "G_a": G_a, "R_h": R_h, "R_a": R_a,
                "fb_h": fb_h, "fb_a": fb_a, "gk_h": gk_h, "gk_a": gk_a,
                "me": {}, "snap": None}

    assert mode == "new", mode
    N = defaultdict(float)
    X = defaultdict(float)
    napp = defaultdict(int)
    first_date = {}
    ent_contrib = defaultdict(lambda: [0.0, 0.0])   # season -> [sum(xga-ga), sum(xga)]
    prev_starter = {}
    xb_sum, xb_cnt = 0.0, 0                          # committed (strictly earlier dates)
    pend_sum, pend_cnt = 0.0, 0
    last_committed_date = ""
    cur_date = None
    me_by_season = {}
    m_E = ME_FALLBACK
    snap_out = None
    prev_season = None

    def is_entrant(gk, today):
        fd = first_date.get(gk)
        return (fd if fd is not None else today) >= ENTRANT_FROM

    for i, g in enumerate(games):
        gid = g["game_id"]
        s = g["season"]
        dstr = g["date"]
        # ---- date boundary: commit xbar evidence from strictly earlier dates ----
        if dstr != cur_date:
            if cur_date is not None:
                assert cur_date < dstr, "games not in date order"
                xb_sum += pend_sum
                xb_cnt += pend_cnt
                pend_sum, pend_cnt = 0.0, 0
                last_committed_date = cur_date
            cur_date = dstr
        assert last_committed_date < dstr          # W3: xbar uses strictly earlier dates
        # ---- season boundary ----
        if prev_season is not None and s != prev_season:
            for t in R:
                R[t] = 1500 + (R[t] - 1500) * (1 - REG_ELO)
            for gk in N:
                N[gk] *= SEASON_DECAY
                X[gk] *= SEASON_DECAY
            done = [ss for ss in ent_contrib if ss < s]
            assert all(ss < s for ss in done)        # W3: completed seasons only
            num_e = sum(ent_contrib[ss][0] for ss in done)
            den_e = sum(ent_contrib[ss][1] for ss in done)
            m_E = (num_e / den_e) if den_e >= ME_MIN_EVID else ME_FALLBACK
            me_by_season[s] = {"m_E": m_E, "evidence_xga": den_e,
                               "from_seasons": sorted(done), "fallback": den_e < ME_MIN_EVID}
        prev_season = s
        if s not in me_by_season:
            me_by_season[s] = {"m_E": m_E, "evidence_xga": 0.0, "from_seasons": [],
                               "fallback": True}
        xbar = (xb_sum / xb_cnt) if xb_cnt else XBAR_INIT

        if snap is not None and snap_out is None and dstr >= snap[0]:
            snap_out = {"date": dstr, "xbar": xbar, "m_E": m_E}
            for gk in snap[1]:
                m = m_E if (is_entrant(gk, dstr) and napp[gk] < ENTRANT_N) else 0.0
                r = (N[gk] + K_PRIOR * m) / (X[gk] + K_PRIOR)
                snap_out[str(gk)] = {"r_per_xga": r, "G_goals": r * xbar,
                                     "appearances": napp[gk], "X": X[gk], "N": N[gk],
                                     "entrant": is_entrant(gk, dstr)}

        st = starters.get(gid, {})
        if starter_override is not None:
            st = starter_override.get(gid, {})
        used = {}
        for side, flag, garr, farr, karr in (("home", 1, G_h, fb_h, gk_h),
                                             ("away", 0, G_a, fb_a, gk_a)):
            team = g[side]
            gk = st.get(flag)
            if gk is None:
                gk = prev_starter.get(team)
                farr[i] = True
            used[side] = gk
            if gk is None:
                garr[i] = 0.0
                continue
            if gd:
                gd.read(("g", gk), i)
            m = m_E if (is_entrant(gk, dstr) and napp[gk] < ENTRANT_N) else 0.0
            r = (N[gk] + K_PRIOR * m) / (X[gk] + K_PRIOR)
            garr[i] = r * xbar
            karr[i] = gk

        if gd:
            gd.read(("R", g["home"]), i)
            gd.read(("R", g["away"]), i)
        rh = R.setdefault(g["home"], 1500.0)
        ra = R.setdefault(g["away"], 1500.0)
        R_h[i] = rh
        R_a[i] = ra
        eh = rh + gamma * G_h[i]
        ea = ra + gamma * G_a[i]
        if use_d:
            eh = eh - delta * d_home[i]
            ea = ea - delta * d_away[i]
        p = 1.0 / (1.0 + 10 ** (-((eh + HA_ELO) - ea) / 400.0))
        out[i] = p
        # ---- updates (after the prediction) ----
        R[g["home"]] += K_ELO * (g["y"] - p)
        R[g["away"]] += K_ELO * ((1 - g["y"]) - (1 - p))
        if gd:
            gd.wrote(("R", g["home"]), i)
            gd.wrote(("R", g["away"]), i)
        for side in ("home", "away"):
            if used[side] is not None:
                prev_starter[g[side]] = used[side]
        team_x = defaultdict(float)
        for gk, team, xga, ga in gg.get(gid, []):
            if gk not in first_date:
                first_date[gk] = dstr
            N[gk] = D_APP * N[gk] + (xga - ga)
            X[gk] = D_APP * X[gk] + xga
            napp[gk] += 1
            if first_date[gk] >= ENTRANT_FROM and napp[gk] <= ENTRANT_N:
                ent_contrib[s][0] += xga - ga
                ent_contrib[s][1] += xga
            team_x[team] += xga
            if gd:
                gd.wrote(("g", gk), i)
        for t, v in team_x.items():
            pend_sum += v
            pend_cnt += 1

    return {"p": out, "G_h": G_h, "G_a": G_a, "R_h": R_h, "R_a": R_a,
            "fb_h": fb_h, "fb_a": fb_a, "gk_h": gk_h, "gk_a": gk_a,
            "me": me_by_season, "snap": snap_out}


def project_starters(games, starters, window=20):
    """A6 serving diagnostic: pre-game PROJECTED starter per team.

    Modal starter over the team's last `window` actual starts (ties -> most
    recent); if the team played yesterday and that goalie started yesterday,
    the most-used OTHER goalie of the last `window` (ties -> most recent).
    Uses only starters of strictly earlier games.  Returns gid -> {1: id, 0: id}.
    """
    from datetime import date, timedelta
    hist = defaultdict(list)        # team -> [goalie ids] (actual starters, oldest first)
    last_day = {}
    last_gk = {}
    proj = {}
    for g in games:
        gid = g["game_id"]
        d = date.fromisoformat(g["date"])
        st = starters.get(gid, {})
        pr = {}
        for side, flag in (("home", 1), ("away", 0)):
            t = g[side]
            h = hist[t][-window:]
            if not h:
                continue
            cnt = defaultdict(int)
            rec = {}
            for j, gk in enumerate(h):
                cnt[gk] += 1
                rec[gk] = j
            order = sorted(cnt, key=lambda k: (cnt[k], rec[k]), reverse=True)
            pick = order[0]
            if last_day.get(t) == d - timedelta(days=1) and last_gk.get(t) == pick \
                    and len(order) > 1:
                pick = order[1]
            pr[flag] = pick
        proj[gid] = pr
        for side, flag in (("home", 1), ("away", 0)):
            t = g[side]
            gk = st.get(flag)
            if gk is not None:
                hist[t].append(gk)
                last_gk[t] = gk
            last_day[t] = d
    return proj
