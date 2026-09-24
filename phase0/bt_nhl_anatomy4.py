"""NHL gap anatomy -- step 5: RETURNS and ADDITIONS (the other half of lineup
deviation). DEV ONLY, descriptive; odds evaluation-only.

The absent-regular measure (ledger row 10, and abs_* here) only sees players who
are MISSING. A results-based team rating is equally wrong when a good player
RETURNS (the rating was learned while he was out) or a veteran goalie comes back
from a spell. Per team-game, pre-game state only:

  add_toi   = sum over tonight's dressed skaters NOT among the trailing-10
              regulars of that player's mean TOI over his previous 20 appearances
              (any team, across seasons) -- i.e. what the newcomers usually play
  net_toi   = add_toi - absent_toi (tonight's lineup minus the usual one, TOI units)
  g_return  = tonight's starter has >= 100 prior starts in the data AND started
              none of the team's previous 3 games (veteran back from a spell)

Output: data/bt_nhl_anatomy4.json
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict, deque

import numpy as np
import pandas as pd

sys.path.insert(0, "phase0")
import nhl_depth_eval as Hd  # noqa: E402
from nhl_glicko2_eval import DEV_END, TEST_START, llv  # noqa: E402
from bt_nhl_anatomy import reproduce, join_odds, logit, boot_mean  # noqa: E402
import bt_nhl_anatomy_build as B  # noqa: E402

OUT = "data/bt_nhl_anatomy4.json"


def lineup_deltas(games):
    ids = set(g["game_id"] for g in games)
    _, gids = B.load_goalie_games(ids)
    dressed = B.load_dressed(ids, gids)
    starters = B.load_starters(ids)
    lineup_hist = defaultdict(lambda: deque(maxlen=10))
    ptoi = defaultdict(lambda: deque(maxlen=20))       # player -> last 20 TOI (min)
    team_starters = defaultdict(lambda: deque(maxlen=3))
    gstarts = defaultdict(int)
    out = {}
    prev = None
    for g in games:
        if prev is not None and g["season"] != prev:
            for t in lineup_hist:
                lineup_hist[t].clear()
        prev = g["season"]
        rec = {}
        st = starters.get(g["game_id"], {})
        for side, flag in (("home", 1), ("away", 0)):
            t = g[side]
            dr = dressed.get(g["game_id"], {}).get(t)
            lh = lineup_hist[t]
            if dr and len(lh) >= 5:
                cnt = defaultdict(int)
                for lu in lh:
                    for pid in lu:
                        cnt[pid] += 1
                need = 7 if len(lh) >= 10 else int(np.ceil(0.7 * len(lh)))
                regs = {p for p in cnt if cnt[p] >= need}
                reg_toi = {p: (np.mean(ptoi[p]) if ptoi[p] else 0.0) for p in regs}
                absent = sum(reg_toi[p] for p in regs if p not in dr)
                add = sum((np.mean(ptoi[p]) if len(ptoi[p]) >= 5 else 8.0)
                          for p in dr if p not in regs)
                rec[f"add_{side}"] = add
                rec[f"net_{side}"] = add - absent
                rec[f"abs_{side}"] = absent
            else:
                rec[f"add_{side}"] = rec[f"net_{side}"] = rec[f"abs_{side}"] = np.nan
            gk = st.get(flag)
            rec[f"gret_{side}"] = float(gk is not None and gstarts[gk] >= 100
                                        and len(team_starters[t]) == 3
                                        and gk not in team_starters[t])
        out[g["game_id"]] = rec
        for side, flag in (("home", 1), ("away", 0)):
            t = g[side]
            dr = dressed.get(g["game_id"], {}).get(t)
            if dr:
                lineup_hist[t].append(set(dr))
                for pid, toi in dr.items():
                    ptoi[pid].append(toi / 60.0)
            gk = st.get(flag)
            if gk is not None:
                team_starters[t].append(gk)
                gstarts[gk] += 1
    return out


def loso(X, y, seas):
    p = np.empty(len(y))
    for s in sorted(set(seas.tolist())):
        tr, te = seas != s, seas == s
        w = Hd.fit_logit(X[tr], y[tr])
        p[te] = Hd._sig(X[te] @ w)
    return p


def main():
    p, y, seas, G, Xcols = reproduce()
    pc, po, _ = join_odds(G, y)
    keep = np.where(~np.isnan(pc))[0]
    p, y, seas, pc, po = p[keep], y[keep], seas[keep], pc[keep], po[keep]
    G = [G[i] for i in keep]
    Hd.assert_dev_only(seas)
    allg = Hd.dev_games()
    L = lineup_deltas(allg)
    get = lambda k: np.array([L[g["game_id"]][k] for g in G], float)  # noqa: E731
    z = np.nan_to_num
    lm, lc, lo_ = logit(p), logit(pc), logit(po)
    dlt = lc - lm
    llm, llc, llo = llv(y, p), llv(y, pc), llv(y, po)
    one = np.ones(len(y))
    base = llv(y, loso(np.column_stack([one, lm]), y, seas))

    D = {"absent_toi_diff (a-h)": z(get("abs_away")) - z(get("abs_home")),
         "add_toi_diff (h-a)": z(get("add_home")) - z(get("add_away")),
         "net_lineup_toi_diff (h-a)": z(get("net_home")) - z(get("net_away")),
         "goalie_return_diff (h-a)": get("gret_home") - get("gret_away")}
    res = {}
    print(f"{'descriptor':<28} {'sd':>7} | {'mkt coef':>9} {'t':>6} | {'out coef':>9} "
          f"{'dLL LOSO':>9} {'CI':>20} | {'vs close':>9} {'z':>5}")
    for name, d in D.items():
        Xa = np.column_stack([one, d, lm])
        ba, *_ = np.linalg.lstsq(Xa, dlt, rcond=None)
        ra = dlt - Xa @ ba
        sea = np.sqrt(np.diag(ra.var() * np.linalg.inv(Xa.T @ Xa)))
        pb = loso(np.column_stack([one, lm, d]), y, seas)
        dd = base - llv(y, pb)
        ci = boot_mean(dd)
        wb = Hd.fit_logit(np.column_stack([one, lm, d]), y)
        Xc = np.column_stack([one, lc, d])
        wc = Hd.fit_logit(Xc, y)
        q = Hd._sig(Xc @ wc)
        cov = np.linalg.inv((Xc * (q * (1 - q))[:, None]).T @ Xc)
        zc = wc[2] / np.sqrt(cov[2, 2])
        res[name] = {"sd": round(float(d.std()), 3), "market_coef": round(float(ba[1]), 4),
                     "market_t": round(float(ba[1] / sea[1]), 1),
                     "outcome_coef": round(float(wb[2]), 4),
                     "dLL_loso": round(float(dd.mean()), 5), "ci": ci,
                     "coef_beyond_close": round(float(wc[2]), 4), "z_beyond_close": round(float(zc), 2)}
        print(f"{name:<28} {d.std():>7.3f} | {ba[1]:>+9.4f} {ba[1] / sea[1]:>+6.1f} | {wb[2]:>+9.4f} "
              f"{dd.mean():>+9.5f} [{ci[0]:+.5f},{ci[1]:+.5f}] | {wc[2]:>+9.4f} {zc:>+5.1f}")

    sl = {}
    gr = (get("gret_home") + get("gret_away")) > 0
    big_add = (np.abs(D["add_toi_diff (h-a)"]) >= 30)
    big_net = (np.abs(D["net_lineup_toi_diff (h-a)"]) >= 30)
    for nm_, m in (("veteran goalie returning (either)", gr),
                   ("|add toi diff| >= 30 min", big_add),
                   ("|net lineup toi diff| >= 30 min", big_net),
                   ("|net lineup toi diff| < 10 min", np.abs(D["net_lineup_toi_diff (h-a)"]) < 10)):
        r = {"n": int(m.sum()), "model_minus_close": round(float((llm - llc)[m].mean()), 5),
             "mc_ci": boot_mean((llm - llc)[m]),
             "model_minus_open": round(float((llm - llo)[m].mean()), 5),
             "open_to_close": round(float((llo - llc)[m].mean()), 5)}
        sl[nm_] = r
        print(f"  {nm_:<36} n={r['n']:>5} model-close {r['model_minus_close']:+.5f} "
              f"[{r['mc_ci'][0]:+.5f},{r['mc_ci'][1]:+.5f}] model-open {r['model_minus_open']:+.5f} "
              f"open->close {r['open_to_close']:+.5f}")
    json.dump({"descriptors": res, "slices": sl}, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
