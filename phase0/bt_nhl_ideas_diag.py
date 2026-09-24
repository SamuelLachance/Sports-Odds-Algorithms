"""Breakthrough program - NHL idea sizing diagnostics. DEV ONLY.

Odds are used ONLY as an evaluation benchmark (gap anatomy / sizing). Nothing
here is a model and nothing is fit to the market for use as a feature.

  1. ceiling: LOSO logistic y ~ [shipped logit, close logit] -> how much the
     close knows that we do not (DEV).
  2. market-taught shift, out-of-fold: coefficients of (close - model) on game
     conditions fit on 6 DEV seasons, applied to the 7th -> LL of model+shift.
     Sizing only: what the close's reaction to those conditions is worth.
  3. goalie talent: split-half reliability of save-above-expected rate for DEV
     goalie-seasons; long-memory EB goalie LEVEL rating; the close's loading on
     tonight's-starter-minus-usual-goalie in rating units.
  4. season-start roster turnover: prior-season TOI of tonight's dressed
     skaters that was earned on ANOTHER team, and prior-season TOI of this team
     not dressed tonight; the close's loading in a team's first 10 games.

Writes data/bt_nhl_ideas_diag.json.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict, deque

import numpy as np

sys.path.insert(0, "phase0")
from nhl_features_eval import build_features  # noqa: E402
from nhl_glicko2_eval import llv, load_games, run_elo  # noqa: E402
from nhl_xg_screen import loso  # noqa: E402

DEV_SEASONS = [20112012, 20122013, 20132014, 20142015, 20152016, 20162017, 20172018]
GFIX = {"L.A": "LAK", "N.J": "NJD", "S.J": "SJS", "T.B": "TBL"}


def mp(team):
    return "ARI" if team == "PHX" else team


def main():
    games = [g for g in load_games() if g["season"] <= 20172018]   # DEV era only
    assert max(g["season"] for g in games) <= 20172018
    n = len(games)
    model = json.load(open("data/nhl_model.json"))
    e_out = run_elo(games, **model["elo_cfg"])
    pe = np.clip(np.array([o[1] for o in e_out]), 1e-9, 1 - 1e-9)
    elogit = np.log(pe / (1 - pe))
    F = build_features(games)
    seas = np.array([g["season"] for g in games])
    y = np.array([g["y"] for g in games])
    keep = {g["game_id"] for g in games}

    # ---------------- goalie rows (DEV ids only) ----------------
    grow = defaultdict(list)
    for r in csv.DictReader(open("data/nhl_goalie_xg.csv", encoding="utf-8")):
        gid = int(r["nhl_game_id"])
        if gid in keep:
            grow[gid].append((GFIX.get(r["team"], r["team"]), int(r["goalie_id"]),
                              float(r["xga"]), float(r["ga"])))
    # starter = defender of the earliest shot (pre-game legitimate proxy)
    first = {}
    with open("data/nhl_shots.csv", encoding="utf-8") as fh:
        rd = csv.reader(fh)
        next(rd)
        for r in rd:
            gid = int(r[0])
            if gid not in keep or not r[9]:
                continue
            key = (gid, r[3])                   # shooter team -> defending goalie
            t = int(r[1]) * 10000 + int(r[2])
            if key not in first or t < first[key][0]:
                first[key] = (t, int(float(r[9])))

    # --- 3a. split-half reliability of goalie save-above-expected rate -------
    gs = defaultdict(lambda: [[0.0, 0.0], [0.0, 0.0]])     # (gk, season) -> [odd,even] [gsax, xga]
    gcount = defaultdict(int)
    for g in games:
        for t, gk, xga, ga in grow.get(g["game_id"], []):
            k = (gk, g["season"])
            h = gcount[k] % 2
            gs[k][h][0] += xga - ga
            gs[k][h][1] += xga
            gcount[k] += 1
    a, b, w = [], [], []
    for k, v in gs.items():
        if gcount[k] >= 30 and v[0][1] > 20 and v[1][1] > 20:
            a.append(v[0][0] / v[0][1]); b.append(v[1][0] / v[1][1]); w.append(gcount[k])
    a, b = np.array(a), np.array(b)
    r_half = float(np.corrcoef(a, b)[0, 1])
    obs_sd = float(np.std(np.concatenate([a, b])))
    print(f"goalie-season split-half r (GSAx/xGA, >=30 GP) = {r_half:.3f}  n={len(a)}  "
          f"obs sd per half {obs_sd:.4f}; implied true sd ~ {obs_sd*np.sqrt(max(r_half,0)):.4f}")

    # --- 3b. long-memory EB goalie LEVEL rating + usual-goalie baseline -------
    DECAY, PRIOR_XGA, SEAS_KEEP = 0.998, 120.0, 0.8      # fixed a priori
    S = defaultdict(float); C = defaultdict(float)
    usual = defaultdict(lambda: deque(maxlen=20))
    gdelta = np.full(n, np.nan); glevel = np.full(n, np.nan)
    backup = np.zeros((n, 2)); known = np.zeros(n, bool)
    prev = None
    for i, g in enumerate(games):
        if prev is not None and g["season"] != prev:
            for k in list(S):
                S[k] *= SEAS_KEEP; C[k] *= SEAS_KEEP
        prev = g["season"]
        st = {}
        for side, opp in (("home", "away"), ("away", "home")):
            f = first.get((g["game_id"], mp(g[opp])))       # opp shoots at our goalie
            st[side] = f[1] if f else None
        if st["home"] and st["away"]:
            vals = {}
            for j, side in enumerate(("home", "away")):
                gk = st[side]
                rt = S[gk] / (C[gk] + PRIOR_XGA)
                dq = usual[g[side]]
                if len(dq) >= 5:
                    cnt = Counter(dq)
                    base = sum(c * (S[k] / (C[k] + PRIOR_XGA)) for k, c in cnt.items()) / len(dq)
                    backup[i, j] = float(gk != cnt.most_common(1)[0][0])
                else:
                    base = rt
                vals[side] = (rt, rt - base)
            glevel[i] = vals["home"][0] - vals["away"][0]
            gdelta[i] = vals["home"][1] - vals["away"][1]
            known[i] = True
        for t, gk, xga, ga in grow.get(g["game_id"], []):
            S[gk] = S[gk] * DECAY + (xga - ga)
            C[gk] = C[gk] * DECAY + xga
        for side in ("home", "away"):
            if st[side]:
                usual[g[side]].append(st[side])
    # scale: GSAx per xGA * ~2.7 xGA/game = goals/game
    print(f"goalie level diff sd {np.nanstd(glevel)*2.7:.4f} goals/g; "
          f"delta-vs-usual sd {np.nanstd(gdelta)*2.7:.4f} goals/g")

    # --- 4. season-start roster turnover from dressed TOI --------------------
    dressed = defaultdict(dict)                 # gid -> team -> {pid: toi}
    for r in csv.DictReader(open("data/nhl_dressed.csv", encoding="utf-8")):
        gid = int(r["game_id"])
        if gid in keep:
            dressed[gid].setdefault(r["team"], {})[r["player_id"]] = float(r["toi_s"])
    gk_ids = {str(gk) for rows in grow.values() for _, gk, _, _ in rows}
    ps_toi = defaultdict(lambda: defaultdict(float))   # season -> pid -> total toi
    ps_team = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))  # season->team->pid->toi
    tgp = defaultdict(int)
    imported = np.full((n, 2), np.nan); lost = np.full((n, 2), np.nan)
    prev = None
    for i, g in enumerate(games):
        s = g["season"]
        if prev is not None and s != prev:
            tgp.clear()
        prev = s
        ls = s - 10001                      # previous season int
        for j, side in enumerate(("home", "away")):
            t = g[side]
            d = dressed.get(g["game_id"], {}).get(t, {})
            if d and ls in ps_toi and tgp[t] < 10:
                tl = ps_team[ls].get(t) or ps_team[ls].get({"ARI": "PHX", "WPG": "ATL"}.get(t, t), {})
                imp = sum(ps_toi[ls].get(p, 0.0) - tl.get(p, 0.0) for p in d if p not in gk_ids)
                lst = sum(v for p, v in tl.items() if p not in d and p not in gk_ids)
                imported[i, j] = imp / 3600.0            # hours of prior-season TOI
                lost[i, j] = lst / 3600.0
        for side in ("home", "away"):
            t = g[side]
            for p, v in dressed.get(g["game_id"], {}).get(t, {}).items():
                ps_toi[s][p] += v
                ps_team[s][t][p] += v
            tgp[t] += 1

    # --------------------------- odds (evaluation) ---------------------------
    odds = {}
    for r in csv.DictReader(open("data/odds_nhl.csv", encoding="utf-8")):
        try:
            ho, ao = float(r["home_close"]), float(r["away_close"])
        except ValueError:
            continue
        if ho > 1 and ao > 1:
            odds[(r["date"], r["home"], r["away"])] = (1 / ho) / (1 / ho + 1 / ao)
    pc = np.array([odds.get((g["date"], g["home"], g["away"]), np.nan) for g in games])

    xg = F["xg_diff"]
    dev = np.isin(seas, DEV_SEASONS) & ~np.isnan(xg)
    X = np.column_stack([elogit, F["rest_diff"], F["b2b_home"], F["b2b_away"], xg])
    pm = np.full(n, np.nan)
    pm[dev] = loso(X[dev], y[dev], seas[dev], DEV_SEASONS)
    m = dev & ~np.isnan(pc)
    assert set(np.unique(seas[m])) <= set(DEV_SEASONS)
    lm = np.log(pm / (1 - pm)); lc = np.log(pc / (1 - pc))
    out = {}
    base_ll = float(llv(y[m], pm[m]).mean()); close_ll = float(llv(y[m], pc[m]).mean())
    out["model_ll"], out["close_ll"] = base_ll, close_ll

    # 1. ceiling
    Xc = np.column_stack([lm, lc])[m]
    pcomb = loso(Xc, y[m], seas[m], DEV_SEASONS)
    out["combo_ll"] = float(llv(y[m], pcomb).mean())
    from sklearn.linear_model import LogisticRegression
    cf = LogisticRegression(C=1e6, max_iter=2000).fit(Xc, y[m]).coef_[0]
    print(f"\nDEV n={m.sum()} model {base_ll:.5f} close {close_ll:.5f} "
          f"LOSO combo {out['combo_ll']:.5f}  (weights model {cf[0]:.3f} close {cf[1]:.3f})")

    # 2. out-of-fold market-taught shift
    tg = np.zeros((n, 2))
    cnt = defaultdict(int); prev = None
    for i, g in enumerate(games):
        if prev is not None and g["season"] != prev:
            cnt.clear()
        prev = g["season"]
        tg[i] = (cnt[g["home"]], cnt[g["away"]])
        cnt[g["home"]] += 1; cnt[g["away"]] += 1
    from nhl_scratch_eval import build_features_scratch
    absent, _ = build_features_scratch(games)
    conds = {
        "backup": np.column_stack([backup[:, 0] * known, backup[:, 1] * known]),
        "gdelta": np.nan_to_num(gdelta)[:, None] * 2.7,
        "absent": np.nan_to_num(absent)[:, None],
        "b2b": np.column_stack([F["b2b_home"], F["b2b_away"]]),
        "turnover": np.column_stack([np.nan_to_num(imported[:, 0] - imported[:, 1]),
                                     np.nan_to_num(lost[:, 0] - lost[:, 1])]),
    }
    r = (lc - lm)

    def oof_shift(keys):
        Z = np.column_stack([np.ones(n)] + [conds[k] for k in keys])
        shift = np.full(n, np.nan)
        for s in DEV_SEASONS:
            tr = m & (seas != s); te = m & (seas == s)
            beta, *_ = np.linalg.lstsq(Z[tr], r[tr], rcond=None)
            shift[te] = Z[te, 1:] @ beta[1:]
        pz = 1 / (1 + np.exp(-(lm + shift)))
        return float(llv(y[m], pz[m]).mean())

    out["oof_shift"] = {}
    for keys in (["backup"], ["gdelta"], ["backup", "gdelta"], ["absent"], ["b2b"],
                 ["turnover"], ["backup", "gdelta", "absent", "b2b", "turnover"]):
        ll = oof_shift(keys)
        out["oof_shift"]["+".join(keys)] = round(base_ll - ll, 5)
        print(f"  OOF market-taught shift {'+'.join(keys):<36} gain {base_ll - ll:+.5f}")

    # 3c. the close's loading on goalie quality (DEV, OLS on residual)
    def ols(cols, mask, names):
        Z = np.column_stack([np.ones(mask.sum())] + [c[mask] for c in cols] + [lm[mask]])
        beta, *_ = np.linalg.lstsq(Z, r[mask], rcond=None)
        res = r[mask] - Z @ beta
        se = np.sqrt(np.diag(np.linalg.inv(Z.T @ Z)) * res.var())
        o = {nm: [round(float(bb), 4), round(float(bb / ss), 2)]
             for nm, bb, ss in zip(["const"] + names + ["model_logit"], beta, se)}
        print("   ", o)
        return o

    mk = m & known
    print("\nclose-minus-model loading on goalie (goals/game units), DEV:")
    out["goalie_load"] = ols([gdelta * 2.7, (glevel - gdelta) * 2.7, backup[:, 0], backup[:, 1]],
                             mk, ["delta_vs_usual", "usual_level_diff", "backup_h", "backup_a"])
    early = m & (tg.max(axis=1) < 10) & ~np.isnan(imported).any(axis=1)
    print(f"\nclose-minus-model loading on season-start turnover (first 10 games, n={early.sum()}):")
    out["turnover_load"] = ols([imported[:, 0], imported[:, 1], lost[:, 0], lost[:, 1]],
                               early, ["imported_h", "imported_a", "lost_h", "lost_a"])
    print(f"  imported hrs sd {np.nanstd(imported[early]):.2f}  lost hrs sd {np.nanstd(lost[early]):.2f}")
    out["goalie_split_half_r"] = r_half
    out["goalie_obs_sd"] = obs_sd
    json.dump(out, open("data/bt_nhl_ideas_diag.json", "w"), indent=1)
    print("wrote data/bt_nhl_ideas_diag.json")


if __name__ == "__main__":
    main()
