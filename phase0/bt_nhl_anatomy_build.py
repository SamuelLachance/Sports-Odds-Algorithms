"""NHL gap anatomy -- step 1: per-game DESCRIPTIVE slice variables, DEV only.

Breakthrough program (documents/breakthrough_program_prereg_2026_09_24.md).
Nothing here is a model feature and nothing here is ever served. The columns are
slice keys for the anatomy in phase0/bt_nhl_anatomy.py, which asks WHERE the
shipped blend loses log-loss to the de-vigged closing line.

PROTOCOL
  * games are truncated to season <= DEV_END (2017-18) BEFORE anything is
    computed; shots/dressed/goalie rows are filtered to DEV game ids on read.
  * every per-team state is read BEFORE it is updated with the current game.
    The only same-game quantities used are tonight's STARTING goalie and
    tonight's DRESSED lineup -- both public before puck drop (goalie confirmed
    at warmups, lineup announced), i.e. information the closing line also has.
    Those are exactly the "day-of" information channel we want to measure.

Output: data/bt_nhl_anatomy_feats.csv (one row per DEV regular-season game).
"""
from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict, deque
from datetime import date

import numpy as np
import pandas as pd

sys.path.insert(0, "phase0")
from nhl_depth_eval import dev_games  # noqa: E402  (asserts season <= DEV_END)
from nhl_glicko2_eval import DEV_END, TEST_START  # noqa: E402
from nhl_features_eval import TEAM_FIX  # noqa: E402

OUT = "data/bt_nhl_anatomy_feats.csv"
MAX_GID = 2018000000          # first TEST-era game id is 2018020001
assert DEV_END == 20172018 and TEST_START == 20182019

# fixed a priori, never searched
WIN_LINEUP = 10      # trailing team games defining "usual lineup"
REG_MIN = 7          # regular = dressed in >= 7 of the last 10
WIN_GOALIE = 20      # trailing team games defining the team's usual goalie
G_DECAY = 0.98       # per-appearance decay on goalie GSAx evidence
G_PRIOR = 20.0       # EB prior weight (appearances) toward 0 GSAx/game


def norm(t):
    return TEAM_FIX.get(t, t)


def load_starters(dev_ids):
    """gid -> {defending team: starting goalie id} = goalie facing the first
    on-goal/attempt of the game for that side (earliest period, per_sec)."""
    sh = pd.read_csv("data/nhl_shots.csv",
                     usecols=["nhl_game_id", "period", "per_sec", "shooter_team",
                              "is_home", "goalie_id"])
    sh = sh[sh.nhl_game_id < MAX_GID]
    sh = sh[sh.nhl_game_id.isin(dev_ids)]
    sh = sh.dropna(subset=["goalie_id"])
    sh = sh.sort_values(["nhl_game_id", "period", "per_sec"])
    sh["def_home"] = 1 - sh["is_home"]
    first = sh.groupby(["nhl_game_id", "def_home"], sort=False).first().reset_index()
    out = defaultdict(dict)
    for r in first.itertuples(index=False):
        out[int(r.nhl_game_id)][int(r.def_home)] = int(r.goalie_id)
    return out


def load_goalie_games(dev_ids):
    gx = pd.read_csv("data/nhl_goalie_xg.csv")
    gx = gx[gx.nhl_game_id < MAX_GID]
    gx = gx[gx.nhl_game_id.isin(dev_ids)]
    out = defaultdict(list)          # gid -> [(goalie_id, team, xga, ga)]
    for r in gx.itertuples(index=False):
        out[int(r.nhl_game_id)].append((int(r.goalie_id), norm(r.team),
                                        float(r.xga), float(r.ga)))
    return out, set(int(x) for x in gx.goalie_id.unique())


def load_dressed(dev_ids, goalie_ids):
    d = pd.read_csv("data/nhl_dressed.csv")
    d = d[d.game_id < MAX_GID]
    d = d[d.game_id.isin(dev_ids)]
    d = d[~d.player_id.isin(goalie_ids)]
    out = defaultdict(dict)          # gid -> team -> {pid: toi_s}
    for r in d.itertuples(index=False):
        out[int(r.game_id)].setdefault(norm(r.team), {})[int(r.player_id)] = float(r.toi_s)
    return out


def main():
    games = dev_games()
    assert max(g["season"] for g in games) <= DEV_END
    ids = set(g["game_id"] for g in games)
    assert max(ids) < MAX_GID
    print(f"{len(games)} DEV regular-season games", flush=True)

    starters = load_starters(ids)
    ggames, goalie_ids = load_goalie_games(ids)
    dressed = load_dressed(ids, goalie_ids)
    print(f"starters for {len(starters)} games, goalie rows {len(ggames)}, "
          f"dressed {len(dressed)}", flush=True)

    # per-team state
    gp = defaultdict(int)                          # team games this season
    last_date = {}
    prev_starter = {}
    starter_hist = defaultdict(lambda: deque(maxlen=WIN_GOALIE))
    lineup_hist = defaultdict(lambda: deque(maxlen=WIN_LINEUP))   # [{pid: toi}]
    # per-goalie state
    g_num = defaultdict(float)                     # decayed sum GSAx
    g_den = defaultdict(float)                     # decayed n appearances
    g_starts = defaultdict(int)                    # career starts in data
    # standings (pts, gp) for points% -- as-of, strictly earlier games
    pts = defaultdict(int)
    # results for form
    res_hist = defaultdict(lambda: deque(maxlen=10))

    raw = list(csv.DictReader(open("data/nhl_games.csv", encoding="utf-8")))
    lastp = {int(r["game_id"]): r["last_period"] for r in raw
             if int(r["game_id"]) < MAX_GID}

    rows = []
    prev_season = None
    for g in games:
        s = g["season"]
        if prev_season is not None and s != prev_season:
            gp.clear()
            pts.clear()
            res_hist.clear()
            for t in lineup_hist:
                lineup_hist[t].clear()          # new season: no "usual lineup"
        prev_season = s
        gid = g["game_id"]
        d = date.fromisoformat(g["date"])
        rec = {"game_id": gid, "date": g["date"], "season": s,
               "home": g["home"], "away": g["away"], "month": d.month}
        st = starters.get(gid, {})
        for side, flag in (("home", 1), ("away", 0)):
            t = g[side]
            rec[f"gp_{side}"] = gp[t]
            rec[f"ptspct_{side}"] = (pts[t] / (2 * gp[t])) if gp[t] else np.nan
            rh = res_hist[t]
            rec[f"form10_{side}"] = (sum(rh) / len(rh)) if len(rh) >= 5 else np.nan
            lg = last_date.get(t)
            rec[f"rest_{side}"] = (d - lg).days if lg else np.nan
            # ---- goalie (tonight's starter vs team's usual) ----
            gk = st.get(flag)
            rec[f"g_known_{side}"] = int(gk is not None)
            hist = starter_hist[t]
            if gk is not None and len(hist) >= 10:
                share = sum(1 for x in hist if x == gk) / len(hist)
                modal = max(set(hist), key=lambda x: sum(1 for y in hist if y == x))
                rec[f"g_share_{side}"] = share
                rec[f"g_backup_{side}"] = int(share < 0.5)
                rec[f"g_notmodal_{side}"] = int(gk != modal)
                rt = g_num[gk] / (g_den[gk] + G_PRIOR)
                rm = g_num[modal] / (g_den[modal] + G_PRIOR)
                rec[f"g_rtg_{side}"] = rt
                rec[f"g_delta_{side}"] = rt - rm
            else:
                for k in ("g_share", "g_backup", "g_notmodal", "g_rtg", "g_delta"):
                    rec[f"{k}_{side}"] = np.nan
            if gk is not None and t in prev_starter:
                rec[f"g_changed_{side}"] = int(prev_starter[t] != gk)
            else:
                rec[f"g_changed_{side}"] = np.nan
            rec[f"g_career_{side}"] = g_starts[gk] if gk is not None else np.nan
            # ---- lineup (tonight's dressed skaters vs usual) ----
            dr = dressed.get(gid, {}).get(t)
            lh = lineup_hist[t]
            if dr and len(lh) >= 5:
                cnt = defaultdict(int)
                tsum = defaultdict(float)
                for lu in lh:
                    for pid, toi in lu.items():
                        cnt[pid] += 1
                        tsum[pid] += toi
                need = REG_MIN if len(lh) >= WIN_LINEUP else int(np.ceil(0.7 * len(lh)))
                regs = {p: tsum[p] / cnt[p] / 60.0 for p in cnt if cnt[p] >= need}
                top = sorted(regs, key=regs.get, reverse=True)[:4]
                miss = [p for p in regs if p not in dr]
                rec[f"abs_n_{side}"] = len(miss)
                rec[f"abs_toi_{side}"] = sum(regs[p] for p in miss)
                rec[f"abs_top4_{side}"] = sum(1 for p in top if p not in dr)
                rec[f"new_n_{side}"] = sum(1 for p in dr if p not in cnt)
            else:
                for k in ("abs_n", "abs_toi", "abs_top4", "new_n"):
                    rec[f"{k}_{side}"] = np.nan
        rows.append(rec)

        # ---------------- updates AFTER reading ----------------
        y = g["y"]
        lp = lastp.get(gid, "REG")
        for side, flag, won in (("home", 1, y), ("away", 0, 1 - y)):
            t = g[side]
            gp[t] += 1
            pts[t] += 2 if won else (1 if lp in ("OT", "SO") else 0)
            res_hist[t].append(won)
            last_date[t] = d
            gk = st.get(flag)
            if gk is not None:
                starter_hist[t].append(gk)
                prev_starter[t] = gk
                g_starts[gk] += 1
            dr = dressed.get(gid, {}).get(t)
            if dr:
                lineup_hist[t].append(dr)
        for gk, team, xga, ga in ggames.get(gid, []):
            g_num[gk] = G_DECAY * g_num[gk] + (xga - ga)
            g_den[gk] = G_DECAY * g_den[gk] + 1.0

    df = pd.DataFrame(rows)
    assert df.season.max() <= DEV_END
    # merge pre-game deadness + context (both keyed by game_id, pre-game builds)
    dead = pd.read_csv("data/nhl_dead.csv")
    dead = dead[dead.game_id < MAX_GID][["game_id", "home_elim", "home_lock",
                                         "home_soft", "away_elim", "away_lock",
                                         "away_soft"]]
    ctx = pd.read_csv("data/nhl_context.csv")
    ctx = ctx[ctx.game_id < MAX_GID][["game_id", "trav_home_km", "trav_away_km",
                                      "tz_away", "away_trip_no", "g3in4_home",
                                      "g3in4_away", "gconsec_home", "gconsec_away"]]
    df = df.merge(dead, on="game_id", how="left").merge(ctx, on="game_id", how="left")
    tmp = OUT + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, OUT)
    print(f"wrote {OUT}: {len(df)} rows, seasons {df.season.min()}..{df.season.max()}")
    for c in ("g_known_home", "g_backup_home", "g_backup_away", "g_changed_home",
              "abs_n_home", "abs_top4_home", "home_elim"):
        print(f"  {c:<16} non-null {df[c].notna().mean():.3f}  mean {df[c].mean():.3f}")


if __name__ == "__main__":
    main()
