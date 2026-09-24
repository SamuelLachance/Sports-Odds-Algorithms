"""Player-value program, NHL -- event-archive quality checks (DEV games ONLY).

Every statistic here is computed on DEV games (gid < 2018000000); TEST rows are
filtered out at load time by pv_nhl_io and asserted absent. Writes
data/pv_nhl_qa.json and prints a summary.

Checks
  A  coverage          DEV spine games reduced OK
  B  goal totals       events (non-SO goals + shootout winner) vs nhl_games.csv,
                       and the raw payload's final score vs nhl_games.csv
  C  box reconcile     per player-game G / A / SOG / BLK / PIM vs the DEV box
                       scores in data/bt_nhl_boxel_box.csv (2010-11 .. 2014-15 reg;
                       that file's parser records no goalie assists / SOG, so
                       goalie rows are reported separately)
  D  rosters           dressed set vs box score; g_start vs the box `starter` flag;
                       shift-chart TOI vs box-score TOI
  E  xG join           share of unblocked non-SO shots carrying MoneyPuck xG, share
                       of MoneyPuck shots used, exact vs fuzzy, goal-flag agreement
  F  attribution       blank team; named players absent from rosterSpots; penalty
                       team vs committed-by player's team; blocker vs shooter team
  G  per-season event volumes, coordinate / situation coverage
  H  team abbrevs      pbp home/away vs nhl_games.csv
  I  geometry          dist / angle sanity for shots
  J  rink-scorer bias  per-arena home/road recorded-count ratio by event type
  K  special           penalty shots, shootout rows, penalties with no committer
"""
from __future__ import annotations

import gzip
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pv_nhl_io import DEV_MAX_GID, load_events, load_rosters  # noqa: E402

OUT = "data/pv_nhl_qa.json"
ROOT = "data/pv_nhl_pbp"
ALIAS = {"PHX": "ARI", "ARI": "ARI"}


def r4(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)


def main():
    qa = {}
    games = pd.read_csv("data/nhl_games.csv")
    games = games[games.game_id < DEV_MAX_GID].copy()
    stat = pd.read_csv("data/pv_nhl_event_games.csv")
    stat = stat[stat.gid < DEV_MAX_GID]
    ev = load_events()
    ro = load_rosters()
    assert ev.gid.max() < DEV_MAX_GID and ro.gid.max() < DEV_MAX_GID
    print(f"DEV events {len(ev):,}  rosters {len(ro):,}  games {len(games):,}")

    # ---- A coverage ------------------------------------------------------
    ok = set(stat.gid[stat.status == "ok"])
    miss = sorted(set(games.game_id) - ok)
    qa["A_coverage"] = {"dev_spine_games": int(len(games)), "reduced_ok": len(ok),
                        "not_ok": len(miss), "not_ok_gids": [int(g) for g in miss[:50]],
                        "status_counts": stat.status.value_counts().to_dict()}

    # ---- B goal totals ---------------------------------------------------
    g = ev[ev.ev == "goal"]
    reg = g[g.ptype != "SO"].groupby(["gid", "is_home"]).size().unstack(fill_value=0)
    so = g[g.ptype == "SO"].groupby(["gid", "is_home"]).size().unstack(fill_value=0)
    reg = reg.reindex(columns=[0, 1], fill_value=0)
    so = so.reindex(columns=[0, 1], fill_value=0)
    gm = games.set_index("game_id").loc[sorted(ok)]
    ea = reg[0].reindex(gm.index, fill_value=0).astype(int)
    eh = reg[1].reindex(gm.index, fill_value=0).astype(int)
    soa = so[0].reindex(gm.index, fill_value=0)
    soh = so[1].reindex(gm.index, fill_value=0)
    is_so = gm.last_period == "SO"
    ea = ea + ((soa > soh) & is_so).astype(int)
    eh = eh + ((soh > soa) & is_so).astype(int)
    match = (ea == gm.away_goals) & (eh == gm.home_goals)
    mm = gm[~match]
    qa["B_goals_events_vs_games_csv"] = {
        "games": int(len(gm)), "exact": int(match.sum()), "rate": r4(match.mean()),
        "shootout_games": int(is_so.sum()),
        "so_games_without_so_goal_events": int((is_so & (soa + soh == 0)).sum()),
        "mismatches": [{"gid": int(i), "csv": f"{int(r.away_goals)}-{int(r.home_goals)}",
                        "events": f"{int(ea[i])}-{int(eh[i])}"}
                       for i, r in mm.head(40).iterrows()]}
    # the payload's own final score
    pay_ok = pay_n = 0
    pay_bad = []
    for gid, season in games[["game_id", "season"]].itertuples(index=False):
        p = f"{ROOT}/{season}/{gid}.json.gz"
        if not os.path.exists(p):
            continue
        with gzip.open(p, "rb") as fh:
            d = json.loads(fh.read())
        pay_n += 1
        row = gm.loc[gid] if gid in gm.index else games.set_index("game_id").loc[gid]
        a, h = d["awayTeam"].get("score"), d["homeTeam"].get("score")
        if a == row.away_goals and h == row.home_goals:
            pay_ok += 1
        elif len(pay_bad) < 40:
            pay_bad.append({"gid": int(gid), "csv": f"{int(row.away_goals)}-{int(row.home_goals)}",
                            "payload": f"{a}-{h}"})
    qa["B_goals_payload_vs_games_csv"] = {"games": pay_n, "exact": pay_ok,
                                          "rate": r4(pay_ok / max(pay_n, 1)),
                                          "mismatches": pay_bad}

    # ---- C box reconcile -------------------------------------------------
    box = pd.read_csv("data/bt_nhl_boxel_box.csv")
    box = box[box.gid < DEV_MAX_GID]
    bg = set(box.gid) & ok
    e2 = ev[ev.gid.isin(bg) & (ev.ptype != "SO")]
    def ids(df, col):
        return df[["gid", col]].dropna().astype("int64").set_axis(["gid", "pid"], axis=1)

    gl = e2[e2.ev == "goal"]
    cnt = {"G": ids(gl, "shooter").groupby(["gid", "pid"]).size(),
           "A": pd.concat([ids(gl, "assist1"), ids(gl, "assist2")])
                  .groupby(["gid", "pid"]).size(),
           "SOG": ids(e2[e2.ev.isin(["goal", "shot-on-goal"])], "shooter")
                  .groupby(["gid", "pid"]).size(),
           "BLK": ids(e2[e2.ev == "blocked-shot"], "blocker").groupby(["gid", "pid"]).size()}
    pen = ev[ev.gid.isin(bg) & (ev.ev == "penalty")][["gid", "pen_by", "pen_min"]].dropna()
    cnt["PIM"] = pen.astype("int64").groupby(["gid", "pen_by"]).pen_min.sum()         .rename_axis(["gid", "pid"])
    b2 = box[box.gid.isin(bg)].set_index(["gid", "pid"])
    rec = {"games": len(bg), "player_games": int(len(b2))}
    for k, s in cnt.items():
        mine = s.reindex(b2.index, fill_value=0).astype(float)
        theirs = b2[k].astype(float)
        eq = (mine.values == theirs.values)
        extra = s[~s.index.isin(b2.index)]
        sk = (b2.is_goalie.values == 0)
        rec[k] = {"player_game_exact": r4(eq.mean()),
                  "skater_player_game_exact": r4(eq[sk].mean()),
                  "n_differ_skaters": int((~eq[sk]).sum()),
                  "n_differ_goalies": int((~eq[~sk]).sum()),
                  "total_events": float(mine.sum()), "total_box": float(theirs.sum()),
                  "credit_to_players_not_in_box": float(extra.sum()),
                  "n_player_games_differ": int((~eq).sum())}
    qa["C_box_reconcile"] = rec

    # ---- D rosters -------------------------------------------------------
    rset = ro[ro.gid.isin(bg)].groupby(["gid", "side"]).pid.apply(frozenset)
    bx = box[box.gid.isin(bg)].copy()
    bx["side"] = bx.side.map({"home": "H", "away": "A"})
    bset = bx.groupby(["gid", "side"]).pid.apply(frozenset)
    j = rset.to_frame("r").join(bset.to_frame("b"), how="inner")
    same = (j.r == j.b)
    gst = ro[(ro.pos == "G") & ro.gid.isin(bg)].merge(
        bx[bx.is_goalie == 1][["gid", "pid", "starter"]], on=["gid", "pid"], how="inner")
    ro_all = ro[ro.gid.isin(ok)]
    per_side = ro_all.groupby(["gid", "side"]).size()
    gper = ro_all[ro_all.pos == "G"].groupby(["gid", "side"]).size()
    starts = ro_all[ro_all.pos == "G"].groupby(["gid", "side"]).g_start.sum()
    qa["D_rosters"] = {
        "team_games_vs_box": int(len(j)), "dressed_set_identical": r4(same.mean()),
        "goalie_rows_vs_box": int(len(gst)),
        "g_start_agrees_with_box_starter": r4((gst.g_start.astype(int) ==
                                                gst.starter.astype(int)).mean()),
        "dressed_per_team_game": {str(k): int(v) for k, v in
                                  per_side.value_counts().sort_index().items()},
        "goalies_per_team_game": {str(k): int(v) for k, v in
                                  gper.value_counts().sort_index().items()},
        "team_games_with_exactly_one_g_start": r4((starts == 1).mean()),
        "players_with_zero_events_share": r4((ro_all.n_ev == 0).mean())}
    tm = ro[ro.gid.isin(bg)].merge(box[["gid", "pid", "toi_s", "is_goalie"]],
                                   on=["gid", "pid"], suffixes=("", "_box"))
    td = tm.toi_s - tm.toi_s_box
    qa["D_rosters"].update({
        "toi_rows_vs_box": int(len(tm)),
        "toi_exact_vs_box": r4((td == 0).mean()),
        "toi_within_60s_vs_box": r4((td.abs() <= 60).mean()),
        "roster_rows_with_toi": r4(ro_all.toi_s.notna().mean()),
        "dressed_with_zero_toi_share": r4((ro_all.toi_s == 0).mean())})

    # ---- E xG join -------------------------------------------------------
    sh = ev[ev.ev.isin(["goal", "shot-on-goal", "missed-shot"]) & (ev.ptype != "SO")]
    mp = pd.read_csv("data/nhl_shots.csv", usecols=["nhl_game_id", "period", "goal", "xg"])
    mp = mp[mp.nhl_game_id < DEV_MAX_GID]
    mp = mp[mp.nhl_game_id.isin(ok)]
    has = sh.xg.notna()
    by_season = sh.assign(h=has).groupby("season").h.mean()
    gj = sh.assign(h=has).groupby("gid").h.mean()
    matched = sh[has]
    mp_per_game = mp.groupby("nhl_game_id").size()
    qa["E_xg_join"] = {
        "pbp_unblocked_nonSO_shots": int(len(sh)),
        "with_xg": int(has.sum()), "pbp_join_rate": r4(has.mean()),
        "moneypuck_shots_in_dev_games": int(len(mp)),
        "moneypuck_used_rate": r4(has.sum() / max(len(mp), 1)),
        "exact_share_of_matched": r4((matched.xg_match == 1).mean()),
        "fuzzy_time_share_of_matched": r4((matched.xg_match == 2).mean()),
        "goal_flag_differs_share_of_matched": r4((matched.xg_match == 3).mean()),
        "pbp_join_rate_by_season": {str(k): r4(v) for k, v in by_season.items()},
        "games_join_rate_lt_0.80": int((gj < 0.80).sum()),
        "games_join_rate_lt_0.80_gids": [int(x) for x in gj[gj < 0.80].index[:40]],
        "dev_games_without_moneypuck_rows": int(len(ok - set(mp_per_game.index))),
        "goals_with_xg_share": r4(sh[sh.ev == "goal"].xg.notna().mean()),
        "mean_xg_goal_vs_nongoal": [r4(matched[matched.ev == "goal"].xg.mean()),
                                    r4(matched[matched.ev != "goal"].xg.mean())]}

    # ---- F attribution ---------------------------------------------------
    rkey = set(zip(ro.gid.astype("int64"), ro.pid.astype("int64")))
    f = {"events": int(len(ev)), "blank_team": int(ev.team.isna().sum())}
    roles = ["shooter", "assist1", "assist2", "goalie", "blocker", "fo_win", "fo_lose",
             "hitter", "hittee", "player", "pen_by", "pen_drawn", "pen_served"]
    notin = {}
    for c in roles:
        s = ev[["gid", c]].dropna()
        if len(s):
            k = pd.Series(list(zip(s.gid.astype("int64"), s[c].astype("int64"))))
            notin[c] = int((~k.isin(rkey)).sum())
    f["named_ids_not_in_rosterSpots"] = notin
    team_of = dict(zip(zip(ro.gid.astype("int64"), ro.pid.astype("int64")), ro.team))
    p = ev[(ev.ev == "penalty") & ev.pen_by.notna()]
    pt = [team_of.get((int(a), int(b))) for a, b in zip(p.gid, p.pen_by)]
    f["penalty_team_eq_committer_team"] = r4(np.mean([x == y for x, y in zip(pt, p.team)]))
    b = ev[(ev.ev == "blocked-shot") & ev.blocker.notna() & ev.shooter.notna()]
    bt = [team_of.get((int(a), int(c))) for a, c in zip(b.gid, b.blocker)]
    f["blocked_blocker_team_ne_shooter_team"] = r4(np.mean([x != y for x, y in
                                                             zip(bt, b.team)]))
    f["missing_primary_player"] = {
        e: int(ev[(ev.ev == e)][c].isna().sum()) for e, c in
        [("goal", "shooter"), ("shot-on-goal", "shooter"), ("missed-shot", "shooter"),
         ("blocked-shot", "shooter"), ("blocked-shot", "blocker"),
         ("faceoff", "fo_win"), ("faceoff", "fo_lose"), ("hit", "hitter"),
         ("hit", "hittee"), ("giveaway", "player"), ("takeaway", "player"),
         ("penalty", "pen_by")]}
    f["shots_on_goal_missing_goalie_nonEN"] = int(
        ev[(ev.ev.isin(["shot-on-goal", "goal"])) & ev.goalie.isna() & (ev.g_ag == 1)
           & (ev.ptype != "SO")].shape[0])
    qa["F_attribution"] = f

    # ---- G volumes -------------------------------------------------------
    ng = ev.groupby("season").gid.nunique()
    vol = ev.groupby(["season", "ev"]).size().unstack(fill_value=0).div(ng, axis=0)
    qa["G_per_game_volume_by_season"] = {str(s): {k: round(float(v), 1) for k, v in
                                                  row.items()} for s, row in vol.iterrows()}
    qa["G_coverage"] = {
        "xy_share": r4(ev.x.notna().mean()),
        "xy_share_by_ev": {k: r4(v) for k, v in ev.x.notna().groupby(ev.ev).mean().items()},
        "sit_share": r4(ev.sit.notna().mean()),
        "shot_type_share_unblocked": r4(sh.shot_type.notna().mean()),
        "strength_5v5_share_of_unblocked": r4(((sh.sk_for == 5) & (sh.sk_ag == 5)).mean())}

    # ---- H abbrevs -------------------------------------------------------
    hm = ro.groupby(["gid", "side"]).team.first().unstack()
    gg = games.set_index("game_id")
    idx = hm.index.intersection(gg.index)
    norm = lambda s: s.map(lambda x: ALIAS.get(x, x))  # noqa: E731
    qa["H_team_abbrev_vs_games_csv"] = {
        "games": int(len(idx)),
        "home_eq": r4((norm(hm.loc[idx, "H"]) == norm(gg.loc[idx, "home"])).mean()),
        "away_eq": r4((norm(hm.loc[idx, "A"]) == norm(gg.loc[idx, "away"])).mean()),
        "pairs_differing": sorted({f"{a}->{b}" for a, b in
                                   zip(hm.loc[idx, "H"], gg.loc[idx, "home"]) if a != b})}

    # ---- I geometry ------------------------------------------------------
    s2 = sh[sh.dist.notna()]
    qa["I_geometry"] = {
        "unblocked_with_dist": r4(sh.dist.notna().mean()),
        "dist_median_goal_vs_nongoal": [r4(s2[s2.ev == "goal"].dist.median()),
                                        r4(s2[s2.ev != "goal"].dist.median())],
        "share_dist_gt_100": r4((s2.dist > 100).mean()),
        "xg_corr_neg_dist": r4(np.corrcoef(s2.dropna(subset=["xg"]).dist,
                                           s2.dropna(subset=["xg"]).xg)[0, 1])}

    # ---- J rink-scorer bias ----------------------------------------------
    # recorded count per game at an arena / the same home team's road-game count,
    # per (season, home team): subjective events (hits, giveaways, takeaways)
    # are known to be arena-scored; a downstream rating must adjust for it.
    gi = games.set_index("game_id")[["season", "home", "away"]]
    j_out = {}
    for e in ["hit", "giveaway", "takeaway", "blocked-shot", "missed-shot",
              "shot-on-goal", "faceoff"]:
        c = ev[ev.ev == e].groupby("gid").size().reindex(sorted(ok), fill_value=0)
        df = gi.loc[c.index].assign(n=c.values)
        home_rate = df.groupby(["season", "home"]).n.mean()
        road_rate = df.groupby(["season", "away"]).n.mean()
        road_rate.index = road_rate.index.set_names(["season", "home"])
        ratio = (home_rate / road_rate).dropna()
        if len(ratio):
            j_out[e] = {"arena_ratio_p05": r4(ratio.quantile(0.05)),
                        "median": r4(ratio.median()),
                        "p95": r4(ratio.quantile(0.95)),
                        "min": r4(ratio.min()), "max": r4(ratio.max())}
    qa["J_rink_scorer_bias_home_over_road_count"] = j_out

    # penalty shots / shootout bookkeeping
    ps = ev[ev.ev.isin(["goal", "shot-on-goal", "missed-shot", "failed-shot-attempt"])
            & (ev.ptype != "SO") & (ev.sk_for == 1) & (ev.sk_ag == 0)]
    qa["K_special"] = {
        "penalty_shot_attempts_nonSO": int(len(ps)),
        "penalty_shot_with_xg": int(ps.xg.notna().sum()),
        "shootout_attempt_rows": int(ev[(ev.ptype == "SO") & ev.ev.isin(
            ["goal", "shot-on-goal", "missed-shot", "failed-shot-attempt"])].shape[0]),
        "penalties_without_committer": int(ev[(ev.ev == "penalty") & ev.pen_by.isna()]
                                           .shape[0]),
        "of_which_bench_BEN": int(ev[(ev.ev == "penalty") & ev.pen_by.isna()
                                     & (ev.pen_code == "BEN")].shape[0])}

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(qa, fh, indent=1, default=int)
    print(json.dumps(qa, indent=1, default=int)[:20000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
