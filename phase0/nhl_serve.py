"""Serve the NHL model into site/data/nhl.json.

Replays the frozen Elo + xG + rest/B2B model through all history (leak-safe:
each game predicted from pre-game state), then emits:

  * schedule  - every game of the current season: played games with their
                frozen pre-game probability and result (playoff finals too,
                `playoff: 1`), and EVERY unplayed game (the 10-day slate plus
                the rest of the season) predicted from current team states.
                All NHL forecasts are EARLY tier.
  * teams     - identity (name, division, conference), the model's current
                Elo / xG ratings, the full standings row (GP, points %, RW,
                ROW, home/away, L10, streak, division / conference / wild-card
                rank, official clinch mark), season xGF/xGA, the GLASSBOX
                lineup rating, the roster, and last season's final line.
  * players   - every player on a current NHL roster, skaters and goalies,
                with bio, rating (or the reason there is none) and production
                lines for this season, last season and career.
  * proj      - a Monte Carlo season projection from the model's own game
                probabilities (phase0/nhl_site_sim.py).
  * prev_season - last season's record, never pooled with the live one. A
                TEST season (2018-19..2025-26) is emitted verbatim from
                data/nhl_prev_season_{season}.json, frozen from the payload
                the site published - its metrics are never recomputed
                (phase0/nhl_site_prev.py).
  * rapm      - the skater-rating window, its build date and the shift-data
                coverage of each fit season (`coverage`, `caveat`).

Market-blind - no odds in the model or the payload (the edge layer adds those
separately, mirroring nfl.json). Needs numpy only. Every display source
(rosters, stats, standings, the season schedule; phase0/nhl_site_fetch.py) is
optional: a missing cache falls back to the previous payload, then to nothing.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "phase0")
from nhl_glicko2_eval import llv, load_games, run_elo  # noqa: E402
from nhl_features_eval import build_features  # noqa: E402
from nhl_contributions import contributions, home_elogit  # noqa: E402
from nhl_serve_guards import needs_boundary_regression, schedule_rest  # noqa: E402
from nhl_season_boundary import (  # noqa: E402
    current_season, display_teams, fmt_metric, is_playoff,
)

EPS = 1e-9
PAYLOAD = "site/data/nhl.json"
FEATURES_REPORT = "data/nhl_features_report.json"   # the one TEST look (blend vs Elo)
ELO_REPORT = "data/nhl_glicko2_report.json"         # the one TEST look (tuned Elo)
# fields the edge layer (market/sched_edges.py) owns on a schedule row; a
# re-serve carries them forward until its next cycle rewrites them
EDGE_ROW_KEYS = ("value", "mkt", "value_blocked")


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def final_states(games, model):
    """Walk Elo + xG-Elo + last-game-date to the end of the spine (mirrors the
    freeze walk) so upcoming games are predicted from CURRENT states."""
    from collections import defaultdict
    from nhl_features_eval import XG_HA, XG_K, XG_REGRESS, load_team_xg
    cfg = model["elo_cfg"]
    R, xg, last = {}, defaultdict(float), {}
    txg = load_team_xg()
    prev = None
    for g in games:
        if prev is not None and g["season"] != prev:
            for t in R:
                R[t] = 1500 + (R[t] - 1500) * (1 - cfg["regress"])
            for t in xg:
                xg[t] *= (1 - XG_REGRESS)
        prev = g["season"]
        rh = R.setdefault(g["home"], 1500.0); ra = R.setdefault(g["away"], 1500.0)
        pe = 1.0 / (1.0 + 10 ** (-((rh + cfg["ha"]) - ra) / 400.0))
        R[g["home"]] += cfg["k"] * (g["y"] - pe)
        R[g["away"]] += cfg["k"] * ((1 - g["y"]) - (1 - pe))
        last[g["home"]] = g["date"]; last[g["away"]] = g["date"]
        tx = txg.get(g["game_id"])
        if tx and g["home"] in tx and g["away"] in tx:
            hxgf, _ = tx[g["home"]]; axgf, _ = tx[g["away"]]
            err = (hxgf - axgf) - (xg[g["home"]] - xg[g["away"]] + XG_HA)
            xg[g["home"]] += XG_K / 100.0 * err
            xg[g["away"]] -= XG_K / 100.0 * err
    return R, xg, last


def _load(path, default=None):
    try:
        return json.load(open(path, encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def started(row, now_iso: str) -> bool:
    """Puck has dropped (or the game is over but not yet in the spine)."""
    if row.get("state") in ("LIVE", "CRIT", "OVER", "OFF", "FINAL"):
        return True
    t = row.get("start_utc")
    return bool(t) and t <= now_iso


def hold_started(sched, ledger, now_iso):
    """Split unplayed rows whose game has STARTED and already has a pre-game
    ledger entry off from the freeze, restoring them from that entry.

    nhl_hp_freeze refreshes every unplayed row's entry on each serve; a serve
    during a game (every 4 hours in-season) would otherwise overwrite the
    pre-game number and its receipt time with a post-puck-drop recompute.
    Returns (rows for the freeze, number held).
    """
    to_freeze, held = [], 0
    for s in sched:
        ent = ledger.get(str(s["id"]))
        if s.get("hs") is None and ent is not None and started(s, now_iso):
            s["hp"] = ent["hp"]
            if ent.get("ct") is not None:
                s["ct"] = ent["ct"]
            s["frozen_at"] = ent.get("t")
            s["tier"] = ent.get("tier", "EARLY")
            held += 1
        else:
            to_freeze.append(s)
    return to_freeze, held


def carry_edge(sched: list[dict], old: dict) -> int:
    """Copy the edge layer's row fields (EDGE_ROW_KEYS) from the previous
    payload onto the same unplayed games. Returns the badges dropped.

    One exception: a `value` badge with no `tier` stamp on an EARLY row. It was
    priced before the EARLY gate (market/sched_edges.py stamps every badge it
    writes), on an older forecast; the site never shows it on an EARLY row
    (nhlBadge) and the edge ledger refuses it, so it is dropped instead of being
    carried forward forever. Its `mkt` price feed is still carried. The
    top-level `value_blocked` (the priced build was stale) is never carried:
    this serve is a fresh build.
    """
    prev = {g["id"]: g for g in old.get("schedule", [])
            if any(k in g for k in EDGE_ROW_KEYS)}
    dropped = 0
    for s in sched:
        o = prev.get(s["id"])
        if not o or s.get("hs") is not None:
            continue
        for k in EDGE_ROW_KEYS:
            if k not in o:
                continue
            if (k == "value" and not (o[k] or {}).get("tier")
                    and s.get("tier", "EARLY") == "EARLY"):
                dropped += 1
                continue
            s[k] = o[k]
    return dropped


def test_card(model: dict, feat_rep: dict | None = None,
              elo_rep: dict | None = None) -> dict:
    """The model card's locked-TEST numbers, all from ONE harness, read-only.

    The one TEST look (phase0/nhl_features_eval.py, data/nhl_features_report.json
    'all') scored the served blend (`with` = test_ll) and a logistic fit on the
    Elo term alone (`base`) on the SAME games; the published gain and its 95% CI
    are base - with. `baseline_elo_test` is that same-harness base, so on the
    card test_ll + gain == baseline_elo_test. (It used to be the tuned Elo's raw
    probabilities from a different harness - 0.66918 on one more game - so the
    card printed 0.6692 - 0.6642 next to a 0.0054 gain.) That raw-Elo number is
    a different reference; it ships as `elo_raw_test`, labelled, never as the
    baseline. Nothing here is scored on TEST: every number was published by the
    one look (locked-split rule), and the model's TEST accuracy and calibration
    were never published, so the card carries none.
    """
    from nhl_glicko2_eval import DEV_END, DEV_WARM_BEFORE, TEST_END, TEST_START
    from nhl_site_teams import season_label
    ll, gain = model["test_ll"], model["test_delta_vs_elo"]
    a = (feat_rep or {}).get("all") or {}
    same = a.get("with") is not None and abs(a["with"] - ll) < 1e-9
    base = a["base"] if same and a.get("base") is not None else round(ll + gain, 5)
    if abs(base - ll - gain) > 1e-5 + 1e-12:
        raise ValueError(f"NHL model card does not reconcile: {base} - {ll} != {gain}")
    er = elo_rep or {}
    return {
        "test_ll": ll,
        "test_delta_vs_elo": gain,
        "test_ci": model.get("test_ci") or (a.get("ci") if same else None),
        "baseline_elo_test": base,
        "baseline_def": (f"a logistic fit on the Elo term alone, fit on "
                         f"{season_label(DEV_WARM_BEFORE)} to {season_label(DEV_END)} like the "
                         "model and scored on the same TEST games"),
        "test_n": a.get("n") if same else None,
        "test_seasons": f"{season_label(TEST_START)} to {season_label(TEST_END)}",
        "elo_raw_test": model.get("baseline_elo_test"),
        "elo_raw_n": er.get("test_n"),
    }


def train_count(games: list[dict], F: dict, as_of: str | None) -> int:
    """Games the served blend was fit on (phase0/nhl_freeze.py): the warm,
    xG-covered regular-season history through the model's as_of date."""
    from nhl_glicko2_eval import DEV_WARM_BEFORE
    xg = F["xg_diff"]
    return int(sum(1 for i, g in enumerate(games)
                   if g["season"] >= DEV_WARM_BEFORE and not np.isnan(xg[i])
                   and (as_of is None or g["date"] <= as_of)))


def team_xg_totals(path="data/nhl_team_xg.csv") -> dict:
    """(season_start, team) -> [xgf, xga, games], regular season only."""
    import csv
    from nhl_features_eval import TEAM_FIX
    out = defaultdict(lambda: [0.0, 0.0, 0])
    try:
        fh = open(path, encoding="utf-8")
    except FileNotFoundError:
        return out
    with fh:
        for r in csv.DictReader(fh):
            if str(r["nhl_game_id"])[4:6] != "02":
                continue
            a = out[(int(r["season"]), TEAM_FIX.get(r["team"], r["team"]))]
            a[0] += float(r["xgf"])
            a[1] += float(r["xga"])
            a[2] += 1
    return out


def main():
    import csv
    import os
    from datetime import datetime, timezone

    import nhl_hp_freeze
    import nhl_names
    import nhl_site_players as SP
    import nhl_site_prev as SPREV
    import nhl_site_teams as ST
    from nhl_features_eval import XG_HA, XG_REGRESS
    from nhl_site_schedule import (
        is_near, ledger_path, playoff_finals, spine_rest, unplayed_rows,
    )
    from nhl_update import et_today

    model = json.load(open("data/nhl_model.json"))
    b = model["blend"]
    c = b["coefs"]
    games = load_games()
    today = et_today()
    _now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    old = _load(PAYLOAD, {}) or {}

    # Upcoming is loaded BEFORE the season is derived, on purpose: the spine
    # holds only COMPLETED games, so deriving the season from it alone leaves
    # the payload a season stale from the day the new schedule posts until the
    # first new final lands. See phase0/nhl_season_boundary.py.
    upcoming = []
    if os.path.exists("data/nhl_upcoming.json"):
        try:
            upcoming = json.load(open("data/nhl_upcoming.json")) or []
        except json.JSONDecodeError:
            upcoming = []
    CUR_SEASON = current_season((g["season"] for g in games),
                                (u["id"] for u in upcoming))
    # The season before the served one (the served one's predecessor both
    # across the boundary and in-season).
    _prior = [s for s in {g["season"] for g in games} if s < CUR_SEASON]
    PREV_SEASON = max(_prior) if _prior else CUR_SEASON
    cur_start = CUR_SEASON // 10000

    raw = {int(r["game_id"]): r for r in csv.DictReader(open("data/nhl_games.csv", encoding="utf-8"))}

    # the whole season's schedule (display + far-horizon forecasts). Fallback:
    # the previous payload's own rows for this season.
    sfile = _load(f"data/nhl_schedule_{CUR_SEASON}.json", {}) or {}
    season_games = sfile.get("games") or []
    sched_src = sfile.get("fetched_at")
    if not season_games and old.get("cur_season") == CUR_SEASON:
        season_games = [{"id": g["id"], "d": g["d"], "home": g["home"], "away": g["away"],
                         "type": 3 if g.get("playoff") else 2, "t": g.get("start_utc"),
                         "state": g.get("state"), "sched": "PPD" if g.get("ppd") else "OK"}
                        for g in old.get("schedule", [])]
        sched_src = old.get("sources", {}).get("schedule")
    start_by_id = {g["id"]: g.get("t") for g in season_games if g.get("t")}

    HOME_ELOGIT = home_elogit(model["elo_cfg"]["ha"])
    e_out = run_elo(games, **model["elo_cfg"])
    p = np.clip(np.array([o[1] for o in e_out]), EPS, 1 - EPS)
    elogit = np.log(p / (1 - p))
    F = build_features(games)
    rest_played = spine_rest(games)

    z = (b["intercept"] + c["elo_logit"] * elogit + c["rest"] * F["rest_diff"]
         + c["b2b_home"] * F["b2b_home"] + c["b2b_away"] * F["b2b_away"]
         + c["xg"] * np.nan_to_num(F["xg_diff"]))
    phome = sig(z)

    # ---- current-season schedule: played games (replay, restored below) ----
    sched = []
    prev_rows = []
    for i, g in enumerate(games):
        if g["season"] == PREV_SEASON and PREV_SEASON != CUR_SEASON:
            r = raw.get(g["game_id"]) or {}
            prev_rows.append([g["game_id"], g["date"], g["home"], g["away"],
                              round(float(phome[i]), 4),
                              int(r["home_goals"]) if r else None,
                              int(r["away_goals"]) if r else None,
                              r.get("last_period") if r else None])
        if g["season"] != CUR_SEASON:
            continue
        rh, ra, hb2b, ab2b = rest_played[i]
        row = {
            "id": g["game_id"], "d": g["date"], "home": g["home"], "away": g["away"],
            "hp": round(float(phome[i]), 4),
            "hs": g["home_goals"] if "home_goals" in g else None,
            "as": g["away_goals"] if "away_goals" in g else None,
            "y": g["y"], "ot": g["so"] or None,
            "playoff": 1 if is_playoff(g["game_id"]) else 0,
            # exact decomposition: 0.5 + sum(ct)/100 == hp (see nhl_contributions);
            # home ice is its own term: the +ha Elo points, and the +XG_HA goals
            # whenever the xG term is on (a game with no xG row serves xg 0)
            "ct": contributions(b["intercept"], c, float(elogit[i]),
                                float(F["rest_diff"][i]), float(F["b2b_home"][i]),
                                float(F["b2b_away"][i]),
                                float(np.nan_to_num(F["xg_diff"])[i]),
                                home_elogit=HOME_ELOGIT,
                                home_xg=0.0 if np.isnan(F["xg_diff"][i]) else XG_HA),
            "hrest": rh, "arest": ra, "hb2b": hb2b, "ab2b": ab2b,
        }
        if start_by_id.get(g["game_id"]):
            row["start_utc"] = start_by_id[g["game_id"]]
        sched.append(row)

    # spine dicts lack goals; reload raw for scores
    for s in sched:
        r = raw.get(s["id"])
        if r:
            s["hs"] = int(r["home_goals"]); s["as"] = int(r["away_goals"])
            s["last"] = r["last_period"]

    # ---- unplayed games: the slate AND the rest of the season, from CURRENT states
    n_upcoming = 0
    R, xg_st, last = final_states(games, model)
    # Season boundary: the walk regresses ratings only when it crosses into a
    # game of the new season. Before opening night there is none, so apply the
    # model's own between-season regression here, once (nhl_serve_guards).
    last_walked = max((g["season"] for g in games), default=CUR_SEASON)
    if needs_boundary_regression(last_walked, CUR_SEASON):
        rg = model["elo_cfg"]["regress"]
        R = {t: 1500 + (r - 1500) * (1 - rg) for t, r in R.items()}
        xg_st = {t: v * (1 - XG_REGRESS) for t, v in xg_st.items()}
        print(f"[nhl_serve] season boundary {last_walked} -> {CUR_SEASON}: "
              f"ratings regressed {rg:.0%} toward the mean before serving")
    rows = unplayed_rows(set(raw), upcoming, season_games, CUR_SEASON)
    served = [u for u in rows if u["home"] in R and u["away"] in R]  # refuse unresolved teams
    # This season's PLAYOFF finals (nhl_site_schedule.playoff_finals): not in
    # the regular-season walk, dropped by unplayed_rows as played. Each is
    # built like an unplayed row - from current states, a fallback the freeze
    # below replaces with its pre-game ledger number (or marks replay) - and
    # then given its result. Standings, the projection and the model card
    # read regular-season rows only.
    po_done = playoff_finals(raw, CUR_SEASON, start_by_id, R)
    # rest/b2b from each team's previous game PLAYED OR SCHEDULED (playoff
    # finals included, so the next playoff game's rest counts them)
    rows_in = served + po_done
    rests = schedule_rest(last, rows_in)
    for u, (rh, ra, hb2b, ab2b) in zip(rows_in, rests):
        h, a = u["home"], u["away"]
        elp = 1.0 / (1.0 + 10 ** (-((R[h] + model["elo_cfg"]["ha"]) - R[a]) / 400.0))
        elogit_u = float(np.log(max(elp, EPS) / max(1 - elp, EPS)))
        xgd = (xg_st.get(h, 0.0) + XG_HA) - xg_st.get(a, 0.0)
        z = (b["intercept"] + c["elo_logit"] * elogit_u + c["rest"] * (rh - ra)
             + c["b2b_home"] * hb2b + c["b2b_away"] * ab2b + c["xg"] * xgd)
        row = {"id": u["id"], "d": u["d"], "home": h, "away": a,
               "hp": round(float(sig(z)), 4), "hs": None, "as": None,
               "y": None, "ot": None, "playoff": u.get("playoff", 0),
               "ct": contributions(b["intercept"], c, elogit_u,
                                   rh - ra, hb2b, ab2b, xgd,
                                   home_elogit=HOME_ELOGIT, home_xg=XG_HA),
               "hrest": int(rh), "arest": int(ra), "hb2b": int(hb2b), "ab2b": int(ab2b)}
        if u.get("t"):
            row["start_utc"] = u["t"]       # puck drop, for the card and the ledger
        if u.get("hs") is not None:         # a playoff final
            row.update({"hs": u["hs"], "as": u["as"], "y": 1 if u["hs"] > u["as"] else 0,
                        "ot": True if u["last"] == "SO" else None, "last": u["last"]})
            sched.append(row)
            continue
        row["near"] = 1 if is_near(u["d"], today) else 0
        if u.get("state") and u["state"] not in ("FUT", "OK"):
            row["state"] = u["state"]
        if u.get("ppd"):
            row["ppd"] = 1
        sched.append(row)
        n_upcoming += 1
    sched.sort(key=lambda s: (s["d"], s["id"]))

    # ---- pre-game freeze: played rows get the number published before puck drop
    _ledger = nhl_hp_freeze.load()
    # played games, the near slate and every row that already has an entry
    # (nhl_site_schedule.ledger_path): payload hp == ledger hp for every key
    far = [s for s in sched if not ledger_path(s, _ledger)]
    for s in far:
        s["tier"] = "EARLY"
    near = [s for s in sched if ledger_path(s, _ledger)]
    to_freeze, _nh = hold_started(near, _ledger, _now)
    # every NHL forecast is EARLY: the model has no lineup or goalie input
    # (documents/pick_policy.md). The tier is stored so a future goalie-aware
    # model can stamp PROJECTED without rewriting history.
    _ledger, _nf, _nr = nhl_hp_freeze.freeze(to_freeze, _ledger, _now, CUR_SEASON, "EARLY")
    nhl_hp_freeze.save(_ledger)
    print(f"[nhl_serve] hp-freeze: {_nf} played games restored to their pre-game hp, "
          f"{_nr} replays, {_nh} started games held at their pre-game entry "
          f"({len(_ledger)} ledger rows)")

    # ---- team identity + standings ----
    st_cache = (_load("data/nhl_site_standings.json", {}) or {}).get("seasons", {})
    api_cur = (st_cache.get(str(CUR_SEASON)) or {}).get("teams") or {}
    api_prev = (st_cache.get(str(PREV_SEASON)) or {}).get("teams") or {}
    meta = ST.team_meta(api_cur or api_prev)
    played_cur = [s for s in sched if s["hs"] is not None and not s["playoff"]]
    table = ST.rank_standings(ST.standings(played_cur, meta), meta)
    prev_played = [{"id": r[0], "d": r[1], "home": r[2], "away": r[3], "hs": r[5],
                    "as": r[6], "last": r[7]} for r in prev_rows if r[5] is not None]
    prev_table = ST.rank_standings(ST.standings(prev_played, meta), meta) if prev_played else {}
    txg = team_xg_totals()

    # current walked states (boundary-regressed), not the model freeze: the
    # frozen table never moved in-season
    elo_r = {t: round(v, 1) for t, v in R.items()}
    xg_r = {t: round(v, 4) for t, v in xg_st.items()}
    # Active teams = those on the current season's schedule (played OR upcoming)
    # unioned with the previous season's. Defunct codes (ATL/PHX/ARI) appear in
    # neither and are dropped.
    cur_teams = {t for s in sched for t in (s["home"], s["away"])}
    prev_teams = {t for g in games if g["season"] == PREV_SEASON
                  for t in (g["home"], g["away"])}
    cur_elo = display_teams(elo_r, cur_teams, prev_teams)
    ranks = {t: i + 1 for i, t in enumerate(sorted(cur_elo, key=lambda x: -cur_elo[x]))}

    # ---- players: every current-roster skater and goalie ----
    names = nhl_names.load()
    if not any(v.get("on_roster") for v in names.values()):
        # no roster answer recorded yet: the previous payload's players are the
        # last known rosters (never the cumulative map - it keeps departed
        # players on their old clubs)
        names = {pid: {"name": q.get("name"), "pos": q.get("pos"), "team": q.get("team"),
                       "on_roster": True, "num": q.get("num"), "born": q.get("born"),
                       "shoots": q.get("shoots"), "ht": q.get("ht"), "wt": q.get("wt"),
                       "nat": q.get("nat"), "img": q.get("img")}
                 for pid, q in (old.get("players") or {}).items() if q.get("team")}
    rapm = _load(SP.RAPM, {}) or {}
    rapm_info = SP.rapm_meta({int(r["season"]) for r in raw.values()}, cur_season=CUR_SEASON)
    st_files = {"cur": _load(f"data/nhl_site_stats_{CUR_SEASON}.json"),
                "prev": _load(f"data/nhl_site_stats_{PREV_SEASON}.json")
                if PREV_SEASON != CUR_SEASON else None}
    career = _load("data/nhl_site_career.json", {}) or {}
    grows = SP.load_goalie_games()
    tau = SP.estimate_tau(grows)
    gq, gmeta = SP.goalie_quality(grows, cur_start, tau)
    SP.rate_goalies(gq, [pid for pid, v in names.items()
                         if v.get("on_roster") and v.get("pos") == "G"])
    # a stats source that could not be fetched falls back to the previous
    # payload's lines for the same season (older, never wrong-season). They go
    # in BEFORE build_players, so the line to lead with and the unrated
    # reasons (which name the seasons a player played) see them too.
    old_players = old.get("players") or {}
    old_ss = old.get("stats_seasons") or {}
    stats_in = dict(st_files)
    career_in = career
    carried = {}
    for key, have, same in (("cur", st_files["cur"], old_ss.get("cur") == CUR_SEASON),
                            ("prev", st_files["prev"], old_ss.get("prev") == PREV_SEASON),
                            ("career", career.get("skaters") or career.get("goalies"),
                             old_ss.get("cur") == CUR_SEASON)):
        if have or not same:
            continue
        carried[key] = True
        fb = {"skaters": {}, "goalies": {}}
        for pid, q in old_players.items():
            ln = (q.get("stats") or {}).get(key)
            if ln:
                fb["goalies" if q.get("pos") == "G" else "skaters"][pid] = ln
        if key == "career":
            career_in = fb
        else:
            stats_in[key] = fb
    players = SP.build_players(names, rapm, stats_in, career_in, gq, SP.season_lines(grows),
                               cur_start, today, rapm_info["window"], gmeta["window"],
                               rapm_info["coverage"])
    blocks = SP.team_blocks(players, list(cur_elo), rapm)

    teams = {}
    for t in cur_elo:
        r = table.get(t) or ST.standings([], [t])[t]
        m = meta.get(t, {})
        x = txg.get((cur_start, t))
        tm = {
            "elo": elo_r[t], "xg": xg_r.get(t, 0.0), "rank": ranks[t],
            "w": r["w"], "l": r["l"], "otl": r["otl"], "gf": r["gf"], "ga": r["ga"],
            "pts": r["pts"],
            "name": m.get("name"), "city": m.get("city"), "nick": m.get("nick"),
            "div": m.get("div"), "conf": m.get("conf"),
        }
        for k in ("gp", "rw", "row", "pts_pct", "gd", "home", "away", "l10", "streak",
                  "league_rank", "conf_rank", "div_rank", "wc_rank", "po"):
            tm[k] = r.get(k)
        old_t = (old.get("teams") or {}).get(t) or {}
        tm["clinch"] = ((api_cur.get(t) or {}).get("clinch") if api_cur
                        else old_t.get("clinch") if old.get("cur_season") == CUR_SEASON
                        else None)
        tm["xgf"] = round(x[0], 1) if x else None
        tm["xga"] = round(x[1], 1) if x else None
        tm["xg_gp"] = x[2] if x else 0
        tm.update(blocks.get(t, {"roster": [], "goalies": [], "top": []}))
        pr = prev_table.get(t)
        if pr:
            px = txg.get((PREV_SEASON // 10000, t))
            tm["prev"] = {"season": PREV_SEASON,
                          **{k: pr.get(k) for k in (
                              "gp", "w", "l", "otl", "pts", "pts_pct", "rw", "row", "gf",
                              "ga", "gd", "home", "away", "l10", "streak", "league_rank",
                              "conf_rank", "div_rank", "wc_rank", "po")},
                          "clinch": ((api_prev.get(t) or {}).get("clinch") if api_prev
                                     else ((old_t.get("prev") or {}).get("clinch")
                                           if (old_t.get("prev") or {}).get("season") == PREV_SEASON
                                           else None)),
                          "xgf": round(px[0], 1) if px else None,
                          "xga": round(px[1], 1) if px else None}
        else:
            tm["prev"] = old_t.get("prev")
        teams[t] = tm

    # ---- realized current-season accuracy / LL ----
    done = [s for s in sched if s["hs"] is not None and not s["playoff"]]
    y = np.array([s["y"] for s in done])
    pp = np.array([s["hp"] for s in done])
    cur_ll = float(llv(y, np.clip(pp, EPS, 1 - EPS)).mean()) if len(y) else None
    cur_acc = float(((pp > 0.5) == (y > 0.5)).mean()) if len(y) else None

    # ---- last season: its record, clearly labelled, never pooled ----
    # A TEST season (2018-19..2025-26) is emitted VERBATIM from the block frozen
    # out of the payload the site published for it - no TEST-season metric is
    # ever computed here (phase0/nhl_site_prev.py). A later season, one this
    # site served live, is computed from its pre-game ledger numbers.
    prev_block = None
    if prev_played:
        if SPREV.is_test(PREV_SEASON):
            prev_block = (SPREV.load(PREV_SEASON)
                          or SPREV.rows_only(PREV_SEASON, prev_rows, model["as_of"]))
        else:
            rows_l, n_led = [], 0
            for r in prev_rows:
                ent = _ledger.get(str(r[0]))
                if ent is not None:
                    r = r[:4] + [ent["hp"]] + r[5:]
                    n_led += 1
                rows_l.append(r)
            done_l = [r for r in rows_l if r[5] is not None]
            py = np.array([1 if r[5] > r[6] else 0 for r in done_l])
            ph = np.array([r[4] for r in done_l])
            kind = ("live" if n_led == len(rows_l) else "replay" if n_led == 0 else "mixed")
            prev_block = {
                "season": PREV_SEASON, "label": ST.season_label(PREV_SEASON),
                "kind": kind,
                "note": {"live": "The live record: every game graded on the probability "
                                 "the site published before puck drop.",
                         "mixed": (f"{n_led} of {len(rows_l)} games graded on the probability "
                                   "published before puck drop; the rest are walk-forward "
                                   "replays."),
                         "replay": SPREV.replay_note(model["as_of"])}[kind]
                        + " Every forecast is EARLY tier (team ratings only).",
                "n": int(len(py)),
                "ll": round(float(llv(py, np.clip(ph, EPS, 1 - EPS)).mean()), 5) if len(py) else None,
                "acc": round(float(((ph > 0.5) == (py > 0.5)).mean()), 4) if len(py) else None,
                "rows_cols": SPREV.ROWS_COLS,
                "rows": rows_l,
            }

    # ---- season projection (Monte Carlo over every remaining regular-season game)
    proj, proj_info = {}, None
    remaining = [s for s in sched if s["hs"] is None and not s["playoff"]]
    if remaining:
        from nhl_site_sim import simulate
        sim_meta = {t: {"div": meta[t]["div"], "conf": meta[t]["conf"]}
                    for t in teams if t in meta}
        seed = (CUR_SEASON * 7919 + len(done) * 104729) % (2 ** 32)
        proj, proj_info = simulate(remaining, R, xg_st, sim_meta, table,
                                   model["elo_cfg"], b, XG_HA, seed=seed)
        proj_info.update({"generated": _now, "tier": "EARLY",
                          "method": ("Every remaining regular-season game played "
                                     f"{proj_info['n_sims']:,} times with the model's own "
                                     "win probability; Elo updated after each simulated game; "
                                     "OT/SO modelled (loser takes a point); top 3 per division "
                                     "+ 2 wild cards per conference make the playoffs.")})

    # ---- dates ----
    cur_rows = [s for s in sched if not s["playoff"]]
    results_through = max((s["d"] for s in done), default=None)
    first_game = min((s["d"] for s in cur_rows), default=None)
    last_game = max((s["d"] for s in cur_rows), default=None)
    if not n_upcoming:
        phase = "offseason"
    elif not done and remaining:
        phase = "preseason"
    elif remaining:
        phase = "regular"
    else:
        phase = "playoffs"

    # carry the edge layer's fields forward (carry_edge): odds.yml owns them,
    # but a re-serve must not blank the badges, the price feed, the reason a
    # quoted game carries no badge, or the last edge run's status line until
    # its next cycle
    n_stale_badges = carry_edge(sched, old)
    if n_stale_badges:
        print(f"[nhl_serve] dropped {n_stale_badges} pre-gate EDGE badge(s) with no tier stamp "
              f"on EARLY rows (never shown, refused by the edge ledger)")
    value_updated = old.get("value_updated")

    _cav = SP.coverage_caveat(rapm_info["coverage"])
    names_at = nhl_names.fetched_at(names) if names else None
    old_src = old.get("sources") or {}
    st_cache_at = (_load("data/nhl_site_standings.json", {}) or {}).get("fetched_at")
    payload = {
        "status": "season" if n_upcoming else "offseason",
        "phase": phase,
        "generated": _now,
        "served_at": _now,
        "as_of": model["as_of"],
        "cur_season": CUR_SEASON,
        "season_label": ST.season_label(CUR_SEASON),
        "first_game": first_game, "last_game": last_game,
        "n_games": len(cur_rows),
        "results_through": results_through,
        "ratings_through": games[-1]["date"] if games else None,
        "divisions": list(ST.DIVISIONS),
        "conferences": ST.CONFERENCES,
        "stats_seasons": {"cur": CUR_SEASON,
                          "prev": PREV_SEASON if PREV_SEASON != CUR_SEASON else None},
        "sources": {
            "rosters": names_at,
            "stats": ((st_files["cur"] or {}).get("fetched_at")
                      or (old_src.get("stats") if carried.get("cur") else None)),
            "career": career.get("fetched_at") or (old_src.get("career")
                                                   if carried.get("career") else None),
            "schedule": sched_src,
            "standings": st_cache_at or (old_src.get("standings")
                                         if old.get("cur_season") == CUR_SEASON else None),
            "rapm_built": rapm_info["built"],
        },
        "rapm": {**rapm_info,
                 "caveat": ((f"The shift data these ratings were fit on is incomplete: "
                             f"{_cav}. 5v5 minutes and ratings from those seasons are "
                             "undercounted until the missing shift charts are re-pulled "
                             "and the ratings refit.") if _cav else None),
                 "n_rated": sum(1 for q in players.values()
                                if q["grp"] != "G" and q["rating"] is not None),
                 "units": ("player toi = 5v5 minutes in the shift data on file (this "
                           "window); season lines: toi_pg = seconds per game (skaters), "
                           "toi_s = total seconds (goalies)"),
                 "scale": "50 = average NHL skater, 15 points = 1 SD; forwards and "
                          "defencemen pooled; 5v5, teammate- and opponent-adjusted; "
                          "display metric, not a model input"},
        "goalie_model": {**gmeta,
                         "n_rated": sum(1 for q in players.values()
                                        if q["grp"] == "G" and q["rating"] is not None),
                         "scale": "50 = league-average goalie, 15 points = 1 SD of rostered "
                                  "goalies; goals saved above expected (MoneyPuck xG), "
                                  "regular season, recent seasons weighted, shrunk toward "
                                  "average; display metric, not a model input"},
        "schedule": sched,
        "teams": teams,
        "players": players,
        "proj": proj,
        "proj_info": proj_info,
        "prev_season": prev_block,
        "model_card": {
            # one harness: test_ll + test_delta_vs_elo == baseline_elo_test
            **test_card(model, _load(FEATURES_REPORT), _load(ELO_REPORT)),
            "home_win_rate": model["home_win_rate"],
            # the home edge inside the inputs; ct.home = these + the intercept
            "home_ice": {"elo": model["elo_cfg"]["ha"], "xg": XG_HA},
            # `is not None`, not truthiness: an accuracy of 0.0 is a real
            # measurement and must not serialize as null.
            "cur_season_ll": round(cur_ll, 5) if cur_ll is not None else None,
            "cur_season_acc": round(cur_acc, 4) if cur_acc is not None else None,
            "cur_season_n": len(done),
            "features": ["Elo (k8,ha30)", "rest", "back-to-back", "xG team rating"],
            "n_games_train": train_count(games, F, model.get("as_of")),
        },
    }
    if value_updated:
        payload["value_updated"] = value_updated
    if old.get("value_status"):
        payload["value_status"] = old["value_status"]
    with open(PAYLOAD + ".tmp", "w", encoding="utf-8") as _fh:
        json.dump(payload, _fh, separators=(",", ":"))
    os.replace(PAYLOAD + ".tmp", PAYLOAD)
    print(f"wrote {PAYLOAD}: {len(sched)} games ({n_upcoming} unplayed), "
          f"{len(teams)} teams, {len(players)} players "
          f"({sum(1 for q in players.values() if q['grp'] == 'G')} goalies)")
    # fmt_metric, not an f-string spec: both are None until the season's first
    # game is played, and `f"{None:.5f}"` raises TypeError.
    print(f"current-season model LL {fmt_metric(cur_ll, 5)}  "
          f"acc {fmt_metric(cur_acc, 4)} (n={len(done)})")
    if proj:
        top = sorted(proj, key=lambda t: -proj[t]["pts"])[:5]
        print("projected points:", [(t, proj[t]["pts"], proj[t]["po"]) for t in top])


if __name__ == "__main__":
    main()
