"""Assemble the NHL day-of goalie SERVING-FEASIBILITY verdict into one file.

Breakthrough program (documents/breakthrough_program_prereg_2026_09_24.md).
Reads the evidence files written by the other bt_nhl_starter_* scripts and adds
the source/terms review and the proposed pipeline, so the verdict and its
evidence travel together:

  data/bt_nhl_starter_probe.jsonl        live polls of api-web + RO (probe)
  data/bt_nhl_starter_timing.json        first-appearance times per signal
  data/bt_nhl_goalie_confirm.jsonl       live watcher archive (prototype feed)
  data/bt_nhl_starter_wayback.json       regular-season RO timing (Internet Archive)
  data/bt_nhl_starter_backfill.json      first-shot proxy vs official starter, DEV
  data/bt_nhl_starter_predictability.json  walk-forward starter rules, DEV

No model metric, no TEST season, no odds.

    python -X utf8 phase0/bt_nhl_starter_feasibility.py
    -> data/bt_nhl_starter_feasibility.json
"""
from __future__ import annotations

import datetime as dt
import json
import os
from collections import defaultdict

OUT = "data/bt_nhl_starter_feasibility.json"


def _load(path, default=None):
    try:
        return json.load(open(path, encoding="utf-8"))
    except (OSError, ValueError):
        return default


def watcher_summary(path="data/bt_nhl_goalie_confirm.jsonl"):
    """Per game: first pre-game read with a confirmed starter, per side."""
    if not os.path.exists(path):
        return None
    first = defaultdict(dict)
    last_unconf = defaultdict(dict)
    games = set()
    for line in open(path, encoding="utf-8"):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("truth"):
            continue
        games.add(r["gid"])
        for side in ("away", "home"):
            s = r.get(side) or {}
            m = r.get("min_to_start")
            if s.get("confirmed") and side not in first[r["gid"]]:
                first[r["gid"]][side] = {"min_before": m, "starter": s.get("starter"),
                                         "starter_id": s.get("starter_id")}
            elif not s.get("confirmed") and side not in first[r["gid"]]:
                last_unconf[r["gid"]][side] = m
    rows = {}
    for g in sorted(games):
        rows[str(g)] = {side: {"first_confirmed_min_before": (first[g].get(side) or {}).get("min_before"),
                               "last_unconfirmed_min_before": last_unconf[g].get(side),
                               "starter": (first[g].get(side) or {}).get("starter"),
                               "starter_id": (first[g].get(side) or {}).get("starter_id")}
                        for side in ("away", "home")}
    n_sides = 2 * len(rows)
    n_conf = sum(1 for r in rows.values() for s in r.values()
                 if s["first_confirmed_min_before"] is not None)
    return {"games": len(rows), "team_sides": n_sides,
            "sides_confirmed_before_scheduled_start": n_conf, "per_game": rows}


SOURCES = [
    {"source": "api-web gamecenter/{id}/landing matchup.goalieComparison",
     "names_starter_pre_game": False,
     "evidence": "lists EVERY rostered goalie (2-4 per team) in ascending playerId order "
                 "at all 18 regular-season pre-game Internet Archive captures (T-696..T-25 min, "
                 "incl. one PRE-state) and at every live preseason poll; the list is a "
                 "season-stats widget, not a probable/confirmed starter"},
    {"source": "api-web gamecenter/{id}/boxscore playerByGameStats.*.goalies[].starter",
     "names_starter_pre_game": False,
     "evidence": "flag absent while LIVE (4 archive captures T+41..T+111) and absent from "
                 "preseason boxscores; present only once the game is OFF/FINAL "
                 "(100% of 300 sampled DEV sides) -> post-game truth, not a feed"},
    {"source": "api-web gamecenter/{id}/play-by-play rosterSpots",
     "names_starter_pre_game": False,
     "evidence": "dressed roster incl. both dressed goalies, no starter mark (see live probe "
                 "timing for when it populates)"},
    {"source": "api-web schedule / score / right-rail",
     "names_starter_pre_game": False,
     "evidence": "no goalie fields pre-game (score has teamLeaders; right-rail has "
                 "scratches/coaches); no key anywhere matching probable/starter/confirmed"},
    {"source": "nhl.com Club Playing Roster report scores/htmlreports/{season}/RO{gg}.HTM",
     "names_starter_pre_game": True,
     "evidence": "posted pre-game (from >= T-57 min in the archive) with the dressed roster; "
                 "'* Starting Lineup in Bold' marks each team's starting goalie only minutes "
                 "before the SCHEDULED start: 0/20 team-sides bold at >= T-20, 2/2 at T-20..-15, "
                 "2/4 at T-15..-10, 12/16 at T-10..-5, 4/4 at T-5..0 (28 captures 2022-23.."
                 "2025-26); actual puck drop is a median 9 min after the scheduled start "
                 "(n=54, range 7..38)"},
    {"source": "DailyFaceoff starting-goalies page (and similar editorial trackers)",
     "names_starter_pre_game": True,
     "evidence": "publishes morning 'confirmed/likely' starters hours before puck drop (the "
                 "information the market moves on); page data list was empty for the "
                 "2026-09-24 preseason slate",
     "terms": "no Terms of Use linked from the site (privacy policy refers to Terms of Use "
              "that are not published at any URL we found); robots.txt disallows /api/; "
              "owned by The Nation Network (Better Collective, a betting-media group). "
              "Editorial determinations re-published on a public site need written "
              "permission -> NOT scraped, NOT recommended without a licence"},
    {"source": "ESPN site API", "names_starter_pre_game": None,
     "evidence": "HTTP 403 Access Denied from this machine; not pursued"},
]

TERMS = {
    "nhl": "NHL.com Terms of Service (last updated 2025-10-29) prohibit 'unauthorized "
           "spidering, scraping, or harvesting' / automated compilation and limit use to "
           "non-commercial, informational, personal use. This applies equally to the "
           "api-web JSON the whole existing NHL pipeline already uses (20.6k-game spine, "
           "4-hourly nhl_update); api-web has no robots.txt, nhl.com robots.txt allows "
           "/scores/htmlreports/. The goalie feed adds ~20 small requests per game night-"
           "game and no new category of use; the site already labels itself non-commercial "
           "with NHL attribution. It is a standing, shared risk, not a new one - but it is "
           "not a licence.",
    "dailyfaceoff": "see SOURCES; do not use without permission",
}


PIPELINE = {
    "tiers": {
        "EARLY": "unchanged: the shipped team-level model; every NHL row until its starters "
                 "are confirmed. Not a pick.",
        "PROJECTED": "NOT recommended for NHL today: no official probable-goalie feed exists, "
                     "and walk-forward starter rules name the starter only ~2/3 of the time "
                     "(data/bt_nhl_starter_predictability.json) - a named-goalie projection "
                     "would be wrong on a third of team-games, and a mixture over goalies "
                     "stays close to the EARLY number. A licensed editorial feed (morning "
                     "'confirmed/likely') is the only route to an honest PROJECTED tier.",
        "CONFIRMED": "both sides' starting goalies marked bold in the official Playing Roster "
                     "report, read BEFORE the scheduled start with gameState FUT/PRE "
                     "(bt_nhl_starter_feed.fetch_confirmed enforces both)."},
    "steps": [
        "1. WATCH (new game-day job, not the 4-hourly refresh): phase0/bt_nhl_starter_watch.py "
        "reads the day's slate and polls each game at T-60,-40,-25,-20,-16,-13,-10,-8,-6,-4,"
        "-2,-1 min before the SCHEDULED start (2 requests per poll: play-by-play + RO; "
        "~20-24 requests per game), thinning to one re-check per 4 min after confirmation "
        "to catch warm-up swaps, never polling at/after the start. Run it as ONE long "
        "GitHub Actions job per slate (a trigger ~90 min before the first start; the job "
        "loops internally until the last start, so cron-trigger delay does not matter; "
        "matinee and evening slates need two triggers; <= 6 h per job) or on an always-on "
        "machine. The 4-hourly cron (00/04/08/12/16/20 UTC) has a nominal run inside "
        "T-19..T-0 for 1 of 1,344 2026-27 regular-season games, so it cannot do this job.",
        "2. ARCHIVE: every changed pre-game observation appended to an append-only JSONL "
        "(prototype data/bt_nhl_goalie_confirm.jsonl): gid, t_fetch UTC, min_to_start, "
        "gameState, RO status + generation stamp, per side the dressed goalies, the bold "
        "starter (sweater, name, playerId via play-by-play rosterSpots), bold skaters and "
        "scratches. Next morning, a truth record per game from the boxscore starter flag "
        "(watch --truth) so the late-swap rate is measured. Committed by the job like "
        "data/nhl_lineup_archive.jsonl.",
        "3. SERVE (only once a goalie-aware model has passed its DEV bar and TEST look): the "
        "watcher's on_confirmed hook re-serves that one game with the confirmed starters and "
        "writes its data/nhl_hp_ledger.json entry {hp, ct, t, tier: CONFIRMED, goalies, src: "
        "RO, ro_gen} before the scheduled start; nhl_serve.hold_started already freezes it "
        "from the scheduled start on. nhl_hp_freeze.freeze must take a per-row tier "
        "(s.get('tier', tier)) instead of one tier for the whole slate. A swap after "
        "confirmation re-serves (still pre-start). A game not confirmed by T-1 keeps its "
        "EARLY entry.",
        "4. PUBLISH: nhl.json row gets tier CONFIRMED + goalie names; one Pages deploy per "
        "confirmation wave (games cluster at :00/:30, ~3-5 deploys a night), sharing the "
        "refresh workflow's concurrency group so pushes never race.",
        "5. EDGE: the EARLY gate stays; a CONFIRMED NHL row may be priced only with odds "
        "fetched after its confirmation time (the badge must not pair a CONFIRMED hp with a "
        "pre-confirmation price).",
    ],
    "backtest": {
        "CONFIRMED": "reconstructable for every historical game: the official starter "
                     "(boxscore starter flag, final RO bold) equals the first-shot goalie the "
                     "round-1 screens already used on 300/300 sampled DEV team-sides - so "
                     "gmar/rapmel ALREADY measured CONFIRMED-grade goalie information "
                     "(gmar +0.00064, CI [-0.00024,+0.00154], n.s.). What history cannot give "
                     "is WHEN it was known and how often a confirmed starter was swapped "
                     "before the drop; the archive measures both going forward.",
        "PROJECTED": "reconstructable walk-forward from the spine (previous/modal/b2b rules, "
                     "no outcomes) - data/bt_nhl_starter_predictability.json.",
        "morning_editorial": "NOT reconstructable: no official archive of morning "
                             "'confirmed/likely' starters; Internet Archive captures are "
                             "sporadic and the source's terms are unpublished.",
        "existing_archive": "data/nhl_lineup_archive.jsonl does not exist yet "
                            "(data/nhl_lineup_inventory.json: 0 records). It is written only "
                            "by the 4-hourly refresh for regular-season games, which for "
                            "19:00 ET games last runs at T-3 h (16:00 ET) - before any "
                            "official goalie mark - so as scheduled it would archive roster "
                            "lists and coaches but never a confirmed starter.",
    },
}

RISKS = [
    "Window: the official mark lands ~5-19 min before the scheduled start (~15-28 min "
    "before the real puck drop); a missed or late poll leaves the game EARLY. Coverage "
    "(share of games with both sides confirmed by T-1) must be measured from the archive "
    "before anyone quotes a CONFIRMED record.",
    "Terms: NHL.com ToS forbid unauthorized automated harvesting and limit use to "
    "non-commercial personal use; the entire existing NHL pipeline shares this exposure "
    "(no new category of use, ~20 requests per game). Not a licence. DailyFaceoff and "
    "similar editorial trackers: terms unpublished, robots disallows /api/, betting-media "
    "owner - do not use without written permission.",
    "Format: RO is legacy HTML ('* Starting Lineup in Bold'); a layout change silently "
    "yields 'no bold goalie' - the feed fails SAFE (stays EARLY) and tests pin the "
    "structure, but a monitor on 'RO posted, 0 bold goalies at T-1' is needed.",
    "Selection: away teams' starting lineups are marked before home teams' (FLA@PIT, SEA@CAR "
    "at T-9: away bold, home not), so a late cutoff confirms away sides first; tier "
    "populations are start-time/venue-selected, as the pick policy already discloses for MLB.",
    "Value: serving feasibility does not create signal. The round-1 screens already had the "
    "actual starter and were null on DEV; a goalie-aware model must clear the prereg bar "
    "before any of this is wired to serving. A CONFIRMED number published ~10 min before "
    "puck drop also has little reader value versus the morning market move.",
    "Existing consumer: phase0/nhl_lineup_features.py treats a ONE-entry goalieComparison "
    "list as an inferred confirmation ('goalie_confirmed_inferred'); goalieComparison is a "
    "roster/season-stats list at every horizon observed, so that inference is unsupported "
    "and should be removed before the October inventory.",
    "Evidence base: live timing today is PRESEASON (split squads, preseason reports); "
    "regular-season timing rests on 28 Internet Archive captures (56 team-sides, 4 seasons, "
    "venue-skewed). Re-run bt_nhl_starter_probe/watch on the 2026-10 regular-season slates.",
]


def main():
    timing = _load("data/bt_nhl_starter_timing.json")
    wayback = _load("data/bt_nhl_starter_wayback.json")
    backfill = _load("data/bt_nhl_starter_backfill.json")
    pred = _load("data/bt_nhl_starter_predictability.json")
    res = {
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "protocol": {"dev_only_metrics": True, "test_touched": False, "market_use": "none",
                     "network": "public JSON/HTML reads only, low volume"},
        "sources": SOURCES,
        "terms": TERMS,
        "live_probe_timing": timing,
        "live_watcher": watcher_summary(),
        "wayback_regular_season_ro": {k: wayback.get(k) for k in
                                      ("n_pregame_captures", "by_minutes_before_start")}
        if wayback else None,
        "backfill_official_vs_first_shot": {k: backfill.get(k) for k in
                                            ("protocol", "counts", "official_flag_coverage",
                                             "first_shot_equals_official")} if backfill else None,
        "starter_predictability_dev": pred.get("all") if pred else None,
        "pipeline": PIPELINE,
        "risks": RISKS,
    }
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
