"""Player-value program (pv), NFL DATA step: player-attributed play events.

Re-pulls the nflverse play-by-play parquet per season and reduces it to one
row per play with every individually-credited player id (passer, target,
rusher, sackers, QB hitters, pass defenders, interceptor, tacklers, forced
fumble, fumbler, recoverer, penalised player, kicker/punter/returners) plus
the down/distance/field/score/clock context and the play outcome.

Why: the shipped NFL player layer is an 11v11 participation TrueSkill on play
EPA (everyone on the field shares each play's outcome -- the plus-minus
problem). The MLB breakthrough rated each player only on events HE controls
(FIP) in opponent-adjusted 1v1 matchups. The local reduced tables
(nfl_plays / nfl_duel_plays / nfl_rapm_plays / nfl_play_ctx2) carry no
defensive credits at all, so this pull is required before any NFL "FIP"
can be built.

Outputs (all rebuildable):
  data/pv_nfl_pbp_raw/play_by_play_{season}.parquet  raw nflverse cache (kept
        so later steps can reach any of the 372 columns without re-pulling)
  data/pv_nfl_events_parts/{season}.parquet          reduced per season
  data/pv_nfl_events.parquet                         all seasons, typed
  data/pv_nfl_events.csv                             same, CSV
  data/pv_nfl_events_coverage.json                   non-null counts per
        column per season -- DEV + warm-up seasons (<= 2015) ONLY
  data/pv_nfl_players_seen.csv                       id -> most frequent pbp
        short name, first/last season seen (identity metadata, no outcomes)

Constitution notes:
  * Market-blind: vegas_wp / vegas_wpa / vegas_home_wp(a) / spread_line /
    total_line are NOT carried; neither are the final-score columns
    (result/total/home_score/away_score) -- the in-play running score
    (posteam_score, defteam_score, score_differential) is kept.
  * Locked split: seasons >= 2016 are TEST. They are downloaded and reduced
    (serving needs them) but this script prints and stores NO statistic for
    them -- only a pass/fail integrity check (schema + non-empty).
  * epa / ep / wp / wpa / cp / cpoe / xpass / pass_oe / xyac_epa are nflfastR
    model outputs fit on pooled multi-season data (incl. post-2015 seasons).
    Same caveat as the shipped EPA features. The raw ingredients (down,
    ydstogo, yardline_100, yards_gained, air_yards, complete_pass, ...) are
    carried so a DEV-only / walk-forward re-fit is possible if needed.

Usage:
  python phase0/pv_nfl_events_pull.py                 # resumable, all seasons
  python phase0/pv_nfl_events_pull.py --refresh 2026  # re-pull given seasons
  python phase0/pv_nfl_events_pull.py --rereduce       # rebuild parts from raw cache
  python phase0/pv_nfl_events_pull.py --combine-only
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

import pandas as pd
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RAW_DIR = os.path.join(DATA, "pv_nfl_pbp_raw")
PART_DIR = os.path.join(DATA, "pv_nfl_events_parts")
OUT_CSV = os.path.join(DATA, "pv_nfl_events.csv")
OUT_PQ = os.path.join(DATA, "pv_nfl_events.parquet")
OUT_COV = os.path.join(DATA, "pv_nfl_events_coverage.json")
OUT_PLAYERS = os.path.join(DATA, "pv_nfl_players_seen.csv")
URL = ("https://github.com/nflverse/nflverse-data/releases/download/pbp/"
       "play_by_play_{y}.parquet")

FIRST_SEASON = 1999
LAST_DEV_SEASON = 2015          # NFL DEV = 2006-2015, TEST = 2016+
MIN_INTERVAL = 0.5              # seconds between HTTP requests (<= 4 req/s)

# ---------------------------------------------------------------- columns
ID_COLS = ["game_id", "play_id", "season", "season_type", "week", "game_date",
           "home_team", "away_team", "posteam", "defteam", "play_type"]

CTX_COLS = ["qtr", "down", "ydstogo", "yardline_100", "goal_to_go",
            "game_seconds_remaining", "half_seconds_remaining",
            "posteam_score", "defteam_score", "score_differential",
            "posteam_timeouts_remaining", "defteam_timeouts_remaining",
            "fixed_drive", "series", "shotgun", "no_huddle"]

OUTCOME_COLS = ["yards_gained", "ep", "epa", "success", "wp", "wpa",
                "first_down", "series_success", "touchdown", "pass_touchdown",
                "rush_touchdown", "return_touchdown", "safety",
                "qb_dropback", "qb_scramble", "qb_kneel", "qb_spike",
                "pass_attempt", "rush_attempt", "sack", "complete_pass",
                "incomplete_pass", "interception", "qb_hit",
                "tackled_for_loss", "fumble", "fumble_forced",
                "fumble_not_forced", "fumble_lost", "penalty",
                "two_point_attempt", "two_point_conv_result",
                "special_teams_play", "aborted_play",
                "air_yards", "yards_after_catch", "pass_length",
                "pass_location", "run_location", "run_gap",
                "cp", "cpoe", "xpass", "pass_oe", "air_epa", "yac_epa",
                "comp_air_epa", "comp_yac_epa", "xyac_epa", "qb_epa",
                "field_goal_result", "kick_distance", "extra_point_result",
                "return_yards", "penalty_type", "penalty_yards",
                "penalty_team", "td_team"]

PLAYER_COLS = [
    # offense, as credited by the gamebook
    "passer_player_id", "receiver_player_id", "rusher_player_id",
    # nflfastR-cleaned ids: passer_id also covers scrambles, sacks and
    # nullified (no_play) dropbacks; rusher_id/receiver_id likewise
    "passer_id", "receiver_id", "rusher_id",
    "lateral_receiver_player_id", "lateral_rusher_player_id",
    "fumbled_1_player_id", "fumbled_1_team",
    "fumbled_2_player_id", "fumbled_2_team",
    "td_player_id",
    # pass rush
    "sack_player_id", "half_sack_1_player_id", "half_sack_2_player_id",
    "qb_hit_1_player_id", "qb_hit_2_player_id",
    "tackle_for_loss_1_player_id", "tackle_for_loss_2_player_id",
    # coverage
    "pass_defense_1_player_id", "pass_defense_2_player_id",
    "interception_player_id",
    # tackling (team kept: after turnovers / on returns the tackler can be
    # an offensive player)
    "solo_tackle_1_player_id", "solo_tackle_1_team",
    "solo_tackle_2_player_id", "solo_tackle_2_team",
    "assist_tackle_1_player_id", "assist_tackle_1_team",
    "assist_tackle_2_player_id", "assist_tackle_2_team",
    "assist_tackle_3_player_id", "assist_tackle_3_team",
    "assist_tackle_4_player_id", "assist_tackle_4_team",
    "tackle_with_assist_1_player_id", "tackle_with_assist_1_team",
    "tackle_with_assist_2_player_id", "tackle_with_assist_2_team",
    # ball security / takeaways
    "forced_fumble_player_1_player_id", "forced_fumble_player_1_team",
    "forced_fumble_player_2_player_id", "forced_fumble_player_2_team",
    "fumble_recovery_1_player_id", "fumble_recovery_1_team",
    # discipline
    "penalty_player_id",
    # special teams
    "kicker_player_id", "punter_player_id", "kickoff_returner_player_id",
    "punt_returner_player_id", "blocked_player_id", "safety_player_id",
]

# player id -> short-name source pairs, for the identity map only
NAME_PAIRS = [("passer_player_id", "passer_player_name"),
              ("receiver_player_id", "receiver_player_name"),
              ("rusher_player_id", "rusher_player_name"),
              ("sack_player_id", "sack_player_name"),
              ("half_sack_1_player_id", "half_sack_1_player_name"),
              ("half_sack_2_player_id", "half_sack_2_player_name"),
              ("qb_hit_1_player_id", "qb_hit_1_player_name"),
              ("qb_hit_2_player_id", "qb_hit_2_player_name"),
              ("pass_defense_1_player_id", "pass_defense_1_player_name"),
              ("pass_defense_2_player_id", "pass_defense_2_player_name"),
              ("interception_player_id", "interception_player_name"),
              ("solo_tackle_1_player_id", "solo_tackle_1_player_name"),
              ("assist_tackle_1_player_id", "assist_tackle_1_player_name"),
              ("tackle_with_assist_1_player_id",
               "tackle_with_assist_1_player_name"),
              ("forced_fumble_player_1_player_id",
               "forced_fumble_player_1_player_name"),
              ("penalty_player_id", "penalty_player_name"),
              ("kicker_player_id", "kicker_player_name"),
              ("punter_player_id", "punter_player_name")]

MARKET_OR_RESULT = {"vegas_wp", "vegas_wpa", "vegas_home_wp", "vegas_home_wpa",
                    "spread_line", "total_line", "result", "total",
                    "home_score", "away_score"}

KEEP = ID_COLS + CTX_COLS + OUTCOME_COLS + PLAYER_COLS
# derived convenience columns (appended after KEEP):
#   target_player_id  pass-play target = receiver_player_id, else the
#                     nflfastR-parsed receiver_id. The gamebook id is MISSING
#                     on incompletions/interceptions in 2003-2008 (~40% of
#                     targets); receiver_id recovers them (agreement 0.9998
#                     where both exist).
#   sack_credit       "id:1" or "id:0.5;id:0.5" on sacks (empty = team sack /
#                     uncredited)
DERIVED = ["target_player_id", "sack_credit"]
OUT_COLS = KEEP + DERIVED
assert not (set(KEEP) & MARKET_OR_RESULT), "market/result column leaked"
assert len(KEEP) == len(set(KEEP)), "duplicate column"

INT_COLS = (["play_id", "season", "week", "qtr", "down", "ydstogo",
             "yardline_100", "goal_to_go", "game_seconds_remaining",
             "half_seconds_remaining", "posteam_score", "defteam_score",
             "score_differential", "posteam_timeouts_remaining",
             "defteam_timeouts_remaining", "fixed_drive", "series",
             "shotgun", "no_huddle", "yards_gained", "first_down",
             "series_success", "touchdown", "pass_touchdown",
             "rush_touchdown", "return_touchdown", "safety", "qb_dropback",
             "qb_scramble", "qb_kneel", "qb_spike", "pass_attempt",
             "rush_attempt", "sack", "complete_pass", "incomplete_pass",
             "interception", "qb_hit", "tackled_for_loss", "fumble",
             "fumble_forced", "fumble_not_forced", "fumble_lost", "penalty",
             "two_point_attempt", "special_teams_play", "aborted_play",
             "air_yards", "yards_after_catch", "kick_distance",
             "return_yards", "penalty_yards"])


def is_test(season):
    return season > LAST_DEV_SEASON


# ---------------------------------------------------------------- network
_last_req = [0.0]


def fetch(season, dest, attempts=6):
    """Download one season parquet atomically. Returns False on 404."""
    url = URL.format(y=season)
    tmp = dest + ".part"
    for k in range(attempts):
        wait = MIN_INTERVAL - (time.time() - _last_req[0])
        if wait > 0:
            time.sleep(wait)
        _last_req[0] = time.time()
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "glassbox-research/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r, \
                    open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            pq.read_schema(tmp)           # validates the parquet footer
            os.replace(tmp, dest)
            return True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False
            err = e
        except Exception as e:           # noqa: BLE001 (network + truncation)
            err = e
        if os.path.exists(tmp):
            os.remove(tmp)
        back = min(60, 2 ** (k + 1))
        print(f"  {season}: attempt {k + 1} failed ({err}); retry in {back}s",
              flush=True)
        time.sleep(back)
    raise RuntimeError(f"{season}: download failed after {attempts} attempts")


# ---------------------------------------------------------------- reduce
def reduce_season(raw_fp):
    names = set(pq.read_schema(raw_fp).names)
    missing = [c for c in KEEP if c not in names]
    name_cols = sorted({n for _, n in NAME_PAIRS if n in names})
    df = pd.read_parquet(raw_fp, columns=[c for c in KEEP if c in names]
                         + name_cols)
    for c in missing:
        df[c] = pd.NA
    # plays only: drop game-start / quarter-end / timeout-only rows that have
    # neither a play type nor a team in possession
    df = df[df.play_type.notna() & df.posteam.notna()].copy()
    for c in INT_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").round().astype("Int32")
    df["season"] = df["season"].astype("Int16")
    # null the gamebook placeholder ids ("0", "XX-0000001") seen in 1999-2000
    for c in [c for c in PLAYER_COLS if c.endswith("_id")]:
        v = df[c].astype("string")
        df[c] = v.mask((v == "0") | v.str.startswith("XX-").fillna(False))
    ident = df[[c for c in name_cols] +
               [i for i, n in NAME_PAIRS if n in names]].copy()
    ident["season"] = df["season"]
    df = df[KEEP].sort_values(["game_id", "play_id"]).reset_index(drop=True)
    is_tgt = (df.play_type == "pass") & (df.sack.fillna(0) == 0)
    df["target_player_id"] = (df.receiver_player_id
                              .fillna(df.receiver_id).where(is_tgt))
    full = df.sack_player_id.notna()
    half = df.half_sack_1_player_id.notna()
    sc = pd.Series(pd.NA, index=df.index, dtype="object")
    sc[full] = df.sack_player_id[full].astype(str) + ":1"
    h2 = df.half_sack_2_player_id.notna()
    sc[half & ~full] = (df.half_sack_1_player_id[half & ~full].astype(str)
                        + ":0.5")
    sc[half & h2 & ~full] = (sc[half & h2 & ~full] + ";"
                             + df.half_sack_2_player_id[half & h2 & ~full]
                             .astype(str) + ":0.5")
    df["sack_credit"] = sc.where(df.sack.fillna(0) == 1)
    return df[OUT_COLS], ident, missing


def identity_rows(ident):
    rows = []
    for i, n in NAME_PAIRS:
        if i not in ident.columns or n not in ident.columns:
            continue
        s = ident[[i, n, "season"]].dropna(subset=[i])
        s.columns = ["player_id", "name", "season"]
        rows.append(s)
    return pd.concat(rows, ignore_index=True) if rows else None


def coverage(df):
    out = {"n_plays": int(len(df))}
    for c in PLAYER_COLS + DERIVED + ["epa", "success", "air_yards",
                            "yards_after_catch", "cpoe", "cp", "xpass"]:
        out[c] = int(df[c].notna().sum())
    for pt, n in df.play_type.value_counts().items():
        out[f"play_type:{pt}"] = int(n)
    return out


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=FIRST_SEASON)
    ap.add_argument("--last", type=int, default=2026)
    ap.add_argument("--refresh", type=int, nargs="*", default=[],
                    help="seasons to re-download and re-reduce")
    ap.add_argument("--combine-only", action="store_true")
    ap.add_argument("--rereduce", action="store_true",
                    help="rebuild every part from the raw cache (no download"
                         " unless the raw file is missing)")
    args = ap.parse_args()
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PART_DIR, exist_ok=True)

    if not args.combine_only:
        for y in range(args.first, args.last + 1):
            raw = os.path.join(RAW_DIR, f"play_by_play_{y}.parquet")
            part = os.path.join(PART_DIR, f"{y}.parquet")
            ident_fp = os.path.join(PART_DIR, f"{y}_ident.parquet")
            fresh = y in args.refresh
            if (os.path.exists(part) and os.path.exists(ident_fp)
                    and not fresh and not args.rereduce):
                print(f"{y}: part exists, skip", flush=True)
                continue
            if fresh or not os.path.exists(raw):
                if not fetch(y, raw):
                    print(f"{y}: not published (404), skip", flush=True)
                    continue
            df, ident, missing = reduce_season(raw)
            idr = identity_rows(ident)
            tmp = part + ".tmp"
            df.to_parquet(tmp, index=False)
            os.replace(tmp, part)
            if idr is not None:
                idr.to_parquet(ident_fp, index=False)
            if is_test(y):
                ok = len(df) > 0 and list(df.columns) == OUT_COLS
                print(f"{y}: TEST season reduced -- integrity "
                      f"{'OK' if ok else 'FAIL'} (stats suppressed)",
                      flush=True)
            else:
                print(f"{y}: {len(df)} plays, {df.game_id.nunique()} games, "
                      f"missing cols {missing}", flush=True)

    # ------------------------------------------------------------ combine
    parts, idents, cov = [], [], {}
    for y in range(FIRST_SEASON, 2100):
        fp = os.path.join(PART_DIR, f"{y}.parquet")
        if not os.path.exists(fp):
            continue
        d = pd.read_parquet(fp)
        assert list(d.columns) == OUT_COLS, f"{y}: schema drift"
        assert len(d) > 0, f"{y}: empty part"
        parts.append(d)
        if not is_test(y):
            cov[str(y)] = coverage(d)
        ifp = os.path.join(PART_DIR, f"{y}_ident.parquet")
        if os.path.exists(ifp):
            idents.append(pd.read_parquet(ifp))
    allp = pd.concat(parts, ignore_index=True)
    assert not allp.duplicated(["game_id", "play_id"]).any(), "dup play key"
    tmp = OUT_PQ + ".tmp"
    allp.to_parquet(tmp, index=False)
    os.replace(tmp, OUT_PQ)
    tmp = OUT_CSV + ".tmp"
    allp.to_csv(tmp, index=False)
    os.replace(tmp, OUT_CSV)
    with open(OUT_COV, "w") as f:
        json.dump({"note": "non-null counts per column; seasons <= "
                           f"{LAST_DEV_SEASON} only (TEST seasons withheld)",
                   "seasons": cov}, f, indent=1)

    if idents:
        ids = pd.concat(idents, ignore_index=True)
        ids = ids[ids.name.notna()]
        nm = (ids.groupby(["player_id", "name"]).size().rename("n")
              .reset_index().sort_values(["player_id", "n"],
                                         ascending=[True, False])
              .drop_duplicates("player_id")[["player_id", "name"]])
        span = ids.groupby("player_id").season.agg(["min", "max"])
        span.columns = ["first_season", "last_season"]
        nm = nm.merge(span, left_on="player_id", right_index=True)
        nm.to_csv(OUT_PLAYERS, index=False)

    seasons = sorted(int(s) for s in allp.season.dropna().unique())
    nondev = allp[allp.season <= LAST_DEV_SEASON]
    print(f"combined: seasons {seasons[0]}-{seasons[-1]} "
          f"({len(seasons)} seasons); plays in seasons <= {LAST_DEV_SEASON}: "
          f"{len(nondev)}; columns {len(OUT_COLS)}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
