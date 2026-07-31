"""Announced-lineup features for NHL — the row-10 reopen, built before the data.

WHY THIS EXISTS
---------------
The NHL model (0.66418, ledger row 7) is statistically tied with the closing
line, but the two forecasts are not the same forecast: our orthogonal component
predicts outcomes and so does the market's, and the market's is the larger of
the two. The weight table says the market discounts our xG-team-rating and
back-to-back signal and substitutes something we carry no column for. The
leading candidates in that "something" are announced scratches and confirmed
starting goalies — information that exists only in the 30-60 minutes before puck
drop and that nobody archives historically.

phase0/nhl_update.py started archiving it (data/nhl_lineup_archive.jsonl) from
this season on. This module is the consumer: it reads that archive and turns it
into pre-game, market-blind features. It is written NOW, against an empty
archive, so that October is a data problem and not also a code problem.

Two ledger rows define the target:

  row 10  scratch/absence, valued by trailing TOI, +0.00077 CI[-0.00009,+0.00164]
          n.s. — the most promising null in the NHL ledger (positive point
          estimate, right-signed coefficient, unlike the flat-zero goalie and
          player nulls). Recorded caveat: "TOI is a weak value proxy (grinder
          minutes = star minutes)". Recorded reopen condition: "announced
          lineups + RAPM-weighted player values".
  row  9  starting-goalie GSAx DELTA, -0.00003 CI[-0.00036,+0.00030] n.s. and
          well-powered. That row kills the goalie-RATING channel. It does not
          test the goalie-IDENTITY-CONFIRMATION channel, because its starter was
          inferred post-hoc from shots faced, which is available for every game
          and therefore carries no news. What a confirmation adds over an
          inference is the timing and the certainty, which is the part the
          market visibly prices.

WHAT IS DIFFERENT FROM ROW 10
-----------------------------
  * value proxy: RAPM net xG/60 (data/nhl_rapm2_ratings.json) instead of
    trailing TOI. The RAPM build was materially corrected on 2026-07-31
    (strength-filtered 5v5 response, side-composition dummies that pin the
    unidentified F-vs-D level, deployment context, lambda 60k with EB
    shrinkage); the pre-correction ratings would have been a WORSE proxy than
    TOI, so this reopen is only worth taking with the corrected file.
  * lineup source: the announced pre-game roster, not "whoever took a shift",
    so the 61.5% 2025-26 shift coverage that hobbled row 10 does not apply and
    scratch-heavy end-of-season weeks are not silently missing.
  * positive information too: who IS dressed (dressed_power_diff), which is
    defined even on nights when the scratch list itself never posts.

PRE-REGISTRATION REQUIRED BEFORE ANY OF THIS IS SCREENED
--------------------------------------------------------
Nothing here has been screened and nothing here may be screened until an
archive with real volume exists. When that day comes:

  1. PRIMARY feature is pre-declared here, in advance, as `scratch_rapm_diff`.
     Everything else in FEATURE_ORDER is secondary and reported as such. Row 10
     pre-declared `absent_toi` the same way; a reopen that shops among six
     columns for the one that clears is not the same test.
  2. Reuse the row-10 harness verbatim — phase0/nhl_scratch_eval.py: increment
     over the shipped Elo+rest+b2b+xG blend, base_z as the single control
     column, LogisticRegression(C=1e6) fit on DEV, scored on TEST, 10k bootstrap
     CI on the per-game log-loss difference, seed 7.
  3. DEV/TEST discipline unchanged: DEV <= 20172018, TEST >= 20182019, masks
     hard-asserted, no TEST-era number printed by anything but the one eval.
  4. Leak safety: every piece of state (UsualStarter especially) is read BEFORE
     the game and updated AFTER, exactly as row 10 does. A snapshot may only be
     used for the game it belongs to.
  5. COST: one TEST look, ledger row 11, spent by the orchestrator and nobody
     else. The archive will be small at first — a season is ~1,300 games, so an
     honest first screen is n~1,300 against row 10's n~9,000. Power, not just
     sign, has to be argued before the look is spent. A likely-underpowered
     look is worth less than the look itself.

Usage:
  python -X utf8 phase0/nhl_lineup_features.py --report    # inventory the real archive
  python -X utf8 phase0/nhl_lineup_features.py --selftest  # synthetic round trip
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import Counter, defaultdict

ARCHIVE_PATH = "data/nhl_lineup_archive.jsonl"
RAPM_PATH = "data/nhl_rapm2_ratings.json"
NAMES_PATH = "data/nhl_player_names.json"
GOALIE_XG_PATH = "data/nhl_goalie_xg.csv"
INVENTORY_PATH = "data/nhl_lineup_inventory.json"

NAN = float("nan")
SIDES = ("homeTeam", "awayTeam")

# Pre-declared primary (see the pre-registration section above). Order is the
# order the report prints; it is not a ranking and it is not a menu to shop in.
PRIMARY_FEATURE = "scratch_rapm_diff"
FEATURE_ORDER = (
    "scratch_rapm_diff",        # PRIMARY
    "scratch_rapm_pos_diff",
    "scratch_n_diff",
    "dressed_power_diff",
    "backup_start_diff",
    "d_count_diff",
    "coach_change_diff",
)

# Fallback teamId -> abbrev. Used ONLY when the data-driven vote below cannot
# resolve a side, and any id missing from it degrades to "unresolved" rather
# than to a guess. The vote is primary precisely so that a stale entry here
# (relocations, expansion) cannot silently mislabel a lineup.
#
# Verified 2026-07-31 against api.nhle.com/stats/rest/en/team (all 34 entries
# match its triCode) and spot-checked against rosterSpots in 12 real 2025-26
# play-by-play payloads. Utah legitimately holds TWO franchise ids — 59 (Utah
# Hockey Club, 2024-25) and 68 (Utah Mammoth, 2025-26 on) — and the archive
# will carry 68, so an expansion/rebrand adding a third is exactly the case the
# vote exists to survive.
TEAM_IDS = {
    1: "NJD", 2: "NYI", 3: "NYR", 4: "PHI", 5: "PIT", 6: "BOS", 7: "BUF",
    8: "MTL", 9: "OTT", 10: "TOR", 11: "ATL", 12: "CAR", 13: "FLA", 14: "TBL",
    15: "WSH", 16: "CHI", 17: "DET", 18: "NSH", 19: "STL", 20: "CGY", 21: "COL",
    22: "EDM", 23: "VAN", 24: "ANA", 25: "DAL", 26: "LAK", 27: "PHX", 28: "SJS",
    29: "CBJ", 30: "MIN", 52: "WPG", 53: "ARI", 54: "VGK", 55: "SEA", 59: "UTA",
    68: "UTA",
}

# Keys the reader knows how to interpret inside record["lineup"]. Anything else
# is recorded in the inventory as an unknown key and otherwise ignored — the
# point of the archive is that we do not yet know the full key set.
KNOWN_LINEUP_KEYS = {
    "rosterSpots", "scratches", "gameInfoScratches", "goalieComparison",
    "confirmedGoalie", "headCoach",
}

USUAL_WINDOW = 10        # trailing confirmed starts used to name a usual starter
USUAL_MIN_OBS = 3        # below this the team has no usual starter -> NaN


# ----------------------------------------------------------------- extraction

def _as_ids(obj):
    """Every player id in an arbitrarily-shaped blob, order-preserving.

    The archive's shape is only partly known: a scratch list may arrive as bare
    ints, as {"id": ...} (what the right-rail endpoint returns for a finished
    game), or as {"playerId": ...} (what rosterSpots returns). Rather than
    guessing one, accept all of them and never raise on the rest.
    """
    out = []
    if isinstance(obj, bool):
        return out
    if isinstance(obj, int):
        return [obj]
    if isinstance(obj, str):
        return [int(obj)] if obj.isdigit() else []
    if isinstance(obj, dict):
        for key in ("playerId", "id"):
            v = obj.get(key)
            if isinstance(v, int) and not isinstance(v, bool):
                return [v]
            if isinstance(v, str) and v.isdigit():
                return [int(v)]
        return []
    if isinstance(obj, (list, tuple)):
        for v in obj:
            out.extend(_as_ids(v))
    return out


def _load_names(path=NAMES_PATH):
    """pid -> team abbrev, for the data-driven side vote. Missing file is fine."""
    try:
        raw = json.load(open(path, encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out = {}
    for k, v in raw.items():
        try:
            pid = int(k)
        except (TypeError, ValueError):
            continue
        if isinstance(v, dict) and v.get("team"):
            out[pid] = v["team"]
    return out


class Snapshot:
    """One archived pre-game observation, normalised into side-keyed views.

    Every field is optional. `.present` names what this snapshot actually
    carried, and every feature builder consults it rather than assuming.
    """

    __slots__ = ("gid", "t_fetch", "home", "away", "state", "raw",
                 "dressed", "positions", "scratches", "goalies", "coach",
                 "present", "unknown_keys", "notes")

    def __init__(self, gid, t_fetch, home, away, state, raw):
        self.gid = gid
        self.t_fetch = t_fetch
        self.home = home
        self.away = away
        self.state = state
        self.raw = raw
        self.dressed = {}       # side -> [pid]
        self.positions = {}     # side -> {pid: positionCode}
        self.scratches = {}     # side -> [pid]
        self.goalies = {}       # side -> [pid]  (list, NOT necessarily a starter)
        self.coach = {}         # side -> str
        self.present = set()
        self.unknown_keys = set()
        self.notes = []

    def __repr__(self):
        return (f"<Snapshot {self.gid} {self.away}@{self.home} {self.t_fetch} "
                f"{sorted(self.present)}>")

    def has(self, *fields):
        return all(f in self.present for f in fields)


def _resolve_sides(team_ids, home, away, names, members):
    """teamId -> "homeTeam"/"awayTeam", by data vote first, static map second.

    Returns {} when the orientation cannot be established. Guessing here would
    silently swap a scratch list between teams and flip the sign of the whole
    feature, which is the single worst failure mode available to this module,
    so the unresolved case is a hard NaN rather than a coin flip.
    """
    if len(team_ids) != 2 or not (home and away):
        return {}
    votes = {}
    for tid in team_ids:
        tally = Counter(names[p] for p in members.get(tid, ()) if p in names)
        votes[tid] = tally
    lab = {}
    for tid in team_ids:
        tally = votes[tid]
        top = tally.most_common(1)
        if top and top[0][1] >= 3:
            lab[tid] = top[0][0]
    if len(lab) < 2 or len(set(lab.values())) < 2:
        lab = {tid: TEAM_IDS[tid] for tid in team_ids if tid in TEAM_IDS}
    if set(lab.values()) != {home, away}:
        return {}
    return {tid: ("homeTeam" if ab == home else "awayTeam") for tid, ab in lab.items()}


def _parse_lineup(snap, lineup, names):
    """Fill a Snapshot's normalised views from whatever `lineup` turns out to be."""
    if not isinstance(lineup, dict):
        snap.notes.append("lineup is not an object")
        return
    snap.unknown_keys = {k for k in lineup if k not in KNOWN_LINEUP_KEYS}

    # -- rosterSpots: the dressed lineup, keyed by numeric teamId --------------
    spots = lineup.get("rosterSpots")
    if isinstance(spots, list) and spots:
        members, pos = defaultdict(list), {}
        for s in spots:
            if not isinstance(s, dict):
                continue
            ids = _as_ids(s)
            tid = s.get("teamId")
            if not ids or not isinstance(tid, int):
                continue
            members[tid].append(ids[0])
            pos[ids[0]] = s.get("positionCode") or "?"
        sides = _resolve_sides(set(members), snap.home, snap.away, names, members)
        if sides:
            for tid, side in sides.items():
                snap.dressed[side] = members[tid]
                snap.positions[side] = {p: pos.get(p, "?") for p in members[tid]}
            snap.present.add("dressed")
        elif members:
            snap.notes.append(f"rosterSpots present but sides unresolved "
                              f"(teamIds {sorted(members)})")

    # -- scratches: two known shapes, plus an unknown-but-tolerated one --------
    gis = lineup.get("gameInfoScratches")
    if isinstance(gis, dict):
        got = {}
        for side in SIDES:
            ids = _as_ids(gis.get(side))
            if ids:
                got[side] = ids
        if got:
            snap.scratches.update(got)
            snap.present.add("scratches")
    flat = lineup.get("scratches")
    if flat and "scratches" not in snap.present:
        if isinstance(flat, dict):
            got = {}
            for side in SIDES:
                ids = _as_ids(flat.get(side))
                if ids:
                    got[side] = ids
            if got:
                snap.scratches.update(got)
                snap.present.add("scratches")
        else:
            ids = _as_ids(flat)
            by_team = defaultdict(list)
            for s in (flat if isinstance(flat, list) else []):
                if isinstance(s, dict) and isinstance(s.get("teamId"), int):
                    got = _as_ids(s)
                    if got:
                        by_team[s["teamId"]].append(got[0])
            sides = _resolve_sides(set(by_team), snap.home, snap.away, names, by_team)
            if sides:
                for tid, side in sides.items():
                    snap.scratches[side] = by_team[tid]
                snap.present.add("scratches")
            elif ids:
                # a flat, unattributed list: usable as a count only, and not
                # even that per-side, so it is recorded and not used
                snap.notes.append(f"flat scratches list of {len(ids)} ids "
                                  f"cannot be attributed to a side")

    # -- goalies: a LIST here, never assumed to be a confirmation -------------
    gc = lineup.get("goalieComparison")
    if isinstance(gc, dict):
        got = {}
        for side in SIDES:
            ids = _as_ids(gc.get(side))
            if ids:
                got[side] = ids
        if got:
            snap.goalies.update(got)
            snap.present.add("goalie_list")
    conf = lineup.get("confirmedGoalie")
    if isinstance(conf, dict):
        got = {}
        for side in SIDES:
            ids = _as_ids(conf.get(side))
            if len(ids) == 1:
                got[side] = ids
        if got:
            snap.goalies.update(got)
            snap.present.add("goalie_confirmed")

    hc = lineup.get("headCoach")
    if isinstance(hc, dict):
        got = {}
        for side in SIDES:
            v = hc.get(side)
            if isinstance(v, dict):
                v = v.get("default")
            if isinstance(v, str) and v.strip():
                got[side] = v.strip()
        if got:
            snap.coach.update(got)
            snap.present.add("coach")

    # A two-entry goalie list is the season-leaders comparison widget, not a
    # confirmation. A ONE-entry list is treated as a confirmation candidate,
    # because that is the shape a posted starter would most plausibly take;
    # this inference is flagged so the October inventory can confirm or kill it.
    if "goalie_confirmed" not in snap.present and snap.goalies:
        if all(len(v) == 1 for v in snap.goalies.values()) and len(snap.goalies) == 2:
            snap.present.add("goalie_confirmed_inferred")


# --------------------------------------------------------------------- reader

def read_archive(path=ARCHIVE_PATH, names=None):
    """Parse data/nhl_lineup_archive.jsonl into Snapshots + an inventory report.

    Contract: never raises on archive content. The file is written append-only
    by a job that can be killed mid-write, so a torn final line is expected, not
    exceptional; likewise an unfamiliar key, because the whole point of the
    archive is that the game-day payload shape was unobservable when the
    archiver was written.
    """
    names = _load_names() if names is None else names
    rep = {
        "path": path, "exists": os.path.exists(path), "lines": 0, "records": 0,
        "bad_json": 0, "not_object": 0, "missing_gid": 0, "no_lineup": 0,
        "games": 0, "snapshots_per_game": {}, "field_counts": Counter(),
        "lineup_keys": Counter(), "unknown_lineup_keys": Counter(),
        "states": Counter(), "notes": Counter(),
    }
    snaps = []
    if not rep["exists"]:
        rep["message"] = f"{path} does not exist yet (expected in the offseason)"
        return snaps, rep
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rep["lines"] += 1
            try:
                rec = json.loads(line)
            except ValueError:
                rep["bad_json"] += 1      # torn tail of an interrupted append
                continue
            if not isinstance(rec, dict):
                rep["not_object"] += 1
                continue
            gid = rec.get("gid")
            if isinstance(gid, str) and gid.isdigit():
                gid = int(gid)
            if not isinstance(gid, int):
                rep["missing_gid"] += 1
                continue
            lineup = rec.get("lineup")
            if not isinstance(lineup, dict) or not lineup:
                rep["no_lineup"] += 1
                continue
            snap = Snapshot(gid, rec.get("t_fetch") or "", rec.get("home") or "",
                            rec.get("away") or "", rec.get("state") or "", lineup)
            _parse_lineup(snap, lineup, names)
            snaps.append(snap)
            rep["records"] += 1
            rep["states"][snap.state] += 1
            rep["lineup_keys"].update(lineup.keys())
            rep["unknown_lineup_keys"].update(snap.unknown_keys)
            rep["field_counts"].update(snap.present)
            rep["notes"].update(snap.notes)
    per = Counter(s.gid for s in snaps)
    rep["games"] = len(per)
    rep["snapshots_per_game"] = dict(Counter(per.values()))
    return snaps, rep


def latest_per_game(snaps):
    """The last snapshot archived for each game, richest-wins on ties.

    Later is better (lineups only firm up), but a later fetch that lost a field
    must not erase an earlier one that had it, so ties and regressions are
    broken by how many fields the snapshot actually carries.
    """
    best = {}
    for s in snaps:
        cur = best.get(s.gid)
        if cur is None or (s.t_fetch, len(s.present)) >= (cur.t_fetch, len(cur.present)):
            best[s.gid] = s
    return best


def merge_snapshots(snaps):
    """Per game, fold every snapshot into one, preferring the latest value of
    each field independently. Use when the archiver's fields arrive in different
    fetches (e.g. rosterSpots early, confirmed goalie late)."""
    order = defaultdict(list)
    for s in snaps:
        order[s.gid].append(s)
    out = {}
    for gid, lst in order.items():
        lst = sorted(lst, key=lambda s: s.t_fetch)
        base = Snapshot(gid, lst[-1].t_fetch, lst[-1].home, lst[-1].away,
                        lst[-1].state, lst[-1].raw)
        for s in lst:
            for attr in ("dressed", "positions", "scratches", "goalies", "coach"):
                for side, val in getattr(s, attr).items():
                    getattr(base, attr)[side] = val
            base.present |= s.present
            base.unknown_keys |= s.unknown_keys
            base.notes.extend(n for n in s.notes if n not in base.notes)
        out[gid] = base
    return out


# ------------------------------------------------------------- value lookups

def load_rapm(path=RAPM_PATH):
    """pid -> corrected RAPM row. Missing file returns {} (features go NaN)."""
    try:
        raw = json.load(open(path, encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out = {}
    for k, v in raw.items():
        try:
            pid = int(k)
        except (TypeError, ValueError):
            continue
        if isinstance(v, dict) and isinstance(v.get("net"), (int, float)):
            out[pid] = v
    return out


class UsualStarter:
    """Who each team has actually been starting in goal lately.

    Rolling window of the last USUAL_WINDOW confirmed starts per team; the modal
    goalie is the "usual" one. Below USUAL_MIN_OBS observations a team has no
    usual starter and every goalie feature for it is NaN — an early-season team
    genuinely has no established starter, and calling the first guy we saw
    "usual" would manufacture backup-start flags out of ignorance. This is the
    same instinct as row 10's first-8-games-of-season mask.

    LEAK RULE: read before the game, observe() after. `seed_from_goalie_xg` takes
    a cutoff for exactly that reason — for a screen it must be the game's own
    date, never the end of the file.
    """

    def __init__(self, window=USUAL_WINDOW, min_obs=USUAL_MIN_OBS):
        self.window = window
        self.min_obs = min_obs
        self.hist = defaultdict(list)

    def observe(self, team, goalie_id):
        if not team or not isinstance(goalie_id, int):
            return
        h = self.hist[team]
        h.append(goalie_id)
        if len(h) > self.window:
            del h[:-self.window]

    def usual(self, team):
        h = self.hist.get(team) or []
        if len(h) < self.min_obs:
            return None
        return Counter(h).most_common(1)[0][0]

    def seen(self, team):
        return len(self.hist.get(team) or [])

    def seed_from_goalie_xg(self, path=GOALIE_XG_PATH, before_date=None,
                            games_path="data/nhl_games.csv"):
        """Backfill history from completed games: starter = most work faced.

        Not a confirmation — it is the row-9 post-hoc proxy — but for the
        BASELINE ("who does this team usually start") a post-hoc identification
        of past starters is exactly right and involves no information about
        tonight. Returns the number of team-games folded in.
        """
        import csv
        try:
            fh = open(path, encoding="utf-8")
        except OSError:
            return 0
        dates = {}
        if before_date:
            try:
                with open(games_path, encoding="utf-8") as gh:
                    for r in csv.DictReader(gh):
                        dates[r["game_id"]] = r["date"]
            except OSError:
                pass
        best = {}
        with fh:
            for r in csv.DictReader(fh):
                gid = r.get("nhl_game_id")
                if before_date and dates.get(gid, "") >= before_date:
                    continue
                try:
                    key = (gid, r["team"])
                    work = float(r["xga"])
                    pid = int(r["goalie_id"])
                except (KeyError, TypeError, ValueError):
                    continue
                if key not in best or work > best[key][0]:
                    best[key] = (work, pid)
        n = 0
        for (gid, team), (_, pid) in sorted(best.items()):
            self.observe(team, pid)
            n += 1
        return n


# ----------------------------------------------------------- feature builders

def feat_scratch_absence_rapm(snap, rapm):
    """PRIMARY. Announced scratches, valued by corrected RAPM net xG/60.

    MECHANISM. The team rating (Elo + xG) prices a team's AVERAGE lineup. What
    it cannot know pre-game is tonight's deviation from that average: a team
    dressing without a top-pair defenceman is weaker tonight than its rating
    says, and the rating will not learn this until after the game. Row 10 tested
    exactly this deviation but valued each missing player by his trailing
    ice-time, which does not separate a shutdown centre from a fourth-line
    grinder who plays the same minutes. RAPM net is a teammate- and
    opponent-adjusted, EB-shrunk estimate of the goal-expectancy swing a player
    produces per 60, so it is the value proxy row 10 asked for.

        scratch_value(side) = sum over announced scratches of RAPM net
        scratch_rapm_diff   = away_value - home_value

    SIGN: identical to row 10's `absent_toi` convention — positive means the
    AWAY team lost more, i.e. the home team is stronger than the ratings say.

    Unrated scratches (below the 400s RAPM floor: call-ups, 13th forwards) score
    0, which is the right default because the RAPM scale is centred within
    position — an unrated player is a replacement-level player and losing him is
    close to a no-op. `n_unrated` reports how often that default fires so the
    assumption stays visible.
    """
    out = {"scratch_rapm_diff": NAN, "scratch_rapm_pos_diff": NAN,
           "scratch_n_diff": NAN, "n_unrated": 0, "n_scratches": 0}
    if "scratches" not in snap.present:
        return out
    vals, pos_vals, counts = {}, {}, {}
    for side in SIDES:
        ids = snap.scratches.get(side)
        if ids is None:
            return out                     # one side only is not a difference
        tot = ptot = 0.0
        for pid in ids:
            row = rapm.get(pid)
            if row is None:
                out["n_unrated"] += 1
                continue
            net = float(row["net"])
            tot += net
            ptot += max(net, 0.0)
        vals[side], pos_vals[side], counts[side] = tot, ptot, len(ids)
    out["scratch_rapm_diff"] = vals["awayTeam"] - vals["homeTeam"]
    out["scratch_rapm_pos_diff"] = pos_vals["awayTeam"] - pos_vals["homeTeam"]
    out["scratch_n_diff"] = float(counts["awayTeam"] - counts["homeTeam"])
    out["n_scratches"] = counts["awayTeam"] + counts["homeTeam"]
    return out


def feat_dressed_power(snap, rapm):
    """Total RAPM value of the skaters who ARE dressed, home minus away.

    MECHANISM. The complement of the scratch feature, and strictly better posed
    where it is available: it uses the positive information (this is tonight's
    actual eighteen) instead of the negative one, so it is defined on nights
    when the scratch list never posts but the roster does, and it does not
    depend on correctly identifying who "should" have played. It is the NHL
    analogue of the NFL serving chain's rpow lineup power.

    Because RAPM net is centred within position, an average lineup sums to
    roughly zero and the difference is a direct "tonight's rosters differ by
    this much goal expectancy per 60" reading. Goalies are excluded (the RAPM
    fit excludes them by construction).

    Note the overlap with the scratch feature: on a night where both are
    defined they are near-complements, so they must NOT be screened as if they
    were independent evidence, and only one may enter a shipped blend.
    """
    out = {"dressed_power_diff": NAN, "dressed_rated_share": NAN,
           "d_count_diff": NAN, "dressed_n": {}}
    if "dressed" not in snap.present:
        return out
    power, nD, rated, total = {}, {}, 0, 0
    for side in SIDES:
        ids = snap.dressed.get(side)
        if not ids:
            return out
        pos = snap.positions.get(side, {})
        tot = 0.0
        d = 0
        for pid in ids:
            if pos.get(pid) == "G":
                continue
            total += 1
            if pos.get(pid) == "D":
                d += 1
            row = rapm.get(pid)
            if row is not None:
                tot += float(row["net"])
                rated += 1
        power[side], nD[side] = tot, d
        out["dressed_n"][side] = len(ids)
    out["dressed_power_diff"] = power["homeTeam"] - power["awayTeam"]
    out["d_count_diff"] = float(nD["homeTeam"] - nD["awayTeam"])
    out["dressed_rated_share"] = rated / total if total else NAN
    return out


def feat_goalie_confirmation(snap, usual):
    """Is tonight's CONFIRMED starter the team's usual starter?

    MECHANISM. Row 9 settled that a goalie's GSAx RATING adds nothing over the
    team rating: NHL backups are still NHL goalies and single-game goaltending
    is mostly noise. It did not settle the identity channel, because its starter
    was identified after the fact from shots faced — information available for
    every game ever played, and therefore not news. A CONFIRMATION is news: it
    arrives at a known time, it resolves an uncertainty the ratings cannot
    resolve, and it is one of the few things the market observably reacts to.
    The feature is therefore deliberately crude and identity-only:

        backup_start(side) = 1 if the confirmed starter is not the team's modal
                             starter over its last 10 confirmed starts, else 0
        backup_start_diff  = away_backup - home_backup

    SIGN: positive = the away team is starting a non-usual goalie = home
    stronger, matching the away-minus-home badness convention above.

    NaN, NOT ZERO, whenever the confirmation is absent or the team has fewer
    than USUAL_MIN_OBS starts on record. Zero would assert "the usual guy is in
    net", which is the most common state and would therefore bury every real
    backup start inside a wall of false negatives.
    """
    out = {"backup_start_diff": NAN, "goalie_source": "none",
           "starters": {}, "usual": {}}
    conf = ("goalie_confirmed" in snap.present
            or "goalie_confirmed_inferred" in snap.present)
    if not conf:
        return out
    out["goalie_source"] = ("confirmed" if "goalie_confirmed" in snap.present
                            else "inferred_single_entry")
    flags = {}
    for side, team in (("homeTeam", snap.home), ("awayTeam", snap.away)):
        ids = snap.goalies.get(side) or []
        if len(ids) != 1:
            return out
        u = usual.usual(team)
        out["starters"][side] = ids[0]
        out["usual"][side] = u
        if u is None:
            return out
        flags[side] = 0.0 if ids[0] == u else 1.0
    out["backup_start_diff"] = flags["awayTeam"] - flags["homeTeam"]
    return out


def feat_coach_change(snap, coach_state):
    """Bench-boss change since the team's last archived game, home minus away.

    MECHANISM. The head coach is pre-game, market-blind, and one of the few
    non-injury shocks a team rating cannot absorb for weeks — a mid-season
    firing changes deployment and effort immediately while Elo moves at k=8.
    Only computable if the archiver ever stores headCoach (the right-rail
    endpoint carries it; the landing endpoint does not), so this returns NaN on
    today's capture and is here so that adding the field is a one-line change.

    coach_state is a mutable {team: last_seen_coach} dict, updated AFTER the
    feature is read.
    """
    out = {"coach_change_diff": NAN, "coach_new": {}}
    if "coach" not in snap.present:
        return out
    flags = {}
    for side, team in (("homeTeam", snap.home), ("awayTeam", snap.away)):
        name = snap.coach.get(side)
        if not name or not team:
            return out
        prev = coach_state.get(team)
        flags[side] = 0.0 if prev is None else (1.0 if name != prev else 0.0)
        out["coach_new"][side] = flags[side]
    out["coach_change_diff"] = flags["homeTeam"] - flags["awayTeam"]
    for side, team in (("homeTeam", snap.home), ("awayTeam", snap.away)):
        coach_state[team] = snap.coach.get(side)
    return out


def snapshot_diagnostics(snaps_for_game):
    """Latency/churn instrumentation — DIAGNOSTIC ONLY, never a model feature.

    n_snapshots and the spread of fetch times measure OUR polling cadence at
    least as much as the team's lineup churn: a 6x/day CI job that happens to
    run twice inside the announcement window will report "two revisions" for a
    game where nothing changed, and a job that misses the window reports one for
    a game where a star was scratched an hour before puck drop. Screening it
    would be screening the cron schedule. It belongs in the archive-health
    report and nowhere else.
    """
    ts = sorted(s.t_fetch for s in snaps_for_game if s.t_fetch)
    return {"n_snapshots": len(snaps_for_game),
            "first_fetch": ts[0] if ts else "", "last_fetch": ts[-1] if ts else ""}


def build_features(snaps, rapm=None, usual=None, coach_state=None):
    """gid -> feature dict, for every game with at least one snapshot.

    Games are processed in archived order and the goalie state is updated AFTER
    each game is scored, mirroring row 10's leak-safe walk. Any feature whose
    inputs are absent is NaN, never 0.
    """
    rapm = load_rapm() if rapm is None else rapm
    usual = UsualStarter() if usual is None else usual
    coach_state = {} if coach_state is None else coach_state
    merged = merge_snapshots(snaps)
    by_game = defaultdict(list)
    for s in snaps:
        by_game[s.gid].append(s)
    out = {}
    for gid in sorted(merged):
        snap = merged[gid]
        row = {"gid": gid, "home": snap.home, "away": snap.away,
               "t_fetch": snap.t_fetch, "fields": sorted(snap.present)}
        row.update(feat_scratch_absence_rapm(snap, rapm))
        row.update(feat_dressed_power(snap, rapm))
        row.update(feat_goalie_confirmation(snap, usual))
        row.update(feat_coach_change(snap, coach_state))
        row.update(snapshot_diagnostics(by_game[gid]))
        out[gid] = row
        # ---- state updates AFTER the read (leak-safe) ----
        for side, team in (("homeTeam", snap.home), ("awayTeam", snap.away)):
            ids = snap.goalies.get(side) or []
            if len(ids) == 1:
                usual.observe(team, ids[0])
    return out


# ------------------------------------------------- capture-gap helper (unused
# by the reader, offered to the archiver)

def lineup_subset_from_endpoints(landing, right_rail=None, play_by_play=None):
    """Build a lineup subset from ALL THREE gamecenter endpoints.

    Measured against the live API on 2026-07-31 (see the module report):
    `landing` does NOT carry rosterSpots or scratches in any observable game
    state — for a finished game its summary.gameInfo is an empty object, and for
    a scheduled game it carries only the season-stat goalie comparison. The
    scratch list lives in `right-rail` -> gameInfo.{home,away}Team.scratches and
    the dressed roster lives in `play-by-play` -> rosterSpots.

    phase0/nhl_update.archive_lineups currently fetches landing only, so on
    today's code path the two fields this module most wants may never be
    archived at all. This function is the drop-in replacement — it produces a
    strict SUPERSET of the current subset shape, so the reader above handles
    both without a version flag.
    """
    out = {}
    landing = landing if isinstance(landing, dict) else {}
    rr = right_rail if isinstance(right_rail, dict) else {}
    pbp = play_by_play if isinstance(play_by_play, dict) else {}

    spots = pbp.get("rosterSpots") or landing.get("rosterSpots")
    if spots:
        out["rosterSpots"] = [
            {k: p.get(k) for k in ("teamId", "playerId", "positionCode",
                                   "sweaterNumber")}
            for p in spots if isinstance(p, dict)]

    for src in (rr.get("gameInfo"), (landing.get("summary") or {}).get("gameInfo")):
        if not isinstance(src, dict):
            continue
        for side in SIDES:
            d = src.get(side)
            if not isinstance(d, dict):
                continue
            ids = _as_ids(d.get("scratches"))
            if ids:
                out.setdefault("gameInfoScratches", {}).setdefault(side, ids)
            hc = d.get("headCoach")
            if isinstance(hc, dict):
                hc = hc.get("default")
            if isinstance(hc, str) and hc.strip():
                out.setdefault("headCoach", {}).setdefault(side, hc.strip())

    gc = (landing.get("matchup") or {}).get("goalieComparison") or {}
    for side in SIDES:
        ids = [g.get("playerId") for g in ((gc.get(side) or {}).get("leaders") or [])
               if isinstance(g, dict) and g.get("playerId")]
        if ids:
            out.setdefault("goalieComparison", {})[side] = ids
    return out


# ----------------------------------------------------------- synthetic + report

def synthetic_archive_lines():
    """A hand-built archive covering every shape the reader claims to survive.

    Game 1 is the target case: an early landing-only snapshot (two goalies per
    side = the comparison widget, no confirmation), then a game-day snapshot
    with a dressed roster, a per-side scratch list, coaches, and a single-entry
    goalie list. Game 2 is the degenerate case: goalie list only, plus an
    unknown key. Then the three malformed lines that a killed append or an API
    change will actually produce.
    """
    # BOS(6) home vs MTL(8) away. 900001.. are home ids, 900101.. away ids.
    home_ids = [900001 + i for i in range(20)]
    away_ids = [900101 + i for i in range(20)]
    pos = (["C"] * 4 + ["L"] * 4 + ["R"] * 4 + ["D"] * 6 + ["G"] * 2)

    def spots():
        rows = []
        for tid, ids in ((6, home_ids), (8, away_ids)):
            for p, pc in zip(ids, pos):
                rows.append({"teamId": tid, "playerId": p, "positionCode": pc,
                             "sweaterNumber": p % 100})
        return rows

    early = {"gid": 2026020001, "t_fetch": "2026-10-08T14:00:00+00:00",
             "home": "BOS", "away": "MTL", "state": "FUT",
             "lineup": {"goalieComparison": {"homeTeam": [900019, 900018],
                                             "awayTeam": [900119, 900118]}}}
    gameday = {"gid": 2026020001, "t_fetch": "2026-10-08T22:40:00+00:00",
               "home": "BOS", "away": "MTL", "state": "PRE",
               "lineup": {
                   "rosterSpots": spots(),
                   # right-rail shape: list of {"id": ...} dicts
                   "gameInfoScratches": {"homeTeam": [{"id": 900050}],
                                         "awayTeam": [{"id": 900150},
                                                      {"id": 900151}]},
                   "headCoach": {"homeTeam": {"default": "A. Coach"},
                                 "awayTeam": "B. Coach"},
                   "confirmedGoalie": {"homeTeam": [900019],
                                       "awayTeam": [900118]}}}
    thin = {"gid": 2026020002, "t_fetch": "2026-10-08T15:00:00+00:00",
            "home": "TOR", "away": "OTT", "state": "FUT",
            "lineup": {"goalieComparison": {"homeTeam": [900219, 900218],
                                            "awayTeam": [900319]},
                       "warmupNotes": {"anything": "the API might add"}}}
    no_lineup = {"gid": 2026020003, "t_fetch": "2026-10-08T15:00:00+00:00",
                 "home": "NYR", "away": "NYI", "state": "FUT", "lineup": {}}
    lines = [json.dumps(r) for r in (early, gameday, thin, no_lineup)]
    lines.append("")                                   # blank line
    lines.append('{"gid": 2026020004, "t_fetch": "2026-10-08T16:0')  # torn tail
    return lines


def _synthetic_rapm():
    """Values for the synthetic players. Deliberately lopsided so the sign of
    every difference feature is unambiguous."""
    r = {}
    for i in range(20):
        r[900001 + i] = {"net": 0.05, "pos": "F", "toi_min": 900.0}
        r[900101 + i] = {"net": 0.05, "pos": "F", "toi_min": 900.0}
    r[900001] = {"net": 0.30, "pos": "F", "toi_min": 1600.0}   # a star BOS dresses
    # the scratched players are deliberately NOT in either dressed roster, which
    # is what a scratch is; if one ever leaks into dressed_power the expectation
    # below moves and the check fires
    r[900050] = {"net": 0.40, "pos": "D", "toi_min": 1500.0}   # BOS scratches a star
    r[900150] = {"net": 0.20, "pos": "F", "toi_min": 1200.0}   # MTL scratches a good
    r[900151] = {"net": -0.05, "pos": "F", "toi_min": 800.0}   # ...and a weak one
    return r


def selftest(verbose=True):
    """End-to-end dry run against the synthetic archive. Returns 0 on success.

    This is the whole point of writing the module in July: the first real
    game-day payload must not be the first time this code path executes.
    """
    import tempfile

    fails = []

    def check(cond, msg):
        if not cond:
            fails.append(msg)
        if verbose:
            print(f"  [{'ok ' if cond else 'FAIL'}] {msg}")

    tmpdir = tempfile.mkdtemp(prefix="nhl_lineup_selftest_")
    path = os.path.join(tmpdir, "archive.jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(synthetic_archive_lines()))

    if verbose:
        print("reader")
    snaps, rep = read_archive(path, names={})
    check(rep["records"] == 3, f"3 usable records parsed (got {rep['records']})")
    check(rep["bad_json"] == 1, f"torn tail counted, not raised (got {rep['bad_json']})")
    check(rep["no_lineup"] == 1, f"empty-lineup record skipped (got {rep['no_lineup']})")
    check(rep["unknown_lineup_keys"].get("warmupNotes") == 1,
          "unknown key recorded rather than crashed")
    check(rep["games"] == 2, f"2 games with usable snapshots (got {rep['games']})")

    merged = merge_snapshots(snaps)
    g1 = merged[2026020001]
    check(g1.has("dressed", "scratches", "coach", "goalie_confirmed"),
          f"game 1 merged fields = {sorted(g1.present)}")
    check(len(g1.dressed["homeTeam"]) == 20 and len(g1.dressed["awayTeam"]) == 20,
          "20 dressed per side, sides resolved from teamId via the static map")
    check(g1.dressed["homeTeam"][0] == 900001,
          "home side got the home teamId's players (orientation not swapped)")
    check(g1.scratches["awayTeam"] == [900150, 900151],
          "right-rail {'id': ...} scratch shape decoded")

    if verbose:
        print("features")
    rapm = _synthetic_rapm()
    usual = UsualStarter(min_obs=1)
    usual.hist["BOS"] = [900019, 900019, 900019]     # BOS usually starts ...019
    usual.hist["MTL"] = [900119, 900119, 900119]     # MTL usually starts ...119
    feats = build_features(snaps, rapm=rapm, usual=usual)

    f1 = feats[2026020001]
    exp = (0.20 - 0.05) - 0.40                       # away lost 0.15, home lost 0.40
    check(abs(f1["scratch_rapm_diff"] - exp) < 1e-9,
          f"scratch_rapm_diff {f1['scratch_rapm_diff']:+.3f} == {exp:+.3f} "
          f"(negative = home lost more, so home is weaker)")
    check(f1["scratch_rapm_diff"] < 0, "sign: scratching the better player hurts that side")
    check(abs(f1["scratch_rapm_pos_diff"] - (0.20 - 0.40)) < 1e-9,
          "positive-only variant ignores the below-average scratch")
    check(f1["scratch_n_diff"] == 1.0, "scratch count difference away-minus-home")
    check(f1["n_unrated"] == 0, "every synthetic scratch is rated")

    # 18 dressed skaters/side (20 minus 2 goalies) at 0.05, except BOS's 900001
    # at 0.30. The scratched players score nowhere here — they are not dressed.
    exp_pow = (17 * 0.05 + 0.30) - (18 * 0.05)
    check(abs(f1["dressed_power_diff"] - exp_pow) < 1e-9,
          f"dressed_power_diff {f1['dressed_power_diff']:+.3f} == {exp_pow:+.3f} "
          f"(positive = home dressed the stronger eighteen)")
    check(f1["d_count_diff"] == 0.0, "both sides dressed 6 D")
    check(abs(f1["dressed_rated_share"] - 1.0) < 1e-9, "all dressed skaters rated")

    check(f1["backup_start_diff"] == 1.0,
          "backup_start_diff +1: MTL started its non-usual goalie, BOS its usual")
    check(f1["goalie_source"] == "confirmed", "confirmation came from an explicit field")
    check(f1["coach_change_diff"] == 0.0, "first sighting of both coaches is not a change")
    check(f1["n_snapshots"] == 2, "both game-1 snapshots counted in diagnostics")

    f2 = feats[2026020002]
    check(math.isnan(f2["scratch_rapm_diff"]), "no scratch data -> NaN, not 0")
    check(math.isnan(f2["dressed_power_diff"]), "no roster -> NaN, not 0")
    check(math.isnan(f2["backup_start_diff"]),
          "goalie comparison widget (2 ids one side) is not a confirmation -> NaN")

    if verbose:
        print("orientation guard")
    bad = json.loads(json.dumps(json.loads(synthetic_archive_lines()[1])))
    bad["home"], bad["away"] = "CGY", "EDM"          # abbrevs that match no teamId here
    s2, _ = read_archive(_write_tmp(tmpdir, "bad.jsonl", [json.dumps(bad)]), names={})
    check(len(s2) == 1 and "dressed" not in s2[0].present,
          "unresolvable sides refuse to guess (dressed absent, features will be NaN)")

    if verbose:
        print("endpoint-merge helper")
    sub = lineup_subset_from_endpoints(
        landing={"matchup": {"goalieComparison": {
            "homeTeam": {"leaders": [{"playerId": 1}, {"playerId": 2}]}}}},
        right_rail={"gameInfo": {"homeTeam": {"scratches": [{"id": 7}],
                                              "headCoach": {"default": "C"}},
                                 "awayTeam": {"scratches": [], "headCoach": None}}},
        play_by_play={"rosterSpots": [{"teamId": 6, "playerId": 11,
                                       "positionCode": "C", "sweaterNumber": 1}]})
    check(sub.get("gameInfoScratches", {}).get("homeTeam") == [7],
          "right-rail scratches reach the subset")
    check(len(sub.get("rosterSpots", [])) == 1, "play-by-play rosterSpots reach the subset")
    check(sub.get("headCoach", {}).get("homeTeam") == "C", "head coach reaches the subset")

    if verbose:
        print("real-file smoke")
    real = load_rapm()
    check(isinstance(real, dict), "load_rapm returns a dict even if the file is missing")
    if real:
        check(all(isinstance(k, int) for k in list(real)[:50]),
              f"real RAPM file loaded, {len(real)} rated players, int keys")
    snaps0, rep0 = read_archive(ARCHIVE_PATH)
    check(rep0["records"] >= 0 and isinstance(build_features(snaps0, rapm=real), dict),
          f"real archive path runs (exists={rep0['exists']}, "
          f"records={rep0['records']})")

    if verbose:
        print(f"\nselftest: {'PASS' if not fails else 'FAIL'} "
              f"({len(fails)} failing check(s))")
        for f in fails:
            print("  - " + f)
    return 0 if not fails else 1


def _write_tmp(tmpdir, name, lines):
    p = os.path.join(tmpdir, name)
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return p


def report(path=ARCHIVE_PATH, write=True):
    """Inventory the real archive: what arrived, what is missing, what is new."""
    snaps, rep = read_archive(path)
    print(f"archive {path}: exists={rep['exists']} lines={rep['lines']} "
          f"records={rep['records']} games={rep['games']}")
    if not rep["exists"]:
        print("  nothing to inventory yet — the archiver only writes on game days.")
    if rep["bad_json"] or rep["not_object"] or rep["missing_gid"] or rep["no_lineup"]:
        print(f"  skipped: bad_json={rep['bad_json']} not_object={rep['not_object']} "
              f"missing_gid={rep['missing_gid']} no_lineup={rep['no_lineup']}")
    if rep["lineup_keys"]:
        print("  lineup keys seen: " + ", ".join(
            f"{k}x{v}" for k, v in rep["lineup_keys"].most_common()))
    if rep["unknown_lineup_keys"]:
        print("  UNKNOWN keys (the archive grew a field this module ignores): "
              + ", ".join(f"{k}x{v}" for k, v in rep["unknown_lineup_keys"].most_common()))
    if rep["field_counts"]:
        print("  normalised fields: " + ", ".join(
            f"{k}x{v}" for k, v in rep["field_counts"].most_common()))
    for note, n in rep["notes"].most_common(10):
        print(f"  note x{n}: {note}")

    rapm = load_rapm()
    usual = UsualStarter()
    feats = build_features(snaps, rapm=rapm, usual=usual)
    cov = {}
    for name in FEATURE_ORDER:
        vals = [f[name] for f in feats.values()
                if isinstance(f.get(name), float) and not math.isnan(f[name])]
        cov[name] = {"n": len(vals),
                     "coverage": round(len(vals) / len(feats), 4) if feats else 0.0,
                     "mean": round(sum(vals) / len(vals), 4) if vals else None}
        tag = " (PRIMARY)" if name == PRIMARY_FEATURE else ""
        print(f"  {name:<24}{tag:<11} n={cov[name]['n']:<6} "
              f"coverage={cov[name]['coverage']:.1%} mean={cov[name]['mean']}")
    out = {"records": rep["records"], "games": rep["games"],
           "bad_json": rep["bad_json"], "no_lineup": rep["no_lineup"],
           "lineup_keys": dict(rep["lineup_keys"]),
           "unknown_lineup_keys": dict(rep["unknown_lineup_keys"]),
           "fields": dict(rep["field_counts"]), "features": cov,
           "rapm_players": len(rapm), "primary": PRIMARY_FEATURE,
           "screened": False,
           "note": "readiness inventory only — no feature here has been screened"}
    if write:
        tmp = INVENTORY_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)
        os.replace(tmp, INVENTORY_PATH)
        print(f"wrote {INVENTORY_PATH}")
    return out


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "--report"
    if mode == "--selftest":
        return selftest()
    if mode == "--report":
        report()
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
