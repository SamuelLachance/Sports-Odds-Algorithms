"""NHL shift charts from the official HTML time-on-ice reports (repair + fallback).

WHY. The shift-chart API (api.nhle.com/stats/rest/en/shiftcharts) returned no rows
for 582 spine games when data/nhl_shifts.csv was pulled (57 in April 2024-25, 525 in
2025-26; list: data/pv_nhl_full_noshift_games.csv). Those games' player minutes feed
every per-60 rate. The NHL also publishes, per game, two HTML time-on-ice reports
(the official scoresheet family):

    https://www.nhl.com/scores/htmlreports/{season}/TH{gg}.HTM   home team
    https://www.nhl.com/scores/htmlreports/{season}/TV{gg}.HTM   visiting team

(gg = last 6 digits of the game id). Each lists, per dressed player (sweater number +
name), every shift with period, start and end (elapsed / remaining) and duration,
followed by a per-period summary (shifts, TOI). This module turns them into rows of
EXACTLY the data/nhl_shifts.csv schema

    game_id, player_id, team, period, start_s, end_s

(start_s/end_s = seconds elapsed in the period, as phase0/nhl_pull_shifts.py writes).

MAPPING. Report players carry a sweater number, not an NHL id; the id comes from the
game's play-by-play rosterSpots (teamId, sweaterNumber, playerId), read from the
cached data/pv_nhl_pbp/{season}/{gid}.json.gz or fetched from api-web.nhle.com when
the cache has no file. Team codes are the pbp homeTeam/awayTeam abbrevs (the same
NHL API family as the shift API's teamAbbrev). Period labels: 1..N as printed;
regular-season overtime is printed "OT" -> 4; playoff overtimes are printed 4,5,6...
A shootout ("SO") is never a shift.

API PARITY -- GOAL ROWS. The shift API also returns one zero-length row per goal
(typeCode 505: playerId = scorer, startTime = endTime = goal time, duration null),
and phase0/nhl_pull_shifts.py stores them unfiltered, so data/nhl_shifts.csv holds
one start_s == end_s row per goal. The HTML report has no such row; it
is re-created from the pbp goal event (scorer, scoring team, period, timeInPeriod;
shootout goals too, as period 5 at 00:00, exactly as the API stores them) so a
backfilled game is row-for-row what the API would have delivered. Consumers
either clip these to zero minutes (pv_nhl_events toi_s), drop them (end_s > start_s
filters) or never cover a half-open [s, s) instant, so they carry no minutes.

VALIDATION (``validate``). On a seeded random sample of games that have BOTH sources
(DEV 2010-11..2017-18, mid 2018-19..2023-24, and recent 2024-25/2025-26 non-gap
games) every player-game's time on ice and shift count from the HTML is compared to
the stored API rows, and the whole row multiset is compared. ``recheck-api``
re-queries the shift API for the gap games themselves (the API has since been
repopulated for some) and compares those to the HTML too. Results:
data/nhl_shifts_html_validation.json.

BACKFILL (``backfill``). The gap games are parsed into a SEPARATE file,
data/nhl_shifts_html_backfill.csv (same header as data/nhl_shifts.csv), with a per-
game QA ledger data/nhl_shifts_html_backfill_games.csv. Only games that pass every
QA check are written. data/nhl_shifts.csv is NOT touched.

MERGE PATH (``merge``; dry-run unless --apply). Deliberately separate, because the
pre-registered player-value pipeline treats gap games as "shift chart unavailable"
(phase0/pv_nhl_full_common.noshift_games, cached list above) and the NHL
forward-test pre-registration (data/pv_nhl_forward_prereg_2026_27.json, data_rule)
allows repairing the shift source BEFORE evaluation, as a deliberate step:

    python phase0/nhl_shifts_html.py merge            # dry run: what would change
    python phase0/nhl_shifts_html.py merge --apply    # rewrite data/nhl_shifts.csv

--apply streams data/nhl_shifts.csv into a temp file, inserting each backfilled
game's rows in spine (chronological) order, skips any game that already has rows
there (e.g. re-pulled from the API since), verifies the row count, atomically
replaces the file, and appends the merged ids to data/nhl_shifts_html_merged.csv.
Afterwards the shift-derived caches must be rebuilt before anything reads them:
phase0/pv_nhl_events.py (roster toi_s), then
pv_nhl_full_common.noshift_games(rebuild=True) (its assertion "roster toi_s blank
== no shift rows" must hold again), then the consumers (nhl_rapm2.py, ...). The
TEST-era numbers already in the ledgers were computed WITH the gaps and are not
re-scored. The merge touches only data/nhl_shifts.csv (not hash-pinned); it must
NOT be followed by regenerating any file pinned in frozen_code_sha256 /
frozen_params_sha256 of data/pv_nhl_forward_prereg_2026_27.json (e.g.
data/pv_nhl_fix_full_compose_weights.json) -- that voids the forward test. When
to merge is the owner's call; nothing in this module merges on its own.
NOTE: the shift API has since been repopulated for many gap games (``recheck-api``
counts them), and phase0/nhl_pull_shifts.py retries API-empty games on every run,
so its next run APPENDS those games from the API regardless of this module; a
later merge then skips them as already present.

FALLBACK. phase0/nhl_pull_shifts.py calls ``fallback_rows`` for a game whose API
response is empty (seasons >= FALLBACK_MIN_SEASON by default): <= 2 requests/s,
resumable (a filled game is in data/nhl_shifts.csv and is not revisited; a report
that is not yet posted / not yet final is simply retried on the next run).

Data repair only: no model feature, rating or metric is computed here.
Market-blind: no odds are read.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import html as htmllib
import json
import os
import random
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def D(*p):
    return os.path.join(ROOT, "data", *p)


REPORT_URL = "https://www.nhl.com/scores/htmlreports/{season}/{kind}{gg}.HTM"
SHIFT_API = "https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId={}"
PBP_API = "https://api-web.nhle.com/v1/gamecenter/{}/play-by-play"
HEADERS = {"User-Agent": "glassbox-nhl/1.0 (research)", "Accept-Encoding": "gzip"}
SCHEMA = ["game_id", "player_id", "team", "period", "start_s", "end_s"]

SHIFTS_CSV = D("nhl_shifts.csv")
GAMES_CSV = D("nhl_games.csv")
NOSHIFT_CSV = D("pv_nhl_full_noshift_games.csv")
PBP_DIR = D("pv_nhl_pbp")
CACHE_DIR = D("nhl_shifts_html_cache")            # {season}/T{H,V}{gg}.HTM.gz (final only)
BACKFILL_CSV = D("nhl_shifts_html_backfill.csv")
BACKFILL_GAMES = D("nhl_shifts_html_backfill_games.csv")
VALIDATION_JSON = D("nhl_shifts_html_validation.json")
VAL_API_CSV = D("nhl_shifts_html_validation_api.csv")   # API rows of the sample
GAP_API_CSV = D("nhl_shifts_html_gap_api_recheck.csv")  # fresh API rows, gap games
MERGED_CSV = D("nhl_shifts_html_merged.csv")
FILLED_CSV = D("nhl_shifts_html_filled.csv")            # fallback provenance
PENDING_JSON = D("nhl_shifts_html_pending.json")        # fallback: not-yet-available

MIN_INTERVAL = 0.5          # seconds between requests -> <= 2 req/s
FALLBACK_MIN_SEASON = 20262027   # puller fallback: future games only (see docstring)
DEV_SEASONS = list(range(2010, 2018))
MID_SEASONS = list(range(2018, 2024))
RECENT_SEASONS = [2024, 2025]


# ----------------------------------------------------------------------------- utils

def season_of(gid):
    y = int(str(gid)[:4])
    return y * 10000 + y + 1


def report_url(gid, kind):
    """kind 'TH' (home) or 'TV' (visitor)."""
    assert kind in ("TH", "TV"), kind
    return REPORT_URL.format(season=season_of(gid), kind=kind, gg=str(gid)[-6:])


def mmss(s):
    """'12:34' -> 754; None when unparseable."""
    try:
        m, sec = str(s).strip().split(":")
        return int(m) * 60 + int(sec)
    except (ValueError, AttributeError):
        return None


def norm_name(s):
    s = unicodedata.normalize("NFKD", str(s or ""))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^A-Z]", "", s.upper())


def utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Throttle:
    """Minimum spacing between consecutive requests (shared by every fetch)."""

    def __init__(self, min_interval=MIN_INTERVAL, clock=time.monotonic, sleep=time.sleep):
        self.min_interval = float(min_interval)
        self.clock, self.sleep = clock, sleep
        self._last = None

    def wait(self):
        if self._last is not None:
            dt = self.clock() - self._last
            if dt < self.min_interval:
                self.sleep(self.min_interval - dt)
        self._last = self.clock()


def http_get(url, throttle, timeout=45, retries=3):
    """-> (status, bytes|None). 404 is final (report not posted); transient errors
    are retried with back-off; status 'error' after the last attempt."""
    last = "error"
    for attempt in range(retries + 1):
        throttle.wait()
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read()
            if raw[:2] == b"\x1f\x8b":
                raw = gzip.decompress(raw)
            return 200, raw
        except urllib.error.HTTPError as ex:
            if ex.code in (403, 404, 410):
                return ex.code, None
            last = f"http{ex.code}"
        except Exception as ex:  # noqa: BLE001 - network flakiness of every kind
            last = f"error:{type(ex).__name__}"
        if attempt < retries:
            time.sleep(2 * (attempt + 1))
    return last, None


# ----------------------------------------------------------------------------- parse

_TD = re.compile(r"<td\b[^>]*>(.*?)</td>", re.S | re.I)
_TR = re.compile(r"<tr\b[^>]*>(.*?)</tr>", re.S | re.I)
_PH = re.compile(r"<td\b[^>]*class=\"playerHeading[^\"]*\"[^>]*>(.*?)</td>", re.S | re.I)
_TAG = re.compile(r"<[^>]+>")
_SCRIPT = re.compile(r"<(script|style)\b.*?</\1>", re.S | re.I)


def _cell(x):
    return re.sub(r"\s+", " ", htmllib.unescape(_TAG.sub(" ", x)).replace("\xa0", " ")).strip()


def decode_report(raw):
    """Reports are UTF-8, except older bilingual ones, which are Latin-1/cp1252."""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace")


def period_label(lbl):
    """'1'..'N' -> int, 'OT' -> 4; 'SO' / anything else -> None (not a shift)."""
    lbl = str(lbl).strip().upper()
    if lbl.isdigit():
        return int(lbl)
    if lbl == "OT":
        return 4
    return None


def parse_report(text):
    """Parse one TH/TV time-on-ice report.

    -> {'title', 'status', 'final', 'team_name', 'players': [
          {'num', 'name', 'shifts': [(period, start_s, end_s, dur_s, event)],
           'summary': {period: (shf, toi_s)}, 'tot': (shf, toi_s) | None,
           'bad_rows': [...unparseable shift rows...]}]}"""
    if isinstance(text, bytes):
        text = decode_report(text)
    text = _SCRIPT.sub(" ", text)
    m = re.search(r"<title>(.*?)</title>", text, re.S | re.I)
    title = _cell(m.group(1)) if m else ""
    heads = list(_PH.finditer(text))
    head_end = heads[0].start() if heads else len(text)
    hcells = [_cell(c) for c in _TD.findall(text[:head_end])]
    status = ""
    for i, c in enumerate(hcells):
        # "Game 1000"; bilingual reports (Canadian arenas, older seasons) print
        # "Match/Game 0570"
        if re.fullmatch(r"(?:Match/)?Game \d+", c) and i + 1 < len(hcells):
            status = hcells[i + 1]
            break
    m = re.search(r"class=\"teamHeading[^\"]*\"[^>]*>(.*?)</td>", text[:head_end], re.S | re.I)
    team_name = _cell(m.group(1)) if m else ""
    players = []
    for k, h in enumerate(heads):
        seg = text[h.end(): heads[k + 1].start() if k + 1 < len(heads) else len(text)]
        label = _cell(h.group(1))
        mm = re.match(r"(\d+)\s+(.*)$", label)
        num, name = (int(mm.group(1)), mm.group(2)) if mm else (None, label)
        shifts, summary, tot, bad = [], {}, None, []
        for tr in _TR.findall(seg):
            cells = [_cell(c) for c in _TD.findall(tr)]
            if len(cells) == 6 and cells[0].isdigit():
                per = period_label(cells[1])
                st = mmss(cells[2].split("/")[0])
                en = mmss(cells[3].split("/")[0])
                du = mmss(cells[4])
                if per is None or st is None or en is None:
                    if str(cells[1]).strip().upper() != "SO":
                        bad.append(cells)
                    continue
                shifts.append((per, st, en, du, cells[5]))
            elif len(cells) == 7 and cells[0] != "Per":
                shf = int(cells[1]) if cells[1].isdigit() else None
                toi = mmss(cells[3])
                if cells[0].upper() == "TOT":
                    tot = (shf, toi)
                else:
                    per = period_label(cells[0])
                    if per is not None:
                        summary[per] = (shf, toi)
        players.append({"num": num, "name": name, "shifts": shifts, "summary": summary,
                        "tot": tot, "bad_rows": bad})
    return {"title": title, "status": status,
            "final": status.strip().lower().startswith("final"),
            "team_name": team_name, "players": players}


def report_qa(rep):
    """Internal consistency of one parsed report -> dict of counts (0 = clean)."""
    q = Counter()
    for p in rep["players"]:
        q["bad_rows"] += len(p["bad_rows"])
        per_toi, per_n = Counter(), Counter()
        for (per, st, en, du, _e) in p["shifts"]:
            if en < st:
                q["neg_shift"] += 1
            if du is not None and en - st != du:
                q["dur_mismatch"] += 1
            per_toi[per] += max(en - st, 0)
            per_n[per] += 1
        for per, (shf, toi) in p["summary"].items():
            if shf is not None and shf != per_n.get(per, 0):
                q["summary_shf_mismatch"] += 1
            if toi is not None and toi != per_toi.get(per, 0):
                q["summary_toi_mismatch"] += 1
        if p["tot"] is not None:
            shf, toi = p["tot"]
            if shf is not None and shf != sum(per_n.values()):
                q["tot_shf_mismatch"] += 1
            if toi is not None and toi != sum(per_toi.values()):
                q["tot_toi_mismatch"] += 1
    return {k: v for k, v in q.items() if v}


# ----------------------------------------------------------------------------- pbp

def load_pbp(gid, throttle=None, fetch=True):
    """Cached pv_nhl_pbp JSON, else (fetch=True) the api-web play-by-play."""
    p = os.path.join(PBP_DIR, str(season_of(gid)), f"{gid}.json.gz")
    if os.path.exists(p):
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            return json.load(fh)
    if not fetch:
        return None
    st, raw = http_get(PBP_API.format(gid), throttle or Throttle())
    if st != 200:
        return None
    return json.loads(raw)


def roster_from_pbp(pbp):
    """-> {'H': {'abbrev', 'id', 'num': {n: (pid, last)}, 'name': {normlast: [pid]}}, 'A': ...}"""
    out = {}
    for side, key in (("H", "homeTeam"), ("A", "awayTeam")):
        t = pbp[key]
        out[side] = {"abbrev": t.get("abbrev", ""), "id": t.get("id"), "num": {}, "name": defaultdict(list)}
    by_id = {out[s]["id"]: s for s in out}
    for r in pbp.get("rosterSpots", []):
        s = by_id.get(r.get("teamId"))
        if s is None or r.get("playerId") is None:
            continue
        last = (r.get("lastName") or {}).get("default", "")
        if r.get("sweaterNumber") is not None:
            out[s]["num"][int(r["sweaterNumber"])] = (int(r["playerId"]), last)
        out[s]["name"][norm_name(last)].append(int(r["playerId"]))
    return out


def goal_rows(gid, pbp, ros):
    """The shift API's zero-length goal rows, re-created from pbp goal events.

    The API writes one per goal INCLUDING shootout goals (period 5 in the regular
    season, time 00:00 -- observed in every shootout game of the validation
    sample), so shootout goals are re-created as well."""
    side_of = {ros[s]["id"]: s for s in ros}
    rows = []
    for p in pbp.get("plays", []):
        if p.get("typeDescKey") != "goal":
            continue
        pd_ = p.get("periodDescriptor", {}) or {}
        det = p.get("details", {}) or {}
        pid, t = det.get("scoringPlayerId"), mmss(p.get("timeInPeriod"))
        side = side_of.get(det.get("eventOwnerTeamId"))
        if pid is None or t is None or side is None or pd_.get("number") is None:
            continue
        rows.append([int(gid), int(pid), ros[side]["abbrev"], int(pd_["number"]), t, t])
    return rows


# ----------------------------------------------------------------------------- game

def build_game(gid, home_html, away_html, pbp, with_goal_rows=True):
    """Parse both reports of one game into data/nhl_shifts.csv rows.

    -> (rows, qa). qa['ok'] is False on any hard failure: a report not final, a
    player with shifts that cannot be mapped to an NHL id, no shifts, or a
    home/away mix-up; soft counts (summary mismatches etc.) are reported."""
    qa = {"gid": int(gid), "ok": True, "reasons": []}
    ros = roster_from_pbp(pbp)
    rows = []
    for side, kind, txt in (("H", "TH", home_html), ("A", "TV", away_html)):
        rep = parse_report(txt)
        want = "Home" if side == "H" else "Away"
        if want.lower() not in rep["title"].lower():
            qa["ok"] = False
            qa["reasons"].append(f"{kind}:title:{rep['title']!r}")
        if not rep["final"]:
            qa["ok"] = False
            qa["reasons"].append(f"{kind}:not_final:{rep['status']!r}")
        for k, v in report_qa(rep).items():
            if v:
                qa[f"{kind}_{k}"] = v
        R = ros[side]
        n_players = 0
        for pl in rep["players"]:
            if not pl["shifts"]:
                continue
            n_players += 1
            hit = R["num"].get(pl["num"])
            last = pl["name"].split(",")[0]
            if hit is not None:
                pid, rlast = hit
                a, b = norm_name(last), norm_name(rlast)
                if a and b and not (a in b or b in a):
                    qa[f"{kind}_name_warn"] = qa.get(f"{kind}_name_warn", 0) + 1
            else:
                cands = R["name"].get(norm_name(last), [])
                if len(cands) == 1:
                    pid = cands[0]
                    qa[f"{kind}_mapped_by_name"] = qa.get(f"{kind}_mapped_by_name", 0) + 1
                else:
                    qa["ok"] = False
                    qa["reasons"].append(f"{kind}:unmapped:{pl['num']} {pl['name']}")
                    continue
            for (per, st, en, _du, _ev) in pl["shifts"]:
                rows.append([int(gid), pid, R["abbrev"], per, st, en])
        qa[f"{kind}_players"] = n_players
        if n_players == 0:
            qa["ok"] = False
            qa["reasons"].append(f"{kind}:no_players")
    if with_goal_rows:
        g = goal_rows(gid, pbp, ros)
        qa["goal_rows"] = len(g)
        rows.extend(g)
    rows.sort(key=lambda r: (r[1], r[4] != r[5], r[3], r[4], r[5]))
    qa["rows"] = len(rows)
    return rows, qa


# ----------------------------------------------------------------------------- cache

def cache_path(gid, kind):
    return os.path.join(CACHE_DIR, str(season_of(gid)), f"{kind}{str(gid)[-6:]}.HTM.gz")


def get_report(gid, kind, throttle, use_cache=True, write_cache=True):
    """-> (status, text|None). Status 200 / 404 (not posted) / 'error...'. Only a
    report whose header says Final is cached (a live report keeps changing)."""
    p = cache_path(gid, kind)
    if use_cache and os.path.exists(p):
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            return 200, fh.read()
    st, raw = http_get(report_url(gid, kind), throttle)
    if st != 200:
        return st, None
    text = decode_report(raw)
    if write_cache and parse_report(text)["final"]:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = p + ".tmp"
        with gzip.open(tmp, "wt", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, p)
    return 200, text


def fallback_rows(gid, throttle, write_cache=True):
    """Puller fallback for ONE game whose shift API response is empty.

    -> (rows, status, qa). status 'ok' (rows ready to append), 'not_posted' (a report
    404s: not yet published), 'not_final' (live report), 'no_pbp', 'qa_fail',
    or 'error:<why>' — every non-ok status means: write nothing, retry next run."""
    try:
        texts = {}
        for kind in ("TH", "TV"):
            st, txt = get_report(gid, kind, throttle, write_cache=write_cache)
            if st in (403, 404, 410):
                return [], "not_posted", None
            if st != 200:
                return [], f"error:{st}", None
            texts[kind] = txt
        if not (parse_report(texts["TH"])["final"] and parse_report(texts["TV"])["final"]):
            return [], "not_final", None
        pbp = load_pbp(gid, throttle)
        if pbp is None:
            return [], "no_pbp", None
        rows, qa = build_game(gid, texts["TH"], texts["TV"], pbp)
        if not qa["ok"] or not rows:
            return [], "qa_fail", qa
        return rows, "ok", qa
    except Exception as ex:  # noqa: BLE001 - the puller must never die on one game
        return [], f"error:{type(ex).__name__}:{ex}"[:200], None


# ----------------------------------------------------------------------------- compare

def player_game_stats(rows):
    """rows -> {(gid, pid): {'toi', 'n', 'goal_rows', 'team'}} counting real shifts
    (end > start) for n; toi = sum of clip(end - start, 0) after exact-duplicate
    removal (the pv_nhl_events.load_toi definition); raw sums are kept too."""
    seen = set()
    out = defaultdict(lambda: {"toi": 0, "toi_raw": 0, "n": 0, "n_raw": 0, "goal_rows": 0,
                               "dups": 0, "team": set()})
    for r in rows:
        g, p, t, per, s0, s1 = int(r[0]), int(r[1]), r[2], int(r[3]), int(r[4]), int(r[5])
        d = out[(g, p)]
        d["team"].add(t)
        d["toi_raw"] += s1 - s0
        key = (g, p, per, s0, s1)
        if s1 > s0:
            d["n_raw"] += 1
        if key in seen:
            d["dups"] += 1
            continue
        seen.add(key)
        if s1 == s0:
            d["goal_rows"] += 1
        elif s1 > s0:
            d["n"] += 1
        d["toi"] += max(s1 - s0, 0)
    return out


def compare_game(api_rows, html_rows):
    """Per player-game agreement of one game's two row sets."""
    A, H = player_game_stats(api_rows), player_game_stats(html_rows)
    keys = set(A) | set(H)
    res = {"player_games": len(keys), "only_api": 0, "only_html": 0, "toi_exact": 0,
           "n_exact": 0, "both_exact": 0, "team_mismatch": 0, "goal_rows_exact": 0,
           "abs_dtoi": [], "api_dups": sum(v["dups"] for v in A.values())}
    only_api_toi = 0
    for k in keys:
        a, h = A.get(k), H.get(k)
        if a is None:
            res["only_html"] += 1
            res["abs_dtoi"].append(h["toi"])
            continue
        if h is None:
            res["only_api"] += 1
            only_api_toi += a["toi"]
            res["abs_dtoi"].append(a["toi"])
            continue
        te, ne = a["toi"] == h["toi"], a["n"] == h["n"]
        res["toi_exact"] += te
        res["n_exact"] += ne
        res["both_exact"] += te and ne
        res["goal_rows_exact"] += a["goal_rows"] == h["goal_rows"]
        res["team_mismatch"] += a["team"] != h["team"]
        res["abs_dtoi"].append(abs(a["toi"] - h["toi"]))
    res["only_api_toi"] = only_api_toi
    ms = lambda rs: Counter((int(r[1]), r[2], int(r[3]), int(r[4]), int(r[5])) for r in rs)  # noqa: E731
    ma, mh = ms(api_rows), ms(html_rows)
    res["rows_api"], res["rows_html"] = sum(ma.values()), sum(mh.values())
    res["rows_common"] = sum((ma & mh).values())
    res["rows_identical"] = ma == mh
    ua = Counter({k: v for k, v in ma.items() if k[3] != k[4]})
    uh = Counter({k: v for k, v in mh.items() if k[3] != k[4]})
    res["shift_rows_identical_dedup"] = set(ua) == set(uh)
    return res


def summarize(per_game):
    tot = Counter()
    dt = []
    for r in per_game:
        for k in ("player_games", "only_api", "only_html", "toi_exact", "n_exact",
                  "both_exact", "team_mismatch", "goal_rows_exact", "rows_api",
                  "rows_html", "rows_common", "api_dups"):
            tot[k] += r[k]
        tot["games_rows_identical"] += bool(r["rows_identical"])
        tot["games_shift_rows_identical_dedup"] += bool(r["shift_rows_identical_dedup"])
        tot["games_all_players_exact"] += r["both_exact"] == r["player_games"]
        dt.extend(r["abs_dtoi"])
    n = max(tot["player_games"], 1)
    dt.sort()
    q = lambda f: dt[min(len(dt) - 1, int(f * len(dt)))] if dt else None  # noqa: E731
    return {"games": len(per_game), **dict(tot),
            "share_player_games_toi_exact": round(tot["toi_exact"] / n, 5),
            "share_player_games_shifts_exact": round(tot["n_exact"] / n, 5),
            "share_player_games_both_exact": round(tot["both_exact"] / n, 5),
            "share_games_all_players_exact": round(tot["games_all_players_exact"] / max(len(per_game), 1), 5),
            "abs_dtoi_s": {"mean": round(sum(dt) / len(dt), 3) if dt else None,
                           "p50": q(0.5), "p99": q(0.99), "max": dt[-1] if dt else None,
                           "n_gt_0": sum(1 for x in dt if x > 0),
                           "n_gt_60": sum(1 for x in dt if x > 60)}}


# ----------------------------------------------------------------------------- IO

def read_spine():
    with open(GAMES_CSV, encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh)]


def gap_games():
    with open(NOSHIFT_CSV, encoding="utf-8") as fh:
        return [int(r["gid"]) for r in csv.DictReader(fh)]


def extract_api_rows(gids, path=SHIFTS_CSV):
    """Stream data/nhl_shifts.csv once, returning {gid: [rows]} for gids."""
    want = {str(g) for g in gids}
    out = defaultdict(list)
    with open(path, encoding="utf-8") as fh:
        rd = csv.reader(fh)
        next(rd)
        for r in rd:
            if r and r[0] in want:
                out[int(r[0])].append([int(r[0]), int(r[1]), r[2], int(r[3]), int(r[4]), int(r[5])])
    return out


def api_rows_now(gid, throttle):
    """Fresh shift-API rows for one game in the nhl_pull_shifts.py row format."""
    st, raw = http_get(SHIFT_API.format(gid), throttle)
    if st != 200:
        return None
    rows = []
    for s in json.loads(raw).get("data", []):
        a, b = mmss(s.get("startTime")), mmss(s.get("endTime"))
        if a is None or b is None or s.get("playerId") is None:
            continue
        rows.append([int(gid), int(s["playerId"]), s.get("teamAbbrev", ""),
                     int(s.get("period", 1)), a, b])
    return rows


def write_csv(path, header, rows):
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    os.replace(tmp, path)


def html_for(gid, throttle):
    out = {}
    for kind in ("TH", "TV"):
        st, txt = get_report(gid, kind, throttle)
        if st != 200:
            return None, f"{kind}:{st}"
        out[kind] = txt
    return out, "ok"


# ----------------------------------------------------------------------------- commands

def sample_games(seed, n_dev, n_mid, n_recent):
    spine = read_spine()
    gaps = set(gap_games())
    pool = defaultdict(list)
    for r in spine:
        g = int(r["game_id"])
        if g in gaps:
            continue
        y = int(r["game_id"][:4])
        grp = "dev" if y in DEV_SEASONS else "mid" if y in MID_SEASONS else \
            "recent" if y in RECENT_SEASONS else None
        if grp:
            pool[(grp, y)].append(g)
    rng = random.Random(seed)
    out = {}
    for grp, n, years in (("dev", n_dev, DEV_SEASONS), ("mid", n_mid, MID_SEASONS),
                          ("recent", n_recent, RECENT_SEASONS)):
        per = [n // len(years) + (1 if i < n % len(years) else 0) for i in range(len(years))]
        for y, k in zip(years, per):
            for g in rng.sample(sorted(pool[(grp, y)]), k):
                out[g] = grp
    return out


def cmd_validate(a):
    thr = Throttle()
    samp = sample_games(a.seed, a.n_dev, a.n_mid, a.n_recent)
    gids = sorted(samp)
    if os.path.exists(VAL_API_CSV):
        api = defaultdict(list)
        with open(VAL_API_CSV, encoding="utf-8") as fh:
            rd = csv.reader(fh)
            next(rd)
            for r in rd:
                api[int(r[0])].append([int(r[0]), int(r[1]), r[2], int(r[3]), int(r[4]), int(r[5])])
        if not set(gids) <= set(api):
            api = None
    else:
        api = None
    if api is None:
        print(f"extracting API rows of {len(gids)} sample games from nhl_shifts.csv ...", flush=True)
        api = extract_api_rows(gids)
        write_csv(VAL_API_CSV, SCHEMA, [r for g in gids for r in api.get(g, [])])
    per_game, fails, qa_soft = [], [], Counter()
    for k, g in enumerate(gids):
        if not api.get(g):
            fails.append({"gid": g, "why": "no_api_rows"})
            continue
        texts, why = html_for(g, thr)
        if texts is None:
            fails.append({"gid": g, "why": why})
            continue
        pbp = load_pbp(g, thr)
        rows, qa = build_game(g, texts["TH"], texts["TV"], pbp)
        for kk, v in qa.items():
            if kk.startswith(("TH_", "TV_")) and not kk.endswith("_players") and v:
                qa_soft[kk[3:]] += v
        c = compare_game(api[g], rows)
        c.update(gid=g, group=samp[g], season=season_of(g), gtype=int(str(g)[4:6]),
                 qa_ok=qa["ok"], qa_reasons=qa["reasons"],
                 qa_soft={kk: v for kk, v in qa.items()
                          if kk.startswith(("TH_", "TV_")) and not kk.endswith("_players")})
        per_game.append(c)
        if (k + 1) % 25 == 0:
            print(f"  {k+1}/{len(gids)}", flush=True)
    out = {"written": utcnow(), "seed": a.seed,
           "definition": ("per player-game: toi = sum(clip(end_s-start_s,0)) after exact-"
                          "duplicate removal (pv_nhl_events.load_toi); shifts = rows with "
                          "end_s > start_s; goal rows = start_s == end_s. API = stored rows "
                          "of data/nhl_shifts.csv; HTML = this parser + pbp goal rows."),
           "sample": {grp: sum(1 for v in samp.values() if v == grp) for grp in ("dev", "mid", "recent")},
           "fetch_failures": fails,
           "html_internal_qa_soft_counts": dict(qa_soft),
           "all": summarize(per_game),
           "by_group": {grp: summarize([r for r in per_game if r["group"] == grp])
                        for grp in ("dev", "mid", "recent")},
           "games_not_all_exact": [
               {k: v for k, v in r.items() if k != "abs_dtoi"} | {"max_abs_dtoi": max(r["abs_dtoi"] or [0])}
               for r in per_game if r["both_exact"] != r["player_games"]],
           "games_with_soft_qa_flags": [
               {"gid": r["gid"], "qa_soft": r["qa_soft"],
                "all_players_exact_vs_api": r["both_exact"] == r["player_games"]}
               for r in per_game if r["qa_soft"]]}
    prev = {}
    if os.path.exists(VALIDATION_JSON):
        with open(VALIDATION_JSON, encoding="utf-8") as fh:
            prev = json.load(fh)
    prev["sample_validation"] = out
    with open(VALIDATION_JSON, "w", encoding="utf-8") as fh:
        json.dump(prev, fh, indent=1)
    s = out["all"]
    print(json.dumps({k: s[k] for k in ("games", "player_games", "share_player_games_toi_exact",
                                         "share_player_games_shifts_exact",
                                         "share_games_all_players_exact", "abs_dtoi_s")}, indent=1))


def cmd_recheck_api(a):
    """Fresh shift-API pull for the gap games (cached to GAP_API_CSV, resumable),
    compared to the HTML parse where the API now has rows."""
    thr = Throttle()
    gids = gap_games()
    have, empty = defaultdict(list), set()
    if os.path.exists(GAP_API_CSV):
        with open(GAP_API_CSV, encoding="utf-8") as fh:
            rd = csv.reader(fh)
            next(rd)
            for r in rd:
                have[int(r[0])].append([int(r[0]), int(r[1]), r[2], int(r[3]), int(r[4]), int(r[5])])
    empty_p = GAP_API_CSV + ".empty.json"
    if os.path.exists(empty_p):
        with open(empty_p, encoding="utf-8") as fh:
            empty = set(json.load(fh))
    todo = [g for g in gids if g not in have and g not in empty]
    print(f"{len(gids)} gap games | {len(have)} with API rows cached | {len(empty)} empty | "
          f"{len(todo)} to query", flush=True)
    for k, g in enumerate(todo):
        rows = api_rows_now(g, thr)
        if rows is None:
            continue
        if rows:
            have[g] = rows
        else:
            empty.add(g)
        if (k + 1) % 50 == 0 or k + 1 == len(todo):
            write_csv(GAP_API_CSV, SCHEMA, [r for gg in sorted(have) for r in have[gg]])
            with open(empty_p, "w", encoding="utf-8") as fh:
                json.dump(sorted(empty), fh)
            print(f"  {k+1}/{len(todo)} (api rows now: {len(have)})", flush=True)
    per_game = []
    for g in sorted(have):
        texts, why = html_for(g, thr)
        if texts is None:
            continue
        rows, qa = build_game(g, texts["TH"], texts["TV"], load_pbp(g, thr))
        c = compare_game(have[g], rows)
        c.update(gid=g, season=season_of(g), qa_ok=qa["ok"])
        per_game.append(c)
    out = {"written": utcnow(), "gap_games": len(gids),
           "api_now_has_rows": len(have), "api_still_empty": len(empty),
           "api_still_empty_by_season": dict(Counter(season_of(g) for g in empty)),
           "compare": summarize(per_game),
           "games_not_all_exact": [
               {k: v for k, v in r.items() if k != "abs_dtoi"} | {"max_abs_dtoi": max(r["abs_dtoi"] or [0])}
               for r in per_game if r["both_exact"] != r["player_games"]]}
    prev = {}
    if os.path.exists(VALIDATION_JSON):
        with open(VALIDATION_JSON, encoding="utf-8") as fh:
            prev = json.load(fh)
    prev["gap_games_api_recheck"] = out
    with open(VALIDATION_JSON, "w", encoding="utf-8") as fh:
        json.dump(prev, fh, indent=1)
    print(json.dumps({k: out[k] for k in ("gap_games", "api_now_has_rows", "api_still_empty")}))
    s = out["compare"]
    print(json.dumps({k: s.get(k) for k in ("games", "player_games", "share_player_games_both_exact",
                                             "share_games_all_players_exact", "abs_dtoi_s")}))


def cmd_backfill(a):
    """Fetch (resumable via the raw-report cache) then parse every gap game;
    rewrite BACKFILL_CSV + BACKFILL_GAMES from scratch (pure function of the cache)."""
    thr = Throttle()
    gids = gap_games() if not a.gids else [int(x) for x in a.gids.split(",")]
    rows_all, ledger = [], []
    for k, g in enumerate(gids):
        if (k + 1) % 50 == 0:
            print(f"  {k+1}/{len(gids)}", flush=True)
        texts, why = html_for(g, thr)
        rec = {"gid": g, "season": season_of(g), "status": "", "rows": 0, "shift_rows": 0,
               "goal_rows": 0, "players_home": 0, "players_away": 0, "soft_qa": "",
               "reasons": ""}
        if texts is None:
            rec["status"] = f"fetch:{why}"
            ledger.append(rec)
            continue
        pbp = load_pbp(g, thr)
        if pbp is None:
            rec["status"] = "no_pbp"
            ledger.append(rec)
            continue
        rows, qa = build_game(g, texts["TH"], texts["TV"], pbp)
        soft = {k: v for k, v in qa.items() if k.startswith(("TH_", "TV_")) and not k.endswith("_players")}
        rec.update(status="ok" if qa["ok"] else "qa_fail", rows=len(rows),
                   shift_rows=sum(1 for r in rows if r[5] > r[4]), goal_rows=qa.get("goal_rows", 0),
                   players_home=qa.get("TH_players", 0), players_away=qa.get("TV_players", 0),
                   soft_qa=json.dumps(soft, sort_keys=True) if soft else "",
                   reasons="; ".join(qa["reasons"]))
        ledger.append(rec)
        if qa["ok"]:
            rows_all.extend(rows)
    write_csv(BACKFILL_CSV, SCHEMA, rows_all)
    cols = list(ledger[0].keys()) if ledger else ["gid"]
    write_csv(BACKFILL_GAMES, cols, [[r[c] for c in cols] for r in ledger])
    c = Counter(r["status"] for r in ledger)
    prev = {}
    if os.path.exists(VALIDATION_JSON):
        with open(VALIDATION_JSON, encoding="utf-8") as fh:
            prev = json.load(fh)
    prev["backfill"] = {
        "written": utcnow(), "file": "data/nhl_shifts_html_backfill.csv",
        "ledger": "data/nhl_shifts_html_backfill_games.csv",
        "games": len(gids), "status": dict(c), "rows": len(rows_all),
        "shift_rows": sum(1 for r in rows_all if r[5] > r[4]),
        "goal_rows": sum(1 for r in rows_all if r[5] == r[4]),
        "by_season": {str(s): sum(1 for r in ledger if r["season"] == s and r["status"] == "ok")
                      for s in sorted({r["season"] for r in ledger})},
        "games_with_soft_qa_flags": [r["gid"] for r in ledger if r["soft_qa"]],
        "merge_path": [
            "python phase0/nhl_shifts_html.py merge            # dry run, writes nothing",
            "python phase0/nhl_shifts_html.py merge --apply    # inserts the backfill into "
            "data/nhl_shifts.csv in spine order; skips games already present; logs "
            "data/nhl_shifts_html_merged.csv",
            "then rebuild shift-derived caches: phase0/pv_nhl_events.py -> "
            "pv_nhl_full_common.noshift_games(rebuild=True) -> consumers (nhl_rapm2.py, ...)",
            "never regenerate a file pinned by data/pv_nhl_forward_prereg_2026_27.json; "
            "ledger TEST numbers are not re-scored"]}
    with open(VALIDATION_JSON, "w", encoding="utf-8") as fh:
        json.dump(prev, fh, indent=1)
    print(f"backfill: {dict(c)} | {len(rows_all):,} rows -> {os.path.relpath(BACKFILL_CSV, ROOT)}")


def _spine_index():
    return {int(r["game_id"]): i for i, r in enumerate(read_spine())}


def merge(shifts_path, backfill_path, spine_idx, apply=False, merged_log=None):
    """Insert backfill games into shifts_path in spine order (see module docstring).
    Games already present in shifts_path are skipped. -> report dict."""
    bf = defaultdict(list)
    with open(backfill_path, encoding="utf-8") as fh:
        rd = csv.reader(fh)
        hdr = next(rd)
        assert hdr == SCHEMA, f"backfill header {hdr} != {SCHEMA}"
        for r in rd:
            if r:
                bf[int(r[0])].append(r)
    present = set()
    order_breaks, n_old, last_idx = 0, 0, -1
    with open(shifts_path, encoding="utf-8") as fh:
        hdr = fh.readline().rstrip("\r\n").split(",")
        assert hdr == SCHEMA, f"shifts header {hdr} != {SCHEMA}"
        prev = None
        for line in fh:
            n_old += 1
            g = line.split(",", 1)[0]
            if g != prev:
                prev = g
                gi = int(g)
                present.add(gi)
                ix = spine_idx.get(gi, last_idx)
                if ix < last_idx:
                    order_breaks += 1
                last_idx = max(last_idx, ix)
    skip = sorted(g for g in bf if g in present)
    add = sorted((g for g in bf if g not in present), key=lambda g: (spine_idx.get(g, 10 ** 9), g))
    n_add = sum(len(bf[g]) for g in add)
    rep = {"shifts_rows_before": n_old, "backfill_games": len(bf),
           "skip_already_present": len(skip), "skip_gids": skip[:50],
           "games_to_insert": len(add), "rows_to_insert": n_add,
           "existing_file_order_breaks": order_breaks, "applied": False}
    if not apply or not add:
        return rep
    tmp = shifts_path + ".html_merge.tmp"
    queue = list(add)
    qi = 0
    n_out = 0
    # newline="" on both sides: existing lines are copied byte-for-byte and the
    # inserted rows use the file's own terminator (csv.writer's CRLF in practice)
    with open(shifts_path, encoding="utf-8", newline="") as src, \
            open(tmp, "w", newline="", encoding="utf-8") as dst:
        head = src.readline()
        term = "\r\n" if head.endswith("\r\n") else "\n"
        dst.write(head)
        w = csv.writer(dst, lineterminator=term)
        prev = None
        for line in src:
            g = line.split(",", 1)[0]
            if g != prev:
                prev = g
                ix = spine_idx.get(int(g))
                while ix is not None and qi < len(queue) and spine_idx.get(queue[qi], 10 ** 9) < ix:
                    w.writerows(bf[queue[qi]])
                    n_out += len(bf[queue[qi]])
                    qi += 1
            dst.write(line if line.endswith("\n") else line + term)
            n_out += 1
        while qi < len(queue):
            w.writerows(bf[queue[qi]])
            n_out += len(bf[queue[qi]])
            qi += 1
    assert n_out == n_old + n_add, (n_out, n_old, n_add)
    os.replace(tmp, shifts_path)
    rep.update(applied=True, shifts_rows_after=n_out)
    if merged_log:
        new = not os.path.exists(merged_log)
        with open(merged_log, "a", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["gid", "rows", "source", "merged_utc"])
            ts = utcnow()
            for g in add:
                w.writerow([g, len(bf[g]), "html_toi_report", ts])
    return rep


def cmd_merge(a):
    rep = merge(SHIFTS_CSV, BACKFILL_CSV, _spine_index(), apply=a.apply, merged_log=MERGED_CSV)
    print(json.dumps(rep, indent=1))
    if not a.apply:
        print("dry run: nothing written (pass --apply to rewrite data/nhl_shifts.csv; then "
              "rebuild pv_nhl_events.py -> noshift_games(rebuild=True) -> consumers)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate")
    v.add_argument("--seed", type=int, default=20260924)
    v.add_argument("--n-dev", type=int, default=64)
    v.add_argument("--n-mid", type=int, default=24)
    v.add_argument("--n-recent", type=int, default=64)
    sub.add_parser("recheck-api")
    b = sub.add_parser("backfill")
    b.add_argument("--gids", default="")
    m = sub.add_parser("merge")
    m.add_argument("--apply", action="store_true")
    a = ap.parse_args(argv)
    {"validate": cmd_validate, "recheck-api": cmd_recheck_api, "backfill": cmd_backfill,
     "merge": cmd_merge}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
