"""NHL players for site/data/nhl.json: every rostered skater AND goalie.

Standard library only (testable without numpy). Display data - nothing here
feeds the game model, which has no player, lineup or goalie input.

Who appears: exactly the players on a current NHL roster
(data/nhl_player_names.json, `on_roster: true`, see phase0/nhl_names.py). A
player who left every roster is not shown as anybody's team member.

Skaters keep the teammate-adjusted RAPM rating (data/nhl_rapm2_ratings.json,
xG/60 at 5v5, 0-100 scale, 50 = average) once they have MIN_TOI 5v5 minutes in
the shift data the ratings were fit on; under that, or with no minutes at all,
`rating` is null and `nr` says why in words. The shift data on file is not
complete for every season (2025-26: 807 of 1,312 regular-season games), so the
coverage of each fit season is recorded next to the ratings (rapm_meta) and a
reason names it whenever the player played in an incomplete season - a
full-season rookie is short of the floor because games are missing, not
because he barely played.

Units: `toi` on a player is 5v5 MINUTES in that shift data (RAPM). Season lines
carry `toi_pg` (skaters, seconds per game) and `toi_s` (goalies, total
seconds); `norm_line` upgrades lines cached before that rename.

Goalies get a rating built here from data/nhl_goalie_xg.csv (MoneyPuck per-game
goals against vs expected goals against, regular season):

  * per goalie-season, D = GSAx relative to that season's league-average goalie
    (so a league-wide xG calibration drift is not credited to anyone), X = xGA
    faced;
  * seasons weighted by recency, DECAY per season, N_AGES seasons;
  * empirical-Bayes shrinkage toward the league average:
        theta = sum(w D) / (sum(w X) + 1 / tau^2)
    with the Poisson approximation var(GA | xGA) ~ xGA, and the between-goalie
    talent spread tau estimated by method of moments on the DEV seasons
    (2011-12..2017-18) only - no TEST-season statistic is computed;
  * `gsax60` = theta x league xGA per team-game (~per 60 minutes): goals saved
    above an average goalie per 60 at a league-average workload;
  * `rating` = 50 + 15 * theta / sd(theta over qualified rostered goalies),
    clipped 1..99, the same "15 points per SD" scale the skaters use; `rel` =
    sum(w X) / (sum(w X) + 1/tau^2), the share of the estimate that is data.
  Goalies under MIN_GP games in the window get rating null with a reason.

Skaters also carry `pv`, the player-value snapshot (data/nhl_site_pv.json, built
locally by phase0/nhl_site_player_value.py from the validated player-value
program): finishing, creation, assists, power play, faceoffs, penalties and
defence, each walk-forward and shrunk, composed into goals per 60 / per game with
a percentile within F / D. It is a DISPLAY rating - the game model does not use
it (its game-level test was not significant; a forward test on 2026-27 is
pre-registered). A static snapshot "as of the end of 2025-26": the serve only
reads the JSON (stdlib), and a missing or unreadable file leaves `pv` null, so
the pages fall back to the RAPM rating. team_blocks adds `pv_lu`, the summed
value of the same 18-skater lineup the GlassBox rating uses.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from datetime import date, datetime, timezone

GOALIE_XG = "data/nhl_goalie_xg.csv"
RAPM = "data/nhl_rapm2_ratings.json"
RAPM_META = "data/nhl_site_rapm_meta.json"
SHIFTS = "data/nhl_shifts.csv"      # local only (~536 MB, never in CI)
SPINE = "data/nhl_games.csv"
PV = "data/nhl_site_pv.json"        # player-value snapshot (phase0/nhl_site_player_value.py)
PV_COMPS = ("cre", "fin", "a1", "a2", "pp", "fo", "pen", "def")

MIN_TOI = 1000.0          # 5v5 minutes for a displayed skater rating
FULL_COVERAGE = 0.99      # a fit season below this share of games with shifts is "incomplete"
PRIOR_RATING = 50.0       # team aggregate only: an unrated lineup slot with no RAPM row
MIN_GP = 10               # goalie games in the window for a displayed rating
DECAY = 0.7               # per-season recency weight
N_AGES = 5                # seasons in the goalie window (current + 4 prior)
DEV_SEASONS = range(2011, 2018)   # start years: the NHL DEV window
TAU_MIN_XGA = 20.0        # goalie-seasons used to estimate tau

FWD = ("C", "L", "R")


def pos_group(pos: str) -> str:
    return "G" if pos == "G" else ("D" if pos == "D" else "F")


def season_label(start_year: int) -> str:
    return f"{start_year}-{str(start_year + 1)[2:]}"


# ------------------------------------------------------------------ goalies --
def load_goalie_games(path: str = GOALIE_XG) -> list[tuple]:
    """Regular-season goalie-game rows: (season_start, pid, team, gid, xga, ga, gsax).
    goalie_id 0 (empty net) is dropped."""
    out = []
    try:
        fh = open(path, encoding="utf-8")
    except FileNotFoundError:
        return out
    with fh:
        for r in csv.DictReader(fh):
            gid = str(r["nhl_game_id"])
            pid = str(r["goalie_id"]).split(".")[0]
            if gid[4:6] != "02" or pid in ("", "0"):
                continue
            out.append((int(r["season"]), pid, r["team"], gid, float(r["xga"]),
                        float(r["ga"]), float(r["gsax"])))
    return out


def league_rates(rows) -> dict:
    """season -> league GSAx per xGA (the average goalie of that season)."""
    x, d = defaultdict(float), defaultdict(float)
    for s, _, _, _, xga, _, gsax in rows:
        x[s] += xga
        d[s] += gsax
    return {s: d[s] / x[s] for s in x if x[s] > 0}


def estimate_tau(rows, seasons=DEV_SEASONS, min_xga: float = TAU_MIN_XGA) -> float:
    """Between-goalie SD of true GSAx-per-xGA, method of moments on DEV seasons:
    tau^2 = mean(r^2 - 1/X) over goalie-seasons with X >= min_xga, r the
    league-centred rate. DEV only, by construction of `seasons`."""
    m = league_rates(rows)
    x, d = defaultdict(float), defaultdict(float)
    for s, pid, _, _, xga, _, gsax in rows:
        if s in seasons:
            x[(pid, s)] += xga
            d[(pid, s)] += gsax
    vals = []
    for k, xx in x.items():
        if xx >= min_xga:
            r = (d[k] - m[k[1]] * xx) / xx
            vals.append(r * r - 1.0 / xx)
    if len(vals) < 30:
        return 0.043        # the DEV estimate (463 goalie-seasons), if DEV is absent
    return math.sqrt(max(sum(vals) / len(vals), 1e-6))


def season_lines(rows) -> dict:
    """season_start -> pid -> {gp, xga, ga, gsax} (raw, regular season)."""
    out = defaultdict(lambda: defaultdict(lambda: {"gp": 0, "xga": 0.0, "ga": 0.0, "gsax": 0.0}))
    for s, pid, _, _, xga, ga, gsax in rows:
        a = out[s][pid]
        a["gp"] += 1
        a["xga"] += xga
        a["ga"] += ga
        a["gsax"] += gsax
    return out


def goalie_quality(rows, cur_start: int, tau: float, decay: float = DECAY,
                   n_ages: int = N_AGES) -> tuple[dict, dict]:
    """pid -> {theta, rel, gsax60, gp, xga, ga, gsax, w_xga} over the window,
    plus window meta. Ratings are added by `rate_goalies` (they need the pool)."""
    m = league_rates(rows)
    lo = cur_start - (n_ages - 1)
    win = [r for r in rows if lo <= r[0] <= cur_start]
    team_games = {(r[3], r[2]) for r in win}
    lg_xga = sum(r[4] for r in win)
    lg_per_game = lg_xga / len(team_games) if team_games else 2.8
    agg = defaultdict(lambda: {"gp": 0, "xga": 0.0, "ga": 0.0, "gsax": 0.0,
                               "wx": 0.0, "wd": 0.0})
    for s, pid, _, _, xga, ga, gsax in win:
        w = decay ** (cur_start - s)
        a = agg[pid]
        a["gp"] += 1
        a["xga"] += xga
        a["ga"] += ga
        a["gsax"] += gsax
        a["wx"] += w * xga
        a["wd"] += w * (gsax - m.get(s, 0.0) * xga)
    prior = 1.0 / (tau * tau)
    out = {}
    for pid, a in agg.items():
        theta = a["wd"] / (a["wx"] + prior)
        out[pid] = {"theta": theta, "rel": a["wx"] / (a["wx"] + prior),
                    "gsax60": theta * lg_per_game, "gp": a["gp"],
                    "xga": a["xga"], "ga": a["ga"], "gsax": a["gsax"]}
    seasons = sorted({r[0] for r in win})
    meta = {"window": (f"{season_label(seasons[0])} to {season_label(seasons[-1])}"
                       if seasons else None),
            "tau": round(tau, 4), "decay": decay, "min_gp": MIN_GP,
            "lg_xga60": round(lg_per_game, 3)}
    return out, meta


def rate_goalies(q: dict, pool_ids) -> float:
    """Add the 0-100 `rating` to every goalie in q; returns the scale SD.
    The SD is taken over qualified goalies in `pool_ids` (the rostered ones)."""
    pool = [q[p]["theta"] for p in pool_ids if p in q and q[p]["gp"] >= MIN_GP]
    if len(pool) >= 5:
        mu = sum(pool) / len(pool)
        sd = math.sqrt(sum((v - mu) ** 2 for v in pool) / (len(pool) - 1))
    else:
        sd = 0.0
    sd = sd if sd > 1e-9 else 0.02
    for v in q.values():
        v["rating"] = round(min(99.0, max(1.0, 50.0 + 15.0 * v["theta"] / sd)), 1)
    return sd


# --------------------------------------------------------------------- RAPM --
def ratings_sha(path: str) -> str:
    """Content hash of the ratings file, line-ending agnostic (a CRLF checkout
    of the same committed file must match its sidecar)."""
    return hashlib.sha256(open(path, "rb").read().replace(b"\r\n", b"\n")).hexdigest()


def shift_coverage(seasons, shifts_path: str = SHIFTS, spine_path: str = SPINE) -> dict | None:
    """season -> {"with_shifts": n, "games": N}: regular-season spine games of
    each season that have ANY shift rows in the shift file, of all of them.
    None when the shift file is absent (CI never has it)."""
    try:
        fh = open(shifts_path, "rb")
    except FileNotFoundError:
        return None
    ids, last = set(), None
    with fh:
        next(fh, None)                                   # header
        for line in fh:
            k = line.split(b",", 1)[0]
            if k != last:                                # rows are grouped by game
                last = k
                ids.add(k)
    want = {int(s) for s in seasons}
    games, have = defaultdict(int), defaultdict(int)
    with open(spine_path, encoding="utf-8") as sp:
        for r in csv.DictReader(sp):
            if r.get("type") != "2" or int(r["season"]) not in want:
                continue
            s = int(r["season"])
            games[s] += 1
            have[s] += r["game_id"].encode() in ids
    return {str(s): {"with_shifts": have[s], "games": games[s]}
            for s in sorted(want) if games[s]}


def rapm_meta(spine_seasons, ratings_path: str = RAPM, meta_path: str = RAPM_META,
              cur_season: int | None = None, shifts_path: str = SHIFTS,
              spine_path: str = SPINE, warn=print) -> dict:
    """Which seasons the committed RAPM ratings were fit on, when, and how much
    of each season's shift data they saw.

    nhl_rapm2.py (local only - the shift file never lives in CI) fits the
    latest four spine seasons and records none of this, so the sidecar
    data/nhl_site_rapm_meta.json keys it to the ratings file's content hash.
    The sidecar is COMMITTED with the ratings; while the hash matches, its
    window, build date and coverage stand.

    Hash mismatch:
      * shift file present (the local machine that refits, where the weekly
        chain serves right after nhl_rapm2): the window is nhl_rapm2's own rule
        applied to this spine, the build date is the ratings file's mtime, and
        the coverage is counted - all written to the sidecar;
      * no shift file (a CI checkout without a matching sidecar): nothing is
        known. The window is INFERRED as the four seasons before the served
        one, `built` is None (never "today": CI does not refit), a warning is
        logged and nothing is written, so a guess can never be persisted.
    A matching sidecar without coverage gets it added when the shift file is
    present.
    """
    empty = {"window": None, "built": None, "min_toi": MIN_TOI, "coverage": None,
             "inferred": False}
    try:
        sha = ratings_sha(ratings_path)
    except FileNotFoundError:
        return empty
    try:
        meta = json.load(open(meta_path, encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        meta = {}
    local = os.path.exists(shifts_path)
    dirty = False
    if meta.get("sha256") != sha:
        if local:
            fit = sorted(set(spine_seasons))[-4:]
            built = datetime.fromtimestamp(os.path.getmtime(ratings_path),
                                           timezone.utc).date().isoformat()
            meta = {"sha256": sha, "fit_seasons": fit, "built": built}
            dirty = True
            warn(f"[rapm_meta] ratings changed (local refit): window re-derived as "
                 f"{fit[0] if fit else None}..{fit[-1] if fit else None}, built {built}")
        else:
            prior = sorted(s for s in set(spine_seasons)
                           if cur_season is None or s < cur_season)
            meta = {"fit_seasons": prior[-4:], "built": None, "inferred": True}
            warn(f"WARNING rapm_meta: {meta_path} does not describe {ratings_path} "
                 f"(missing or stale sidecar, no shift file to re-derive it); window "
                 f"inferred as the four seasons before {cur_season}, build date unknown")
    if local and not meta.get("inferred") and "coverage" not in meta:
        cov = shift_coverage(meta.get("fit_seasons") or [], shifts_path, spine_path)
        if cov is not None:
            meta["coverage"] = cov
            dirty = True
    if dirty:
        try:
            with open(meta_path, "w", encoding="utf-8", newline="\n") as fh:
                json.dump(meta, fh, indent=1)
                fh.write("\n")
        except OSError:
            pass
    fit = meta.get("fit_seasons") or []
    window = (f"{season_label(fit[0] // 10000)} to {season_label(fit[-1] // 10000)}"
              if fit else None)
    cov = None
    if meta.get("coverage"):
        cov = {}
        for s, c in sorted(meta["coverage"].items()):
            n, w = int(c.get("games") or 0), int(c.get("with_shifts") or 0)
            cov[season_label(int(s) // 10000)] = {
                "with_shifts": w, "games": n, "share": round(w / n, 3) if n else None}
    return {"window": window, "built": meta.get("built"), "min_toi": MIN_TOI,
            "coverage": cov, "inferred": bool(meta.get("inferred"))}


def incomplete_seasons(coverage: dict | None) -> dict:
    """start year -> (with_shifts, games) for fit seasons under FULL_COVERAGE.
    `coverage` is rapm_meta()'s, keyed by season label."""
    out = {}
    for lbl, c in (coverage or {}).items():
        if c.get("games") and c["with_shifts"] / c["games"] < FULL_COVERAGE:
            out[int(lbl[:4])] = (c["with_shifts"], c["games"])
    return out


def coverage_caveat(coverage: dict | None, seasons=None) -> str | None:
    """'2025-26 shift data covers 807 of 1,312 games' for each incomplete fit
    season (restricted to `seasons`, start years, when given), newest first."""
    inc = incomplete_seasons(coverage)
    keys = sorted((s for s in inc if seasons is None or s in seasons), reverse=True)
    parts = [f"{season_label(s)} shift data covers {inc[s][0]:,} of {inc[s][1]:,} games"
             for s in keys]
    return "; ".join(parts) or None


# ------------------------------------------------------------- player value --
def load_pv(path: str = PV, warn=print) -> dict:
    """pid -> player-value block from the static snapshot, or {} when the file is
    absent or unreadable (CI keeps serving; the pages fall back to RAPM). The
    snapshot is built locally (phase0/nhl_site_player_value.py, from gitignored
    parquets) and CI only reads it, so a missing file is logged loudly rather
    than letting the feature disappear in silence."""
    def gone(why: str) -> dict:
        warn(f"WARNING nhl_site_players: player-value snapshot {path} {why}; skater pages "
             f"fall back to on-ice xG (RAPM). Commit data/nhl_site_pv.json with the serve.")
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
    except OSError:
        return gone("is missing")
    except json.JSONDecodeError:
        return gone("is unreadable")
    pl = d.get("players") if isinstance(d, dict) else None
    if not isinstance(pl, dict):
        return gone("has no players block")
    out = {str(k): v for k, v in pl.items() if isinstance(v, dict) and v.get("v") is not None}
    return out or gone("has no valued skater")


def pv_lineup(fwd: list, dmen: list) -> dict | None:
    """Summed player value of a lineup (goals per game above a lineup of average
    regulars). A slot with no value (no NHL game on file) counts as average, 0,
    and `n` says how many of the `slots` carry one. `k` splits the total by
    component (each player's contribution x his minutes / 60); the parts can miss
    the total by rounding."""
    slots = list(fwd) + list(dmen)
    if not slots:
        return None

    def g(p):
        return float((p.get("pv") or {}).get("g") or 0.0)

    got = [p for p in slots if p.get("pv")]
    comp = {k: round(sum(float((p["pv"].get("c") or {}).get(k) or 0.0)
                         * float(p["pv"].get("toi") or 0.0) / 60.0 for p in got), 3)
            for k in PV_COMPS}
    return {"tot": round(sum(g(p) for p in slots), 3),
            "f": round(sum(g(p) for p in fwd), 3),
            "d": round(sum(g(p) for p in dmen), 3),
            "n": len(got), "slots": len(slots), "k": comp,
            "top": [p["id"] for p in sorted(got, key=lambda p: (-g(p), p["name"] or ""))[:3]]}


# ------------------------------------------------------------------ players --
def _age(born: str | None, today: date) -> int | None:
    try:
        b = date.fromisoformat(born)
    except (TypeError, ValueError):
        return None
    return today.year - b.year - ((today.month, today.day) < (b.month, b.day))


def _r(v, nd):
    return None if v is None else round(v, nd)


def norm_line(ln: dict | None, grp: str) -> dict | None:
    """A season line with its ice time under an explicit unit: skaters
    `toi_pg` (seconds per game), goalies `toi_s` (total seconds). Lines cached
    or served before the rename carried both as `toi`."""
    if not ln or "toi" not in ln:
        return ln
    ln = dict(ln)
    v = ln.pop("toi")
    ln.setdefault("toi_s" if grp == "G" else "toi_pg", v)
    return ln


def usage(p: dict, key: str) -> float:
    """Ice time in season line `key` ("cur" or "prev"), in seconds: skaters
    TOI/GP x GP, goalies total TOI. A player with no line in that season sorts
    after everyone who has one, ordered among themselves by RAPM 5v5 minutes."""
    ln = norm_line((p.get("stats") or {}).get(key), p["grp"])
    if ln and ln.get("gp"):
        if p["grp"] == "G":
            return float(ln.get("toi_s") or ln["gp"] * 3600)
        return float((ln.get("toi_pg") or 0) * ln["gp"]) + ln["gp"]
    return float(p.get("toi") or 0.0) / 1e6


def main_line(lines: dict) -> str | None:
    """Which season line a page should lead with: this season once it has
    10 games, otherwise last season (early October), otherwise whatever exists."""
    cur = lines.get("cur")
    if cur and (cur.get("gp") or 0) >= 10:
        return "cur"
    if lines.get("prev"):
        return "prev"
    return "cur" if cur else None


def build_players(names: dict, rapm: dict, stats: dict, career: dict,
                  gq: dict, gseason: dict, cur_start: int, today: date,
                  rapm_window: str | None, goalie_window: str | None,
                  rapm_coverage: dict | None = None, pv: dict | None = None) -> dict:
    """pid -> player block for every current-roster player.

    stats:  {"cur": {"skaters": {...}, "goalies": {...}} | None, "prev": ...}
    career: {"skaters": {...}, "goalies": {...}} | {}
    gq:     goalie_quality() output with ratings (rate_goalies)
    gseason: season_lines() output, adds GSAx/xGA to goalie season lines
    rapm_coverage: rapm_meta()["coverage"]; an unrated skater who played in an
            incomplete fit season gets that season's coverage in his reason
    pv:     load_pv() output (read from PV when None); skaters get `pv`, their
            block or null
    """
    pv = load_pv() if pv is None else pv
    players = {}
    where = f" ({rapm_window})" if rapm_window else ""
    for pid, v in names.items():
        if not v.get("on_roster") or not v.get("team"):
            continue
        pos = v.get("pos") or ""
        grp = pos_group(pos)
        kind = "goalies" if grp == "G" else "skaters"
        p = {"id": pid, "name": v.get("name"), "team": v["team"], "pos": pos, "grp": grp,
             "num": v.get("num"), "born": v.get("born"), "age": _age(v.get("born"), today),
             "shoots": v.get("shoots"), "ht": v.get("ht"), "wt": v.get("wt"),
             "nat": v.get("nat"), "img": v.get("img"),
             "rating": None, "rtype": "gsax" if grp == "G" else "rapm", "nr": None,
             "off": None, "def": None, "net": None, "toi": None, "rel": None,
             "pv": dict(pv[pid]) if grp != "G" and pid in pv else None}
        if grp == "G":
            q = gq.get(pid)
            if q is None:
                p["nr"] = f"no NHL regular-season games in the rating window ({goalie_window})"
            else:
                p["g_win"] = {"gp": q["gp"], "xga": round(q["xga"], 1),
                              "ga": int(round(q["ga"])), "gsax": round(q["gsax"], 1)}
                p["rel"] = round(q["rel"], 3)
                if q["gp"] >= MIN_GP:
                    p["rating"] = q["rating"]
                    p["gsax60"] = round(q["gsax60"], 3)
                else:
                    p["nr"] = (f"fewer than {MIN_GP} NHL games in the rating window "
                               f"({q['gp']} so far)")
        lines = {}
        for k in ("cur", "prev"):
            src = (stats.get(k) or {}).get(kind) or {}
            ln = norm_line(src.get(pid), grp)
            if ln is not None and grp == "G":
                ln = dict(ln)
                ms = gseason.get(cur_start - (0 if k == "cur" else 1), {}).get(pid)
                if ms:
                    ln["gsax"] = round(ms["gsax"], 1)
                    ln["xga"] = round(ms["xga"], 1)
            lines[k] = ln
        lines["career"] = norm_line((career.get(kind) or {}).get(pid), grp)
        lines["main"] = main_line(lines)
        p["stats"] = lines
        if grp != "G":
            # seasons he has a line in; the caveat names only those of them
            # whose shift data on file is incomplete
            played = {cur_start - (0 if k == "cur" else 1) for k in ("cur", "prev")
                      if (lines.get(k) or {}).get("gp")}
            cav = coverage_caveat(rapm_coverage, played)
            r = rapm.get(pid)
            if r is None:
                p["nr"] = ("no 5v5 minutes in the shift data on file"
                           + (f" ({rapm_window}; {cav})" if cav and rapm_window
                              else f" ({cav})" if cav else where))
            else:
                p["off"] = r.get("off")
                p["def"] = r.get("def")
                p["net"] = r.get("net")
                p["toi"] = r.get("toi_min")
                p["rel"] = r.get("rel")
                if (r.get("toi_min") or 0) >= MIN_TOI:
                    p["rating"] = r.get("rating")
                else:
                    p["nr"] = (f"under {MIN_TOI:,.0f} 5v5 minutes in the shift data on file "
                               f"({(r.get('toi_min') or 0):,.0f}" + (f"; {cav}" if cav else "")
                               + ")")
        players[pid] = p

    # ranks among displayed ratings: skaters overall and within F / D; goalies
    sk = sorted((p for p in players.values() if p["grp"] != "G" and p["rating"] is not None),
                key=lambda p: (-p["rating"], p["name"] or ""))
    for i, p in enumerate(sk):
        p["rk"] = i + 1
    for g in ("F", "D"):
        for i, p in enumerate([q for q in sk if q["grp"] == g]):
            p["rk_pos"] = i + 1
    gl = sorted((p for p in players.values() if p["grp"] == "G" and p["rating"] is not None),
                key=lambda p: (-p["rating"], p["name"] or ""))
    for i, p in enumerate(gl):
        p["rk"] = i + 1
        p["rk_pos"] = i + 1
    return players


def lineup_value(p: dict, rapm: dict | None) -> tuple[float, str]:
    """A lineup slot's rating for the TEAM aggregate, and where it came from:
    the displayed rating ("rated"); else the player's ridge- and EB-shrunk RAPM
    rating from the ratings file, which exists below the 1,000-minute display
    floor ("rapm"); else the league-average prior PRIOR_RATING ("prior")."""
    if p.get("rating") is not None:
        return float(p["rating"]), "rated"
    r = (rapm or {}).get(p["id"])
    if r and r.get("rating") is not None:
        return float(r["rating"]), "rapm"
    return PRIOR_RATING, "prior"


def team_blocks(players: dict, teams, rapm: dict | None = None) -> dict:
    """Per team: ordered roster ids, goalies, top-5 skaters, the GLASSBOX
    lineup rating and the lineup's player value `pv_lu` (pv_lineup; `rk` among
    the teams whose lineup carries any value) - display only, not model inputs.

    Lineup = the 12 forwards and 6 defencemen with the most ice time THIS
    season once the team has played (last season before that - one season
    for the whole team, so the two are never mixed). EVERY lineup slot counts
    (lineup_value): a displayed rating, else the player's shrunk RAPM rating
    under the display floor, else the league-average prior 50 - so a team of
    rookies is rated on its whole lineup, not on the few veterans who clear
    the floor, and teams are comparable. gb_f / gb_d = mean over the slots;
    glassbox = mean over all 18; n_f / n_d = slots with a DISPLAYED rating;
    `cov` = their share of the slots; `fill` = slots filled from the RAPM file
    and from the prior. gb_g = the most-used goalie's rating. `use` says which
    season chose the lineup.
    """
    by = defaultdict(list)
    for pid, p in players.items():
        by[p["team"]].append(p)
    out = {}
    for t in teams:
        members = by.get(t, [])
        key = "cur" if any(((p.get("stats") or {}).get("cur") or {}).get("gp")
                           for p in members) else "prev"
        ps = sorted(members, key=lambda p: (-usage(p, key), p["name"] or ""))
        f = [p for p in ps if p["grp"] == "F"]
        d = [p for p in ps if p["grp"] == "D"]
        g = [p for p in ps if p["grp"] == "G"]
        fs = [lineup_value(p, rapm) for p in f[:12]]
        ds = [lineup_value(p, rapm) for p in d[:6]]
        fr = [v for v, _ in fs]
        dr = [v for v, _ in ds]
        allr = fr + dr
        srcs = [s for _, s in fs + ds]
        n_f = sum(1 for _, s in fs if s == "rated")
        n_d = sum(1 for _, s in ds if s == "rated")
        rated_sk = [p for p in ps if p["grp"] != "G" and p["net"] is not None
                    and p["rating"] is not None]
        out[t] = {
            "roster": [p["id"] for p in f + d + g],
            "goalies": [p["id"] for p in g],
            "top": [p["id"] for p in sorted(rated_sk, key=lambda p: -p["net"])[:5]],
            "glassbox": round(sum(allr) / len(allr), 1) if allr else None,
            "gb_f": round(sum(fr) / len(fr), 1) if fr else None,
            "gb_d": round(sum(dr) / len(dr), 1) if dr else None,
            "gb_g": g[0]["rating"] if g else None,
            "gb_goalie": g[0]["id"] if g else None,
            "n_f": n_f, "n_d": n_d,
            "cov": round((n_f + n_d) / len(allr), 3) if allr else None,
            "fill": {"rapm": srcs.count("rapm"), "prior": srcs.count("prior")},
            "use": key,
            "pv_lu": pv_lineup(f[:12], d[:6]),
        }
    ranked = sorted((t for t in out if out[t]["glassbox"] is not None),
                    key=lambda t: -out[t]["glassbox"])
    for i, t in enumerate(ranked):
        out[t]["gb_rank"] = i + 1
    for t in out:
        out[t].setdefault("gb_rank", None)
    # player-value lineup rank: only once any lineup carries a value
    pvr = sorted((t for t in out if out[t]["pv_lu"] and out[t]["pv_lu"]["n"]),
                 key=lambda t: -out[t]["pv_lu"]["tot"])
    for i, t in enumerate(pvr):
        out[t]["pv_lu"]["rk"] = i + 1
    return out
