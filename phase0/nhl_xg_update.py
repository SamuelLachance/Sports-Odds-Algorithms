"""Incremental MoneyPuck xG refresh: re-pull the CURRENT season only.

The full pull (nhl_pull_xg.py) downloads 16 season zips; in-season only the
current season's file grows, so this re-downloads exactly one zip (always
fresh — no cache; the file changes weekly) and rewrites the current season's
rows in place, leaving every other season's rows untouched:

  data/nhl_team_xg.csv    current-season rows replaced
  data/nhl_goalie_xg.csv  current-season rows replaced
  data/nhl_shots.csv      current-season rows replaced (only when
                          data/nhl_shifts.csv exists — shots are consumed by
                          the shift-based models, which only run where the
                          local shifts artifact lives; skipped in CI)

Aggregation logic is a faithful replica of nhl_pull_xg.py / nhl_pull_shots.py —
in particular the game-id join: MoneyPuck game_id = type*10000 + game_num, and
nhl_game_id = int(f"{season}{g_rel//10000:02d}{g_rel%10000:04d}") with season =
start year (2025 = 2025-26). Because the id embeds the season start year, the
current season's ids sort strictly after all prior seasons', so replacing the
tail block preserves the files' global sort.

Graceful no-op (exit 0) when MoneyPuck is unreachable or not yet publishing the
season (offseason 404). Standard library only.
"""
from __future__ import annotations

import csv
import io
import os
import sys
import urllib.error
import urllib.request
import zipfile
from collections import defaultdict

sys.path.insert(0, "phase0")
from nhl_pull_xg import HEADERS, URL, fnum  # noqa: E402  (identical join/parse)

TEAM = "data/nhl_team_xg.csv"
GOALIE = "data/nhl_goalie_xg.csv"
SHOTS = "data/nhl_shots.csv"
SHIFTS = "data/nhl_shifts.csv"


def current_season() -> int:
    """MoneyPuck season = start year of the max NHL season in the spine."""
    with open("data/nhl_games.csv", encoding="utf-8") as fh:
        mx = max(int(r["season"]) for r in csv.DictReader(fh))
    return mx // 10000          # 20252026 -> 2025


def fetch_zip_fresh(season: int) -> bytes:
    """Always a fresh download — in-season the zip grows every day."""
    req = urllib.request.Request(URL.format(season), headers=HEADERS)
    return urllib.request.urlopen(req, timeout=180).read()


def aggregate(season: int, raw: bytes, want_shots: bool):
    """Replica of the per-season loops in nhl_pull_xg.main (team/goalie rows)
    and nhl_pull_shots.main (per-shot rows). Do not 'improve' — output must be
    byte-identical to the full pull for unchanged games."""
    z = zipfile.ZipFile(io.BytesIO(raw))
    name = next(n for n in z.namelist() if n.endswith(".csv"))
    tg = defaultdict(lambda: defaultdict(lambda: [0.0, 0, 0]))     # [xgf, shots, gf]
    meta = {}                                                       # gid -> (home, away)
    gl = defaultdict(lambda: [0.0, 0, "", ""])                      # (gid,goalie)->[xga,ga,name,team]
    shot_rows = []
    with z.open(name) as fh:
        rd = csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8", errors="replace"))
        for r in rd:
            try:
                g_rel = int(float(r["game_id"]))
            except (KeyError, ValueError):
                continue
            # MoneyPuck game_id = type*10000 + game_num; NHL API gameId is
            # {season}{type:02d}{game:04d} (e.g. 20001 -> 2010020001).
            gid = int(f"{season}{g_rel // 10000:02d}{g_rel % 10000:04d}")
            shooter = (r.get("teamCode") or "").strip()
            home = (r.get("homeTeamCode") or "").strip()
            away = (r.get("awayTeamCode") or "").strip()
            gk_raw = (r.get("goalieIdForShot") or "").strip()
            xg = fnum(r.get("xGoal"))
            goal = 1 if fnum(r.get("goal")) >= 0.5 else 0
            if want_shots:
                try:
                    period = int(float(r["period"]))
                    tsec = int(float(r["time"]))
                except (KeyError, ValueError):
                    period = None
                if period is not None:
                    shot_rows.append({
                        "nhl_game_id": gid, "period": period,
                        "per_sec": tsec - (period - 1) * 1200,
                        "shooter_team": shooter,
                        "is_home": 1 if fnum(r.get("isHomeTeam")) >= 0.5 else 0,
                        "xg": round(xg, 4), "goal": goal,
                        "home_sk": int(fnum(r.get("homeSkatersOnIce"))),
                        "away_sk": int(fnum(r.get("awaySkatersOnIce"))),
                        "goalie_id": gk_raw if gk_raw not in ("", "NA") else ""})
            if not shooter or not home or not away:
                continue
            meta[gid] = (home, away)
            cell = tg[gid][shooter]
            cell[0] += xg; cell[1] += 1; cell[2] += goal
            if gk_raw and gk_raw not in ("", "NA"):
                try:
                    gk = int(float(gk_raw))
                except ValueError:
                    gk = None
                if gk is not None:
                    defending = away if shooter == home else home
                    gg = gl[(gid, gk)]
                    gg[0] += xg; gg[1] += goal
                    gg[2] = (r.get("goalieNameForShot") or "").strip()
                    gg[3] = defending
    team_rows, goalie_rows = [], []
    for gid, sides in tg.items():
        home, away = meta[gid]
        for tcode, (xgf, shots_f, gf) in sides.items():
            opp = away if tcode == home else home
            oxgf, oshots, oga = sides.get(opp, [0.0, 0, 0])
            team_rows.append({
                "nhl_game_id": gid, "season": season, "team": tcode,
                "is_home": 1 if tcode == home else 0,
                "xgf": round(xgf, 4), "xga": round(oxgf, 4),
                "shots_for": shots_f, "shots_against": oshots,
                "gf": gf, "ga": oga})
    for (gid, gk), (xga, ga, nm, tm) in gl.items():
        goalie_rows.append({
            "nhl_game_id": gid, "season": season, "goalie_id": gk,
            "goalie_name": nm, "team": tm,
            "xga": round(xga, 4), "ga": ga,
            "gsax": round(xga - ga, 4)})
    team_rows.sort(key=lambda r: (r["nhl_game_id"], r.get("team", "")))
    goalie_rows.sort(key=lambda r: (r["nhl_game_id"], r.get("team", "")))
    shot_rows.sort(key=lambda r: (r["nhl_game_id"], r["period"], r["per_sec"]))
    return team_rows, goalie_rows, shot_rows


def replace_season(path: str, season: int, new_rows: list[dict]):
    """Stage a rewrite of `path`: keep every non-current-season row exactly as
    read, append the fresh current-season block (ids sort after prior seasons).

    Writes to `path + '.tmp'` and returns a zero-arg commit callable that
    os.replace()s it into place. Two-phase so main() can stage ALL files first
    and then publish them back-to-back: an interruption (timeout kill,
    disk-full) mid-write can never leave a truncated tracked CSV — prior-season
    rows are only re-obtainable via a full nhl_pull_xg/nhl_pull_shots re-run —
    and the team/goalie files can't diverge for the current season."""
    n_old = 0
    with open(path, encoding="utf-8") as fh:
        rd = csv.reader(fh)
        header = next(rd)
        if "season" in header:
            si = header.index("season")
            is_cur = lambda row: int(row[si]) == season          # noqa: E731
        else:                       # nhl_shots.csv has no season column; the
            gi = header.index("nhl_game_id")  # 10-digit id starts with the year
            is_cur = lambda row: int(row[gi]) // 1000000 == season  # noqa: E731
        keep = []
        for row in rd:
            if not row:
                continue
            n_old += 1
            if not is_cur(row):
                keep.append(row)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(keep)
        dw = csv.DictWriter(fh, fieldnames=header)
        dw.writerows(new_rows)

    def commit() -> None:
        os.replace(tmp, path)       # atomic on POSIX and Windows
        print(f"  {path}: kept {len(keep):,} other-season rows, "
              f"replaced {n_old - len(keep):,} -> {len(new_rows):,} "
              f"season-{season} rows")
    return commit


def main() -> int:
    season = current_season()
    print(f"[nhl_xg_update] current season {season} ({season}{season+1})")
    try:
        raw = fetch_zip_fresh(season)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as ex:
        print(f"[nhl_xg_update] MoneyPuck shots_{season}.zip unavailable ({ex}) — "
              "no-op (offseason or not yet publishing)")
        return 0
    want_shots = os.path.exists(SHIFTS)
    team_rows, goalie_rows, shot_rows = aggregate(season, raw, want_shots)
    if not team_rows:
        print(f"[nhl_xg_update] zip parsed but no {season} games — no-op")
        return 0
    # Stage every rewrite first, then publish atomically back-to-back so a
    # mid-run kill leaves all three files either fully old or fully new.
    commits = [replace_season(TEAM, season, team_rows),
               replace_season(GOALIE, season, goalie_rows)]
    if want_shots:
        commits.append(replace_season(SHOTS, season, shot_rows))
    for commit in commits:
        commit()
    if not want_shots:
        print(f"  {SHOTS}: skipped ({SHIFTS} absent — shift models not run here)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
