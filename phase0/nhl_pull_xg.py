"""Aggregate MoneyPuck shot-level xG into team-game and goalie-game tables.

MoneyPuck publishes free shot-by-shot data with a modelled expected-goals value
(`xGoal`) per unblocked shot attempt, back to 2007-08. We reduce it to:

  data/nhl_team_xg.csv    nhl_game_id, season, team, is_home, xgf, xga,
                          shots_for, shots_against, gf, ga
  data/nhl_goalie_xg.csv  nhl_game_id, season, goalie_id, goalie_name, team,
                          shots_faced, xga, ga   (GSAx = xga - ga, saved-above-exp)

The join key to the NHL-API spine is nhl_game_id = int(f"{season}{game_id:05d}")
(MoneyPuck game_id is season-relative: 20001 -> 2023020001).

Expected goals is the market-blind hockey analog of MLB's FIP: it separates
shot QUALITY generated/allowed from the luck of which pucks went in. The prior
established pre-game EXPECTED shot differential predicts wins (corr 0.20) while
REALIZED shot differential does not (corr -0.02) — so only xG is trustworthy.
"""
from __future__ import annotations

import csv
import io
import time
import urllib.request
import zipfile
from collections import defaultdict

URL = "https://peter-tanner.com/moneypuck/downloads/shots_{}.zip"
HEADERS = {"User-Agent": "glassbox-nhl/1.0 (research)"}
SCRATCH = ("C:/Users/Admin/AppData/Local/Temp/claude/"
           "C--Users-Admin-Projects-Sports-Odds-Algorithms/"
           "c82ef32b-d2a5-4470-9d81-a522f1e0284f/scratchpad")
SEASONS = list(range(2010, 2026))     # MoneyPuck season = start year (2025 = 2025-26)


def fetch_zip(season: int) -> bytes:
    import os
    cache = f"{SCRATCH}/shots_{season}.zip"
    if os.path.exists(cache) and os.path.getsize(cache) > 100000:
        return open(cache, "rb").read()
    for attempt in range(4):
        try:
            req = urllib.request.Request(URL.format(season), headers=HEADERS)
            data = urllib.request.urlopen(req, timeout=180).read()
            open(cache, "wb").write(data)
            return data
        except Exception as ex:  # noqa: BLE001
            if attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def main() -> None:
    team_rows, goalie_rows = [], []
    t0 = time.time()
    for season in SEASONS:
        try:
            raw = fetch_zip(season)
        except Exception as ex:  # noqa: BLE001
            print(f"  {season}: download FAILED ({ex}); skipping", flush=True)
            continue
        z = zipfile.ZipFile(io.BytesIO(raw))
        name = next(n for n in z.namelist() if n.endswith(".csv"))
        # team-game: gid -> teamCode -> aggregates
        tg = defaultdict(lambda: defaultdict(lambda: [0.0, 0, 0]))     # [xgf, shots, gf]
        meta = {}                                                       # gid -> (home, away)
        gl = defaultdict(lambda: [0.0, 0, "", ""])                      # (gid,gid_goalie)->[xga,ga,name,team]
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
                if not shooter or not home or not away:
                    continue
                meta[gid] = (home, away)
                xg = fnum(r.get("xGoal"))
                goal = 1 if fnum(r.get("goal")) >= 0.5 else 0
                cell = tg[gid][shooter]
                cell[0] += xg; cell[1] += 1; cell[2] += goal
                gidr = (r.get("goalieIdForShot") or "").strip()
                if gidr and gidr not in ("", "NA"):
                    try:
                        gk = int(float(gidr))
                    except ValueError:
                        gk = None
                    if gk is not None:
                        defending = away if shooter == home else home
                        gg = gl[(gid, gk)]
                        gg[0] += xg; gg[1] += goal
                        gg[2] = (r.get("goalieNameForShot") or "").strip()
                        gg[3] = defending
        # emit team-game rows (xGA = opponent xGF)
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
        print(f"  {season}: {len(tg):,} games, {len(team_rows):,} team-rows total "
              f"[{time.time()-t0:.0f}s]", flush=True)

    for path, rows in (("data/nhl_team_xg.csv", team_rows),
                       ("data/nhl_goalie_xg.csv", goalie_rows)):
        rows.sort(key=lambda r: (r["nhl_game_id"], r.get("team", "")))
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {path}: {len(rows):,} rows")


if __name__ == "__main__":
    main()
