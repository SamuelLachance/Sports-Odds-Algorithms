"""Extract per-shot rows (time + xGoal + on-ice counts) from cached MoneyPuck zips.

The team-game aggregation (nhl_pull_xg.py) collapses shots; here we keep each
unblocked shot attempt as its own row so it can be matched to the on-ice unit
from the shift charts and scored as a TrueSkill match.

Output data/nhl_shots.csv: nhl_game_id, period, per_sec (seconds within period),
shooter_team, is_home, xg, goal, home_sk, away_sk, goalie_id.
"""
from __future__ import annotations

import csv
import io
import os
import zipfile

SCRATCH = ("C:/Users/Admin/AppData/Local/Temp/claude/"
           "C--Users-Admin-Projects-Sports-Odds-Algorithms/"
           "c82ef32b-d2a5-4470-9d81-a522f1e0284f/scratchpad")
SEASONS = range(2010, 2026)


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def main():
    out = []
    period_max = {}          # period -> max observed 'time' (to detect cumulative vs relative)
    for season in SEASONS:
        z = f"{SCRATCH}/shots_{season}.zip"
        if not os.path.exists(z):
            continue
        zf = zipfile.ZipFile(z)
        name = next(n for n in zf.namelist() if n.endswith(".csv"))
        with zf.open(name) as fh:
            rd = csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8", errors="replace"))
            for r in rd:
                try:
                    g_rel = int(float(r["game_id"]))
                    period = int(float(r["period"]))
                    tsec = int(float(r["time"]))
                except (KeyError, ValueError):
                    continue
                gid = int(f"{season}{g_rel // 10000:02d}{g_rel % 10000:04d}")
                period_max[period] = max(period_max.get(period, 0), tsec)
                # cumulative game-seconds -> within-period seconds
                per_sec = tsec - (period - 1) * 1200
                gk = (r.get("goalieIdForShot") or "").strip()
                out.append({
                    "nhl_game_id": gid, "period": period, "per_sec": per_sec,
                    "shooter_team": (r.get("teamCode") or "").strip(),
                    "is_home": 1 if fnum(r.get("isHomeTeam")) >= 0.5 else 0,
                    "xg": round(fnum(r.get("xGoal")), 4),
                    "goal": 1 if fnum(r.get("goal")) >= 0.5 else 0,
                    "home_sk": int(fnum(r.get("homeSkatersOnIce"))),
                    "away_sk": int(fnum(r.get("awaySkatersOnIce"))),
                    "goalie_id": gk if gk not in ("", "NA") else ""})
        print(f"  {season}: {len(out):,} shots total", flush=True)

    out.sort(key=lambda r: (r["nhl_game_id"], r["period"], r["per_sec"]))
    with open("data/nhl_shots.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(f"\nwrote data/nhl_shots.csv: {len(out):,} shots")
    print("max 'time' per period (cumulative if P2~2400, relative if P2~1200):",
          dict(sorted(period_max.items())))


if __name__ == "__main__":
    main()
