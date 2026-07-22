"""Load Retrosheet games + starter FIP lines for training.

Retrosheet game logs supply the matchup, outcome and starting-pitcher ids; the
parsed event appearances supply each starter's FIP components. Both are on disk
under data/retrosheet and data/retro_events (see phase0/parse_events.py, whose
output is validated by phase0/validate_events.py).

Attribution required by Retrosheet's terms, reproduced on the site's methodology
page: "The information used here was obtained free of charge from and is
copyrighted by Retrosheet. Interested parties may contact Retrosheet at
www.retrosheet.org."
"""

from __future__ import annotations

import csv
import glob
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
GL_DIR = PROJECT / "data" / "retrosheet"
APPEAR = PROJECT / "data" / "retro_events" / "appearances.csv"
FLYBALLS = PROJECT / "data" / "retro_events" / "flyballs.csv"

F_DATE, F_VIS, F_HOME, F_VIS_R, F_HOME_R, F_VIS_SP, F_HOME_SP = 0, 3, 6, 9, 10, 101, 103

# Set by load_games when fly-ball data is present: league HR per fly ball, the
# constant that turns a start into its xFIP line. Frozen into the model artifact.
LEAGUE_HRFB = None


def load_fly_balls() -> dict:
    """(game_id, pitcher) -> fly balls allowed (non-HR), from phase0/fb_parse.py."""
    fb = {}
    if FLYBALLS.is_file():
        with open(FLYBALLS, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                fb[(r["game_id"], r["pitcher"])] = int(r["fb"])
    return fb


def to_xfip(line, fb: float, lg_hrfb: float):
    """FIP line (outs,HR,BB,HBP,SO) -> xFIP line: actual HR replaced by expected
    HR = (fly balls + HR) * league HR/FB. HR are fly balls, so FB_total = fb + HR."""
    outs, hr, bb, hbp, so = line
    return (outs, (fb + hr) * lg_hrfb, bb, hbp, so)


def load_starter_lines() -> dict:
    """(game_id, pitcher) -> (outs, HR, BB, HBP, SO) for starters."""
    out = {}
    with open(APPEAR, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["is_starter"] != "1":
                continue
            out[(r["game_id"], r["pitcher"])] = (
                int(r["outs"]), int(r["HR"]), int(r["BB"]), int(r["HBP"]), int(r["SO"]))
    return out


def pitcher_names() -> dict:
    """retro_id -> most recent display name, from the game logs."""
    names = {}
    for path in sorted(glob.glob(str(GL_DIR / "gl*.txt"))):
        for r in csv.reader(open(path, encoding="latin-1")):
            if len(r) < 105:
                continue
            if r[F_VIS_SP].strip():
                names[r[F_VIS_SP].strip()] = r[102]
            if r[F_HOME_SP].strip():
                names[r[F_HOME_SP].strip()] = r[104]
    return names


def load_games(year_min=None, year_max=None) -> list[dict]:
    lines = load_starter_lines()
    rows = []
    for path in sorted(glob.glob(str(GL_DIR / "gl*.txt"))):
        yr = int(path[-8:-4])
        if year_min and yr < year_min:
            continue
        if year_max and yr > year_max:
            continue
        for r in csv.reader(open(path, encoding="latin-1")):
            if len(r) < 105:
                continue
            try:
                vr, hr = int(r[F_VIS_R]), int(r[F_HOME_R])
            except ValueError:
                continue
            if vr == hr:
                continue
            gid = r[F_HOME] + r[F_DATE] + r[1]
            hsp, asp = r[F_HOME_SP].strip(), r[F_VIS_SP].strip()
            rows.append({
                "game_id": gid, "date": r[F_DATE], "season": yr,
                "home": r[F_HOME], "away": r[F_VIS],
                "home_sp": hsp, "away_sp": asp, "y": 1.0 if hr > vr else 0.0,
                "home_line": lines.get((gid, hsp)), "away_line": lines.get((gid, asp)),
            })
    rows.sort(key=lambda g: (g["date"], g["game_id"]))

    # Adopt xFIP as the pitcher core: replace each starter's actual HR with expected
    # HR from his fly balls. league HR/FB is computed over the loaded games and frozen
    # (train.py) so serving uses the same constant. Falls back to plain FIP if the
    # fly-ball parse is absent.
    fb = load_fly_balls()
    if fb:
        global LEAGUE_HRFB
        tot_hr = tot_fb = 0
        for g in rows:
            for sp, ln in ((g["home_sp"], g["home_line"]), (g["away_sp"], g["away_line"])):
                if ln:
                    tot_hr += ln[1]
                    tot_fb += fb.get((g["game_id"], sp), 0) + ln[1]
        LEAGUE_HRFB = tot_hr / tot_fb if tot_fb else 0.11
        for g in rows:
            if g["home_line"]:
                g["home_line"] = to_xfip(g["home_line"], fb.get((g["game_id"], g["home_sp"]), 0), LEAGUE_HRFB)
            if g["away_line"]:
                g["away_line"] = to_xfip(g["away_line"], fb.get((g["game_id"], g["away_sp"]), 0), LEAGUE_HRFB)
    return rows


def league_fip_core(games) -> float:
    HR = BB = HBP = SO = outs = 0
    for g in games:
        for ln in (g["home_line"], g["away_line"]):
            if ln:
                o, hr, bb, hbp, so = ln
                outs += o; HR += hr; BB += bb; HBP += hbp; SO += so
    ip = outs / 3.0
    return (13 * HR + 3 * (BB + HBP) - 2 * SO) / ip if ip else 1.2
