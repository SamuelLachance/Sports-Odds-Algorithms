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

F_DATE, F_VIS, F_HOME, F_VIS_R, F_HOME_R, F_VIS_SP, F_HOME_SP = 0, 3, 6, 9, 10, 101, 103


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
