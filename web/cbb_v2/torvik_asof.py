"""Point-in-time Torvik ratings lookup (as-of morning before tip).

Loads ``data/supplemental/torvik-asof/full_YYYY.csv`` archives (toRvik-data).
For seasons without an archive, callers must use ESPN walk-forward efficiency
already in the feature engine — never impute Torvik with zeros/NA fills.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ASOF_DIR = PROJECT_ROOT / "data" / "supplemental" / "torvik-asof"

TORVIK_FEATURE_COLUMNS: tuple[str, ...] = (
    "torvik_adj_o_diff",
    "torvik_adj_d_diff",
    "torvik_adj_t_diff",
    "torvik_barthag_diff",
    "torvik_rank_diff",
    "torvik_wab_diff",
    "torvik_home_o_vs_away_d",
    "torvik_away_o_vs_home_d",
    "torvik_exp_poss",
    "torvik_known",
)

# ESPN abbr / common aliases -> Torvik short team name (archive spelling).
_ABBR_ALIASES: dict[str, str] = {
    "app": "Appalachian St.",
    "csub": "Cal St. Bakersfield",
    "csuf": "Cal St. Fullerton",
    "csun": "Cal St. Northridge",
    "cofc": "College of Charleston",
    "etam": "Texas A&M Commerce",
    "gram": "Grambling St.",
    "iuin": "IUPUI",
    "iupu": "IUPUI",
    "lem": "Le Moyne",
    "liu": "LIU Brooklyn",
    "ul": "Louisiana Lafayette",
    "l-md": "Loyola MD",
    "loyola-md": "Loyola MD",
    "mcn": "McNeese St.",
    "mia": "Miami FL",
    "ncsu": "North Carolina St.",
    "ncst": "North Carolina St.",
    "nich": "Nicholls St.",
    "miss": "Mississippi",
    "oma": "Nebraska Omaha",
    "sela": "Southeastern Louisiana",
    "sfpa": "St. Francis PA",
    "shsu": "Sam Houston St.",
    "sjsu": "San Jose St.",
    "ualb": "Albany",
    "conn": "Connecticut",
    "uic": "Illinois Chicago",
    "ulm": "Louisiana Monroe",
    "utm": "Tennessee Martin",
    "wga": "West Georgia",
    "nhvn": "New Haven",
    "merc": "Mercyhurst",
    "bry": "Bryant",
    "detm": "Detroit",
    "kc": "Kansas City",
}


def _norm(name: str) -> str:
    return "".join(ch for ch in str(name or "").lower() if ch.isalnum())


@dataclass(frozen=True)
class TorvikAsOf:
    adj_o: float
    adj_d: float
    adj_t: float
    barthag: float
    rank: float
    wab: float


@lru_cache(maxsize=16)
def _load_year(year: int) -> pd.DataFrame | None:
    path = ASOF_DIR / f"full_{year}.csv"
    if not path.is_file():
        return None
    frame = pd.read_csv(path, low_memory=False)
    if "date" not in frame.columns:
        return None
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date
    frame = frame.dropna(subset=["date", "team"])
    frame["team_norm"] = frame["team"].map(_norm)
    frame = frame.sort_values("date").drop_duplicates(["date", "team_norm"], keep="last")
    return frame


@lru_cache(maxsize=8)
def _team_index(year: int) -> dict[str, str]:
    """Map normalized Torvik name -> canonical Torvik display name for a year."""
    frame = _load_year(year)
    if frame is None or frame.empty:
        return {}
    out: dict[str, str] = {}
    for team, tn in zip(frame["team"], frame["team_norm"], strict=False):
        out[str(tn)] = str(team)
    return out


@lru_cache(maxsize=1)
def _espn_abbr_display() -> dict[str, str]:
    try:
        from web.season_games import _load_espn_team_ids
    except Exception:
        return {}
    out: dict[str, str] = {}
    for _eid, abbr, display in _load_espn_team_ids("cbb"):
        out[str(abbr).lower()] = str(display)
    return out


@lru_cache(maxsize=8)
def _abbr_to_torvik_norm(year: int) -> dict[str, str]:
    """Build ESPN abbr -> Torvik team_norm using aliases + longest-prefix match."""
    teams = _team_index(year)
    if not teams:
        return {}
    team_norms = sorted(teams.keys(), key=len, reverse=True)
    out: dict[str, str] = {}

    for abbr, torvik_name in _ABBR_ALIASES.items():
        tn = _norm(torvik_name)
        if tn in teams:
            out[abbr.lower()] = tn

    for abbr, display in _espn_abbr_display().items():
        if abbr in out:
            continue
        dn = _norm(display)
        # Longest Torvik name that is a prefix of ESPN display (strips mascots).
        hit = None
        for tn in team_norms:
            if len(tn) < 4:
                continue
            if dn.startswith(tn) or tn.startswith(dn):
                hit = tn
                break
            # St. / State variants
            alt = tn.replace("st", "state") if "st" in tn else tn.replace("state", "st")
            if dn.startswith(alt) or alt.startswith(dn):
                hit = tn
                break
        if hit:
            out[abbr] = hit
    return out


def available_years() -> list[int]:
    if not ASOF_DIR.is_dir():
        return []
    years: list[int] = []
    for path in ASOF_DIR.glob("full_*.csv"):
        try:
            years.append(int(path.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return sorted(years)


def asof_date_for_game(game_date: date) -> date:
    """Torvik archive rows are morning-of ratings for that calendar date."""
    return game_date


def _resolve_team_norm(team_name: str, team_abbr: str, year: int) -> str | None:
    teams = _team_index(year)
    if not teams:
        return None
    abbr = str(team_abbr or "").lower()
    mapped = _abbr_to_torvik_norm(year).get(abbr)
    if mapped:
        return mapped

    targets = {_norm(team_name), _norm(team_abbr)}
    targets.discard("")
    for t in list(targets):
        targets.add(t.replace("state", "st"))
        targets.add(t.replace("st", "state"))
    for t in targets:
        if t in teams:
            return t

    # Longest prefix match against ESPN-style display names
    dn = _norm(team_name) or _norm(_espn_abbr_display().get(abbr, ""))
    if not dn:
        return None
    best = None
    for tn in sorted(teams.keys(), key=len, reverse=True):
        if len(tn) < 4:
            continue
        if dn.startswith(tn) or tn.startswith(dn):
            best = tn
            break
    return best


def lookup(
    *,
    team_name: str,
    team_abbr: str,
    asof: date,
    season: int,
) -> TorvikAsOf | None:
    """Return ratings from the most recent archive morning on/before ``asof``.

    If nothing exists on/before ``asof``, uses the earliest archive morning on/after
    ``asof`` within a 3-day window (covers season-open tips).
    """
    day = None
    year_used = None
    for year in (season, season - 1, season + 1):
        frame = _load_year(year)
        if frame is None or frame.empty:
            continue
        on_or_before = frame[frame["date"] <= asof]
        if not on_or_before.empty:
            asof_date = on_or_before["date"].max()
        else:
            after = frame[(frame["date"] >= asof) & (frame["date"] <= asof + timedelta(days=3))]
            if after.empty:
                continue
            asof_date = after["date"].min()
        day = frame[frame["date"] == asof_date]
        if not day.empty:
            year_used = year
            break
    if day is None or day.empty or year_used is None:
        return None

    team_norm = _resolve_team_norm(team_name, team_abbr, year_used)
    if not team_norm:
        return None
    hit = day[day["team_norm"] == team_norm]
    if hit.empty:
        return None
    row = hit.iloc[0]

    def _f(col: str) -> float | None:
        if col not in row.index:
            return None
        try:
            out = float(row[col])
        except (TypeError, ValueError):
            return None
        if out != out:  # NaN
            return None
        return out

    adj_o = _f("adj_o")
    adj_d = _f("adj_d")
    adj_t = _f("adj_tempo")
    if adj_t is None:
        adj_t = _f("adj_t")
    if adj_t is None:
        # Early-season archive rows omit tempo; league mean pace is a public constant.
        adj_t = 68.0
    barthag = _f("barthag")
    if adj_o is None or adj_d is None or barthag is None:
        return None
    rank = _f("rank")
    if rank is None:
        rank = _f("cur_rk")
    if rank is None:
        rank = 200.0
    wab = _f("wab")
    if wab is None:
        wab = 0.0
    return TorvikAsOf(
        adj_o=adj_o,
        adj_d=adj_d,
        adj_t=adj_t,
        barthag=barthag,
        rank=float(rank),
        wab=float(wab),
    )


def torvik_feature_dict(home: TorvikAsOf, away: TorvikAsOf) -> dict[str, float]:
    return {
        "torvik_adj_o_diff": home.adj_o - away.adj_o,
        "torvik_adj_d_diff": home.adj_d - away.adj_d,
        "torvik_adj_t_diff": home.adj_t - away.adj_t,
        "torvik_barthag_diff": home.barthag - away.barthag,
        "torvik_rank_diff": away.rank - home.rank,
        "torvik_wab_diff": home.wab - away.wab,
        "torvik_home_o_vs_away_d": home.adj_o - away.adj_d,
        "torvik_away_o_vs_home_d": away.adj_o - home.adj_d,
        "torvik_exp_poss": (home.adj_t + away.adj_t) / 2.0,
        "torvik_known": 1.0,
    }


def espn_proxy_torvik_features(
    *,
    home_ortg: float,
    home_drtg: float,
    home_pace: float,
    away_ortg: float,
    away_drtg: float,
    away_pace: float,
    home_elo: float,
    away_elo: float,
) -> dict[str, float]:
    """PIT ESPN walk-forward stand-in when Torvik archive year is missing.

    Uses the same column names so training never invents empty cells; values
    are real pre-tip efficiency estimates from completed games only.
    """
    home_net = home_ortg - home_drtg
    away_net = away_ortg - away_drtg

    def _barth(elo: float) -> float:
        return 1.0 / (1.0 + 10.0 ** (-(elo - 1500.0) / 400.0))

    return {
        "torvik_adj_o_diff": home_ortg - away_ortg,
        "torvik_adj_d_diff": home_drtg - away_drtg,
        "torvik_adj_t_diff": home_pace - away_pace,
        "torvik_barthag_diff": _barth(home_elo) - _barth(away_elo),
        "torvik_rank_diff": (away_elo - home_elo) / 10.0,
        "torvik_wab_diff": (home_net - away_net) / 10.0,
        "torvik_home_o_vs_away_d": home_ortg - away_drtg,
        "torvik_away_o_vs_home_d": away_ortg - home_drtg,
        "torvik_exp_poss": (home_pace + away_pace) / 2.0,
        "torvik_known": 0.0,  # marks ESPN proxy (still real numbers, not NA)
    }
