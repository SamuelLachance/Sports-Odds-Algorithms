"""Third-party game/sim overall ratings (2K, Madden, NHL, MLB The Show, EA FC, FM).

Curated CSV files live in data/ratings/ and are matched to ESPN rosters by name + team.
Update those CSVs periodically from published game rating lists — do not derive ratings
from ESPN season stats.

To expand coverage: add rows to the league CSV (player_name, team_abbr, position, overall)
or drop a new file and register it in LEAGUE_RATING_FILES below.
"""

from __future__ import annotations

import csv
import html
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Any

from web.league_profiles import LEAGUE_PROFILES, SUPPORTED_LEAGUES
from web.sports_db.rating_scrape import ScrapedPlayer, fuzzy_search_external_player, scrape_league_team_players

RATINGS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "ratings"

# League → curated rating file (versioned CSV in data/ratings/).
LEAGUE_RATING_FILES: dict[str, str] = {
    "nba": "2k_nba_2026.csv",
    "wnba": "2k_wnba_2026.csv",
    "cbb": "2k_cbb_2026.csv",
    "nfl": "madden_nfl_2026.csv",
    "cfb": "cfb_2026.csv",
    "nhl": "nhl_2026.csv",
    "ncaah": "nhl_ncaa_2026.csv",
    "ncaawh": "nhl_ncaa_2026.csv",
    "mlb": "mlb_show_2026.csv",
    "ncaabb": "mlb_ncaa_2026.csv",
    "dwl": "mlb_winter_2026.csv",
    "pwl": "mlb_winter_2026.csv",
    "vwl": "mlb_winter_2026.csv",
    "lmp": "mlb_winter_2026.csv",
    "wbc": "mlb_wbc_2026.csv",
    "epl": "fc_2026.csv",
    "mls": "fc_2026.csv",
    "laliga": "fc_2026.csv",
    "bundesliga": "fc_2026.csv",
    "seriea": "fc_2026.csv",
    "ligue1": "fc_2026.csv",
    "ucl": "fc_2026.csv",
    "worldcup": "fc_intl_2026.csv",
    "fifa_friendlies": "fc_intl_2026.csv",
    "concacaf_wcq": "fc_intl_2026.csv",
    "concacaf_gold": "fc_intl_2026.csv",
    "concacaf_nations": "fc_intl_2026.csv",
    "uefa_euro": "fc_intl_2026.csv",
    "uefa_nations": "fc_intl_2026.csv",
    "copa_america": "fc_intl_2026.csv",
}

# ESPN abbr aliases → canonical abbr used in CSV rows.
TEAM_ABBR_ALIASES: dict[str, dict[str, str]] = {
    "nba": {"gsw": "gs", "sas": "sa", "nop": "no", "nyk": "ny", "was": "wsh", "utah": "utah"},
    "nfl": {"lar": "la", "lv": "lv"},
    "mlb": {"az": "az", "chw": "chw", "tb": "tb", "sf": "sf", "sd": "sd", "stl": "stl", "wsh": "wsh", "oak": "oak"},
    "nhl": {"njd": "nj", "tbl": "tb", "sjs": "sj", "lak": "la", "nyr": "nyr", "nyi": "nyi", "wsh": "wsh"},
    "wnba": {"gsv": "gs", "lv": "lv", "conn": "conn", "phx": "phx", "ny": "ny"},
    "epl": {"manu": "mun", "manc": "mci", "tot": "tot", "new": "new"},
    "mls": {"la": "la", "lafc": "lafc", "atl": "atl", "mia": "mia"},
}

_NAME_SUFFIXES = re.compile(
    r"\b(jr\.?|sr\.?|ii+|iii+|iv|v)\.?\s*$",
    re.I,
)

RATING_BASELINE = 50.0
RATING_PRIOR_SOURCE = "prior"
RATING_DERIVED_SOURCE = "derived"

SOURCE_LABELS: dict[str, str] = {
    "2k": "2K",
    "madden": "Madden",
    "nhl": "NHL",
    "mlb_ts": "MLB The Show",
    "fc": "EA FC",
    "fm": "FM",
    "cfb": "CFB 25",
    "cbb": "2K",
    RATING_DERIVED_SOURCE: "Team est.",
    RATING_PRIOR_SOURCE: "Missing",
}

_FUZZY_TEAM_THRESHOLD = 0.86
_FUZZY_LEAGUE_THRESHOLD = 0.90
_FUZZY_SOCCER_THRESHOLD = 0.88


@dataclass(frozen=True)
class PlayerRatingResult:
    rating: float
    rating_source: str
    rating_year: int
    matched: bool


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def normalize_player_name(name: str) -> str:
    """Lowercase alphanumeric key for exact lookups."""
    cleaned = html.unescape(name or "")
    cleaned = _strip_accents(cleaned)
    cleaned = _NAME_SUFFIXES.sub("", cleaned.strip())
    return re.sub(r"[^a-z0-9]", "", cleaned.lower())


def _name_tokens(name: str) -> list[str]:
    cleaned = _strip_accents(name or "")
    parts = re.split(r"[\s.\-'']+", cleaned.strip())
    return [p.lower() for p in parts if p]


def name_signature(name: str) -> str | None:
    """Last name + first initial, e.g. 'Natasha Howard' → 'howard|n'."""
    tokens = _name_tokens(name)
    if len(tokens) < 2:
        return None
    first = tokens[0]
    last = tokens[-1]
    if len(first) == 1 or (len(first) == 2 and first.endswith(".")):
        initial = first[0]
    else:
        initial = first[0]
    return f"{last}|{initial}"


def normalize_team_abbr(abbr: str | None, league: str | None = None) -> str:
    team = re.sub(r"[^a-z0-9]", "", (abbr or "").lower())
    if league:
        aliases = TEAM_ABBR_ALIASES.get(league.lower(), {})
        team = aliases.get(team, team)
    return team


def _parse_overall(raw: str) -> float | None:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    # Game OVR is 0–99; keep on same scale for tier colors (elite ≈ 85+).
    return round(min(99.0, max(1.0, value)), 1)


@dataclass(frozen=True)
class _RatingRow:
    player_name: str
    team_abbr: str
    position: str
    overall: float
    espn_id: str
    rating_source: str
    rating_year: int
    norm_name: str
    signature: str | None


def _load_csv(path: Path) -> list[_RatingRow]:
    if not path.is_file():
        return []
    rows: list[_RatingRow] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for record in reader:
            overall = _parse_overall(record.get("overall", ""))
            if overall is None:
                continue
            name = (record.get("player_name") or "").strip()
            if not name:
                continue
            try:
                year = int(record.get("rating_year") or 0)
            except ValueError:
                year = 0
            rows.append(
                _RatingRow(
                    player_name=name,
                    team_abbr=normalize_team_abbr(record.get("team_abbr")),
                    position=(record.get("position") or "").strip().upper(),
                    overall=overall,
                    espn_id=str(record.get("espn_id") or "").strip(),
                    rating_source=(record.get("rating_source") or "unknown").strip().lower(),
                    rating_year=year,
                    norm_name=normalize_player_name(name),
                    signature=name_signature(name),
                )
            )
    return rows


@lru_cache(maxsize=32)
def _league_table(league: str) -> tuple[_RatingRow, ...]:
    league = league.lower()
    filename = LEAGUE_RATING_FILES.get(league)
    if not filename:
        return ()
    return tuple(_load_csv(RATINGS_DIR / filename))


def _scraped_to_rows(
    league: str,
    team_abbr: str,
    scraped: list[ScrapedPlayer],
) -> tuple[_RatingRow, ...]:
    team = normalize_team_abbr(team_abbr, league)
    rows: list[_RatingRow] = []
    for player in scraped:
        rows.append(
            _RatingRow(
                player_name=player.name,
                team_abbr=team,
                position=player.position,
                overall=player.overall,
                espn_id="",
                rating_source=player.rating_source,
                rating_year=player.rating_year,
                norm_name=normalize_player_name(player.name),
                signature=name_signature(player.name),
            )
        )
    return tuple(rows)


@lru_cache(maxsize=256)
def _team_scraped_table(league: str, team_abbr: str) -> tuple[_RatingRow, ...]:
    league = league.lower()
    if league not in SUPPORTED_LEAGUES:
        return ()
    scraped = scrape_league_team_players(league, team_abbr or "")
    return _scraped_to_rows(league, team_abbr, scraped)


def _combined_team_rows(league: str, team_abbr: str | None) -> tuple[_RatingRow, ...]:
    league = league.lower()
    team = normalize_team_abbr(team_abbr, league)
    csv_rows = _league_table(league)
    if team:
        team_rows = [row for row in csv_rows if row.team_abbr == team]
    else:
        team_rows = list(csv_rows)
    scraped = _team_scraped_table(league, team) if team else ()
    if not scraped:
        return tuple(team_rows) if team else tuple(csv_rows)
    seen = {row.norm_name for row in team_rows}
    merged = list(team_rows)
    for row in scraped:
        if row.norm_name not in seen:
            merged.append(row)
            seen.add(row.norm_name)
    return tuple(merged)


def _combined_league_rows(league: str) -> tuple[_RatingRow, ...]:
    return _league_table(league)


def _fuzzy_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _position_adjustment(
    league: str,
    position: str | None,
    roster_meta: dict[str, Any] | None,
    *,
    scale: float = 1.0,
) -> float:
    """Small offset from team/league baseline by position and experience — no ESPN stats."""
    category = LEAGUE_PROFILES.get(league.lower(), {}).get("category", "basketball")
    meta = roster_meta or {}
    delta = 0.0

    try:
        exp = float(meta.get("experience") or 0)
    except (TypeError, ValueError):
        exp = 0.0
    delta += min(4.0, exp * 0.4) * scale

    pos = (position or "").upper()
    if category == "football":
        if pos in {"QB"}:
            delta += 2.5 * scale
        elif pos in {"WR", "RB", "TE", "EDGE", "DE", "CB", "LB"}:
            delta += 1.0 * scale
        elif pos in {"K", "P", "LS"}:
            delta -= 2.0 * scale
    elif category == "baseball":
        if pos in {"SP", "RP", "P"}:
            delta += 1.0 * scale
        elif pos in {"C"}:
            delta += 0.5 * scale
    elif category == "hockey":
        if pos in {"G", "GK"}:
            delta += 1.5 * scale
        elif pos in {"C", "LW", "RW", "D"}:
            delta += 0.5 * scale
    elif category == "basketball":
        if pos in {"G", "PG", "SG"}:
            delta += 0.75 * scale
        elif pos in {"F", "SF", "PF", "C"}:
            delta += 0.5 * scale
    elif category == "soccer":
        if pos in {"F", "FW", "ST", "CF"}:
            delta += 1.0 * scale
        elif pos in {"M", "MF", "AM", "CM", "DM"}:
            delta += 0.5 * scale

    try:
        age = float(meta.get("age") or 0)
        if 24 <= age <= 30:
            delta += 0.5 * scale
        elif age > 0 and age < 22:
            delta -= 0.5 * scale
    except (TypeError, ValueError):
        pass

    return delta


def _roster_prior(
    league: str,
    position: str | None,
    roster_meta: dict[str, Any] | None,
) -> float:
    """Conservative estimate when no game rating exists — never uses ESPN stats."""
    base = RATING_BASELINE + _position_adjustment(league, position, roster_meta)
    return round(max(42.0, min(58.0, base)), 1)


def team_external_roster_stats(
    league: str,
    team_abbr: str,
) -> dict[str, Any] | None:
    """Average OVR and metadata from CSV + live-scraped rows for one team."""
    league = (league or "").lower()
    team = normalize_team_abbr(team_abbr, league)
    if not team:
        return None
    rows = list(_combined_team_rows(league, team))
    if not rows:
        return None
    ovrs = [row.overall for row in rows]
    sources = {row.rating_source for row in rows if row.rating_source}
    source = next(iter(sources)) if len(sources) == 1 else "curated"
    year = max((row.rating_year for row in rows if row.rating_year), default=_default_year_for_league(league))
    return {
        "team_abbr": team,
        "average": round(sum(ovrs) / len(ovrs), 1),
        "count": len(rows),
        "rating_source": source,
        "rating_year": year,
    }


def _default_year_for_league(league: str) -> int:
    rows = _league_table(league)
    if rows:
        return max(row.rating_year for row in rows if row.rating_year)
    return 2026


def _match_row(
    rows: tuple[_RatingRow, ...],
    *,
    player_name: str,
    team_abbr: str | None,
    espn_id: str | None,
    league: str | None = None,
) -> _RatingRow | None:
    if not rows:
        return None

    norm = normalize_player_name(player_name)
    sig = name_signature(player_name)
    team = normalize_team_abbr(team_abbr, league)
    pid = str(espn_id or "").strip()
    category = LEAGUE_PROFILES.get((league or "").lower(), {}).get("category", "")
    league_threshold = _FUZZY_SOCCER_THRESHOLD if category == "soccer" else _FUZZY_LEAGUE_THRESHOLD

    if pid:
        for row in rows:
            if row.espn_id and row.espn_id == pid:
                return row

    if norm:
        team_hits = [row for row in rows if row.norm_name == norm and (not team or row.team_abbr == team)]
        if team_hits:
            return team_hits[0]
        any_hits = [row for row in rows if row.norm_name == norm]
        if any_hits:
            return any_hits[0]

    if sig and team:
        sig_hits = [row for row in rows if row.signature == sig and row.team_abbr == team]
        if len(sig_hits) == 1:
            return sig_hits[0]
        if not sig_hits:
            sig_hits = [row for row in rows if row.signature == sig]
            if len(sig_hits) == 1:
                return sig_hits[0]

    if norm and team:
        best: _RatingRow | None = None
        best_score = 0.0
        for row in rows:
            if row.team_abbr != team:
                continue
            score = _fuzzy_ratio(norm, row.norm_name)
            if score > best_score:
                best_score = score
                best = row
        if best and best_score >= _FUZZY_TEAM_THRESHOLD:
            return best

    if norm:
        best = None
        best_score = 0.0
        for row in rows:
            score = _fuzzy_ratio(norm, row.norm_name)
            if score > best_score:
                best_score = score
                best = row
        if best and best_score >= league_threshold:
            return best

    return None


def match_external_rating(
    league: str,
    player_name: str | None,
    team_abbr: str | None = None,
    espn_id: str | None = None,
    *,
    nationality: str = "",
) -> PlayerRatingResult | None:
    """Return a game OVR when CSV or live-scraped roster matches."""
    league = (league or "").lower()
    row = _match_row(
        _combined_team_rows(league, team_abbr),
        player_name=player_name or "",
        team_abbr=team_abbr,
        espn_id=espn_id,
        league=league,
    )
    if not row:
        row = _match_row(
            _combined_league_rows(league),
            player_name=player_name or "",
            team_abbr=team_abbr,
            espn_id=espn_id,
            league=league,
        )
    if not row:
        fuzzy = fuzzy_search_external_player(
            league,
            player_name or "",
            team_abbr=team_abbr,
            nationality=nationality,
        )
        if fuzzy:
            row = _RatingRow(
                player_name=fuzzy.name,
                team_abbr=normalize_team_abbr(team_abbr, league),
                position=fuzzy.position,
                overall=fuzzy.overall,
                espn_id="",
                rating_source=fuzzy.rating_source,
                rating_year=fuzzy.rating_year,
                norm_name=normalize_player_name(fuzzy.name),
                signature=name_signature(fuzzy.name),
            )
    if not row:
        return None
    return PlayerRatingResult(
        rating=row.overall,
        rating_source=row.rating_source,
        rating_year=row.rating_year or _default_year_for_league(league),
        matched=True,
    )


def derived_team_rating(
    league: str,
    team_abbr: str | None,
    *,
    position: str | None = None,
    roster_meta: dict[str, Any] | None = None,
) -> PlayerRatingResult | None:
    """Estimate from curated team roster average plus position offset."""
    league = (league or "").lower()
    stats = team_external_roster_stats(league, team_abbr or "")
    if not stats or stats["count"] < 3:
        return None
    team_avg = stats["average"]
    pos_delta = _position_adjustment(league, position, roster_meta, scale=0.55)
    rating = round(max(40.0, min(88.0, team_avg - 2.0 + pos_delta)), 1)
    return PlayerRatingResult(
        rating=rating,
        rating_source=RATING_DERIVED_SOURCE,
        rating_year=stats["rating_year"],
        matched=False,
    )


def conservative_prior_rating(
    league: str,
    position: str | None = None,
    roster_meta: dict[str, Any] | None = None,
) -> PlayerRatingResult:
    """Conservative 50± estimate when no game OVR or team CSV context exists."""
    league = (league or "").lower()
    prior = _roster_prior(league, position, roster_meta)
    return PlayerRatingResult(
        rating=prior,
        rating_source=RATING_PRIOR_SOURCE,
        rating_year=_default_year_for_league(league),
        matched=False,
    )


def team_csv_ovr(league: str, team_abbr: str, *, top_n: int = 8) -> dict[str, Any] | None:
    """Top-N average OVR from CSV + scrape for predictions."""
    league = (league or "").lower()
    team = normalize_team_abbr(team_abbr, league)
    if not team:
        return None
    rows = list(_combined_team_rows(league, team))
    if len(rows) < 5:
        return None
    ovrs = sorted((row.overall for row in rows), reverse=True)[:top_n]
    if not ovrs:
        return None
    sources = {row.rating_source for row in rows if row.rating_source}
    source = next(iter(sources)) if len(sources) == 1 else "curated"
    year = max((row.rating_year for row in rows if row.rating_year), default=_default_year_for_league(league))
    return {
        "source": f"data/ratings ({SOURCE_LABELS.get(source, source)})",
        "team_abbr": team,
        "rating": round(sum(ovrs) / len(ovrs), 2),
        "method": f"csv_top{top_n}_ovr",
        "players_matched": len(rows),
        "rating_year": year,
    }


def lookup_player_rating(
    league: str,
    player_name: str | None,
    team_abbr: str | None = None,
    espn_id: str | None = None,
    *,
    position: str | None = None,
    roster_meta: dict[str, Any] | None = None,
    nationality: str = "",
) -> PlayerRatingResult:
    """Resolve a 0–99 game-style overall from external publisher data only."""
    league = (league or "").lower()
    meta = roster_meta or {}
    nat = nationality or str(meta.get("nationality") or meta.get("country") or "")
    hit = match_external_rating(
        league,
        player_name,
        team_abbr=team_abbr,
        espn_id=espn_id,
        nationality=nat,
    )
    if hit:
        return hit
    derived = derived_team_rating(
        league,
        team_abbr,
        position=meta.get("position"),
        roster_meta=meta,
    )
    if derived:
        return derived
    return conservative_prior_rating(league, meta.get("position"), meta)


def rating_source_label(source: str | None, year: int | None = None) -> str:
    """Human label for UI tooltips, e.g. 2K '26."""
    key = (source or RATING_PRIOR_SOURCE).lower()
    label = SOURCE_LABELS.get(key, key.upper())
    if year:
        return f"{label} '{str(year)[-2:]}"
    return label


def clear_rating_cache() -> None:
    """Drop cached CSV/scrape tables (for tests)."""
    _league_table.cache_clear()
    _team_scraped_table.cache_clear()
