"""Fetch player OVR lists from public game-rating sites (cached on disk)."""

from __future__ import annotations

import html as html_lib
import http.client
import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Any

from web.league_profiles import LEAGUE_PROFILES
from web.team_registry import fetch_espn_teams

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = PROJECT_ROOT / "data" / "db_ratings_cache"
CACHE_TTL = timedelta(days=7)
EA_FC_CACHE_TTL = timedelta(days=14)
USER_AGENT = "Mozilla/5.0 (compatible; Sports-Odds-Algorithms/2.0)"
REQUEST_DELAY = 0.35
_FUZZY_NAME_THRESHOLD = 0.86

# ESPN displayName → EA FC team.label (licensed naming differs).
SOCCER_ESPN_TO_EA_LABEL: dict[str, str] = {
    "ac milan": "Milano FC",
    "ajax amsterdam": "Ajax",
    "arsenal": "Arsenal",
    "atalanta": "Bergamo Calcio",
    "athletic club": "Athletic Club",
    "bayern munich": "FC Bayern München",
    "borussia monchengladbach": "M'gladbach",
    "borussia mönchengladbach": "M'gladbach",
    "brighton & hove albion": "Brighton & Hove Albion",
    "cf montreal": "CF Montréal",
    "cf montréal": "CF Montréal",
    "fc barcelona": "FC Barcelona",
    "fk qarabag": "Qarabağ FK",
    "internazionale": "Lombardia FC",
    "inter miami cf": "Inter Miami CF",
    "lazio": "Latium",
    "manchester city": "Manchester City",
    "manchester united": "Manchester United",
    "new york red bulls": "Red Bulls",
    "nottingham forest": "Nott'm Forest",
    "paris saint-germain": "Paris SG",
    "psv eindhoven": "PSV",
    "rb leipzig": "RB Leipzig",
    "real madrid": "Real Madrid",
    "red bull new york": "Red Bulls",
    "slavia prague": "Slavia Praha",
    "sporting kansas city": "Sporting KC",
    "tottenham hotspur": "Spurs",
    "union st.-gilloise": "R. Union St.-G.",
    "vancouver whitecaps": "Whitecaps FC",
    "west ham united": "West Ham",
    "wolverhampton wanderers": "Wolves",
}

# ESPN abbr → rating-site slug overrides when auto-slug fails.
TEAM_SLUG_OVERRIDES: dict[str, dict[str, str]] = {
    "nba": {
        "gs": "golden-state-warriors",
        "ny": "new-york-knicks",
        "sa": "san-antonio-spurs",
        "no": "new-orleans-pelicans",
        "utah": "utah-jazz",
        "wsh": "washington-wizards",
    },
    "nfl": {
        "kc": "kansas-city-chiefs",
        "gb": "green-bay-packers",
        "tb": "tampa-bay-buccaneers",
        "sf": "san-francisco-49ers",
        "la": "los-angeles-rams",
    },
    "mlb": {
        "chw": "chicago-white-sox",
        "chc": "chicago-cubs",
        "tb": "tampa-bay-rays",
        "sf": "san-francisco-giants",
        "sd": "san-diego-padres",
        "kc": "kansas-city-royals",
        "stl": "st-louis-cardinals",
        "wsh": "washington-nationals",
        "oak": "athletics",
        "az": "arizona-diamondbacks",
    },
    "nhl": {
        "tb": "tampa-bay-lightning",
        "sj": "san-jose-sharks",
        "la": "los-angeles-kings",
        "nyr": "new-york-rangers",
        "nyi": "new-york-islanders",
        "nj": "new-jersey-devils",
        "wsh": "washington-capitals",
        "stl": "st-louis-blues",
    },
    "wnba": {
        "lv": "las-vegas-aces",
        "gs": "golden-state-valkyries",
        "ny": "new-york-liberty",
        "conn": "connecticut-sun",
        "phx": "phoenix-mercury",
    },
}


@dataclass(frozen=True)
class ScrapedPlayer:
    name: str
    overall: float
    position: str = ""
    rating_source: str = "unknown"
    rating_year: int = 2026
    team_label: str = ""
    nationality: str = ""


@dataclass(frozen=True)
class EaFcPlayer:
    name: str
    overall: float
    position: str
    team_label: str
    nationality: str
    player_id: int = 0


def _fetch_html(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=25) as response:
        return response.read().decode("utf-8", errors="replace")


def _player_cache_path(source: str, team_key: str) -> Path:
    safe = re.sub(r"[^a-z0-9._-]+", "_", team_key.lower())
    return CACHE_DIR / f"{source}-players" / f"{safe}.json"


def _read_player_cache(source: str, team_key: str) -> list[ScrapedPlayer] | None:
    path = _player_cache_path(source, team_key)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    fetched = payload.get("fetched_at")
    if not fetched:
        return None
    ts = datetime.fromisoformat(fetched)
    if datetime.now(timezone.utc) - ts > CACHE_TTL:
        return None
    rows: list[ScrapedPlayer] = []
    for row in payload.get("players") or []:
        try:
            overall = float(row.get("overall"))
        except (TypeError, ValueError):
            continue
        name = (row.get("name") or "").strip()
        if not name:
            continue
        rows.append(
            ScrapedPlayer(
                name=name,
                overall=overall,
                position=(row.get("position") or "").strip().upper(),
                rating_source=(row.get("rating_source") or payload.get("source") or "unknown"),
                rating_year=int(row.get("rating_year") or payload.get("rating_year") or 2026),
            )
        )
    return rows or None


def _write_player_cache(source: str, team_key: str, players: list[ScrapedPlayer], meta: dict[str, Any]) -> None:
    path = _player_cache_path(source, team_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **meta,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "players": [
            {
                "name": p.name,
                "overall": p.overall,
                "position": p.position,
                "rating_source": p.rating_source,
                "rating_year": p.rating_year,
            }
            for p in players
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _norm_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


def _canonical_soccer_label(label: str) -> str:
    key = re.sub(r"[^a-z0-9]", "", (label or "").lower())
    for espn_key, ea_label in SOCCER_ESPN_TO_EA_LABEL.items():
        if key == re.sub(r"[^a-z0-9]", "", espn_key):
            return ea_label
    return label


def _team_label_match(a: str, b: str) -> bool:
    if not a or not b:
        return False
    a = _canonical_soccer_label(a)
    b = _canonical_soccer_label(b)
    na = re.sub(r"[^a-z0-9]", "", a.lower())
    nb = re.sub(r"[^a-z0-9]", "", b.lower())
    if na == nb or na in nb or nb in na:
        return True
    return SequenceMatcher(None, na, nb).ratio() >= 0.82


def _dedupe_players(players: list[tuple[str, str, str]]) -> list[ScrapedPlayer]:
    seen: set[str] = set()
    rows: list[ScrapedPlayer] = []
    for name, ovr, pos in players:
        key = re.sub(r"[^a-z0-9]", "", name.lower())
        if not key or key in seen:
            continue
        seen.add(key)
        try:
            overall = float(ovr)
        except ValueError:
            continue
        if overall <= 0:
            continue
        rows.append(ScrapedPlayer(name=name.strip(), overall=round(min(99.0, overall), 1), position=pos))
    return rows


def _parse_2k_table(html: str) -> list[tuple[str, str, str]]:
    chunk_start = html.find('style="min-width: 200px;">Player</th>')
    if chunk_start < 0:
        chunk_start = html.find(">Player</th>")
    chunk = html[chunk_start : chunk_start + 180000] if chunk_start > 0 else html
    return [
        (name, ovr, "")
        for name, ovr in re.findall(
            r"<tr><td>\d+</td><td>([^<]+)</td><td class=\"text-center\">\s*"
            r"<span data-order=\"[\d.]+\"[^>]*>(\d{2,3})</span>",
            chunk,
        )
    ]


def _parse_ea_site_table(html: str) -> list[tuple[str, str, str]]:
    """Madden / NHL / The Show roster tables."""
    rows: list[tuple[str, str, str]] = []
    for block in re.findall(r"<tr>.*?</tr>", html, re.S):
        name_match = re.search(
            r'<span class="entry-font[^"]*">\s*<a[^>]+>([^<]+)</a>',
            block,
        )
        if not name_match:
            continue
        ovr_match = re.search(
            r'<span class="[\d.]+ attribute-box[^"]*">(\d{2,3})</span>',
            block,
        )
        if not ovr_match:
            continue
        pos_match = re.search(r"#\d+\s*<a[^>]+>([A-Z0-9/]+)</a>", block)
        if not pos_match:
            pos_match = re.search(r'title="([^"]+)">([A-Z]{1,3})</a>\s*\|', block)
        pos = pos_match.group(1) if pos_match else ""
        if pos_match and pos_match.lastindex and pos_match.lastindex >= 2:
            pos = pos_match.group(2) or pos
        rows.append((name_match.group(1).strip(), ovr_match.group(1), pos))
    return rows


def _parse_json_ld(html: str, rating_label: str = "") -> list[tuple[str, str, str]]:
    if rating_label:
        pattern = (
            r'"name":\s*"([^"]+)"[^}]*"name":\s*"' + re.escape(rating_label) + r'"[^}]*"value":\s*(\d+)'
        )
        rows = [(html_lib.unescape(name), ovr, "") for name, ovr in re.findall(pattern, html)]
        if rows:
            return rows
    rows = []
    for name, ovr in re.findall(
        r'"name":\s*"([^"]+)"[^}]*"name":\s*"[^"]*2K\d+ Rating"[^}]*"value":\s*(\d+)',
        html,
    ):
        rows.append((html_lib.unescape(name), ovr, ""))
    return rows


def _team_slug(league: str, team_abbr: str) -> str | None:
    abbr = (team_abbr or "").lower()
    overrides = TEAM_SLUG_OVERRIDES.get(league, {})
    if abbr in overrides:
        return overrides[abbr]
    for team in fetch_espn_teams(league):
        if team["abbr"] == abbr:
            return team["slug"]
    return None


def scrape_2k_team(league: str, team_abbr: str, *, wnba: bool = False) -> list[ScrapedPlayer]:
    slug = _team_slug(league, team_abbr)
    if not slug:
        return []
    source = "2kratings-wnba" if wnba else "2kratings-nba"
    cached = _read_player_cache(source, slug)
    if cached:
        return cached

    prefix = "/wnba-teams" if wnba else "/teams"
    url = f"https://www.2kratings.com{prefix}/{slug}"
    try:
        html = _fetch_html(url)
    except (urllib.error.URLError, TimeoutError):
        return []

    players = _dedupe_players(_parse_2k_table(html))
    if not players:
        players = _dedupe_players(_parse_json_ld(html))
    for idx, player in enumerate(players):
        players[idx] = ScrapedPlayer(
            name=html_lib.unescape(player.name),
            overall=player.overall,
            position=player.position,
            rating_source="2k",
            rating_year=2026,
        )
    if players:
        _write_player_cache(
            source,
            slug,
            players,
            {"source": "2kratings.com", "team_slug": slug, "url": url},
        )
    return players


def scrape_madden_team(team_abbr: str) -> list[ScrapedPlayer]:
    slug = _team_slug("nfl", team_abbr)
    if not slug:
        return []
    source = "maddenratings-players"
    cached = _read_player_cache(source, slug)
    if cached:
        return cached

    url = f"https://www.maddenratings.com/teams/{slug}"
    try:
        html = _fetch_html(url)
    except (urllib.error.URLError, TimeoutError):
        return []

    idx = html.find("Below are the")
    chunk = html[idx : idx + 250000] if idx > 0 else html
    players = _dedupe_players(_parse_ea_site_table(chunk))
    for idx, player in enumerate(players):
        players[idx] = ScrapedPlayer(
            name=player.name,
            overall=player.overall,
            position=player.position,
            rating_source="madden",
            rating_year=2026,
        )
    if players:
        _write_player_cache(source, slug, players, {"source": "maddenratings.com", "team_slug": slug, "url": url})
    return players


def scrape_theshow_team(league: str, team_abbr: str) -> list[ScrapedPlayer]:
    slug = _team_slug(league, team_abbr) or _team_slug("mlb", team_abbr)
    if not slug:
        return []
    source = "theshowratings-players"
    cached = _read_player_cache(source, slug)
    if cached:
        return cached

    url = f"https://www.theshowratings.com/teams/{slug}"
    try:
        html = _fetch_html(url)
    except (urllib.error.URLError, TimeoutError):
        return []

    players = _dedupe_players(_parse_ea_site_table(html))
    for idx, player in enumerate(players):
        players[idx] = ScrapedPlayer(
            name=player.name,
            overall=player.overall,
            position=player.position,
            rating_source="mlb_ts",
            rating_year=2026,
        )
    if players:
        _write_player_cache(source, slug, players, {"source": "theshowratings.com", "team_slug": slug, "url": url})
    return players


def scrape_nhl_team(team_abbr: str) -> list[ScrapedPlayer]:
    slug = _team_slug("nhl", team_abbr)
    if not slug:
        return []
    source = "nhlratings-players"
    cached = _read_player_cache(source, slug)
    if cached:
        return cached

    url = f"https://www.nhlratings.net/teams/{slug}"
    try:
        html = _fetch_html(url)
    except (urllib.error.URLError, TimeoutError):
        return []

    players = _dedupe_players(_parse_ea_site_table(html))
    for idx, player in enumerate(players):
        players[idx] = ScrapedPlayer(
            name=player.name,
            overall=player.overall,
            position=player.position,
            rating_source="nhl",
            rating_year=2026,
        )
    if players:
        _write_player_cache(source, slug, players, {"source": "nhlratings.net", "team_slug": slug, "url": url})
    return players


# League → scrape function (league, abbr) -> players
SCRAPE_HANDLERS: dict[str, Any] = {}


def _register_handlers() -> None:
    if SCRAPE_HANDLERS:
        return
    for league in ("nba", "cbb"):
        SCRAPE_HANDLERS[league] = lambda lg, abbr, _l=league: scrape_2k_team(_l, abbr)
    SCRAPE_HANDLERS["wnba"] = lambda lg, abbr: scrape_2k_team("wnba", abbr, wnba=True)
    for league in ("nfl", "cfb"):
        SCRAPE_HANDLERS[league] = lambda lg, abbr, _l=league: scrape_madden_team(abbr)
    for league in ("nhl", "ncaah", "ncaawh"):
        SCRAPE_HANDLERS[league] = lambda lg, abbr, _l=league: scrape_nhl_team(abbr)
    for league in ("mlb", "ncaabb", "dwl", "pwl", "vwl", "lmp", "wbc"):
        SCRAPE_HANDLERS[league] = lambda lg, abbr, _l=league: scrape_theshow_team(_l, abbr)
    for league in LEAGUE_PROFILES:
        if LEAGUE_PROFILES[league]["category"] == "soccer":
            SCRAPE_HANDLERS[league] = lambda lg, abbr: scrape_ea_fc_team(lg, abbr)


def _ea_fc_cache_path() -> Path:
    return CACHE_DIR / "ea-fc-all-players.json"


def _ea_fc_build_id() -> str | None:
    try:
        html = _fetch_html("https://www.ea.com/games/ea-sports-fc/ratings")
    except (urllib.error.URLError, TimeoutError):
        return None
    m = re.search(r'"buildId":"([^"]+)"', html)
    if m:
        return m.group(1)
    m = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.+?)</script>', html, re.S)
    if m:
        return json.loads(m.group(1)).get("buildId")
    return None


def _parse_ea_fc_item(item: dict[str, Any]) -> EaFcPlayer | None:
    try:
        overall = float(item.get("overallRating") or 0)
    except (TypeError, ValueError):
        return None
    if overall <= 0:
        return None
    name = item.get("commonName") or f"{item.get('firstName', '')} {item.get('lastName', '')}".strip()
    if not name:
        return None
    pos = item.get("position") or ""
    if isinstance(pos, dict):
        pos = pos.get("abbr") or pos.get("name") or ""
    team = item.get("team") or {}
    team_label = ""
    if isinstance(team, dict):
        team_label = str(team.get("label") or "")
    nat = item.get("nationality") or {}
    nationality = ""
    if isinstance(nat, dict):
        nationality = str(nat.get("label") or nat.get("name") or "")
    return EaFcPlayer(
        name=html_lib.unescape(str(name).strip()),
        overall=round(min(99.0, overall), 1),
        position=str(pos).strip().upper(),
        team_label=team_label,
        nationality=nationality,
        player_id=int(item.get("id") or 0),
    )


def fetch_ea_fc_all_players(*, force: bool = False) -> list[EaFcPlayer]:
    """Download full EA FC 26 ratings database (~18k players) from ea.com."""
    path = _ea_fc_cache_path()
    if not force and path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            fetched = payload.get("fetched_at")
            if fetched:
                ts = datetime.fromisoformat(fetched)
                if datetime.now(timezone.utc) - ts <= EA_FC_CACHE_TTL:
                    rows = []
                    for row in payload.get("players") or []:
                        parsed = _parse_ea_fc_item(row)
                        if parsed:
                            rows.append(parsed)
                    if rows:
                        return rows
        except (json.JSONDecodeError, OSError, ValueError):
            pass

    build_id = _ea_fc_build_id()
    if not build_id:
        return []

    all_items: list[dict[str, Any]] = []
    page = 1
    total_items = None
    while True:
        url = f"https://www.ea.com/_next/data/{build_id}/games/ea-sports-fc/ratings.json?page={page}"
        items: list[dict[str, Any]] = []
        for attempt in range(4):
            try:
                request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(request, timeout=60) as response:
                    data = json.loads(response.read().decode("utf-8", errors="replace"))
                details = data.get("pageProps", {}).get("ratingDetails") or {}
                items = details.get("items") or []
                if total_items is None:
                    total_items = int(details.get("totalItems") or 0)
                break
            except (
                urllib.error.URLError,
                TimeoutError,
                json.JSONDecodeError,
                ValueError,
                http.client.IncompleteRead,
            ):
                if attempt >= 3:
                    break
                time.sleep(1.5 * (attempt + 1))
        if not items:
            break
        all_items.extend(items)
        if total_items and len(all_items) >= total_items:
            break
        if len(items) < 50:
            break
        page += 1
        time.sleep(REQUEST_DELAY)

    players = [p for item in all_items if (p := _parse_ea_fc_item(item))]
    if players:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "source": "ea.com",
                    "url": "https://www.ea.com/games/ea-sports-fc/ratings",
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                    "count": len(players),
                    "players": [
                        {
                            "id": p.player_id,
                            "commonName": p.name,
                            "overallRating": p.overall,
                            "position": p.position,
                            "team": {"label": p.team_label},
                            "nationality": {"label": p.nationality},
                        }
                        for p in players
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return players


@lru_cache(maxsize=1)
def _ea_fc_player_index() -> tuple[EaFcPlayer, ...]:
    return tuple(fetch_ea_fc_all_players())


def _espn_team_label(league: str, team_abbr: str) -> str:
    abbr = (team_abbr or "").lower()
    for team in fetch_espn_teams(league):
        if team["abbr"] == abbr:
            label = team.get("label") or team.get("slug") or abbr
            return _canonical_soccer_label(label)
    return abbr


def _ea_fc_team_players(league: str, team_abbr: str) -> list[EaFcPlayer]:
    label = _espn_team_label(league, team_abbr)
    return [p for p in _ea_fc_player_index() if _team_label_match(p.team_label, label)]


def _fuzzy_match_ea_fc(
    name: str,
    *,
    team_label: str = "",
    nationality: str = "",
    allow_global: bool = True,
) -> EaFcPlayer | None:
    norm = _norm_name(name)
    if not norm:
        return None
    pool = list(_ea_fc_player_index())
    scoped_team = team_label
    if team_label:
        team_pool = [p for p in pool if _team_label_match(p.team_label, team_label)]
        if team_pool:
            pool = team_pool
    exact = [p for p in pool if _norm_name(p.name) == norm]
    if len(exact) == 1:
        return exact[0]
    if exact:
        return max(exact, key=lambda p: p.overall)
    best: EaFcPlayer | None = None
    best_score = 0.0
    for player in pool:
        score = SequenceMatcher(None, norm, _norm_name(player.name)).ratio()
        if score > best_score:
            best_score = score
            best = player
    if best and best_score >= _FUZZY_NAME_THRESHOLD:
        return best
    if scoped_team and allow_global:
        return _fuzzy_match_ea_fc(name, nationality=nationality, allow_global=False)
    if nationality:
        nat_pool = [p for p in _ea_fc_player_index() if _team_label_match(p.nationality, nationality)]
        for player in nat_pool:
            if _norm_name(player.name) == norm:
                return player
    return None


def scrape_ea_fc_team(league: str, team_abbr: str) -> list[ScrapedPlayer]:
    """EA FC roster for one club from the full ea.com ratings database."""
    source = "ea-fc-team"
    team_key = f"{league}-{team_abbr}".lower()
    cached = _read_player_cache(source, team_key)
    if cached:
        return cached

    rows: list[ScrapedPlayer] = []
    for player in _ea_fc_team_players(league, team_abbr):
        rows.append(
            ScrapedPlayer(
                name=player.name,
                overall=player.overall,
                position=player.position,
                rating_source="fc",
                rating_year=2026,
                team_label=player.team_label,
                nationality=player.nationality,
            )
        )
    if rows:
        _write_player_cache(
            source,
            team_key,
            rows,
            {"source": "ea.com", "league": league, "team_abbr": team_abbr},
        )
    return rows


def scrape_ea_fc_global() -> list[ScrapedPlayer]:
    """Full EA FC player pool (all clubs)."""
    source = "ea-fc-global"
    cached = _read_player_cache(source, "global")
    if cached:
        return cached

    players = [
        ScrapedPlayer(
            name=p.name,
            overall=p.overall,
            position=p.position,
            rating_source="fc",
            rating_year=2026,
            team_label=p.team_label,
            nationality=p.nationality,
        )
        for p in _ea_fc_player_index()
    ]
    if players:
        _write_player_cache(
            source,
            "global",
            players,
            {"source": "ea.com", "url": "https://www.ea.com/games/ea-sports-fc/ratings"},
        )
    return players


def fuzzy_search_external_player(
    league: str,
    player_name: str,
    *,
    team_abbr: str | None = None,
    nationality: str = "",
) -> ScrapedPlayer | None:
    """Last-resort fuzzy lookup in publisher databases (EA FC global pool, etc.)."""
    league = (league or "").lower()
    category = LEAGUE_PROFILES.get(league, {}).get("category")
    if category == "soccer":
        team_label = _espn_team_label(league, team_abbr or "") if team_abbr else ""
        hit = _fuzzy_match_ea_fc(player_name, team_label=team_label, nationality=nationality)
        if hit:
            return ScrapedPlayer(
                name=hit.name,
                overall=hit.overall,
                position=hit.position,
                rating_source="fc",
                rating_year=2026,
                team_label=hit.team_label,
                nationality=hit.nationality,
            )
    return None


_register_handlers()


def scrape_league_team_players(league: str, team_abbr: str) -> list[ScrapedPlayer]:
    """Return cached/scraped player OVRs for one team."""
    league = (league or "").lower()
    handler = SCRAPE_HANDLERS.get(league)
    if not handler:
        return []
    try:
        return handler(league, team_abbr)
    except Exception:
        return []


def scrape_source_for_league(league: str) -> str | None:
    league = league.lower()
    category = LEAGUE_PROFILES.get(league, {}).get("category")
    if league in {"nba", "wnba", "cbb"}:
        return "2k"
    if league in {"nfl", "cfb"}:
        return "madden"
    if category == "hockey":
        return "nhl"
    if category == "baseball":
        return "mlb_ts"
    if category == "soccer":
        return "fc"
    return None


def clear_scrape_cache() -> None:
    """No-op placeholder for tests (disk cache persists)."""
    return None


__all__ = [
    "EaFcPlayer",
    "ScrapedPlayer",
    "fetch_ea_fc_all_players",
    "fuzzy_search_external_player",
    "scrape_ea_fc_global",
    "scrape_ea_fc_team",
    "scrape_league_team_players",
    "scrape_source_for_league",
    "clear_scrape_cache",
]
