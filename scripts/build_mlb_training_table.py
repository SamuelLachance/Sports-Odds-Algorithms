"""Build the MLB v2 training table from cached statsapi history + enrichments.

Hard-gates FEATURE_COLUMNS: zero NaNs. Missing opens/weather/ump/IL/statcast use
has_* flags and neutral defaults from MlbFeatureEngine — never invent open=close.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.bet_advisor import devig_two_way_probs  # noqa: E402
from web.closing_odds_db import closing_odds_lookup  # noqa: E402
from web.mlb_stats_api import ESPN_TO_MLB_TEAM_ID  # noqa: E402
from web.mlb_v2.feature_engine import FEATURE_COLUMNS, MlbFeatureEngine  # noqa: E402
from web.mlb_v2.replay import replay_season  # noqa: E402

CACHE_ROOT = PROJECT_ROOT / ".build-cache" / "mlb-history"
OUT_DIR = PROJECT_ROOT / "data" / "mlb_history"
WEATHER_OPENMETEO = PROJECT_ROOT / "data" / "supplemental" / "mlb-weather" / "game_weather.csv"
WEATHER_MLB = PROJECT_ROOT / "data" / "supplemental" / "mlb-weather" / "mlb_api_weather.csv"
ODDS_CSV = PROJECT_ROOT / "data" / "supplemental" / "closing-odds" / "mlb.csv"
UMP_CSV = PROJECT_ROOT / "data" / "supplemental" / "mlb-umpires" / "game_umpires.csv"
IL_CSV = PROJECT_ROOT / "data" / "supplemental" / "mlb-injuries" / "team_il_daily.csv"
SP_XERA_CSV = PROJECT_ROOT / "data" / "supplemental" / "mlb-statcast" / "pitcher_prev_xera.csv"
TEAM_XWOBA_CSV = PROJECT_ROOT / "data" / "supplemental" / "mlb-statcast" / "team_prev_xwoba.csv"

TEAM_ID_TO_ABBR: dict[int, str] = {
    team_id: abbr for abbr, team_id in ESPN_TO_MLB_TEAM_ID.items() if abbr != "ath"
}


def binary_home_win(home_score: int, away_score: int) -> int | None:
    if home_score == away_score:
        return None
    return 1 if home_score > away_score else 0


def load_season(season: int) -> dict[str, Any] | None:
    season_dir = CACHE_ROOT / str(season)
    paths = {
        "games": season_dir / "games.json",
        "pitchers": season_dir / "pitchers.json",
        "team_hitting": season_dir / "team_hitting.json",
        "team_pitching": season_dir / "team_pitching.json",
    }
    if not all(p.is_file() for p in paths.values()):
        return None
    return {key: json.loads(path.read_text(encoding="utf-8")) for key, path in paths.items()}


def _load_weather() -> dict[tuple[str, int], dict[str, float]]:
    """Prefer Open-Meteo (humidity/precip); fall back to MLB schedule weather."""
    out: dict[tuple[str, int], dict[str, float]] = {}
    if WEATHER_OPENMETEO.is_file():
        frame = pd.read_csv(WEATHER_OPENMETEO)
        for row in frame.to_dict(orient="records"):
            day = str(row.get("date") or "")[:10]
            try:
                venue = int(row.get("venue_id"))
            except (TypeError, ValueError):
                continue
            if float(row.get("has_weather") or 0) < 0.5:
                continue
            out[(day, venue)] = {
                "temp_c": float(row.get("temp_c") or 22.0),
                "humidity": float(row.get("humidity") or 50.0),
                "precip_mm": float(row.get("precip_mm") or 0.0),
                "wind_kph": float(row.get("wind_kph") or 0.0),
                "has_weather": 1.0,
                "is_roof": float(row.get("is_roof") or 0.0),
            }
    if WEATHER_MLB.is_file():
        frame = pd.read_csv(WEATHER_MLB)
        for row in frame.to_dict(orient="records"):
            day = str(row.get("date") or "")[:10]
            try:
                venue = int(row.get("venue_id"))
            except (TypeError, ValueError):
                continue
            if float(row.get("has_weather") or 0) < 0.5:
                continue
            key = (day, venue)
            if key in out:
                # Keep Open-Meteo humidity/precip; optionally refresh roof flag.
                if float(row.get("is_roof") or 0) >= 0.5:
                    out[key]["is_roof"] = 1.0
                continue
            out[key] = {
                "temp_c": float(row.get("temp_c") or 22.0),
                "humidity": 50.0,
                "precip_mm": 0.0,
                "wind_kph": float(row.get("wind_kph") or 0.0),
                "has_weather": 1.0,
                "is_roof": float(row.get("is_roof") or 0.0),
            }
    return out


def _load_odds_lookup() -> dict[tuple[str, str, str], dict[str, Any]]:
    if not ODDS_CSV.is_file():
        return {}
    frame = pd.read_csv(ODDS_CSV)
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in frame.to_dict(orient="records"):
        key = (
            str(row.get("date") or "")[:10],
            str(row.get("home_key") or "").lower().strip(),
            str(row.get("away_key") or "").lower().strip(),
        )
        out[key] = row
    return out


def _load_umps() -> dict[int, dict[str, Any]]:
    if not UMP_CSV.is_file():
        return {}
    frame = pd.read_csv(UMP_CSV)
    out: dict[int, dict[str, Any]] = {}
    for row in frame.to_dict(orient="records"):
        try:
            pk = int(row.get("game_pk"))
        except (TypeError, ValueError):
            continue
        out[pk] = row
    return out


def _load_il() -> dict[tuple[str, int], dict[str, float]]:
    if not IL_CSV.is_file():
        return {}
    frame = pd.read_csv(IL_CSV)
    out: dict[tuple[str, int], dict[str, float]] = {}
    for row in frame.to_dict(orient="records"):
        day = str(row.get("date") or "")[:10]
        try:
            tid = int(row.get("team_id"))
        except (TypeError, ValueError):
            continue
        out[(day, tid)] = {
            "il_count": float(row.get("il_count") or 0.0),
            "il_pitchers": float(row.get("il_pitchers") or 0.0),
            "il_60day": float(row.get("il_60day") or 0.0),
            "has_il": float(row.get("has_il") or 1.0),
        }
    return out


def _load_sp_xera() -> dict[tuple[int, int], float]:
    """(for_season, player_id) -> xera"""
    if not SP_XERA_CSV.is_file():
        return {}
    frame = pd.read_csv(SP_XERA_CSV)
    out: dict[tuple[int, int], float] = {}
    for row in frame.to_dict(orient="records"):
        try:
            season = int(row.get("for_season") or (int(row["season"]) + 1))
            pid = int(row["player_id"])
            xera = float(row["xera"])
        except (TypeError, ValueError, KeyError):
            continue
        out[(season, pid)] = xera
    return out


def _load_team_xwoba() -> dict[tuple[int, int], float]:
    if not TEAM_XWOBA_CSV.is_file():
        return {}
    frame = pd.read_csv(TEAM_XWOBA_CSV)
    out: dict[tuple[int, int], float] = {}
    for row in frame.to_dict(orient="records"):
        try:
            season = int(row.get("for_season") or (int(row["season"]) + 1))
            tid = int(row["team_id"])
            xw = float(row["xwoba"])
        except (TypeError, ValueError, KeyError):
            continue
        out[(season, tid)] = xw
    return out


def _attach_enrichments(
    game: dict[str, Any],
    *,
    season: int,
    home_key: str,
    away_key: str,
    odds_lookup: dict[tuple[str, str, str], dict[str, Any]],
    weather: dict[tuple[str, int], dict[str, float]],
    umps: dict[int, dict[str, Any]],
    il: dict[tuple[str, int], dict[str, float]],
    sp_xera: dict[tuple[int, int], float],
    team_xwoba: dict[tuple[int, int], float],
) -> None:
    day = str(game.get("date") or "")[:10]
    row = odds_lookup.get((day, home_key, away_key))
    if row is None:
        try:
            base = date.fromisoformat(day)
            for delta in (-1, 1):
                cand = (base + timedelta(days=delta)).isoformat()
                row = odds_lookup.get((cand, home_key, away_key))
                if row is not None:
                    break
        except ValueError:
            row = None
    if row:
        for field in (
            "home_open_ml",
            "away_open_ml",
            "home_close_ml",
            "away_close_ml",
            "home_open_spread",
            "away_open_spread",
            "home_close_spread",
            "away_close_spread",
            "open_total",
            "close_total",
        ):
            val = row.get(field)
            if val is None or (isinstance(val, float) and val != val):
                continue
            game[field] = val

    venue = game.get("venue_id")
    if venue is not None:
        wx = weather.get((day, int(venue)))
        if wx:
            game.update(wx)

    pk = game.get("gamePk")
    if pk is not None and int(pk) in umps:
        u = umps[int(pk)]
        if float(u.get("has_ump") or 0) >= 0.5 and int(u.get("hp_ump_id") or 0):
            game["hp_ump_id"] = int(u["hp_ump_id"])

    home_id = int(game["home_id"])
    away_id = int(game["away_id"])
    hil = il.get((day, home_id))
    ail = il.get((day, away_id))
    if hil and ail:
        game["home_il_count"] = hil["il_count"]
        game["away_il_count"] = ail["il_count"]
        game["home_il_pitchers"] = hil["il_pitchers"]
        game["away_il_pitchers"] = ail["il_pitchers"]
        game["has_il"] = 1.0

    home_pp = game.get("home_pp_id")
    away_pp = game.get("away_pp_id")
    if home_pp:
        x = sp_xera.get((season, int(home_pp)))
        if x is not None:
            game["home_sp_prev_xera"] = x
    if away_pp:
        x = sp_xera.get((season, int(away_pp)))
        if x is not None:
            game["away_sp_prev_xera"] = x

    hx = team_xwoba.get((season, home_id))
    ax = team_xwoba.get((season, away_id))
    if hx is not None:
        game["home_prev_xwoba"] = hx
    if ax is not None:
        game["away_prev_xwoba"] = ax


def build_rows(start_season: int, end_season: int) -> pd.DataFrame:
    engine = MlbFeatureEngine()
    rows: list[dict[str, Any]] = []
    weather = _load_weather()
    odds_lookup = _load_odds_lookup()
    umps = _load_umps()
    il = _load_il()
    sp_xera = _load_sp_xera()
    team_xwoba = _load_team_xwoba()
    print(
        f"weather={len(weather)} odds={len(odds_lookup)} umps={len(umps)} "
        f"il={len(il)} sp_xera={len(sp_xera)} team_xwoba={len(team_xwoba)}",
        flush=True,
    )

    for season in range(start_season, end_season + 1):
        data = load_season(season)
        if data is None:
            print(f"season {season}: cache missing, skipped", flush=True)
            continue

        def on_row(game: dict[str, Any], feats: dict[str, float]) -> None:
            home_win = binary_home_win(int(game["home_score"]), int(game["away_score"]))
            if home_win is None:
                return
            home_id = int(game["home_id"])
            away_id = int(game["away_id"])
            record: dict[str, Any] = {
                "season": season,
                "date": game["date"],
                "game_pk": game.get("gamePk"),
                "home_id": home_id,
                "away_id": away_id,
                "home_key": TEAM_ID_TO_ABBR.get(home_id, str(home_id)),
                "away_key": TEAM_ID_TO_ABBR.get(away_id, str(away_id)),
                "home_pp_id": game.get("home_pp_id"),
                "away_pp_id": game.get("away_pp_id"),
                "home_score": game.get("home_score"),
                "away_score": game.get("away_score"),
                "home_win": home_win,
                "venue_id": game.get("venue_id"),
                "hp_ump_id": game.get("hp_ump_id"),
            }
            for field in (
                "home_open_ml",
                "away_open_ml",
                "home_close_ml",
                "away_close_ml",
                "home_open_spread",
                "away_open_spread",
                "home_close_spread",
                "away_close_spread",
                "open_total",
                "close_total",
            ):
                if field in game:
                    record[field] = game[field]
            record.update(feats)
            rows.append(record)

        games = data["games"]
        for game in games:
            home_id = int(game["home_id"])
            away_id = int(game["away_id"])
            _attach_enrichments(
                game,
                season=season,
                home_key=TEAM_ID_TO_ABBR.get(home_id, str(home_id)),
                away_key=TEAM_ID_TO_ABBR.get(away_id, str(away_id)),
                odds_lookup=odds_lookup,
                weather=weather,
                umps=umps,
                il=il,
                sp_xera=sp_xera,
                team_xwoba=team_xwoba,
            )

        replay_season(
            engine,
            season,
            games,
            data["pitchers"],
            data["team_hitting"],
            data["team_pitching"],
            on_row=on_row,
        )
        print(f"season {season}: {sum(1 for r in rows if r['season'] == season)} rows", flush=True)

    return pd.DataFrame(rows)


def merge_odds_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if "market_home_prob" in frame.columns and frame["market_home_prob"].notna().any():
        return frame
    probs: list[float | None] = []
    for row in frame.itertuples(index=False):
        home_ml = getattr(row, "home_close_ml", None)
        away_ml = getattr(row, "away_close_ml", None)
        market_home: float | None = None
        if home_ml is not None and away_ml is not None and home_ml == home_ml and away_ml == away_ml:
            try:
                _, market_home = devig_two_way_probs(int(away_ml), int(home_ml))
                if market_home is not None:
                    market_home = round(market_home / 100.0, 6)
            except (TypeError, ValueError):
                market_home = None
        if market_home is None:
            odds = closing_odds_lookup("mlb", row.date, row.home_key, row.away_key) or {}
            hml, aml = odds.get("home_close_ml"), odds.get("away_close_ml")
            if hml is not None and aml is not None:
                _, market_home = devig_two_way_probs(int(aml), int(hml))
                if market_home is not None:
                    market_home = round(market_home / 100.0, 6)
        probs.append(market_home)
    frame["market_home_prob"] = probs
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description="Build MLB v2 training table")
    parser.add_argument("--start-season", type=int, default=2018)
    parser.add_argument("--end-season", type=int, default=date.today().year)
    args = parser.parse_args()

    frame = build_rows(args.start_season, args.end_season)
    if frame.empty:
        print("no rows built; run scripts/fetch_mlb_history.py first")
        return 1
    frame = merge_odds_columns(frame)

    missing = [c for c in FEATURE_COLUMNS if c not in frame.columns]
    if missing:
        raise SystemExit(f"missing FEATURE_COLUMNS: {missing[:20]}")
    na = frame[list(FEATURE_COLUMNS)].isna().sum()
    if int(na.sum()) > 0:
        print("ERROR NaNs in FEATURE_COLUMNS:")
        print(na[na > 0].head(30).to_string())
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "training_table.csv"
    frame.to_csv(out_path, index=False)

    total = len(frame)
    with_odds = int(frame["market_home_prob"].notna().sum()) if "market_home_prob" in frame.columns else 0
    steam = float(frame["has_steam"].mean()) if "has_steam" in frame.columns else 0.0
    wx = float(frame["has_weather"].mean()) if "has_weather" in frame.columns else 0.0
    ump = float(frame["has_ump"].mean()) if "has_ump" in frame.columns else 0.0
    ilc = float(frame["has_il"].mean()) if "has_il" in frame.columns else 0.0
    sc = float(frame["has_statcast_sp"].mean()) if "has_statcast_sp" in frame.columns else 0.0
    print(
        f"\nwrote {out_path} ({total} rows, odds={with_odds / max(total,1):.1%}, "
        f"steam={steam:.1%}, weather={wx:.1%}, ump={ump:.1%}, il={ilc:.1%}, "
        f"statcast_sp={sc:.1%}, n_features={len(FEATURE_COLUMNS)})"
    )
    print(frame.groupby("season").size().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
