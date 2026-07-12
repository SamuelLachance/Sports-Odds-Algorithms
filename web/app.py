"""FastAPI application serving the Sports Odds betting platform."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from web.daily_service import get_daily_slate
from web.predict_service import (
    get_leagues,
    get_seasons,
    get_teams,
    predict_match,
)
from web.team_service import build_teams_index, get_team_profile
from web.tracking_service import build_tracking_response, load_store, update_tracking

DB_DIR = Path(__file__).resolve().parent.parent / "docs" / "api" / "db"

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(
    title="Sports Odds Algorithms",
    description="Algo-driven sports betting service with tracking across all major leagues.",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    league: str = Field(examples=["nba"])
    away_team: str = Field(examples=["portland-trail-blazers"])
    home_team: str = Field(examples=["golden-state-warriors"])
    date: str = Field(examples=["4-16-2017"])
    season_year: str = Field(examples=["2017"])
    algorithm: str = Field(default="Algo_V2", examples=["Algo_V2"])


def _v2_artifacts_status() -> dict[str, bool]:
    """Report whether each sport's GradientBoost v2 artifacts are on disk."""
    checks: tuple[tuple[str, str], ...] = (
        ("nba", "web.nba_v2.live"),
        ("wnba", "web.wnba_v2.live"),
        ("nhl", "web.nhl_v2.live"),
        ("mlb", "web.mlb_v2.live"),
        ("soccer", "web.soccer_v2.live"),
        ("nfl", "web.nfl_v2.live"),
        ("cfb", "web.cfb_v2.live"),
        ("cbb", "web.cbb_v2.live"),
    )
    status: dict[str, bool] = {}
    for key, module_path in checks:
        try:
            module = __import__(module_path, fromlist=["artifacts_available"])
            status[key] = bool(module.artifacts_available())
        except Exception:  # noqa: BLE001 — health must stay non-fatal
            status[key] = False
    return status


@app.get("/api/health")
def health() -> dict:
    artifacts = _v2_artifacts_status()
    return {
        "status": "ok",
        "v2_artifacts": artifacts,
        "v2_artifacts_ready": sum(1 for ok in artifacts.values() if ok),
        "v2_artifacts_total": len(artifacts),
    }


@app.get("/api/leagues")
def leagues() -> list[dict[str, str]]:
    return get_leagues()


@app.get("/api/leagues/{league}/teams")
def teams(league: str) -> list[dict[str, str]]:
    try:
        return get_teams(league)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/leagues/{league}/seasons")
def seasons(league: str) -> list[str]:
    return get_seasons(league)


@app.get("/api/teams")
def teams_index() -> dict:
    return build_teams_index()


@app.get("/api/teams/{league}/{abbr}")
def team_profile(league: str, abbr: str) -> dict:
    try:
        return get_team_profile(league, abbr)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/daily/slate")
def daily_slate(days_ahead: int = 0) -> dict:
    try:
        return get_daily_slate(days_ahead=max(0, min(days_ahead, 3)))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/tracking")
def tracking(refresh: bool = False) -> dict:
    try:
        if refresh:
            slate = get_daily_slate()
            return update_tracking(slate)
        return build_tracking_response(load_store())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _read_db_json(rel: str) -> dict:
    path = DB_DIR / rel
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Database snapshot not found")
    import json

    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/db/manifest")
def db_manifest() -> dict:
    return _read_db_json("manifest.json")


@app.get("/api/db/{league}/league")
def db_league(league: str) -> dict:
    return _read_db_json(f"{league.lower()}/league.json")


@app.get("/api/db/{league}/teams")
def db_teams_index(league: str) -> dict:
    return _read_db_json(f"{league.lower()}/teams/index.json")


@app.get("/api/db/{league}/teams/{abbr}")
def db_team(league: str, abbr: str) -> dict:
    return _read_db_json(f"{league.lower()}/teams/{abbr.lower()}.json")


@app.get("/api/db/{league}/teams/{abbr}/players")
def db_team_players(league: str, abbr: str) -> dict:
    return _read_db_json(f"{league.lower()}/teams/{abbr.lower()}/players.json")


@app.get("/api/db/{league}/players/{player_id}")
def db_player(league: str, player_id: str) -> dict:
    return _read_db_json(f"{league.lower()}/players/{player_id}.json")


@app.post("/api/tracking/sync")
def tracking_sync() -> dict:
    try:
        slate = get_daily_slate()
        return update_tracking(slate)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/predict")
def predict(body: PredictRequest) -> dict:
    try:
        return predict_match(
            league=body.league,
            away_slug=body.away_team,
            home_slug=body.home_team,
            date=body.date,
            season_year=body.season_year,
            algo_version=body.algorithm,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
