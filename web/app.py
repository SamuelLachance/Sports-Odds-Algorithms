"""FastAPI application serving the Sports Odds betting platform."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from web.daily_service import get_daily_slate
from web.world_cup_service import get_world_cup_hub
from web.predict_service import (
    get_leagues,
    get_seasons,
    get_teams,
    predict_match,
)
from web.team_service import build_teams_index, get_team_profile
from web.tracking_service import build_tracking_response, load_store, update_tracking

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


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


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


@app.get("/api/worldcup")
def world_cup_hub() -> dict:
    try:
        return get_world_cup_hub()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


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
