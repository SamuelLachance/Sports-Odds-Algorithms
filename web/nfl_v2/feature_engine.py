"""Walk-forward NFL feature engine shared by offline training and live inference.

Consumes NFL games chronologically. For each game it emits a feature row using
only state accumulated strictly before that date, then folds the result into
state. State is JSON-serializable so training can snapshot end-of-season state
and the live path replays only the current season.

Game dict schema:
  date (ISO), season, home / away (lowercase abbr), home_score, away_score,
  optional: week, game_type, location, div_game, roof, surface, temp, wind,
  weekday, home_rest, away_rest, home_qb_id, away_qb_id, home_coach, away_coach.

Elo defaults: CFB-style K=20 with NFL HFA=48 and POINTS_PER_ELO=25 so
elo_diff is consistent with elo_to_prob(z=401.62). QB/coach Elo updated
only after games (leak-free).
"""

from __future__ import annotations

import math
from datetime import date as date_cls
from typing import Any

LEAGUE_ELO = 1500.0
ELO_K = 20.0
ELO_HOME_ADV = 48.0  # NFL HFA (nfelo-aligned Elo points)
ELO_SEASON_CARRYOVER = 0.70
POINTS_PER_ELO = 25.0
ELO_Z = 401.62

QB_ELO_K = 15.0
COACH_ELO_K = 12.0

ALPHA_FAST = 0.25
ALPHA_SLOW = 0.08
ALPHA_WIN = 0.15
ALPHA_H2H = 0.35
ALPHA_LEAGUE = 0.02

LEAGUE_PPG = 22.5
CLOSE_GAME_MARGIN = 8.0
BLOWOUT_MARGIN = 17.0
DEFAULT_MARGIN_VOL = 14.0
MARGIN_HIST_LEN = 8
SEASON_GAMES_NOMINAL = 17.0
DEFAULT_TEMP = 70.0
DEFAULT_WIND = 0.0
DEFAULT_REST = 7.0

FEATURE_COLUMNS: tuple[str, ...] = (
    # team strength
    "elo_diff",
    "pf_fast_diff",
    "pa_fast_diff",
    "pf_slow_diff",
    "pa_slow_diff",
    "win_ewma_diff",
    "home_split_win_ewma",
    "away_split_win_ewma",
    "win_pct_diff",
    "margin_pg_diff",
    "pythag_diff",
    "prev_win_pct_diff",
    "prev_margin_pg_diff",
    "sos_elo_diff",
    "pf_trend_diff",
    "pa_trend_diff",
    "elo_mom5_diff",
    "home_streak",
    "away_streak",
    "home_games_played",
    "away_games_played",
    "games_played_min",
    # context
    "home_rest_days",
    "away_rest_days",
    "rest_diff",
    "short_week",
    "week",
    "season_frac",
    "is_playoff",
    "div_game",
    "roof_dome",
    "roof_outdoor",
    "surface_turf",
    "temp",
    "wind",
    "weekday_thu",
    "neutral_site",
    "league_ppg",
    "exp_total_env",
    "elo_x_season_frac",
    # matchup history
    "h2h_home_win_rate",
    "h2h_margin_ewma",
    # QB / coach
    "qb_elo_diff",
    "qb_games_diff",
    "qb_win_ewma_diff",
    "home_qb_known",
    "away_qb_known",
    "home_qb_streak",
    "away_qb_streak",
    "coach_elo_diff",
    "coach_games_diff",
    "coach_win_ewma_diff",
    "home_coach_known",
    "away_coach_known",
    # score-based luck / volatility proxies
    "close_win_ewma_diff",
    "blowout_net_ewma_diff",
    "blowout_rate_diff",
    "margin_vol_diff",
    "scoring_vol_diff",
    "one_score_rate_diff",
    "home_off_vs_away_def",
    "away_off_vs_home_def",
    "net_x_season_frac",
)


def nfl_season_of(day: date_cls) -> int:
    """NFL season year: Sep–Dec belong to that year; Jan–Feb playoffs to prior."""
    return day.year if day.month >= 3 else day.year - 1


def _parse_date(value: str) -> date_cls | None:
    try:
        return date_cls.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _safe_float(value: Any, default: float) -> float:
    if value is None:
        return default
    try:
        if isinstance(value, float) and math.isnan(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_missing_id(value: Any) -> bool:
    if value is None:
        return True
    text = str(value).strip().lower()
    return text in ("", "nan", "none", "null")


def infer_neutral_site(game: dict[str, Any]) -> bool:
    if game.get("neutral_site") is not None:
        return bool(game["neutral_site"])
    location = str(game.get("location") or "").strip().lower()
    if location:
        return location != "home"
    return False


def infer_playoff(game: dict[str, Any]) -> bool:
    game_type = str(game.get("game_type") or "").strip().upper()
    if game_type:
        return game_type != "REG"
    return False


def _roof_flags(roof: str) -> tuple[float, float]:
    roof = (roof or "").strip().lower()
    dome = 1.0 if roof in {"dome", "closed"} else 0.0
    outdoor = 1.0 if roof in {"outdoors", "open"} else 0.0
    return dome, outdoor


def _is_turf(surface: str) -> float:
    surface = (surface or "").strip().lower()
    if not surface or surface == "grass":
        return 0.0
    return 1.0


class TeamState:
    __slots__ = (
        "key", "elo", "pf_fast", "pa_fast", "pf_slow", "pa_slow",
        "win_ewma", "home_win_ewma", "away_win_ewma",
        "wins", "losses", "ties", "points_for", "points_against",
        "prev_win_pct", "prev_margin_pg", "sos_elo_sum",
        "last_game_date", "recent_dates", "streak",
        "games_played", "season_seen", "first_season", "h2h", "h2h_margin",
        "close_win_ewma", "blowout_net_ewma", "blowout_games", "blowout_wins",
        "one_score_games", "one_score_wins", "recent_margins", "recent_pf",
        "elo_pre_hist", "current_qb_id", "qb_streak", "current_coach", "coach_streak",
    )

    def __init__(self, key: str):
        self.key = key
        self.elo = LEAGUE_ELO
        self.pf_fast = LEAGUE_PPG
        self.pa_fast = LEAGUE_PPG
        self.pf_slow = LEAGUE_PPG
        self.pa_slow = LEAGUE_PPG
        self.win_ewma = 0.5
        self.home_win_ewma = 0.5
        self.away_win_ewma = 0.5
        self.wins = 0
        self.losses = 0
        self.ties = 0
        self.points_for = 0.0
        self.points_against = 0.0
        self.prev_win_pct = 0.5
        self.prev_margin_pg = 0.0
        self.sos_elo_sum = 0.0
        self.last_game_date = ""
        self.recent_dates: list[str] = []
        self.streak = 0
        self.games_played = 0
        self.season_seen = -1
        self.first_season = -1
        self.h2h: dict[str, list[int]] = {}
        self.h2h_margin: dict[str, float] = {}
        self.close_win_ewma = 0.5
        self.blowout_net_ewma = 0.0
        self.blowout_games = 0
        self.blowout_wins = 0
        self.one_score_games = 0
        self.one_score_wins = 0
        self.recent_margins: list[float] = []
        self.recent_pf: list[float] = []
        self.elo_pre_hist: list[float] = []
        self.current_qb_id = ""
        self.qb_streak = 0
        self.current_coach = ""
        self.coach_streak = 0

    def roll_season(self, season: int) -> None:
        if self.season_seen == season:
            return
        if self.first_season < 0:
            self.first_season = season
        if self.games_played > 0:
            total = self.wins + self.losses + self.ties
            self.prev_win_pct = (self.wins + 0.5 * self.ties) / total if total else 0.5
            self.prev_margin_pg = (
                (self.points_for - self.points_against) / total if total else 0.0
            )
            self.elo = LEAGUE_ELO + ELO_SEASON_CARRYOVER * (self.elo - LEAGUE_ELO)
        self.wins = 0
        self.losses = 0
        self.ties = 0
        self.points_for = 0.0
        self.points_against = 0.0
        self.sos_elo_sum = 0.0
        self.last_game_date = ""
        self.recent_dates = []
        self.streak = 0
        self.games_played = 0
        self.season_seen = season
        self.recent_margins = []
        self.recent_pf = []
        self.elo_pre_hist = []
        self.blowout_games = 0
        self.blowout_wins = 0
        self.one_score_games = 0
        self.one_score_wins = 0

    def win_pct(self) -> float:
        total = self.wins + self.losses + self.ties
        return (self.wins + 0.5 * self.ties) / total if total else 0.5

    def margin_pg(self) -> float:
        total = self.wins + self.losses + self.ties
        if not total:
            return 0.0
        return (self.points_for - self.points_against) / total

    def pythag(self) -> float:
        pf = max(self.points_for, 1.0)
        pa = max(self.points_against, 1.0)
        exp = 2.37
        return pf**exp / (pf**exp + pa**exp)

    def sos_elo(self) -> float:
        total = self.wins + self.losses + self.ties
        return self.sos_elo_sum / total if total else LEAGUE_ELO

    def rest_days(self, game_date: date_cls) -> float:
        prior = _parse_date(self.last_game_date)
        if prior is None:
            return DEFAULT_REST
        # Floor at 0: inverted/out-of-order dates must not invent negative rest.
        return float(min(max((game_date - prior).days, 0), 21))

    def elo_momentum(self) -> float:
        if not self.elo_pre_hist:
            return 0.0
        return self.elo - self.elo_pre_hist[0]

    def margin_volatility(self) -> float:
        if len(self.recent_margins) < 3:
            return DEFAULT_MARGIN_VOL
        mean = sum(self.recent_margins) / len(self.recent_margins)
        var = sum((m - mean) ** 2 for m in self.recent_margins) / len(self.recent_margins)
        return math.sqrt(var)

    def scoring_volatility(self) -> float:
        if len(self.recent_pf) < 3:
            return 10.0
        mean = sum(self.recent_pf) / len(self.recent_pf)
        var = sum((p - mean) ** 2 for p in self.recent_pf) / len(self.recent_pf)
        return math.sqrt(var)

    def blowout_rate(self) -> float:
        if self.blowout_games <= 0:
            return 0.25
        return self.blowout_wins / self.blowout_games

    def one_score_rate(self) -> float:
        if self.one_score_games <= 0:
            return 0.5
        return self.one_score_wins / self.one_score_games

    def qb_streak_for(self, qb_id: str) -> float:
        if _is_missing_id(qb_id) or not self.current_qb_id:
            return 0.0
        if str(qb_id).strip() != self.current_qb_id:
            return 0.0
        return float(self.qb_streak)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "elo": self.elo,
            "pf_fast": self.pf_fast,
            "pa_fast": self.pa_fast,
            "pf_slow": self.pf_slow,
            "pa_slow": self.pa_slow,
            "win_ewma": self.win_ewma,
            "home_win_ewma": self.home_win_ewma,
            "away_win_ewma": self.away_win_ewma,
            "wins": self.wins,
            "losses": self.losses,
            "ties": self.ties,
            "points_for": self.points_for,
            "points_against": self.points_against,
            "prev_win_pct": self.prev_win_pct,
            "prev_margin_pg": self.prev_margin_pg,
            "sos_elo_sum": self.sos_elo_sum,
            "last_game_date": self.last_game_date,
            "recent_dates": list(self.recent_dates[-10:]),
            "streak": self.streak,
            "games_played": self.games_played,
            "season_seen": self.season_seen,
            "first_season": self.first_season,
            "h2h": {k: list(v) for k, v in self.h2h.items()},
            "h2h_margin": dict(self.h2h_margin),
            "close_win_ewma": self.close_win_ewma,
            "blowout_net_ewma": self.blowout_net_ewma,
            "blowout_games": self.blowout_games,
            "blowout_wins": self.blowout_wins,
            "one_score_games": self.one_score_games,
            "one_score_wins": self.one_score_wins,
            "recent_margins": list(self.recent_margins[-MARGIN_HIST_LEN:]),
            "recent_pf": list(self.recent_pf[-MARGIN_HIST_LEN:]),
            "elo_pre_hist": list(self.elo_pre_hist[-5:]),
            "current_qb_id": self.current_qb_id,
            "qb_streak": self.qb_streak,
            "current_coach": self.current_coach,
            "coach_streak": self.coach_streak,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TeamState":
        state = cls(str(payload.get("key") or payload.get("franchise") or ""))
        for key, value in payload.items():
            if key in ("key", "franchise"):
                continue
            if key == "recent_dates":
                state.recent_dates = [str(v) for v in value]
            elif key == "h2h":
                state.h2h = {str(k): [int(x) for x in v] for k, v in dict(value).items()}
            elif key == "h2h_margin":
                state.h2h_margin = {str(k): float(v) for k, v in dict(value).items()}
            elif key == "recent_margins":
                state.recent_margins = [float(v) for v in value]
            elif key == "recent_pf":
                state.recent_pf = [float(v) for v in value]
            elif key == "elo_pre_hist":
                state.elo_pre_hist = [float(v) for v in value]
            elif hasattr(state, key):
                setattr(state, key, value)
        return state


class EntityElo:
    """Leak-free Elo tracker for QBs or coaches."""

    __slots__ = ("key", "elo", "games", "win_ewma", "wins", "losses", "ties")

    def __init__(self, key: str):
        self.key = key
        self.elo = LEAGUE_ELO
        self.games = 0
        self.win_ewma = 0.5
        self.wins = 0
        self.losses = 0
        self.ties = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "elo": self.elo,
            "games": self.games,
            "win_ewma": self.win_ewma,
            "wins": self.wins,
            "losses": self.losses,
            "ties": self.ties,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EntityElo":
        ent = cls(str(payload.get("key") or ""))
        for key, value in payload.items():
            if key == "key":
                continue
            if hasattr(ent, key):
                setattr(ent, key, value)
        return ent


class NflFeatureEngine:
    """League-wide walk-forward state with leak-free feature extraction."""

    def __init__(self) -> None:
        self.teams: dict[str, TeamState] = {}
        self.qbs: dict[str, EntityElo] = {}
        self.coaches: dict[str, EntityElo] = {}
        self.league_ppg = LEAGUE_PPG

    def team(self, key: str) -> TeamState:
        key = str(key or "").lower().strip()
        state = self.teams.get(key)
        if state is None:
            state = TeamState(key)
            self.teams[key] = state
        return state

    def qb(self, key: str) -> EntityElo:
        key = str(key or "").strip()
        state = self.qbs.get(key)
        if state is None:
            state = EntityElo(key)
            self.qbs[key] = state
        return state

    def coach(self, key: str) -> EntityElo:
        key = str(key or "").strip().lower()
        state = self.coaches.get(key)
        if state is None:
            state = EntityElo(key)
            self.coaches[key] = state
        return state

    def _rest_pair(self, game: dict[str, Any], home: TeamState, away: TeamState, game_date: date_cls) -> tuple[float, float]:
        home_rest = _safe_float(game.get("home_rest"), -1.0)
        away_rest = _safe_float(game.get("away_rest"), -1.0)
        if home_rest < 0:
            home_rest = home.rest_days(game_date)
        if away_rest < 0:
            away_rest = away.rest_days(game_date)
        return float(min(home_rest, 21.0)), float(min(away_rest, 21.0))

    def features_for_game(self, game: dict[str, Any]) -> dict[str, float]:
        season = int(game.get("season") or 0)
        game_date = _parse_date(str(game.get("date") or "")) or date_cls(season, 9, 1)
        home = self.team(str(game["home"]))
        away = self.team(str(game["away"]))
        home.roll_season(season)
        away.roll_season(season)

        neutral = infer_neutral_site(game)
        is_playoff = 1.0 if infer_playoff(game) else 0.0
        div_game = 1.0 if bool(game.get("div_game")) else 0.0
        roof_dome, roof_outdoor = _roof_flags(str(game.get("roof") or ""))
        surface_turf = _is_turf(str(game.get("surface") or ""))
        temp = _safe_float(game.get("temp"), DEFAULT_TEMP)
        wind = _safe_float(game.get("wind"), DEFAULT_WIND)
        weekday = str(game.get("weekday") or "").strip().lower()
        weekday_thu = 1.0 if weekday.startswith("thu") else 0.0

        home_rest, away_rest = self._rest_pair(game, home, away, game_date)
        short_week = 1.0 if (home_rest < 7.0 or away_rest < 7.0) else 0.0
        week = _safe_float(game.get("week"), float(min(max(home.games_played + 1, 1), 22)))
        season_frac = min(home.games_played, away.games_played) / SEASON_GAMES_NOMINAL
        elo_diff = home.elo - away.elo + (0.0 if neutral else ELO_HOME_ADV)

        h2h_record = home.h2h.get(away.key) or [0, 0]
        h2h_total = h2h_record[0] + h2h_record[1]
        h2h_rate = h2h_record[0] / h2h_total if h2h_total else 0.5

        home_qb_id = str(game.get("home_qb_id") or "").strip()
        away_qb_id = str(game.get("away_qb_id") or "").strip()
        home_qb_known = 0.0 if _is_missing_id(home_qb_id) else 1.0
        away_qb_known = 0.0 if _is_missing_id(away_qb_id) else 1.0
        home_qb = self.qb(home_qb_id) if home_qb_known else None
        away_qb = self.qb(away_qb_id) if away_qb_known else None
        home_qb_elo = home_qb.elo if home_qb else LEAGUE_ELO
        away_qb_elo = away_qb.elo if away_qb else LEAGUE_ELO
        home_qb_games = float(home_qb.games) if home_qb else 0.0
        away_qb_games = float(away_qb.games) if away_qb else 0.0
        home_qb_win = home_qb.win_ewma if home_qb else 0.5
        away_qb_win = away_qb.win_ewma if away_qb else 0.5

        home_coach_name = str(game.get("home_coach") or "").strip()
        away_coach_name = str(game.get("away_coach") or "").strip()
        home_coach_known = 0.0 if _is_missing_id(home_coach_name) else 1.0
        away_coach_known = 0.0 if _is_missing_id(away_coach_name) else 1.0
        home_coach = self.coach(home_coach_name) if home_coach_known else None
        away_coach = self.coach(away_coach_name) if away_coach_known else None
        home_coach_elo = home_coach.elo if home_coach else LEAGUE_ELO
        away_coach_elo = away_coach.elo if away_coach else LEAGUE_ELO
        home_coach_games = float(home_coach.games) if home_coach else 0.0
        away_coach_games = float(away_coach.games) if away_coach else 0.0
        home_coach_win = home_coach.win_ewma if home_coach else 0.5
        away_coach_win = away_coach.win_ewma if away_coach else 0.5

        features: dict[str, float] = {
            "elo_diff": elo_diff,
            "pf_fast_diff": home.pf_fast - away.pf_fast,
            "pa_fast_diff": home.pa_fast - away.pa_fast,
            "pf_slow_diff": home.pf_slow - away.pf_slow,
            "pa_slow_diff": home.pa_slow - away.pa_slow,
            "win_ewma_diff": home.win_ewma - away.win_ewma,
            "home_split_win_ewma": home.home_win_ewma,
            "away_split_win_ewma": away.away_win_ewma,
            "win_pct_diff": home.win_pct() - away.win_pct(),
            "margin_pg_diff": home.margin_pg() - away.margin_pg(),
            "pythag_diff": home.pythag() - away.pythag(),
            "prev_win_pct_diff": home.prev_win_pct - away.prev_win_pct,
            "prev_margin_pg_diff": home.prev_margin_pg - away.prev_margin_pg,
            "sos_elo_diff": home.sos_elo() - away.sos_elo(),
            "pf_trend_diff": (home.pf_fast - home.pf_slow) - (away.pf_fast - away.pf_slow),
            "pa_trend_diff": (home.pa_fast - home.pa_slow) - (away.pa_fast - away.pa_slow),
            "elo_mom5_diff": home.elo_momentum() - away.elo_momentum(),
            "home_streak": float(max(min(home.streak, 8), -8)),
            "away_streak": float(max(min(away.streak, 8), -8)),
            "home_games_played": float(home.games_played),
            "away_games_played": float(away.games_played),
            "games_played_min": float(min(home.games_played, away.games_played)),
            "home_rest_days": home_rest,
            "away_rest_days": away_rest,
            "rest_diff": home_rest - away_rest,
            "short_week": short_week,
            "week": week,
            "season_frac": season_frac,
            "is_playoff": is_playoff,
            "div_game": div_game,
            "roof_dome": roof_dome,
            "roof_outdoor": roof_outdoor,
            "surface_turf": surface_turf,
            "temp": temp,
            "wind": wind,
            "weekday_thu": weekday_thu,
            "neutral_site": 1.0 if neutral else 0.0,
            "league_ppg": self.league_ppg,
            "exp_total_env": (home.pf_fast + home.pa_fast + away.pf_fast + away.pa_fast) / 2.0,
            "elo_x_season_frac": elo_diff * season_frac,
            "h2h_home_win_rate": h2h_rate,
            "h2h_margin_ewma": home.h2h_margin.get(away.key, 0.0),
            "qb_elo_diff": home_qb_elo - away_qb_elo,
            "qb_games_diff": home_qb_games - away_qb_games,
            "qb_win_ewma_diff": home_qb_win - away_qb_win,
            "home_qb_known": home_qb_known,
            "away_qb_known": away_qb_known,
            "home_qb_streak": home.qb_streak_for(home_qb_id),
            "away_qb_streak": away.qb_streak_for(away_qb_id),
            "coach_elo_diff": home_coach_elo - away_coach_elo,
            "coach_games_diff": home_coach_games - away_coach_games,
            "coach_win_ewma_diff": home_coach_win - away_coach_win,
            "home_coach_known": home_coach_known,
            "away_coach_known": away_coach_known,
            "close_win_ewma_diff": home.close_win_ewma - away.close_win_ewma,
            "blowout_net_ewma_diff": home.blowout_net_ewma - away.blowout_net_ewma,
            "blowout_rate_diff": home.blowout_rate() - away.blowout_rate(),
            "margin_vol_diff": home.margin_volatility() - away.margin_volatility(),
            "scoring_vol_diff": home.scoring_volatility() - away.scoring_volatility(),
            "one_score_rate_diff": home.one_score_rate() - away.one_score_rate(),
            "home_off_vs_away_def": home.pf_fast - away.pa_fast,
            "away_off_vs_home_def": away.pf_fast - home.pa_fast,
            "net_x_season_frac": (home.margin_pg() - away.margin_pg()) * season_frac,
        }
        return features

    def _update_entity(
        self,
        entity: EntityElo,
        *,
        won: bool,
        tied: bool,
        edge: float,
        k: float,
    ) -> None:
        exp = 1.0 / (1.0 + 10.0 ** (-edge / 400.0))
        if tied:
            score = 0.5
        else:
            score = 1.0 if won else 0.0
        entity.elo += k * (score - exp)
        entity.win_ewma = ALPHA_WIN * score + (1.0 - ALPHA_WIN) * entity.win_ewma
        entity.games += 1
        if tied:
            entity.ties += 1
        elif won:
            entity.wins += 1
        else:
            entity.losses += 1

    def update_after_game(self, game: dict[str, Any]) -> None:
        season = int(game.get("season") or 0)
        game_date = _parse_date(str(game.get("date") or "")) or date_cls(season, 9, 1)
        home = self.team(str(game["home"]))
        away = self.team(str(game["away"]))
        home.roll_season(season)
        away.roll_season(season)

        home_score = float(game["home_score"])
        away_score = float(game["away_score"])
        tied = home_score == away_score
        home_win = home_score > away_score
        signed_margin = home_score - away_score
        abs_margin = abs(signed_margin)
        neutral = infer_neutral_site(game)

        home_edge = home.elo - away.elo + (0.0 if neutral else ELO_HOME_ADV)
        exp_home = 1.0 / (1.0 + 10.0 ** (-home_edge / 400.0))
        if tied:
            score_home = 0.5
            mov_mult = 1.0
        else:
            score_home = 1.0 if home_win else 0.0
            winner_edge = home_edge if home_win else -home_edge
            mov_mult = math.log(max(abs_margin, 1.0) + 1.0) * (
                2.2 / (0.001 * max(winner_edge, 0.0) + 2.2)
            )
        delta = ELO_K * mov_mult * (score_home - exp_home)

        home.elo_pre_hist.append(home.elo)
        away.elo_pre_hist.append(away.elo)
        if len(home.elo_pre_hist) > 5:
            home.elo_pre_hist = home.elo_pre_hist[-5:]
        if len(away.elo_pre_hist) > 5:
            away.elo_pre_hist = away.elo_pre_hist[-5:]

        pre_home_elo = home.elo
        pre_away_elo = away.elo
        home.elo += delta
        away.elo -= delta

        def _ewma(old: float, new: float, alpha: float) -> float:
            return alpha * new + (1.0 - alpha) * old

        for team, pf, pa, won, is_home in (
            (home, home_score, away_score, home_win and not tied, True),
            (away, away_score, home_score, (not home_win) and not tied, False),
        ):
            team.pf_fast = _ewma(team.pf_fast, pf, ALPHA_FAST)
            team.pa_fast = _ewma(team.pa_fast, pa, ALPHA_FAST)
            team.pf_slow = _ewma(team.pf_slow, pf, ALPHA_SLOW)
            team.pa_slow = _ewma(team.pa_slow, pa, ALPHA_SLOW)
            result_score = 0.5 if tied else (1.0 if won else 0.0)
            team.win_ewma = _ewma(team.win_ewma, result_score, ALPHA_WIN)
            if is_home and not neutral:
                team.home_win_ewma = _ewma(team.home_win_ewma, result_score, ALPHA_WIN)
            elif not is_home and not neutral:
                team.away_win_ewma = _ewma(team.away_win_ewma, result_score, ALPHA_WIN)

            team.points_for += pf
            team.points_against += pa
            if tied:
                team.ties += 1
                team.streak = 0
            elif won:
                team.wins += 1
                team.streak = team.streak + 1 if team.streak > 0 else 1
            else:
                team.losses += 1
                team.streak = team.streak - 1 if team.streak < 0 else -1

            team_margin = pf - pa
            team.recent_margins.append(team_margin)
            team.recent_pf.append(pf)
            if len(team.recent_margins) > MARGIN_HIST_LEN:
                team.recent_margins = team.recent_margins[-MARGIN_HIST_LEN:]
            if len(team.recent_pf) > MARGIN_HIST_LEN:
                team.recent_pf = team.recent_pf[-MARGIN_HIST_LEN:]

            if abs_margin <= CLOSE_GAME_MARGIN:
                team.close_win_ewma = _ewma(team.close_win_ewma, result_score, ALPHA_WIN)
                team.one_score_games += 1
                if won:
                    team.one_score_wins += 1
            if abs_margin >= BLOWOUT_MARGIN:
                team.blowout_games += 1
                if won:
                    team.blowout_wins += 1
                signed = team_margin if won else -abs_margin
                team.blowout_net_ewma = _ewma(team.blowout_net_ewma, signed, ALPHA_WIN)

            iso = game_date.isoformat()
            team.last_game_date = iso
            team.recent_dates.append(iso)
            if len(team.recent_dates) > 10:
                team.recent_dates = team.recent_dates[-10:]
            team.games_played += 1

        home.sos_elo_sum += pre_away_elo
        away.sos_elo_sum += pre_home_elo

        record = home.h2h.setdefault(away.key, [0, 0])
        if home_win and not tied:
            record[0] += 1
        elif not tied:
            record[1] += 1
        prev = home.h2h_margin.get(away.key, 0.0)
        home.h2h_margin[away.key] = ALPHA_H2H * signed_margin + (1.0 - ALPHA_H2H) * prev
        away_prev = away.h2h_margin.get(home.key, 0.0)
        away.h2h_margin[home.key] = (
            ALPHA_H2H * (-signed_margin) + (1.0 - ALPHA_H2H) * away_prev
        )

        # QB / coach Elo — after game only
        home_qb_id = str(game.get("home_qb_id") or "").strip()
        away_qb_id = str(game.get("away_qb_id") or "").strip()
        if not _is_missing_id(home_qb_id) and not _is_missing_id(away_qb_id):
            hq = self.qb(home_qb_id)
            aq = self.qb(away_qb_id)
            qb_edge = hq.elo - aq.elo + (0.0 if neutral else ELO_HOME_ADV * 0.25)
            self._update_entity(
                hq, won=home_win and not tied, tied=tied, edge=qb_edge, k=QB_ELO_K
            )
            self._update_entity(
                aq, won=(not home_win) and not tied, tied=tied, edge=-qb_edge, k=QB_ELO_K
            )
        if not _is_missing_id(home_qb_id):
            if home.current_qb_id == home_qb_id:
                home.qb_streak += 1
            else:
                home.current_qb_id = home_qb_id
                home.qb_streak = 1
        if not _is_missing_id(away_qb_id):
            if away.current_qb_id == away_qb_id:
                away.qb_streak += 1
            else:
                away.current_qb_id = away_qb_id
                away.qb_streak = 1

        home_coach_name = str(game.get("home_coach") or "").strip()
        away_coach_name = str(game.get("away_coach") or "").strip()
        if not _is_missing_id(home_coach_name) and not _is_missing_id(away_coach_name):
            hc = self.coach(home_coach_name)
            ac = self.coach(away_coach_name)
            coach_edge = hc.elo - ac.elo + (0.0 if neutral else ELO_HOME_ADV * 0.25)
            self._update_entity(
                hc, won=home_win and not tied, tied=tied, edge=coach_edge, k=COACH_ELO_K
            )
            self._update_entity(
                ac, won=(not home_win) and not tied, tied=tied, edge=-coach_edge, k=COACH_ELO_K
            )
        if not _is_missing_id(home_coach_name):
            key = home_coach_name.lower()
            if home.current_coach == key:
                home.coach_streak += 1
            else:
                home.current_coach = key
                home.coach_streak = 1
        if not _is_missing_id(away_coach_name):
            key = away_coach_name.lower()
            if away.current_coach == key:
                away.coach_streak += 1
            else:
                away.current_coach = key
                away.coach_streak = 1

        self.league_ppg = (
            ALPHA_LEAGUE * ((home_score + away_score) / 2.0)
            + (1.0 - ALPHA_LEAGUE) * self.league_ppg
        )

    update_game = update_after_game

    def to_dict(self) -> dict[str, Any]:
        return {
            "league_ppg": self.league_ppg,
            "teams": {key: team.to_dict() for key, team in self.teams.items()},
            "qbs": {key: qb.to_dict() for key, qb in self.qbs.items()},
            "coaches": {key: coach.to_dict() for key, coach in self.coaches.items()},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "NflFeatureEngine":
        engine = cls()
        engine.league_ppg = float(payload.get("league_ppg", LEAGUE_PPG))
        for key, team_payload in dict(payload.get("teams") or {}).items():
            if "key" not in team_payload:
                team_payload = dict(team_payload)
                team_payload["key"] = key
            engine.teams[str(key)] = TeamState.from_dict(team_payload)
        for key, qb_payload in dict(payload.get("qbs") or {}).items():
            if "key" not in qb_payload:
                qb_payload = dict(qb_payload)
                qb_payload["key"] = key
            engine.qbs[str(key)] = EntityElo.from_dict(qb_payload)
        for key, coach_payload in dict(payload.get("coaches") or {}).items():
            if "key" not in coach_payload:
                coach_payload = dict(coach_payload)
                coach_payload["key"] = key
            engine.coaches[str(key)] = EntityElo.from_dict(coach_payload)
        return engine
