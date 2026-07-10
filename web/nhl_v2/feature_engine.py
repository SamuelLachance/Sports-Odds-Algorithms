"""Walk-forward NHL feature engine shared by offline training and live inference.

Consumes games chronologically. For each game it can produce a feature row
using only state accumulated strictly before that date (morning-of-slate
semantics), then folds the result into state.

State is JSON-serializable so a trained artifact can snapshot end-of-season
state and the live path can replay only the current season on top of it.

Data model per game (already merged from NHL Stats API + MoneyPuck):
  gameId, date (ISO), game_type (2 reg / 3 playoff), home, away,
  home_goals, away_goals, home_win, home_shots, away_shots,
  home_pp_pct/pk_pct/faceoff_pct (per game), mp: situational xG metrics,
  home_goalie_id, away_goalie_id (starters, when known).
"""

from __future__ import annotations

import math
from datetime import date as date_cls
from typing import Any

from web.nhl_v2.arenas import ARENA_COORDS

LEAGUE_ELO = 1500.0
ELO_K = 6.0
ELO_HOME_ADV = 33.0
ELO_SEASON_CARRYOVER = 0.72

ALPHA_FAST = 0.18
ALPHA_SLOW = 0.05
ALPHA_WIN = 0.10
ALPHA_SPECIAL = 0.10
ALPHA_GOALIE = 0.12
ALPHA_LEAGUE = 0.01

LEAGUE_GPG = 3.05          # goals per team per game
LEAGUE_XGPG = 2.90
LEAGUE_SOG = 30.5
LEAGUE_SV_PCT = 0.905
LEAGUE_SH_PCT = 0.095
LEAGUE_PP_XG = 0.85        # 5on4 xGF per game
LEAGUE_PK_XGA = 0.85
LEAGUE_FO_PCT = 0.50

GOALIE_PRIOR_SHOTS = 600.0  # shrink goalie sv% toward league over ~20 starts

FEATURE_COLUMNS: tuple[str, ...] = (
    "elo_diff",
    "gf_fast_diff",
    "ga_fast_diff",
    "gf_slow_diff",
    "ga_slow_diff",
    "xgf_fast_diff",
    "xga_fast_diff",
    "xgf_slow_diff",
    "xga_slow_diff",
    "xg55_pct_fast_diff",
    "xg55_pct_slow_diff",
    "corsi_pct_fast_diff",
    "corsi_pct_slow_diff",
    "shoot_luck_diff",
    "home_xgf_venue",
    "away_xgf_venue",
    "hd_share_diff",
    "sh_pct_ewma_diff",
    "sv_pct_ewma_diff",
    "pdo_ewma_diff",
    "sog_for_diff",
    "sog_against_diff",
    "pp_xgf_diff",
    "pk_xga_diff",
    "home_pp_vs_away_pk",
    "away_pp_vs_home_pk",
    "faceoff_diff",
    "team_gsax_diff",
    "win_ewma_diff",
    "home_split_win_ewma",
    "away_split_win_ewma",
    "points_pct_diff",
    "gd_pg_diff",
    "pythag_diff",
    "prev_points_pct_diff",
    "prev_gd_pg_diff",
    "home_rest_days",
    "away_rest_days",
    "rest_diff",
    "home_b2b",
    "away_b2b",
    "home_games_last6",
    "away_games_last6",
    "home_travel_km",
    "away_travel_km",
    "home_games_played",
    "away_games_played",
    "season_frac",
    "is_playoff",
    "league_gpg",
    "exp_total_env",
    "home_goalie_gsax",
    "away_goalie_gsax",
    "goalie_gsax_diff",
    "home_goalie_starts",
    "away_goalie_starts",
    "home_goalie_rest",
    "away_goalie_rest",
    "home_goalie_b2b",
    "away_goalie_b2b",
    "home_goalie_known",
    "away_goalie_known",
    "home_goalie_workload10",
    "away_goalie_workload10",
)


def _parse_date(value: str) -> date_cls | None:
    try:
        return date_cls.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _travel_km(a: tuple[float, float] | None, b: tuple[float, float] | None) -> float:
    if not a or not b:
        return 0.0
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    h = (
        math.sin((lat2 - lat1) / 2.0) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2.0) ** 2
    )
    return 6371.0 * 2.0 * math.asin(min(1.0, math.sqrt(h)))


def _ewma(current: float, observation: float, alpha: float) -> float:
    return (1.0 - alpha) * current + alpha * observation


class TeamState:
    __slots__ = (
        "elo",
        "gf_fast", "ga_fast", "gf_slow", "ga_slow",
        "xgf_fast", "xga_fast", "xgf_slow", "xga_slow",
        "xg55_pct_fast", "xg55_pct_slow",
        "corsi_pct_fast", "corsi_pct_slow",
        "shoot_luck",
        "xgf_home_venue", "xgf_away_venue",
        "hd_share",
        "sh_pct", "sv_pct",
        "sog_for", "sog_against",
        "pp_xgf", "pk_xga",
        "faceoff",
        "gsax",
        "win_ewma", "home_win_ewma", "away_win_ewma",
        "season_wins", "season_losses", "season_ot_losses",
        "season_gf", "season_ga",
        "prev_points_pct", "prev_gd_pg",
        "last_game_date", "recent_dates",
        "last_venue",
        "games_played",
    )

    def __init__(self) -> None:
        self.elo = LEAGUE_ELO
        self.gf_fast = LEAGUE_GPG
        self.ga_fast = LEAGUE_GPG
        self.gf_slow = LEAGUE_GPG
        self.ga_slow = LEAGUE_GPG
        self.xgf_fast = LEAGUE_XGPG
        self.xga_fast = LEAGUE_XGPG
        self.xgf_slow = LEAGUE_XGPG
        self.xga_slow = LEAGUE_XGPG
        self.xg55_pct_fast = 0.5
        self.xg55_pct_slow = 0.5
        self.corsi_pct_fast = 0.5
        self.corsi_pct_slow = 0.5
        self.shoot_luck = 0.0
        self.xgf_home_venue = LEAGUE_XGPG
        self.xgf_away_venue = LEAGUE_XGPG
        self.hd_share = 0.5
        self.sh_pct = LEAGUE_SH_PCT
        self.sv_pct = LEAGUE_SV_PCT
        self.sog_for = LEAGUE_SOG
        self.sog_against = LEAGUE_SOG
        self.pp_xgf = LEAGUE_PP_XG
        self.pk_xga = LEAGUE_PK_XGA
        self.faceoff = LEAGUE_FO_PCT
        self.gsax = 0.0
        self.win_ewma = 0.5
        self.home_win_ewma = 0.54
        self.away_win_ewma = 0.46
        self.season_wins = 0
        self.season_losses = 0
        self.season_ot_losses = 0
        self.season_gf = 0
        self.season_ga = 0
        self.prev_points_pct = 0.5
        self.prev_gd_pg = 0.0
        self.last_game_date: str | None = None
        self.recent_dates: list[str] = []
        self.last_venue: str | None = None
        self.games_played = 0

    # -- season aggregates ------------------------------------------------

    def points_pct(self) -> float:
        games = self.season_wins + self.season_losses + self.season_ot_losses
        if games < 5:
            return self.prev_points_pct
        points = 2 * self.season_wins + self.season_ot_losses
        return points / (2.0 * games)

    def gd_pg(self) -> float:
        games = self.season_wins + self.season_losses + self.season_ot_losses
        if games < 5:
            return self.prev_gd_pg
        return (self.season_gf - self.season_ga) / games

    def pythag(self) -> float:
        gf = max(float(self.season_gf), 1.0)
        ga = max(float(self.season_ga), 1.0)
        games = self.season_wins + self.season_losses + self.season_ot_losses
        if games < 8:
            return self.prev_points_pct
        exp = 2.15
        return gf**exp / (gf**exp + ga**exp)

    def rest_days(self, game_date: str) -> float:
        if not self.last_game_date:
            return 5.0
        current = _parse_date(game_date)
        previous = _parse_date(self.last_game_date)
        if not current or not previous:
            return 3.0
        return float(min((current - previous).days, 10))

    def games_last6(self, game_date: str) -> int:
        current = _parse_date(game_date)
        if not current:
            return 0
        count = 0
        for iso in self.recent_dates:
            past = _parse_date(iso)
            if past and 0 < (current - past).days <= 6:
                count += 1
        return count

    def to_dict(self) -> dict[str, Any]:
        return {slot: getattr(self, slot) for slot in self.__slots__}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TeamState":
        state = cls()
        for slot in cls.__slots__:
            if slot in payload:
                setattr(state, slot, payload[slot])
        return state


class GoalieState:
    __slots__ = (
        "sv_delta",          # shrunk save% above league, per-shot units
        "shots_seen",
        "starts",
        "last_start_date",
        "recent_start_dates",
        "team",
    )

    def __init__(self) -> None:
        self.sv_delta = 0.0
        self.shots_seen = 0.0
        self.starts = 0
        self.last_start_date: str | None = None
        self.recent_start_dates: list[str] = []
        self.team: str | None = None

    def quality(self) -> float:
        """Save% above league expectation, shrunk by career shot volume."""
        weight = self.shots_seen / (self.shots_seen + GOALIE_PRIOR_SHOTS)
        return self.sv_delta * weight

    def rest_days(self, game_date: str) -> float:
        if not self.last_start_date:
            return 6.0
        current = _parse_date(game_date)
        previous = _parse_date(self.last_start_date)
        if not current or not previous:
            return 4.0
        return float(min((current - previous).days, 14))

    def starts_last10(self, game_date: str) -> int:
        current = _parse_date(game_date)
        if not current:
            return 0
        count = 0
        for iso in self.recent_start_dates:
            past = _parse_date(iso)
            if past and 0 < (current - past).days <= 10:
                count += 1
        return count

    def to_dict(self) -> dict[str, Any]:
        return {slot: getattr(self, slot) for slot in self.__slots__}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GoalieState":
        state = cls()
        for slot in cls.__slots__:
            if slot in payload:
                setattr(state, slot, payload[slot])
        return state


class NhlFeatureEngine:
    def __init__(self) -> None:
        self.teams: dict[str, TeamState] = {}
        self.goalies: dict[str, GoalieState] = {}
        self.league_gpg = LEAGUE_GPG
        self.league_sv = LEAGUE_SV_PCT
        self.current_season: int | None = None

    # -- persistence -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "teams": {key: state.to_dict() for key, state in self.teams.items()},
            "goalies": {key: state.to_dict() for key, state in self.goalies.items()},
            "league_gpg": self.league_gpg,
            "league_sv": self.league_sv,
            "current_season": self.current_season,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "NhlFeatureEngine":
        engine = cls()
        engine.teams = {
            key: TeamState.from_dict(value)
            for key, value in (payload.get("teams") or {}).items()
        }
        engine.goalies = {
            key: GoalieState.from_dict(value)
            for key, value in (payload.get("goalies") or {}).items()
        }
        engine.league_gpg = float(payload.get("league_gpg") or LEAGUE_GPG)
        engine.league_sv = float(payload.get("league_sv") or LEAGUE_SV_PCT)
        engine.current_season = payload.get("current_season")
        return engine

    # -- accessors ----------------------------------------------------------

    def team(self, abbr: str) -> TeamState:
        return self.teams.setdefault(abbr, TeamState())

    def goalie(self, goalie_id: Any) -> GoalieState:
        return self.goalies.setdefault(str(goalie_id), GoalieState())

    # -- season boundaries ---------------------------------------------------

    def start_season(self, season: int) -> None:
        if self.current_season == season:
            return
        self.current_season = season
        for state in self.teams.values():
            games = state.season_wins + state.season_losses + state.season_ot_losses
            if games >= 20:
                points = 2 * state.season_wins + state.season_ot_losses
                state.prev_points_pct = points / (2.0 * games)
                state.prev_gd_pg = (state.season_gf - state.season_ga) / games
            state.season_wins = 0
            state.season_losses = 0
            state.season_ot_losses = 0
            state.season_gf = 0
            state.season_ga = 0
            state.games_played = 0
            state.elo = LEAGUE_ELO + ELO_SEASON_CARRYOVER * (state.elo - LEAGUE_ELO)
            state.last_game_date = None
            state.recent_dates = []
            state.last_venue = None

    # -- features -------------------------------------------------------------

    def features_for_game(self, game: dict[str, Any]) -> dict[str, float]:
        home = self.team(game["home"])
        away = self.team(game["away"])
        game_date = str(game.get("date") or "")

        home_rest = home.rest_days(game_date)
        away_rest = away.rest_days(game_date)

        home_coords = ARENA_COORDS.get(game["home"])
        home_prev = ARENA_COORDS.get(home.last_venue or game["home"])
        away_prev = ARENA_COORDS.get(away.last_venue or game["away"])

        month = _parse_date(game_date).month if _parse_date(game_date) else 1
        # NHL season runs Oct(10) -> Jun(6)
        season_frac = {10: 0.0, 11: 0.12, 12: 0.25, 1: 0.4, 2: 0.55, 3: 0.7,
                       4: 0.85, 5: 0.95, 6: 1.0}.get(month, 0.5)

        env_total = (
            (home.xgf_fast + away.xga_fast) / 2.0
            + (away.xgf_fast + home.xga_fast) / 2.0
        ) * (self.league_gpg / max(LEAGUE_XGPG, 1e-6))

        home_goalie_id = game.get("home_goalie_id")
        away_goalie_id = game.get("away_goalie_id")
        home_goalie = self.goalie(home_goalie_id) if home_goalie_id else None
        away_goalie = self.goalie(away_goalie_id) if away_goalie_id else None

        hg_gsax = home_goalie.quality() if home_goalie else 0.0
        ag_gsax = away_goalie.quality() if away_goalie else 0.0
        hg_rest = home_goalie.rest_days(game_date) if home_goalie else 4.0
        ag_rest = away_goalie.rest_days(game_date) if away_goalie else 4.0

        features: dict[str, float] = {
            "elo_diff": (home.elo + ELO_HOME_ADV - away.elo) / 100.0,
            "gf_fast_diff": home.gf_fast - away.gf_fast,
            "ga_fast_diff": home.ga_fast - away.ga_fast,
            "gf_slow_diff": home.gf_slow - away.gf_slow,
            "ga_slow_diff": home.ga_slow - away.ga_slow,
            "xgf_fast_diff": home.xgf_fast - away.xgf_fast,
            "xga_fast_diff": home.xga_fast - away.xga_fast,
            "xgf_slow_diff": home.xgf_slow - away.xgf_slow,
            "xga_slow_diff": home.xga_slow - away.xga_slow,
            "xg55_pct_fast_diff": home.xg55_pct_fast - away.xg55_pct_fast,
            "xg55_pct_slow_diff": home.xg55_pct_slow - away.xg55_pct_slow,
            "corsi_pct_fast_diff": home.corsi_pct_fast - away.corsi_pct_fast,
            "corsi_pct_slow_diff": home.corsi_pct_slow - away.corsi_pct_slow,
            "shoot_luck_diff": home.shoot_luck - away.shoot_luck,
            "home_xgf_venue": home.xgf_home_venue,
            "away_xgf_venue": away.xgf_away_venue,
            "hd_share_diff": home.hd_share - away.hd_share,
            "sh_pct_ewma_diff": home.sh_pct - away.sh_pct,
            "sv_pct_ewma_diff": home.sv_pct - away.sv_pct,
            "pdo_ewma_diff": (home.sh_pct + home.sv_pct) - (away.sh_pct + away.sv_pct),
            "sog_for_diff": home.sog_for - away.sog_for,
            "sog_against_diff": home.sog_against - away.sog_against,
            "pp_xgf_diff": home.pp_xgf - away.pp_xgf,
            "pk_xga_diff": home.pk_xga - away.pk_xga,
            "home_pp_vs_away_pk": home.pp_xgf - away.pk_xga,
            "away_pp_vs_home_pk": away.pp_xgf - home.pk_xga,
            "faceoff_diff": home.faceoff - away.faceoff,
            "team_gsax_diff": home.gsax - away.gsax,
            "win_ewma_diff": home.win_ewma - away.win_ewma,
            "home_split_win_ewma": home.home_win_ewma,
            "away_split_win_ewma": away.away_win_ewma,
            "points_pct_diff": home.points_pct() - away.points_pct(),
            "gd_pg_diff": home.gd_pg() - away.gd_pg(),
            "pythag_diff": home.pythag() - away.pythag(),
            "prev_points_pct_diff": home.prev_points_pct - away.prev_points_pct,
            "prev_gd_pg_diff": home.prev_gd_pg - away.prev_gd_pg,
            "home_rest_days": home_rest,
            "away_rest_days": away_rest,
            "rest_diff": home_rest - away_rest,
            "home_b2b": 1.0 if home_rest <= 1.0 else 0.0,
            "away_b2b": 1.0 if away_rest <= 1.0 else 0.0,
            "home_games_last6": float(home.games_last6(game_date)),
            "away_games_last6": float(away.games_last6(game_date)),
            "home_travel_km": _travel_km(home_prev, home_coords) / 1000.0,
            "away_travel_km": _travel_km(away_prev, home_coords) / 1000.0,
            "home_games_played": float(home.games_played),
            "away_games_played": float(away.games_played),
            "season_frac": season_frac,
            "is_playoff": 1.0 if int(game.get("game_type") or 2) == 3 else 0.0,
            "league_gpg": self.league_gpg,
            "exp_total_env": env_total,
            "home_goalie_gsax": hg_gsax * 100.0,
            "away_goalie_gsax": ag_gsax * 100.0,
            "goalie_gsax_diff": (hg_gsax - ag_gsax) * 100.0,
            "home_goalie_starts": float(home_goalie.starts) if home_goalie else 0.0,
            "away_goalie_starts": float(away_goalie.starts) if away_goalie else 0.0,
            "home_goalie_rest": hg_rest,
            "away_goalie_rest": ag_rest,
            "home_goalie_b2b": 1.0 if (home_goalie and hg_rest <= 1.0) else 0.0,
            "away_goalie_b2b": 1.0 if (away_goalie and ag_rest <= 1.0) else 0.0,
            "home_goalie_known": 1.0 if home_goalie else 0.0,
            "away_goalie_known": 1.0 if away_goalie else 0.0,
            "home_goalie_workload10": float(
                home_goalie.starts_last10(game_date)) if home_goalie else 0.0,
            "away_goalie_workload10": float(
                away_goalie.starts_last10(game_date)) if away_goalie else 0.0,
        }
        return features

    # -- state updates -----------------------------------------------------------

    def update_after_game(self, game: dict[str, Any]) -> None:
        home = self.team(game["home"])
        away = self.team(game["away"])
        game_date = str(game.get("date") or "")

        home_goals = int(game.get("home_goals") or 0)
        away_goals = int(game.get("away_goals") or 0)
        home_win = int(game.get("home_win") or (1 if home_goals > away_goals else 0))

        # Elo with margin multiplier
        expected_home = 1.0 / (
            1.0 + 10.0 ** (-((home.elo + ELO_HOME_ADV) - away.elo) / 400.0)
        )
        margin = abs(home_goals - away_goals)
        margin_mult = math.log1p(max(margin, 1)) * (
            2.2 / ((abs(home.elo - away.elo) * 0.001) + 2.2)
        )
        shift = ELO_K * margin_mult * (home_win - expected_home)
        home.elo += shift
        away.elo -= shift

        mp = game.get("mp") or {}
        home_mp = mp.get(game["home"]) or {}
        away_mp = mp.get(game["away"]) or {}

        for state, own_mp, gf, ga, sog_f, sog_a, at_home in (
            (home, home_mp, home_goals, away_goals,
             game.get("home_shots"), game.get("away_shots"), True),
            (away, away_mp, away_goals, home_goals,
             game.get("away_shots"), game.get("home_shots"), False),
        ):
            state.gf_fast = _ewma(state.gf_fast, gf, ALPHA_FAST)
            state.ga_fast = _ewma(state.ga_fast, ga, ALPHA_FAST)
            state.gf_slow = _ewma(state.gf_slow, gf, ALPHA_SLOW)
            state.ga_slow = _ewma(state.ga_slow, ga, ALPHA_SLOW)

            xgf = own_mp.get("all_scoreVenueAdjustedxGoalsFor")
            xga = own_mp.get("all_scoreVenueAdjustedxGoalsAgainst")
            if xgf is None:
                xgf = own_mp.get("all_xGoalsFor")
            if xga is None:
                xga = own_mp.get("all_xGoalsAgainst")
            if xgf is not None and xga is not None:
                state.xgf_fast = _ewma(state.xgf_fast, float(xgf), ALPHA_FAST)
                state.xga_fast = _ewma(state.xga_fast, float(xga), ALPHA_FAST)
                state.xgf_slow = _ewma(state.xgf_slow, float(xgf), ALPHA_SLOW)
                state.xga_slow = _ewma(state.xga_slow, float(xga), ALPHA_SLOW)
                state.gsax = _ewma(state.gsax, float(xga) - ga, ALPHA_GOALIE)
                state.shoot_luck = _ewma(
                    state.shoot_luck, gf - float(xgf), ALPHA_SLOW
                )
                if at_home:
                    state.xgf_home_venue = _ewma(
                        state.xgf_home_venue, float(xgf), ALPHA_FAST
                    )
                else:
                    state.xgf_away_venue = _ewma(
                        state.xgf_away_venue, float(xgf), ALPHA_FAST
                    )

            xgf55 = own_mp.get("5on5_xGoalsFor")
            xga55 = own_mp.get("5on5_xGoalsAgainst")
            if xgf55 is not None and xga55 is not None:
                total55 = float(xgf55) + float(xga55)
                if total55 > 0.2:
                    pct = float(xgf55) / total55
                    state.xg55_pct_fast = _ewma(state.xg55_pct_fast, pct, ALPHA_FAST)
                    state.xg55_pct_slow = _ewma(state.xg55_pct_slow, pct, ALPHA_SLOW)

            cf = own_mp.get("5on5_scoreAdjustedShotsAttemptsFor")
            ca = own_mp.get("5on5_scoreAdjustedShotsAttemptsAgainst")
            if cf is not None and ca is not None:
                total_c = float(cf) + float(ca)
                if total_c > 5:
                    corsi = float(cf) / total_c
                    state.corsi_pct_fast = _ewma(state.corsi_pct_fast, corsi, ALPHA_FAST)
                    state.corsi_pct_slow = _ewma(state.corsi_pct_slow, corsi, ALPHA_SLOW)

            hd_f = own_mp.get("all_highDangerxGoalsFor")
            hd_a = own_mp.get("all_highDangerxGoalsAgainst")
            if hd_f is not None and hd_a is not None:
                hd_total = float(hd_f) + float(hd_a)
                if hd_total > 0.1:
                    state.hd_share = _ewma(
                        state.hd_share, float(hd_f) / hd_total, ALPHA_FAST
                    )

            pp = own_mp.get("5on4_xGoalsFor")
            if pp is not None:
                state.pp_xgf = _ewma(state.pp_xgf, float(pp), ALPHA_SPECIAL)
            pk = own_mp.get("4on5_xGoalsAgainst")
            if pk is not None:
                state.pk_xga = _ewma(state.pk_xga, float(pk), ALPHA_SPECIAL)

            sog_for = float(sog_f or own_mp.get("all_shotsOnGoalFor") or LEAGUE_SOG)
            sog_against = float(sog_a or own_mp.get("all_shotsOnGoalAgainst") or LEAGUE_SOG)
            state.sog_for = _ewma(state.sog_for, sog_for, ALPHA_FAST)
            state.sog_against = _ewma(state.sog_against, sog_against, ALPHA_FAST)
            if sog_for > 5:
                state.sh_pct = _ewma(state.sh_pct, gf / sog_for, ALPHA_SLOW)
            if sog_against > 5:
                state.sv_pct = _ewma(state.sv_pct, 1.0 - ga / sog_against, ALPHA_SLOW)

        home_fo = game.get("home_faceoff_pct")
        away_fo = game.get("away_faceoff_pct")
        if home_fo:
            home.faceoff = _ewma(home.faceoff, float(home_fo), ALPHA_SPECIAL)
        if away_fo:
            away.faceoff = _ewma(away.faceoff, float(away_fo), ALPHA_SPECIAL)

        home.win_ewma = _ewma(home.win_ewma, float(home_win), ALPHA_WIN)
        away.win_ewma = _ewma(away.win_ewma, float(1 - home_win), ALPHA_WIN)
        home.home_win_ewma = _ewma(home.home_win_ewma, float(home_win), ALPHA_WIN)
        away.away_win_ewma = _ewma(away.away_win_ewma, float(1 - home_win), ALPHA_WIN)

        is_regular = int(game.get("game_type") or 2) == 2
        if is_regular:
            home_ot_loss = int(game.get("home_ot_loss") or 0)
            away_ot_loss = int(game.get("away_ot_loss") or 0)
            if home_win:
                home.season_wins += 1
                if away_ot_loss:
                    away.season_ot_losses += 1
                else:
                    away.season_losses += 1
            else:
                away.season_wins += 1
                if home_ot_loss:
                    home.season_ot_losses += 1
                else:
                    home.season_losses += 1
            home.season_gf += home_goals
            home.season_ga += away_goals
            away.season_gf += away_goals
            away.season_ga += home_goals

        for state in (home, away):
            state.games_played += 1
            state.last_game_date = game_date
            state.recent_dates = (state.recent_dates + [game_date])[-8:]
            state.last_venue = game["home"]

        self.league_gpg = _ewma(
            self.league_gpg, (home_goals + away_goals) / 2.0, ALPHA_LEAGUE
        )

        # goalie updates (starters only). Preferred observation: per-shot GSAx
        # from MoneyPuck shot xG (xg_faced - goals_allowed) / shots; fallback:
        # save% above league average.
        for goalie_id, team_key, shots_key, goals_allowed in (
            (game.get("home_goalie_id"), game["home"], "away_shots", away_goals),
            (game.get("away_goalie_id"), game["away"], "home_shots", home_goals),
        ):
            if not goalie_id:
                continue
            shots_against = game.get(f"{team_key}_goalie_shots")
            saves = game.get(f"{team_key}_goalie_saves")
            xg_faced = game.get(f"{team_key}_goalie_xg")
            xg_shots = game.get(f"{team_key}_goalie_xg_shots")
            xg_goals = game.get(f"{team_key}_goalie_xg_goals")
            if shots_against is None:
                shots_against = game.get(shots_key)
            state = self.goalie(goalie_id)
            state.team = team_key
            state.starts += 1
            state.last_start_date = game_date
            state.recent_start_dates = (state.recent_start_dates + [game_date])[-12:]

            observed_delta: float | None = None
            shots_f = 0.0
            if xg_faced is not None and xg_shots:
                try:
                    shots_f = float(xg_shots)
                    goals_f = float(xg_goals if xg_goals is not None else goals_allowed)
                    if shots_f >= 5:
                        observed_delta = (float(xg_faced) - goals_f) / shots_f
                except (TypeError, ValueError):
                    observed_delta = None
            if observed_delta is None:
                try:
                    shots_f = float(shots_against or 0.0)
                except (TypeError, ValueError):
                    shots_f = 0.0
                if shots_f >= 5:
                    if saves is not None:
                        sv = float(saves) / shots_f
                    else:
                        sv = 1.0 - float(goals_allowed) / shots_f
                    observed_delta = sv - self.league_sv
                    self.league_sv = _ewma(self.league_sv, sv, ALPHA_LEAGUE)
            if observed_delta is not None and shots_f >= 5:
                weight = min(shots_f / 30.0, 1.5)
                state.sv_delta = _ewma(
                    state.sv_delta, observed_delta, min(ALPHA_GOALIE * weight, 0.35)
                )
                state.shots_seen += shots_f
