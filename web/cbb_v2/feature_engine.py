"""Walk-forward CBB feature engine (ESPN completed games only, PIT-safe).

Consumes games chronologically. Features use state accumulated strictly before
each tip; Torvik / season-end ratings are never read. Efficiency proxies use
an assumed possession rate when box scores are absent (same constant as the
legacy CBB fallback path), optionally overridden when home/away boxes supply
FGA/ORB/TOV/FTA.
"""

from __future__ import annotations

import math
from datetime import date as date_cls
from typing import Any

LEAGUE_ELO = 1500.0
ELO_K = 20.0
ELO_HOME_ADV = 75.0
ELO_SEASON_CARRYOVER = 0.60
ASSUMED_POSSESSIONS = 68.0
LEAGUE_PPG = 72.0
LEAGUE_ORTG = 106.0
LEAGUE_PACE = 68.0

ALPHA_FAST = 0.20
ALPHA_SLOW = 0.07
ALPHA_WIN = 0.12
ALPHA_LEAGUE = 0.015
ALPHA_H2H_MARGIN = 0.28
PLAYER_PROXY_ALPHA = 0.25

SEASON_GAMES_NOMINAL = 32.0
CLOSE_MARGIN = 5.0
BLOWOUT_MARGIN = 18.0
DEFAULT_MARGIN_VOL = 12.0

FEATURE_COLUMNS: tuple[str, ...] = (
    # strength / efficiency
    "elo_diff",
    "pf_fast_diff",
    "pa_fast_diff",
    "pf_slow_diff",
    "pa_slow_diff",
    "ortg_fast_diff",
    "drtg_fast_diff",
    "ortg_slow_diff",
    "drtg_slow_diff",
    "net_rtg_fast_diff",
    "net_rtg_slow_diff",
    "pace_sum",
    "pace_diff",
    # form / records
    "win_ewma_diff",
    "home_split_win_ewma",
    "away_split_win_ewma",
    "win_pct_diff",
    "margin_pg_diff",
    "pythag_diff",
    "prev_win_pct_diff",
    "prev_net_rtg_diff",
    "sos_elo_diff",
    # schedule / context
    "home_rest_days",
    "away_rest_days",
    "rest_diff",
    "home_b2b",
    "away_b2b",
    "home_games_last7",
    "away_games_last7",
    "home_3in4",
    "away_3in4",
    "home_streak",
    "away_streak",
    "home_games_played",
    "away_games_played",
    "games_played_min",
    "season_frac",
    "is_playoff",
    "neutral_site",
    "is_conference",
    "conf_win_pct_diff",
    "nonconf_win_pct_diff",
    # environment / interactions
    "league_ppg",
    "exp_total_env",
    "elo_x_season_frac",
    "elo_x_is_conf",
    # h2h / trends
    "h2h_home_win_rate",
    "h2h_margin_ewma",
    "net_rtg_trend_diff",
    "ortg_trend_diff",
    "drtg_trend_diff",
    "elo_mom5_diff",
    "adj_margin_ewma_diff",
    # player-proxy / style (score-derived)
    "blowout_rate_diff",
    "close_win_ewma_diff",
    "margin_vol_diff",
    "scoring_vol_diff",
    "last5_win_pct_diff",
    "home_stand_len",
    "away_road_trip_len",
    "home_conf_sos",
    "away_conf_sos",
)


def _parse_date(value: str) -> date_cls | None:
    try:
        return date_cls.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _std(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / n)


def _possessions(box: dict[str, float], opp_box: dict[str, float]) -> float | None:
    fga = box.get("fga")
    if fga is None:
        return None
    orb = box.get("orb", 0.0)
    tov = box.get("tov", 0.0)
    fta = box.get("fta", 0.0)
    return float(fga) - float(orb) + float(tov) + 0.44 * float(fta)


class TeamState:
    __slots__ = (
        "franchise", "elo", "pf_fast", "pa_fast", "pf_slow", "pa_slow",
        "ortg_fast", "drtg_fast", "ortg_slow", "drtg_slow", "pace_ewma",
        "win_ewma", "home_win_ewma", "away_win_ewma",
        "wins", "losses", "points_for", "points_against",
        "prev_win_pct", "prev_net_rtg", "sos_elo_sum",
        "last_game_date", "recent_dates", "streak", "games_played",
        "season_seen", "first_season", "h2h", "h2h_margin",
        "elo_hist", "close_win_ewma", "blowout_ewma", "recent_margins",
        "recent_pf", "recent_results", "venue_streak", "adj_margin_ewma",
        "conf_wins", "conf_losses", "nonconf_wins", "nonconf_losses",
        "conf_sos_sum", "conf_games", "conference_id",
    )

    def __init__(self, franchise: str):
        self.franchise = franchise
        self.elo = LEAGUE_ELO
        self.pf_fast = LEAGUE_PPG
        self.pa_fast = LEAGUE_PPG
        self.pf_slow = LEAGUE_PPG
        self.pa_slow = LEAGUE_PPG
        self.ortg_fast = LEAGUE_ORTG
        self.drtg_fast = LEAGUE_ORTG
        self.ortg_slow = LEAGUE_ORTG
        self.drtg_slow = LEAGUE_ORTG
        self.pace_ewma = LEAGUE_PACE
        self.win_ewma = 0.5
        self.home_win_ewma = 0.5
        self.away_win_ewma = 0.5
        self.wins = 0
        self.losses = 0
        self.points_for = 0.0
        self.points_against = 0.0
        self.prev_win_pct = 0.5
        self.prev_net_rtg = 0.0
        self.sos_elo_sum = 0.0
        self.last_game_date = ""
        self.recent_dates: list[str] = []
        self.streak = 0
        self.games_played = 0
        self.season_seen = -1
        self.first_season = -1
        self.h2h: dict[str, list[int]] = {}
        self.h2h_margin: dict[str, float] = {}
        self.elo_hist: list[float] = []
        self.close_win_ewma = 0.5
        self.blowout_ewma = 0.0
        self.recent_margins: list[float] = []
        self.recent_pf: list[float] = []
        self.recent_results: list[int] = []
        self.venue_streak = 0
        self.adj_margin_ewma = 0.0
        self.conf_wins = 0
        self.conf_losses = 0
        self.nonconf_wins = 0
        self.nonconf_losses = 0
        self.conf_sos_sum = 0.0
        self.conf_games = 0
        self.conference_id = ""

    def roll_season(self, season: int) -> None:
        if self.season_seen == season:
            return
        if self.first_season < 0:
            self.first_season = season
        if self.games_played > 0:
            total = self.wins + self.losses
            self.prev_win_pct = self.wins / total if total else 0.5
            self.prev_net_rtg = self.ortg_slow - self.drtg_slow
            self.elo = LEAGUE_ELO + ELO_SEASON_CARRYOVER * (self.elo - LEAGUE_ELO)
            self.adj_margin_ewma *= ELO_SEASON_CARRYOVER
        self.wins = 0
        self.losses = 0
        self.points_for = 0.0
        self.points_against = 0.0
        self.sos_elo_sum = 0.0
        self.last_game_date = ""
        self.recent_dates = []
        self.streak = 0
        self.games_played = 0
        self.season_seen = season
        self.elo_hist = []
        self.recent_margins = []
        self.recent_pf = []
        self.recent_results = []
        self.venue_streak = 0
        self.conf_wins = 0
        self.conf_losses = 0
        self.nonconf_wins = 0
        self.nonconf_losses = 0
        self.conf_sos_sum = 0.0
        self.conf_games = 0

    def win_pct(self) -> float:
        total = self.wins + self.losses
        return self.wins / total if total else 0.5

    def conf_win_pct(self) -> float:
        total = self.conf_wins + self.conf_losses
        return self.conf_wins / total if total else 0.5

    def nonconf_win_pct(self) -> float:
        total = self.nonconf_wins + self.nonconf_losses
        return self.nonconf_wins / total if total else 0.5

    def margin_pg(self) -> float:
        total = self.wins + self.losses
        if not total:
            return 0.0
        return (self.points_for - self.points_against) / total

    def pythag(self) -> float:
        pf = max(self.points_for, 1.0)
        pa = max(self.points_against, 1.0)
        return pf**11.5 / (pf**11.5 + pa**11.5)

    def sos_elo(self) -> float:
        total = self.wins + self.losses
        return self.sos_elo_sum / total if total else LEAGUE_ELO

    def conf_sos(self) -> float:
        return self.conf_sos_sum / self.conf_games if self.conf_games else LEAGUE_ELO

    def rest_days(self, game_date: date_cls) -> float:
        prior = _parse_date(self.last_game_date)
        if prior is None:
            return 5.0
        return float(min((game_date - prior).days, 14))

    def games_in_last7(self, game_date: date_cls) -> int:
        count = 0
        for iso in self.recent_dates:
            played = _parse_date(iso)
            if played and 0 < (game_date - played).days <= 7:
                count += 1
        return count

    def is_3in4(self, game_date: date_cls) -> bool:
        count = 0
        for iso in self.recent_dates:
            played = _parse_date(iso)
            if played and 0 < (game_date - played).days <= 3:
                count += 1
        return count >= 2

    def elo_momentum5(self) -> float:
        if not self.elo_hist:
            return 0.0
        anchor = self.elo_hist[-6] if len(self.elo_hist) >= 6 else self.elo_hist[0]
        return self.elo - anchor

    def margin_volatility(self) -> float:
        if len(self.recent_margins) < 4:
            return DEFAULT_MARGIN_VOL
        return _std(self.recent_margins)

    def scoring_volatility(self) -> float:
        if len(self.recent_pf) < 4:
            return 10.0
        return _std(self.recent_pf)

    def last5_win_pct(self) -> float:
        if not self.recent_results:
            return 0.5
        window = self.recent_results[-5:]
        return sum(window) / len(window)

    def to_dict(self) -> dict[str, Any]:
        return {
            "franchise": self.franchise,
            "elo": self.elo,
            "pf_fast": self.pf_fast, "pa_fast": self.pa_fast,
            "pf_slow": self.pf_slow, "pa_slow": self.pa_slow,
            "ortg_fast": self.ortg_fast, "drtg_fast": self.drtg_fast,
            "ortg_slow": self.ortg_slow, "drtg_slow": self.drtg_slow,
            "pace_ewma": self.pace_ewma,
            "win_ewma": self.win_ewma,
            "home_win_ewma": self.home_win_ewma, "away_win_ewma": self.away_win_ewma,
            "wins": self.wins, "losses": self.losses,
            "points_for": self.points_for, "points_against": self.points_against,
            "prev_win_pct": self.prev_win_pct, "prev_net_rtg": self.prev_net_rtg,
            "sos_elo_sum": self.sos_elo_sum,
            "last_game_date": self.last_game_date,
            "recent_dates": list(self.recent_dates[-14:]),
            "streak": self.streak,
            "games_played": self.games_played,
            "season_seen": self.season_seen,
            "first_season": self.first_season,
            "h2h": {k: list(v) for k, v in self.h2h.items()},
            "h2h_margin": dict(self.h2h_margin),
            "elo_hist": list(self.elo_hist[-8:]),
            "close_win_ewma": self.close_win_ewma,
            "blowout_ewma": self.blowout_ewma,
            "recent_margins": list(self.recent_margins[-12:]),
            "recent_pf": list(self.recent_pf[-12:]),
            "recent_results": list(self.recent_results[-10:]),
            "venue_streak": self.venue_streak,
            "adj_margin_ewma": self.adj_margin_ewma,
            "conf_wins": self.conf_wins, "conf_losses": self.conf_losses,
            "nonconf_wins": self.nonconf_wins, "nonconf_losses": self.nonconf_losses,
            "conf_sos_sum": self.conf_sos_sum, "conf_games": self.conf_games,
            "conference_id": self.conference_id,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TeamState":
        state = cls(str(payload.get("franchise") or ""))
        for key, value in payload.items():
            if key == "recent_dates":
                state.recent_dates = [str(v) for v in value]
            elif key == "h2h":
                state.h2h = {str(k): [int(x) for x in v] for k, v in dict(value).items()}
            elif key == "h2h_margin":
                state.h2h_margin = {str(k): float(v) for k, v in dict(value).items()}
            elif key == "elo_hist":
                state.elo_hist = [float(v) for v in value]
            elif key == "recent_margins":
                state.recent_margins = [float(v) for v in value]
            elif key == "recent_pf":
                state.recent_pf = [float(v) for v in value]
            elif key == "recent_results":
                state.recent_results = [int(v) for v in value]
            elif hasattr(state, key):
                setattr(state, key, value)
        return state


class CbbFeatureEngine:
    """League-wide walk-forward state with leak-free feature extraction."""

    def __init__(self) -> None:
        self.teams: dict[str, TeamState] = {}
        self.league_ppg = LEAGUE_PPG
        self.league_pace = LEAGUE_PACE
        self.abbr_to_id: dict[str, str] = {}

    def team(self, franchise: str) -> TeamState:
        state = self.teams.get(franchise)
        if state is None:
            state = TeamState(franchise)
            self.teams[franchise] = state
        return state

    def register_abbr(self, abbr: str, team_id: str) -> None:
        key = str(abbr or "").lower().strip()
        eid = str(team_id or "").strip()
        if key and eid:
            self.abbr_to_id[key] = eid

    def resolve_team_id(self, abbr: str, espn_id: str | None = None) -> str | None:
        if espn_id and str(espn_id) in self.teams:
            return str(espn_id)
        mapped = self.abbr_to_id.get(str(abbr or "").lower().strip())
        if mapped and mapped in self.teams:
            return mapped
        # last resort: treat abbr itself as key (unit tests / synthetic games)
        key = str(abbr or "").lower().strip()
        if key in self.teams:
            return key
        if espn_id:
            return str(espn_id)
        return key or None

    def features_for_game(self, game: dict[str, Any]) -> dict[str, float]:
        season = int(game.get("season") or 0)
        game_date = _parse_date(str(game.get("date") or "")) or date_cls(season, 1, 15)
        home = self.team(str(game["home"]))
        away = self.team(str(game["away"]))
        home.roll_season(season)
        away.roll_season(season)

        neutral = bool(game.get("neutral_site"))
        is_conf = bool(game.get("conference_game"))
        if not is_conf and home.conference_id and away.conference_id:
            is_conf = home.conference_id == away.conference_id and home.conference_id != ""

        home_rest = home.rest_days(game_date)
        away_rest = away.rest_days(game_date)

        h2h_record = home.h2h.get(away.franchise) or [0, 0]
        h2h_total = h2h_record[0] + h2h_record[1]
        h2h_rate = h2h_record[0] / h2h_total if h2h_total else 0.5

        season_frac = min(home.games_played, away.games_played) / SEASON_GAMES_NOMINAL
        season_frac = float(min(max(season_frac, 0.0), 1.5))
        elo_diff = home.elo - away.elo + (0.0 if neutral else ELO_HOME_ADV)

        home_stand = float(min(home.venue_streak, 7) + 1) if home.venue_streak > 0 else 1.0
        road_trip = float(min(-away.venue_streak, 7) + 1) if away.venue_streak < 0 else 1.0

        features: dict[str, float] = {
            "elo_diff": elo_diff,
            "pf_fast_diff": home.pf_fast - away.pf_fast,
            "pa_fast_diff": home.pa_fast - away.pa_fast,
            "pf_slow_diff": home.pf_slow - away.pf_slow,
            "pa_slow_diff": home.pa_slow - away.pa_slow,
            "ortg_fast_diff": home.ortg_fast - away.ortg_fast,
            "drtg_fast_diff": home.drtg_fast - away.drtg_fast,
            "ortg_slow_diff": home.ortg_slow - away.ortg_slow,
            "drtg_slow_diff": home.drtg_slow - away.drtg_slow,
            "net_rtg_fast_diff": (home.ortg_fast - home.drtg_fast)
            - (away.ortg_fast - away.drtg_fast),
            "net_rtg_slow_diff": (home.ortg_slow - home.drtg_slow)
            - (away.ortg_slow - away.drtg_slow),
            "pace_sum": home.pace_ewma + away.pace_ewma,
            "pace_diff": home.pace_ewma - away.pace_ewma,
            "win_ewma_diff": home.win_ewma - away.win_ewma,
            "home_split_win_ewma": home.home_win_ewma,
            "away_split_win_ewma": away.away_win_ewma,
            "win_pct_diff": home.win_pct() - away.win_pct(),
            "margin_pg_diff": home.margin_pg() - away.margin_pg(),
            "pythag_diff": home.pythag() - away.pythag(),
            "prev_win_pct_diff": home.prev_win_pct - away.prev_win_pct,
            "prev_net_rtg_diff": home.prev_net_rtg - away.prev_net_rtg,
            "sos_elo_diff": home.sos_elo() - away.sos_elo(),
            "home_rest_days": home_rest,
            "away_rest_days": away_rest,
            "rest_diff": home_rest - away_rest,
            "home_b2b": 1.0 if home_rest <= 1.0 else 0.0,
            "away_b2b": 1.0 if away_rest <= 1.0 else 0.0,
            "home_games_last7": float(home.games_in_last7(game_date)),
            "away_games_last7": float(away.games_in_last7(game_date)),
            "home_3in4": 1.0 if home.is_3in4(game_date) else 0.0,
            "away_3in4": 1.0 if away.is_3in4(game_date) else 0.0,
            "home_streak": float(max(min(home.streak, 10), -10)),
            "away_streak": float(max(min(away.streak, 10), -10)),
            "home_games_played": float(home.games_played),
            "away_games_played": float(away.games_played),
            "games_played_min": float(min(home.games_played, away.games_played)),
            "season_frac": season_frac,
            "is_playoff": 1.0 if int(game.get("season_type") or 2) == 3 else 0.0,
            "neutral_site": 1.0 if neutral else 0.0,
            "is_conference": 1.0 if is_conf else 0.0,
            "conf_win_pct_diff": home.conf_win_pct() - away.conf_win_pct(),
            "nonconf_win_pct_diff": home.nonconf_win_pct() - away.nonconf_win_pct(),
            "league_ppg": self.league_ppg,
            "exp_total_env": (home.pf_fast + home.pa_fast + away.pf_fast + away.pa_fast) / 2.0,
            "elo_x_season_frac": elo_diff * season_frac,
            "elo_x_is_conf": elo_diff * (1.0 if is_conf else 0.0),
            "h2h_home_win_rate": h2h_rate,
            "h2h_margin_ewma": home.h2h_margin.get(away.franchise, 0.0),
            "net_rtg_trend_diff": (
                (home.ortg_fast - home.drtg_fast) - (home.ortg_slow - home.drtg_slow)
            )
            - ((away.ortg_fast - away.drtg_fast) - (away.ortg_slow - away.drtg_slow)),
            "ortg_trend_diff": (home.ortg_fast - home.ortg_slow)
            - (away.ortg_fast - away.ortg_slow),
            "drtg_trend_diff": (home.drtg_fast - home.drtg_slow)
            - (away.drtg_fast - away.drtg_slow),
            "elo_mom5_diff": home.elo_momentum5() - away.elo_momentum5(),
            "adj_margin_ewma_diff": home.adj_margin_ewma - away.adj_margin_ewma,
            "blowout_rate_diff": home.blowout_ewma - away.blowout_ewma,
            "close_win_ewma_diff": home.close_win_ewma - away.close_win_ewma,
            "margin_vol_diff": home.margin_volatility() - away.margin_volatility(),
            "scoring_vol_diff": home.scoring_volatility() - away.scoring_volatility(),
            "last5_win_pct_diff": home.last5_win_pct() - away.last5_win_pct(),
            "home_stand_len": home_stand,
            "away_road_trip_len": road_trip,
            "home_conf_sos": home.conf_sos(),
            "away_conf_sos": away.conf_sos(),
        }
        return features

    def update_after_game(self, game: dict[str, Any]) -> None:
        season = int(game.get("season") or 0)
        game_date = _parse_date(str(game.get("date") or "")) or date_cls(season, 1, 15)
        home = self.team(str(game["home"]))
        away = self.team(str(game["away"]))
        home.roll_season(season)
        away.roll_season(season)

        if game.get("home_abbr"):
            self.register_abbr(str(game["home_abbr"]), str(game["home"]))
        if game.get("away_abbr"):
            self.register_abbr(str(game["away_abbr"]), str(game["away"]))

        home_conf = str(game.get("home_conference_id") or home.conference_id or "")
        away_conf = str(game.get("away_conference_id") or away.conference_id or "")
        if home_conf:
            home.conference_id = home_conf
        if away_conf:
            away.conference_id = away_conf

        home_score = float(game["home_score"])
        away_score = float(game["away_score"])
        home_win = home_score > away_score
        margin = abs(home_score - away_score)
        signed_margin = home_score - away_score
        neutral = bool(game.get("neutral_site"))
        is_conf = bool(game.get("conference_game"))
        if not is_conf and home_conf and away_conf:
            is_conf = home_conf == away_conf

        home_edge = home.elo - away.elo + (0.0 if neutral else ELO_HOME_ADV)
        exp_home = 1.0 / (1.0 + 10.0 ** (-home_edge / 400.0))
        winner_edge = home_edge if home_win else -home_edge
        mov_mult = math.log(max(margin, 1.0) + 1.0) * (
            2.2 / (0.001 * max(winner_edge, 0.0) + 2.2)
        )
        delta = ELO_K * mov_mult * ((1.0 if home_win else 0.0) - exp_home)
        pre_home_elo = home.elo
        pre_away_elo = away.elo
        home.elo += delta
        away.elo -= delta
        for team in (home, away):
            team.elo_hist.append(team.elo)
            if len(team.elo_hist) > 8:
                team.elo_hist = team.elo_hist[-8:]

        for team, pf, pa in ((home, home_score, away_score), (away, away_score, home_score)):
            team.pf_fast += ALPHA_FAST * (pf - team.pf_fast)
            team.pa_fast += ALPHA_FAST * (pa - team.pa_fast)
            team.pf_slow += ALPHA_SLOW * (pf - team.pf_slow)
            team.pa_slow += ALPHA_SLOW * (pa - team.pa_slow)

        home_box = game.get("home_box")
        away_box = game.get("away_box")
        home_poss = away_poss = None
        if isinstance(home_box, dict) and isinstance(away_box, dict):
            home_poss = _possessions(home_box, away_box)
            away_poss = _possessions(away_box, home_box)
        if not home_poss or not away_poss or home_poss < 40 or away_poss < 40:
            # Score-based pace proxy: scale assumed possessions by game total.
            game_total = home_score + away_score
            scale = game_total / max(2.0 * self.league_ppg, 1.0)
            home_poss = away_poss = ASSUMED_POSSESSIONS * max(0.75, min(scale, 1.35))

        pace = (float(home_poss) + float(away_poss)) / 2.0
        for team, pf, pa, poss, opp_poss in (
            (home, home_score, away_score, float(home_poss), float(away_poss)),
            (away, away_score, home_score, float(away_poss), float(home_poss)),
        ):
            team.pace_ewma += ALPHA_FAST * (pace - team.pace_ewma)
            ortg = pf / poss * 100.0
            drtg = pa / opp_poss * 100.0
            team.ortg_fast += ALPHA_FAST * (ortg - team.ortg_fast)
            team.drtg_fast += ALPHA_FAST * (drtg - team.drtg_fast)
            team.ortg_slow += ALPHA_SLOW * (ortg - team.ortg_slow)
            team.drtg_slow += ALPHA_SLOW * (drtg - team.drtg_slow)

        for team, won, pf, signed_for in (
            (home, home_win, home_score, signed_margin),
            (away, not home_win, away_score, -signed_margin),
        ):
            team.win_ewma += ALPHA_WIN * ((1.0 if won else 0.0) - team.win_ewma)
            team.recent_results.append(1 if won else 0)
            if len(team.recent_results) > 10:
                team.recent_results = team.recent_results[-10:]
            team.recent_margins.append(signed_for)
            if len(team.recent_margins) > 12:
                team.recent_margins = team.recent_margins[-12:]
            team.recent_pf.append(pf)
            if len(team.recent_pf) > 12:
                team.recent_pf = team.recent_pf[-12:]
            if margin <= CLOSE_MARGIN:
                team.close_win_ewma += PLAYER_PROXY_ALPHA * (
                    (1.0 if won else 0.0) - team.close_win_ewma
                )
            team.blowout_ewma += ALPHA_WIN * (
                (1.0 if margin >= BLOWOUT_MARGIN else 0.0) - team.blowout_ewma
            )
            if won:
                team.streak = team.streak + 1 if team.streak >= 0 else 1
                team.wins += 1
            else:
                team.streak = team.streak - 1 if team.streak <= 0 else -1
                team.losses += 1
            team.points_for += pf
            team.points_against += (away_score if team is home else home_score)
            team.games_played += 1
            team.last_game_date = game_date.isoformat()
            team.recent_dates.append(game_date.isoformat())
            if len(team.recent_dates) > 14:
                team.recent_dates = team.recent_dates[-14:]

        if not neutral:
            home.home_win_ewma += ALPHA_WIN * ((1.0 if home_win else 0.0) - home.home_win_ewma)
            away.away_win_ewma += ALPHA_WIN * ((0.0 if home_win else 1.0) - away.away_win_ewma)
            home.venue_streak = home.venue_streak + 1 if home.venue_streak > 0 else 1
            away.venue_streak = away.venue_streak - 1 if away.venue_streak < 0 else -1
        else:
            home.venue_streak = 0
            away.venue_streak = 0

        home.sos_elo_sum += pre_away_elo
        away.sos_elo_sum += pre_home_elo

        if is_conf:
            if home_win:
                home.conf_wins += 1
                away.conf_losses += 1
            else:
                home.conf_losses += 1
                away.conf_wins += 1
            home.conf_sos_sum += pre_away_elo
            away.conf_sos_sum += pre_home_elo
            home.conf_games += 1
            away.conf_games += 1
        else:
            if home_win:
                home.nonconf_wins += 1
                away.nonconf_losses += 1
            else:
                home.nonconf_losses += 1
                away.nonconf_wins += 1

        # opponent-adjusted margin (Elo residual proxy)
        expected_margin = (home_edge / 400.0) * 11.0  # rough Elo->points
        home.adj_margin_ewma += ALPHA_FAST * (
            (signed_margin - expected_margin) - home.adj_margin_ewma
        )
        away.adj_margin_ewma += ALPHA_FAST * (
            (-signed_margin + expected_margin) - away.adj_margin_ewma
        )

        # h2h from home perspective
        record = home.h2h.setdefault(away.franchise, [0, 0])
        if home_win:
            record[0] += 1
        else:
            record[1] += 1
        away_record = away.h2h.setdefault(home.franchise, [0, 0])
        if home_win:
            away_record[1] += 1
        else:
            away_record[0] += 1
        prev = home.h2h_margin.get(away.franchise, 0.0)
        home.h2h_margin[away.franchise] = prev + ALPHA_H2H_MARGIN * (signed_margin - prev)
        away.h2h_margin[home.franchise] = -home.h2h_margin[away.franchise]

        self.league_ppg += ALPHA_LEAGUE * (
            (home_score + away_score) / 2.0 - self.league_ppg
        )
        self.league_pace += ALPHA_LEAGUE * (pace - self.league_pace)

    def to_dict(self) -> dict[str, Any]:
        return {
            "league_ppg": self.league_ppg,
            "league_pace": self.league_pace,
            "abbr_to_id": dict(self.abbr_to_id),
            "teams": {key: team.to_dict() for key, team in self.teams.items()},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CbbFeatureEngine":
        engine = cls()
        engine.league_ppg = float(payload.get("league_ppg") or LEAGUE_PPG)
        engine.league_pace = float(payload.get("league_pace") or LEAGUE_PACE)
        engine.abbr_to_id = {
            str(k): str(v) for k, v in dict(payload.get("abbr_to_id") or {}).items()
        }
        for key, team_payload in dict(payload.get("teams") or {}).items():
            engine.teams[str(key)] = TeamState.from_dict(team_payload)
        return engine
