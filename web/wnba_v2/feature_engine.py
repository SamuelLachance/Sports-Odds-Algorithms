"""Walk-forward WNBA feature engine shared by offline training and live inference.

Consumes games chronologically (1997+). For each game it can produce a feature
row using only state accumulated strictly before that date, then folds the
result into state. State is JSON-serializable so training can snapshot
end-of-season state and the live path replays only the current season.

Game dict schema (merged from data.wnba.com results + ESPN boxes):
  date (ISO), season, season_type (2 reg / 3 post), home, away (franchise keys),
  home_score, away_score, neutral_site (optional),
  home_box / away_box (optional dicts: fgm fga tpm tpa ftm fta orb drb tov ast).
"""

from __future__ import annotations

import math
from datetime import date as date_cls
from typing import Any

from web.wnba_v2.arenas import market_coords

LEAGUE_ELO = 1500.0
ELO_K = 24.0
ELO_HOME_ADV = 70.0
ELO_SEASON_CARRYOVER = 0.70

ALPHA_FAST = 0.22
ALPHA_SLOW = 0.08
ALPHA_WIN = 0.12
ALPHA_LEAGUE = 0.02

LEAGUE_PPG = 78.0
LEAGUE_PACE = 78.0
LEAGUE_ORTG = 100.0
LEAGUE_EFG = 0.47
LEAGUE_TOV_RATE = 0.155
LEAGUE_ORB_PCT = 0.28
LEAGUE_FT_RATE = 0.22

SEASON_GAMES_NOMINAL = 40.0

FEATURE_COLUMNS: tuple[str, ...] = (
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
    "efg_for_diff",
    "efg_against_diff",
    "tov_for_diff",
    "tov_against_diff",
    "orb_for_diff",
    "orb_against_diff",
    "ftr_for_diff",
    "ftr_against_diff",
    "win_ewma_diff",
    "home_split_win_ewma",
    "away_split_win_ewma",
    "win_pct_diff",
    "margin_pg_diff",
    "pythag_diff",
    "prev_win_pct_diff",
    "prev_net_rtg_diff",
    "sos_elo_diff",
    "home_rest_days",
    "away_rest_days",
    "rest_diff",
    "home_b2b",
    "away_b2b",
    "home_games_last7",
    "away_games_last7",
    "home_travel_km",
    "away_travel_km",
    "home_streak",
    "away_streak",
    "home_games_played",
    "away_games_played",
    "games_played_min",
    "season_frac",
    "is_playoff",
    "neutral_site",
    "home_expansion",
    "away_expansion",
    "league_ppg",
    "exp_total_env",
    "elo_x_season_frac",
    "h2h_home_win_rate",
)


def _parse_date(value: str) -> date_cls | None:
    try:
        return date_cls.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    return 6371.0 * 2.0 * math.asin(min(1.0, math.sqrt(h)))


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
        "efg_for", "efg_against", "tov_for", "tov_against",
        "orb_for", "orb_against", "ftr_for", "ftr_against",
        "win_ewma", "home_win_ewma", "away_win_ewma",
        "wins", "losses", "points_for", "points_against",
        "prev_win_pct", "prev_net_rtg", "sos_elo_sum",
        "last_game_date", "recent_dates", "last_market", "streak",
        "games_played", "season_seen", "first_season", "h2h",
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
        self.efg_for = LEAGUE_EFG
        self.efg_against = LEAGUE_EFG
        self.tov_for = LEAGUE_TOV_RATE
        self.tov_against = LEAGUE_TOV_RATE
        self.orb_for = LEAGUE_ORB_PCT
        self.orb_against = LEAGUE_ORB_PCT
        self.ftr_for = LEAGUE_FT_RATE
        self.ftr_against = LEAGUE_FT_RATE
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
        self.last_market: tuple[float, float] | None = None
        self.streak = 0
        self.games_played = 0
        self.season_seen = -1
        self.first_season = -1
        self.h2h: dict[str, list[int]] = {}

    # -- season lifecycle ---------------------------------------------------

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

    # -- helpers -------------------------------------------------------------

    def win_pct(self) -> float:
        total = self.wins + self.losses
        return self.wins / total if total else 0.5

    def margin_pg(self) -> float:
        total = self.wins + self.losses
        if not total:
            return 0.0
        return (self.points_for - self.points_against) / total

    def pythag(self) -> float:
        pf = max(self.points_for, 1.0)
        pa = max(self.points_against, 1.0)
        return pf**10.0 / (pf**10.0 + pa**10.0)

    def sos_elo(self) -> float:
        total = self.wins + self.losses
        return self.sos_elo_sum / total if total else LEAGUE_ELO

    def rest_days(self, game_date: date_cls) -> float:
        prior = _parse_date(self.last_game_date)
        if prior is None:
            return 5.0
        return float(min((game_date - prior).days, 10))

    def games_in_last7(self, game_date: date_cls) -> int:
        count = 0
        for iso in self.recent_dates:
            played = _parse_date(iso)
            if played and 0 < (game_date - played).days <= 7:
                count += 1
        return count

    def to_dict(self) -> dict[str, Any]:
        return {
            "franchise": self.franchise,
            "elo": self.elo,
            "pf_fast": self.pf_fast, "pa_fast": self.pa_fast,
            "pf_slow": self.pf_slow, "pa_slow": self.pa_slow,
            "ortg_fast": self.ortg_fast, "drtg_fast": self.drtg_fast,
            "ortg_slow": self.ortg_slow, "drtg_slow": self.drtg_slow,
            "pace_ewma": self.pace_ewma,
            "efg_for": self.efg_for, "efg_against": self.efg_against,
            "tov_for": self.tov_for, "tov_against": self.tov_against,
            "orb_for": self.orb_for, "orb_against": self.orb_against,
            "ftr_for": self.ftr_for, "ftr_against": self.ftr_against,
            "win_ewma": self.win_ewma,
            "home_win_ewma": self.home_win_ewma, "away_win_ewma": self.away_win_ewma,
            "wins": self.wins, "losses": self.losses,
            "points_for": self.points_for, "points_against": self.points_against,
            "prev_win_pct": self.prev_win_pct, "prev_net_rtg": self.prev_net_rtg,
            "sos_elo_sum": self.sos_elo_sum,
            "last_game_date": self.last_game_date,
            "recent_dates": list(self.recent_dates[-12:]),
            "last_market": list(self.last_market) if self.last_market else None,
            "streak": self.streak,
            "games_played": self.games_played,
            "season_seen": self.season_seen,
            "first_season": self.first_season,
            "h2h": {k: list(v) for k, v in self.h2h.items()},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TeamState":
        state = cls(str(payload.get("franchise") or ""))
        for key, value in payload.items():
            if key == "recent_dates":
                state.recent_dates = [str(v) for v in value]
            elif key == "h2h":
                state.h2h = {str(k): [int(x) for x in v] for k, v in dict(value).items()}
            elif key == "last_market":
                state.last_market = (float(value[0]), float(value[1])) if value else None
            elif hasattr(state, key):
                setattr(state, key, value)
        return state


class WnbaFeatureEngine:
    """League-wide walk-forward state with leak-free feature extraction."""

    def __init__(self) -> None:
        self.teams: dict[str, TeamState] = {}
        self.league_ppg = LEAGUE_PPG
        self.league_pace = LEAGUE_PACE

    def team(self, franchise: str) -> TeamState:
        state = self.teams.get(franchise)
        if state is None:
            state = TeamState(franchise)
            self.teams[franchise] = state
        return state

    # -- features ------------------------------------------------------------

    def features_for_game(self, game: dict[str, Any]) -> dict[str, float]:
        season = int(game.get("season") or 0)
        game_date = _parse_date(str(game.get("date") or "")) or date_cls(season, 7, 1)
        home = self.team(str(game["home"]))
        away = self.team(str(game["away"]))
        home.roll_season(season)
        away.roll_season(season)

        neutral = bool(game.get("neutral_site"))
        home_rest = home.rest_days(game_date)
        away_rest = away.rest_days(game_date)

        home_market = market_coords(home.franchise, season)
        away_market = market_coords(away.franchise, season)
        venue = home_market
        home_travel = 0.0
        away_travel = 0.0
        if venue:
            home_from = home.last_market or home_market
            away_from = away.last_market or away_market
            if home_from:
                home_travel = _haversine_km(home_from, venue)
            if away_from:
                away_travel = _haversine_km(away_from, venue)

        h2h_record = home.h2h.get(away.franchise) or [0, 0]
        h2h_total = h2h_record[0] + h2h_record[1]
        h2h_rate = h2h_record[0] / h2h_total if h2h_total else 0.5

        season_frac = min(home.games_played, away.games_played) / SEASON_GAMES_NOMINAL
        elo_diff = home.elo - away.elo + (0.0 if neutral else ELO_HOME_ADV)

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
            "efg_for_diff": home.efg_for - away.efg_for,
            "efg_against_diff": home.efg_against - away.efg_against,
            "tov_for_diff": home.tov_for - away.tov_for,
            "tov_against_diff": home.tov_against - away.tov_against,
            "orb_for_diff": home.orb_for - away.orb_for,
            "orb_against_diff": home.orb_against - away.orb_against,
            "ftr_for_diff": home.ftr_for - away.ftr_for,
            "ftr_against_diff": home.ftr_against - away.ftr_against,
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
            "home_travel_km": home_travel,
            "away_travel_km": away_travel,
            "home_streak": float(max(min(home.streak, 8), -8)),
            "away_streak": float(max(min(away.streak, 8), -8)),
            "home_games_played": float(home.games_played),
            "away_games_played": float(away.games_played),
            "games_played_min": float(min(home.games_played, away.games_played)),
            "season_frac": season_frac,
            "is_playoff": 1.0 if int(game.get("season_type") or 2) == 3 else 0.0,
            "neutral_site": 1.0 if neutral else 0.0,
            "home_expansion": 1.0 if home.first_season == season else 0.0,
            "away_expansion": 1.0 if away.first_season == season else 0.0,
            "league_ppg": self.league_ppg,
            "exp_total_env": (home.pf_fast + home.pa_fast + away.pf_fast + away.pa_fast)
            / 2.0,
            "elo_x_season_frac": elo_diff * season_frac,
            "h2h_home_win_rate": h2h_rate,
        }
        return features

    # -- state update ---------------------------------------------------------

    def update_after_game(self, game: dict[str, Any]) -> None:
        season = int(game.get("season") or 0)
        game_date = _parse_date(str(game.get("date") or "")) or date_cls(season, 7, 1)
        home = self.team(str(game["home"]))
        away = self.team(str(game["away"]))
        home.roll_season(season)
        away.roll_season(season)

        home_score = float(game["home_score"])
        away_score = float(game["away_score"])
        home_win = home_score > away_score
        margin = abs(home_score - away_score)
        neutral = bool(game.get("neutral_site"))

        # Elo with margin-of-victory multiplier (538 style)
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

        # score EWMAs
        for team, pf, pa in ((home, home_score, away_score), (away, away_score, home_score)):
            team.pf_fast += ALPHA_FAST * (pf - team.pf_fast)
            team.pa_fast += ALPHA_FAST * (pa - team.pa_fast)
            team.pf_slow += ALPHA_SLOW * (pf - team.pf_slow)
            team.pa_slow += ALPHA_SLOW * (pa - team.pa_slow)

        # box-derived efficiency + four factors
        home_box = game.get("home_box")
        away_box = game.get("away_box")
        if isinstance(home_box, dict) and isinstance(away_box, dict):
            home_poss = _possessions(home_box, away_box)
            away_poss = _possessions(away_box, home_box)
            if home_poss and away_poss and home_poss > 20 and away_poss > 20:
                pace = (home_poss + away_poss) / 2.0
                for team, box, opp_box, pf, pa, poss, opp_poss in (
                    (home, home_box, away_box, home_score, away_score, home_poss, away_poss),
                    (away, away_box, home_box, away_score, home_score, away_poss, home_poss),
                ):
                    team.pace_ewma += ALPHA_FAST * (pace - team.pace_ewma)
                    ortg = pf / poss * 100.0
                    drtg = pa / opp_poss * 100.0
                    team.ortg_fast += ALPHA_FAST * (ortg - team.ortg_fast)
                    team.drtg_fast += ALPHA_FAST * (drtg - team.drtg_fast)
                    team.ortg_slow += ALPHA_SLOW * (ortg - team.ortg_slow)
                    team.drtg_slow += ALPHA_SLOW * (drtg - team.drtg_slow)

                    fga = float(box.get("fga") or 0.0)
                    if fga > 0:
                        efg = (float(box.get("fgm") or 0.0) + 0.5 * float(box.get("tpm") or 0.0)) / fga
                        team.efg_for += ALPHA_FAST * (efg - team.efg_for)
                        team.ftr_for += ALPHA_FAST * (
                            float(box.get("fta") or 0.0) / fga - team.ftr_for
                        )
                    opp_fga = float(opp_box.get("fga") or 0.0)
                    if opp_fga > 0:
                        opp_efg = (
                            float(opp_box.get("fgm") or 0.0)
                            + 0.5 * float(opp_box.get("tpm") or 0.0)
                        ) / opp_fga
                        team.efg_against += ALPHA_FAST * (opp_efg - team.efg_against)
                        team.ftr_against += ALPHA_FAST * (
                            float(opp_box.get("fta") or 0.0) / opp_fga - team.ftr_against
                        )
                    if poss > 0:
                        team.tov_for += ALPHA_FAST * (
                            float(box.get("tov") or 0.0) / poss - team.tov_for
                        )
                    if opp_poss > 0:
                        team.tov_against += ALPHA_FAST * (
                            float(opp_box.get("tov") or 0.0) / opp_poss - team.tov_against
                        )
                    orb = float(box.get("orb") or 0.0)
                    opp_drb = float(opp_box.get("drb") or 0.0)
                    if orb + opp_drb > 0:
                        team.orb_for += ALPHA_FAST * (orb / (orb + opp_drb) - team.orb_for)
                    opp_orb = float(opp_box.get("orb") or 0.0)
                    drb = float(box.get("drb") or 0.0)
                    if opp_orb + drb > 0:
                        team.orb_against += ALPHA_FAST * (
                            opp_orb / (opp_orb + drb) - team.orb_against
                        )

        # win EWMAs / records / streaks / rest bookkeeping
        for team, won, was_home, opp_pre_elo in (
            (home, home_win, True, pre_away_elo),
            (away, not home_win, False, pre_home_elo),
        ):
            result = 1.0 if won else 0.0
            team.win_ewma += ALPHA_WIN * (result - team.win_ewma)
            if was_home:
                team.home_win_ewma += ALPHA_WIN * (result - team.home_win_ewma)
            else:
                team.away_win_ewma += ALPHA_WIN * (result - team.away_win_ewma)
            if won:
                team.wins += 1
                team.streak = team.streak + 1 if team.streak >= 0 else 1
            else:
                team.losses += 1
                team.streak = team.streak - 1 if team.streak <= 0 else -1
            team.sos_elo_sum += opp_pre_elo
            team.last_game_date = str(game.get("date") or "")
            team.recent_dates.append(str(game.get("date") or ""))
            if len(team.recent_dates) > 12:
                team.recent_dates = team.recent_dates[-12:]
            team.games_played += 1

        home.points_for += home_score
        home.points_against += away_score
        away.points_for += away_score
        away.points_against += home_score

        venue = market_coords(home.franchise, season)
        if venue:
            home.last_market = venue
            away.last_market = venue

        # head-to-head (recency-capped)
        record = home.h2h.setdefault(away.franchise, [0, 0])
        record[0 if home_win else 1] += 1
        if record[0] + record[1] > 12:
            scale = 12.0 / (record[0] + record[1])
            record[0] = int(round(record[0] * scale))
            record[1] = int(round(record[1] * scale))
        rev = away.h2h.setdefault(home.franchise, [0, 0])
        rev[1 if home_win else 0] += 1
        if rev[0] + rev[1] > 12:
            scale = 12.0 / (rev[0] + rev[1])
            rev[0] = int(round(rev[0] * scale))
            rev[1] = int(round(rev[1] * scale))

        # league scoring environment
        game_total = (home_score + away_score) / 2.0
        self.league_ppg += ALPHA_LEAGUE * (game_total - self.league_ppg)

    # -- serialization ---------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "teams": {key: team.to_dict() for key, team in self.teams.items()},
            "league_ppg": self.league_ppg,
            "league_pace": self.league_pace,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "WnbaFeatureEngine":
        engine = cls()
        engine.league_ppg = float(payload.get("league_ppg") or LEAGUE_PPG)
        engine.league_pace = float(payload.get("league_pace") or LEAGUE_PACE)
        for key, team_payload in dict(payload.get("teams") or {}).items():
            engine.teams[str(key)] = TeamState.from_dict(team_payload)
        return engine
