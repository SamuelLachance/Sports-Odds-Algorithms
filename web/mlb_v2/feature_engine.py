"""Walk-forward MLB feature engine shared by offline training and live inference.

The engine consumes seasons chronologically. For each date it can produce
feature rows for that day's games using only state accumulated strictly
before that date (morning-of-slate semantics, no leakage), then folds the
day's results into state.

State is JSON-serializable so a trained artifact can snapshot end-of-season
state and the live path can replay only the current season on top of it.
"""

from __future__ import annotations

import math
from typing import Any

from web.mlb_v2.venues import VENUE_COORDS
from web.v2_schedule_utils import count_games_in_last_n_days

LEAGUE_ELO = 1500.0
ELO_K = 4.0
ELO_HOME_ADV = 24.0
ELO_SEASON_CARRYOVER = 0.70

ALPHA_RUNS_FAST = 0.20
ALPHA_RUNS_SLOW = 0.06
ALPHA_BATTING = 0.10
ALPHA_BULLPEN = 0.10
ALPHA_WIN = 0.08
ALPHA_LEAGUE = 0.02
ALPHA_SP_FORM = 0.25

LEAGUE_RPG_DEFAULT = 4.45
LEAGUE_OBP = 0.315
LEAGUE_SLG = 0.405
LEAGUE_BA = 0.245
LEAGUE_OPS = LEAGUE_OBP + LEAGUE_SLG  # ~0.720
LEAGUE_ISO = LEAGUE_SLG - LEAGUE_BA  # ~0.160
LEAGUE_XBH_RATE = 0.078
LEAGUE_HR_PG = 1.15
LEAGUE_BB_RATE = 0.082
LEAGUE_SO_RATE = 0.222
LEAGUE_BULLPEN_RA9 = 4.20
LEAGUE_BULLPEN_K9 = 8.8

FIP_CONSTANT = 3.10
LEAGUE_FIP = 4.20
ROOKIE_FIP = 4.55
LEAGUE_SP_K_PCT = 0.215
LEAGUE_SP_BB_PCT = 0.077
SP_PRIOR_WEIGHT_OUTS = 120  # ~40 IP of prior belief
SP_PRIOR_SEASON_SHRINK_OUTS = 150

PARK_WINDOW_GAMES = 250
PARK_SHRINK_GAMES = 100

FEATURE_COLUMNS: tuple[str, ...] = (
    "elo_diff",
    "net_fast_diff",
    "net_slow_diff",
    "rs_fast_diff",
    "ra_fast_diff",
    "obp_diff",
    "slg_diff",
    "hr_pg_diff",
    "bb_rate_diff",
    "so_rate_diff",
    "bullpen_ra9_diff",
    "bullpen_k9_diff",
    "win_ewma_diff",
    "home_split_win_ewma",
    "away_split_win_ewma",
    "prev_win_pct_diff",
    "prev_run_diff_pg_diff",
    "home_rest_days",
    "away_rest_days",
    "home_games_last7",
    "away_games_last7",
    "home_games_played",
    "away_games_played",
    "season_frac",
    "is_night",
    "is_double_header",
    "park_factor",
    "league_rpg",
    "sp_fip_blend_diff",
    "sp_fip_form_diff",
    "sp_k_pct_diff",
    "sp_bb_pct_diff",
    "sp_ip_season_diff",
    "sp_days_rest_diff",
    "home_sp_fip_blend",
    "away_sp_fip_blend",
    "home_sp_k_pct",
    "away_sp_k_pct",
    "home_sp_is_lhp",
    "away_sp_is_lhp",
    "home_sp_known",
    "away_sp_known",
    "home_sp_days_rest",
    "away_sp_days_rest",
    "sp_kbb_diff",
    "home_sp_k_x_away_so",
    "away_sp_k_x_home_so",
    "home_sp_starts",
    "away_sp_starts",
    "home_bullpen_outs_last3",
    "away_bullpen_outs_last3",
    "exp_total_env",
    "season_win_pct_diff",
    "pythag_diff",
    "home_travel_km",
    "away_travel_km",
    "venue_elevation_kft",
    "series_game_num",
    "adj_net_diff",
    "adj_rs_diff",
    "adj_ra_diff",
    "sp_hr_bf_diff",
    "sp_whip_bf_diff",
    "sp_outs_per_start_diff",
    "home_sp_outs_per_start",
    "away_sp_outs_per_start",
    # minimal high-signal additions (leakage-free, no kitchen sink)
    "iso_diff",
    "xbh_rate_diff",
    "elo_momentum_5_diff",
    "home_bullpen_outs_last5",
    "away_bullpen_outs_last5",
    "sp_fip_trend_diff",
    "sp_last_outs_diff",
    # platoon interactions (SP handedness × lineup rates)
    "platoon_k_matchup",
    "platoon_obp_matchup",
    # market (open/close when present; never invent open=close)
    "has_open_line",
    "has_steam",
    "ml_steam_pp",
    "spread_move",
    "total_move",
    "has_total",
    "mkt_home_prob",
    "has_market",
    "mkt_home_spread",
    "has_spread",
    # offense extras
    "ba_diff",
    "ops_diff",
    "pythag_luck_diff",
    # form / streaks
    "win_streak_diff",
    "loss_streak_diff",
    "form10_net_diff",
    "rs_l10_diff",
    "ra_l10_diff",
    # travel / schedule heuristics
    "travel_diff",
    "tz_shift_home",
    "tz_shift_away",
    "tz_diff",
    "home_stand_len",
    "away_road_trip_len",
    "cross_country",
    "is_series_opener",
    "is_rubber_game",
    "rest_diff",
    "games_last7_diff",
    # SP heuristics
    "home_sp_short_rest",
    "away_sp_short_rest",
    "home_sp_long_rest",
    "away_sp_long_rest",
    "home_sp_soft_debut",
    "away_sp_soft_debut",
    "sp_short_rest_diff",
    "home_sp_expected_short",
    "away_sp_expected_short",
    # bullpen heuristics
    "home_bp_fried",
    "away_bp_fried",
    "home_bp_outs_last1",
    "away_bp_outs_last1",
    "bp_fried_diff",
    "bp_fried_x_home_sp_short",
    # park / weather hooks
    "temp_c",
    "humidity",
    "precip_mm",
    "wind_kph",
    "has_weather",
    "is_roof",
    "cold_game",
    "hot_game",
    "windy_game",
    "coors_extreme",
    # cross-terms / seasonality
    "sp_fip_x_park_diff",
    "elo_x_season_frac",
    "steam_x_sp_known",
    "hr_env_x_sp_hr_diff",
    "early_season",
    "late_season",
    "april_noise",
    # umpire (rolling HP bias; has_ump=0 → league-neutral defaults)
    "has_ump",
    "ump_home_win_rate",
    "ump_rpg",
    "ump_games",
    # IL burden (point-in-time from transactions)
    "home_il_count",
    "away_il_count",
    "il_diff",
    "home_il_pitchers",
    "away_il_pitchers",
    "has_il",
    # Statcast priors (previous season only)
    "home_sp_prev_xera",
    "away_sp_prev_xera",
    "sp_xera_diff",
    "has_statcast_sp",
    "home_prev_xwoba",
    "away_prev_xwoba",
    "xwoba_diff",
    "has_statcast_offense",
    # market / bullpen heuristics
    "steam_to_fav",
    "reverse_line_move",
    "chalk_steam",
    "home_no_closer_proxy",
    "away_no_closer_proxy",
)


def _ewma(current: float, value: float, alpha: float) -> float:
    return (1.0 - alpha) * current + alpha * value


def _travel_km(from_venue: int | None, to_venue: int | None) -> float:
    if not from_venue or not to_venue or from_venue == to_venue:
        return 0.0
    src = VENUE_COORDS.get(int(from_venue))
    dst = VENUE_COORDS.get(int(to_venue))
    if not src or not dst:
        return 0.0
    lat1, lon1, _ = src
    lat2, lon2, _ = dst
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    return 6371.0 * 2.0 * math.asin(min(math.sqrt(a), 1.0))


def _tz_hours(venue_id: int | None) -> float:
    """Rough UTC offset hours from venue longitude (15 deg ≈ 1 hour)."""
    if not venue_id:
        return -5.0
    coords = VENUE_COORDS.get(int(venue_id))
    if not coords:
        return -5.0
    return float(coords[1]) / 15.0


def _amer_implied(ml: float) -> float:
    if not math.isfinite(ml) or abs(ml) < 100:
        return 0.5
    return (-ml) / (-ml + 100.0) if ml < 0 else 100.0 / (ml + 100.0)


def _optional_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _mean_or(values: list[float], default: float) -> float:
    if not values:
        return default
    return float(sum(values) / len(values))


class TeamState:
    __slots__ = (
        "elo",
        "rs_fast",
        "rs_slow",
        "ra_fast",
        "ra_slow",
        "obp",
        "slg",
        "hr_pg",
        "bb_rate",
        "so_rate",
        "bullpen_ra9",
        "bullpen_k9",
        "win_ewma",
        "home_win_ewma",
        "away_win_ewma",
        "last_game_date",
        "recent_dates",
        "games_played",
        "prev_win_pct",
        "prev_run_diff_pg",
        "season_wins",
        "season_losses",
        "season_rs",
        "season_ra",
        "bullpen_recent",
        "last_venue_id",
        "recent_opponents",
        "adj_rs",
        "adj_ra",
        "iso",
        "xbh_rate",
        "elo_deltas",
        "ba",
        "ops",
        "recent_results",
        "recent_rs",
        "recent_ra",
        "home_stand_len",
        "road_trip_len",
    )

    def __init__(self) -> None:
        self.elo = LEAGUE_ELO
        self.rs_fast = LEAGUE_RPG_DEFAULT
        self.rs_slow = LEAGUE_RPG_DEFAULT
        self.ra_fast = LEAGUE_RPG_DEFAULT
        self.ra_slow = LEAGUE_RPG_DEFAULT
        self.obp = LEAGUE_OBP
        self.slg = LEAGUE_SLG
        self.hr_pg = LEAGUE_HR_PG
        self.bb_rate = LEAGUE_BB_RATE
        self.so_rate = LEAGUE_SO_RATE
        self.bullpen_ra9 = LEAGUE_BULLPEN_RA9
        self.bullpen_k9 = LEAGUE_BULLPEN_K9
        self.win_ewma = 0.5
        self.home_win_ewma = 0.5
        self.away_win_ewma = 0.5
        self.last_game_date: str | None = None
        self.recent_dates: list[str] = []
        self.games_played = 0
        self.prev_win_pct = 0.5
        self.prev_run_diff_pg = 0.0
        self.season_wins = 0
        self.season_losses = 0
        self.season_rs = 0.0
        self.season_ra = 0.0
        self.bullpen_recent: list[list[Any]] = []  # [date, outs] most recent last
        self.last_venue_id: int | None = None
        self.recent_opponents: list[int] = []  # most recent last
        self.adj_rs = 0.0  # runs scored vs opponent-allowed baseline (EWMA)
        self.adj_ra = 0.0  # runs allowed vs opponent-scored baseline (EWMA)
        self.iso = LEAGUE_ISO
        self.xbh_rate = LEAGUE_XBH_RATE
        self.elo_deltas: list[float] = []  # post-game elo deltas, most recent last
        self.ba = LEAGUE_BA
        self.ops = LEAGUE_OPS
        self.recent_results: list[int] = []  # 1/0 last 10, most recent last
        self.recent_rs: list[float] = []
        self.recent_ra: list[float] = []
        self.home_stand_len = 0
        self.road_trip_len = 0

    def bullpen_outs_since(self, floor_date: str) -> float:
        return float(sum(outs for d, outs in self.bullpen_recent if d >= floor_date))

    def bullpen_outs_last_n(self, n: int = 5) -> float:
        if not self.bullpen_recent:
            return 0.0
        return float(sum(outs for _, outs in self.bullpen_recent[-n:]))

    def elo_momentum(self, n: int = 5) -> float:
        if not self.elo_deltas:
            return 0.0
        return float(sum(self.elo_deltas[-n:]))

    def win_streak(self) -> float:
        count = 0
        for result in reversed(self.recent_results):
            if result:
                count += 1
            else:
                break
        return float(count)

    def loss_streak(self) -> float:
        count = 0
        for result in reversed(self.recent_results):
            if not result:
                count += 1
            else:
                break
        return float(count)

    def form10_net(self) -> float:
        if not self.recent_rs or not self.recent_ra:
            return 0.0
        return _mean_or(self.recent_rs, 0.0) - _mean_or(self.recent_ra, 0.0)

    def season_win_pct(self) -> float:
        games = self.season_wins + self.season_losses
        if games < 10:
            # blend with previous season for early stability
            return (self.season_wins + 10.0 * self.prev_win_pct) / (games + 10.0)
        return self.season_wins / games

    def pythag(self) -> float:
        if self.season_rs + self.season_ra < 40:
            return 0.5 + 0.08 * self.prev_run_diff_pg
        rs2 = self.season_rs**1.83
        ra2 = self.season_ra**1.83
        return rs2 / (rs2 + ra2) if (rs2 + ra2) > 0 else 0.5

    def pythag_luck(self) -> float:
        return self.season_win_pct() - self.pythag()

    def to_dict(self) -> dict[str, Any]:
        return {slot: getattr(self, slot) for slot in self.__slots__}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TeamState":
        state = cls()
        for slot in cls.__slots__:
            if slot in payload:
                setattr(state, slot, payload[slot])
        return state


class PitcherState:
    __slots__ = (
        "hand",
        "outs",
        "er",
        "so",
        "bb",
        "h",
        "hr",
        "hbp",
        "bf",
        "starts",
        "fip_form",
        "last_start_date",
        "prior_fip",
        "prior_outs",
        "prior_k_pct",
        "prior_bb_pct",
        "last_start_outs",
    )

    def __init__(self, hand: str = "R") -> None:
        self.hand = hand
        self.outs = 0
        self.er = 0
        self.so = 0
        self.bb = 0
        self.h = 0
        self.hr = 0
        self.hbp = 0
        self.bf = 0
        self.starts = 0
        self.fip_form = ROOKIE_FIP
        self.last_start_date: str | None = None
        self.prior_fip = ROOKIE_FIP
        self.prior_outs = 0.0
        self.prior_k_pct = LEAGUE_SP_K_PCT
        self.prior_bb_pct = LEAGUE_SP_BB_PCT
        self.last_start_outs = 16.0

    def to_dict(self) -> dict[str, Any]:
        return {slot: getattr(self, slot) for slot in self.__slots__}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PitcherState":
        state = cls()
        for slot in cls.__slots__:
            if slot in payload:
                setattr(state, slot, payload[slot])
        return state

    # -- derived quantities -------------------------------------------------
    def season_fip(self) -> float | None:
        if self.outs < 3:
            return None
        ip = self.outs / 3.0
        return (13.0 * self.hr + 3.0 * (self.bb + self.hbp) - 2.0 * self.so) / ip + FIP_CONSTANT

    def fip_blend(self) -> float:
        season = self.season_fip()
        if season is None:
            return self.prior_fip
        weight = self.outs / (self.outs + SP_PRIOR_WEIGHT_OUTS)
        return weight * season + (1.0 - weight) * self.prior_fip

    def k_pct(self) -> float:
        if self.bf < 20:
            return self.prior_k_pct
        season = self.so / self.bf
        weight = self.bf / (self.bf + 150.0)
        return weight * season + (1.0 - weight) * self.prior_k_pct

    def bb_pct(self) -> float:
        if self.bf < 20:
            return self.prior_bb_pct
        season = self.bb / self.bf
        weight = self.bf / (self.bf + 150.0)
        return weight * season + (1.0 - weight) * self.prior_bb_pct

    def hr_per_bf(self) -> float:
        league = 0.032
        if self.bf < 20:
            return league
        weight = self.bf / (self.bf + 250.0)
        return weight * (self.hr / self.bf) + (1.0 - weight) * league

    def whip_per_bf(self) -> float:
        """(H+BB)/BF shrunk to league — baserunners allowed rate."""
        league = 0.30
        if self.bf < 20:
            return league
        weight = self.bf / (self.bf + 150.0)
        return weight * ((self.h + self.bb) / self.bf) + (1.0 - weight) * league

    def outs_per_start(self) -> float:
        if self.starts < 2:
            return 16.0
        return self.outs / max(self.starts, 1)

    def add_appearance(self, log: dict[str, Any]) -> None:
        self.outs += int(log.get("outs") or 0)
        self.er += int(log.get("er") or 0)
        self.so += int(log.get("so") or 0)
        self.bb += int(log.get("bb") or 0)
        self.h += int(log.get("h") or 0)
        self.hr += int(log.get("hr") or 0)
        self.hbp += int(log.get("hbp") or 0)
        self.bf += int(log.get("bf") or 0)
        outs = int(log.get("outs") or 0)
        if int(log.get("gs") or 0) >= 1:
            self.starts += 1
            self.last_start_date = str(log.get("date") or self.last_start_date or "")
            self.last_start_outs = float(outs)
            if outs >= 3:
                ip = outs / 3.0
                game_fip = (
                    13.0 * int(log.get("hr") or 0)
                    + 3.0 * (int(log.get("bb") or 0) + int(log.get("hbp") or 0))
                    - 2.0 * int(log.get("so") or 0)
                ) / ip + FIP_CONSTANT
                game_fip = min(max(game_fip, -2.0), 12.0)
                self.fip_form = _ewma(self.fip_form, game_fip, ALPHA_SP_FORM)

    def roll_season(self) -> None:
        """Fold the finished season into the prior, then reset counters."""
        season = self.season_fip()
        if season is not None and self.outs > 0:
            weight = self.outs / (self.outs + SP_PRIOR_SEASON_SHRINK_OUTS)
            season_component = weight * season + (1.0 - weight) * LEAGUE_FIP
            if self.prior_outs > 0:
                self.prior_fip = 0.65 * season_component + 0.35 * self.prior_fip
            else:
                self.prior_fip = season_component
            self.prior_outs = 0.6 * self.prior_outs + self.outs
        if self.bf > 0:
            w = self.bf / (self.bf + 200.0)
            self.prior_k_pct = w * (self.so / self.bf) + (1 - w) * self.prior_k_pct
            self.prior_bb_pct = w * (self.bb / self.bf) + (1 - w) * self.prior_bb_pct
        self.outs = self.er = self.so = self.bb = self.h = self.hr = self.hbp = self.bf = 0
        self.starts = 0
        self.fip_form = 0.5 * self.fip_form + 0.5 * self.prior_fip
        self.last_start_date = None
        self.last_start_outs = 16.0


class VenueState:
    __slots__ = ("totals", "games")

    def __init__(self) -> None:
        self.totals: list[float] = []
        self.games = 0

    def add_game(self, total_runs: float) -> None:
        self.totals.append(total_runs)
        if len(self.totals) > PARK_WINDOW_GAMES:
            self.totals.pop(0)
        self.games += 1

    def factor(self, league_rpg: float) -> float:
        if not self.totals:
            return 1.0
        expected = 2.0 * league_rpg
        observed = sum(self.totals) / len(self.totals)
        n = len(self.totals)
        shrunk = (observed * n + expected * PARK_SHRINK_GAMES) / (n + PARK_SHRINK_GAMES)
        return shrunk / expected if expected > 0 else 1.0

    def to_dict(self) -> dict[str, Any]:
        return {"totals": self.totals, "games": self.games}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "VenueState":
        state = cls()
        state.totals = [float(x) for x in payload.get("totals") or []]
        state.games = int(payload.get("games") or 0)
        return state


class UmpState:
    """Rolling home-plate umpire environment (PIT-safe EWMA)."""

    __slots__ = ("home_wins", "games", "total_runs", "home_win_rate", "rpg")

    def __init__(self) -> None:
        self.home_wins = 0.0
        self.games = 0
        self.total_runs = 0.0
        self.home_win_rate = 0.5
        self.rpg = LEAGUE_RPG_DEFAULT * 2.0

    def observe(self, home_won: float, total_runs: float) -> None:
        self.games += 1
        self.home_wins += home_won
        self.total_runs += total_runs
        self.home_win_rate = _ewma(self.home_win_rate, home_won, 0.05)
        self.rpg = _ewma(self.rpg, total_runs, 0.05)

    def to_dict(self) -> dict[str, Any]:
        return {slot: getattr(self, slot) for slot in self.__slots__}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "UmpState":
        state = cls()
        for slot in cls.__slots__:
            if slot in payload:
                setattr(state, slot, payload[slot])
        return state


class MlbFeatureEngine:
    def __init__(self) -> None:
        self.teams: dict[int, TeamState] = {}
        self.pitchers: dict[int, PitcherState] = {}
        self.venues: dict[int, VenueState] = {}
        self.umps: dict[int, UmpState] = {}
        self.league_rpg = LEAGUE_RPG_DEFAULT
        self.season: int | None = None

    # -- state accessors ----------------------------------------------------
    def team(self, team_id: int) -> TeamState:
        if team_id not in self.teams:
            self.teams[team_id] = TeamState()
        return self.teams[team_id]

    def pitcher(self, pitcher_id: int, hand: str = "R") -> PitcherState:
        if pitcher_id not in self.pitchers:
            state = PitcherState(hand=hand)
            self.pitchers[pitcher_id] = state
        return self.pitchers[pitcher_id]

    def venue(self, venue_id: int) -> VenueState:
        if venue_id not in self.venues:
            self.venues[venue_id] = VenueState()
        return self.venues[venue_id]

    def ump(self, ump_id: int) -> UmpState:
        if ump_id not in self.umps:
            self.umps[ump_id] = UmpState()
        return self.umps[ump_id]

    # -- season transitions ---------------------------------------------------
    def begin_season(self, season: int) -> None:
        if self.season is not None and season != self.season:
            for team in self.teams.values():
                games = team.season_wins + team.season_losses
                if games > 0:
                    team.prev_win_pct = team.season_wins / games
                    team.prev_run_diff_pg = (team.season_rs - team.season_ra) / games
                team.elo = ELO_SEASON_CARRYOVER * team.elo + (1 - ELO_SEASON_CARRYOVER) * LEAGUE_ELO
                team.season_wins = team.season_losses = 0
                team.season_rs = team.season_ra = 0.0
                team.games_played = 0
                team.last_game_date = None
                team.recent_dates = []
                team.bullpen_recent = []
                team.last_venue_id = None
                team.recent_opponents = []
                team.elo_deltas = []
                team.recent_results = []
                team.recent_rs = []
                team.recent_ra = []
                team.home_stand_len = 0
                team.road_trip_len = 0
                team.adj_rs = 0.6 * team.adj_rs
                team.adj_ra = 0.6 * team.adj_ra
                # partial mean-reversion of form EWMAs toward league between seasons
                for attr, league_value in (
                    ("rs_fast", self.league_rpg),
                    ("rs_slow", self.league_rpg),
                    ("ra_fast", self.league_rpg),
                    ("ra_slow", self.league_rpg),
                    ("obp", LEAGUE_OBP),
                    ("slg", LEAGUE_SLG),
                    ("hr_pg", LEAGUE_HR_PG),
                    ("bb_rate", LEAGUE_BB_RATE),
                    ("so_rate", LEAGUE_SO_RATE),
                    ("bullpen_ra9", LEAGUE_BULLPEN_RA9),
                    ("bullpen_k9", LEAGUE_BULLPEN_K9),
                    ("iso", LEAGUE_ISO),
                    ("xbh_rate", LEAGUE_XBH_RATE),
                    ("ba", LEAGUE_BA),
                    ("ops", LEAGUE_OPS),
                ):
                    setattr(team, attr, 0.6 * getattr(team, attr) + 0.4 * league_value)
                team.win_ewma = 0.5 + 0.5 * (team.win_ewma - 0.5)
                team.home_win_ewma = 0.5 + 0.5 * (team.home_win_ewma - 0.5)
                team.away_win_ewma = 0.5 + 0.5 * (team.away_win_ewma - 0.5)
            for pitcher in self.pitchers.values():
                pitcher.roll_season()
        self.season = season

    # -- feature construction -------------------------------------------------
    def _rest_days(self, team: TeamState, game_date: str) -> float:
        if not team.last_game_date:
            return 3.0
        try:
            from datetime import date

            last = date.fromisoformat(team.last_game_date)
            current = date.fromisoformat(game_date)
            return float(min(max((current - last).days, 0), 10))
        except ValueError:
            return 1.0

    def _games_last7(self, team: TeamState, game_date: str) -> float:
        # Same morning-slate contract as NHL/NBA: prior games in (date-7, date).
        return float(count_games_in_last_n_days(team.recent_dates, game_date, days=7))

    def _pitcher_features(
        self, pitcher_id: int | None, game_date: str
    ) -> dict[str, float]:
        if pitcher_id is None or pitcher_id not in self.pitchers:
            return {
                "fip_blend": LEAGUE_FIP if pitcher_id is None else ROOKIE_FIP,
                "fip_form": LEAGUE_FIP if pitcher_id is None else ROOKIE_FIP,
                "k_pct": LEAGUE_SP_K_PCT,
                "bb_pct": LEAGUE_SP_BB_PCT,
                "ip_season": 0.0,
                "days_rest": 5.0,
                "is_lhp": 0.0,
                "known": 0.0 if pitcher_id is None else 0.5,
                "starts": 0.0,
                "hr_bf": 0.032,
                "whip_bf": 0.30,
                "outs_start": 16.0,
                "last_outs": 16.0,
                "fip_trend": 0.0,
            }
        state = self.pitchers[pitcher_id]
        if state.last_start_date:
            try:
                from datetime import date

                rest = (
                    date.fromisoformat(game_date) - date.fromisoformat(state.last_start_date)
                ).days
                days_rest = float(min(max(rest, 0), 15))
            except ValueError:
                days_rest = 5.0
        else:
            days_rest = 8.0
        fip_blend = state.fip_blend()
        fip_form = state.fip_form
        return {
            "fip_blend": fip_blend,
            "fip_form": fip_form,
            "k_pct": state.k_pct(),
            "bb_pct": state.bb_pct(),
            "ip_season": state.outs / 3.0,
            "days_rest": days_rest,
            "is_lhp": 1.0 if state.hand == "L" else 0.0,
            "known": 1.0,
            "starts": float(state.starts),
            "hr_bf": state.hr_per_bf(),
            "whip_bf": state.whip_per_bf(),
            "outs_start": state.outs_per_start(),
            "last_outs": float(state.last_start_outs),
            "fip_trend": fip_form - fip_blend,
        }

    def _bp_outs_last1(self, team: TeamState, game_date: str) -> float:
        """Bullpen outs on the prior calendar day (morning-slate safe)."""
        try:
            from datetime import date, timedelta

            floor1 = (date.fromisoformat(game_date) - timedelta(days=1)).isoformat()
        except ValueError:
            floor1 = game_date
        since = team.bullpen_outs_since(floor1)
        if since > 0:
            return float(since)
        if team.bullpen_recent and team.last_game_date:
            last_date, outs = team.bullpen_recent[-1]
            if last_date == team.last_game_date and last_date >= floor1:
                return float(outs)
        return 0.0

    def features_for_game(self, game: dict[str, Any]) -> dict[str, float]:
        """Build the model feature dict. State must NOT include this game yet."""
        home = self.team(int(game["home_id"]))
        away = self.team(int(game["away_id"]))
        game_date = str(game["date"])
        home_sp = self._pitcher_features(game.get("home_pp_id"), game_date)
        away_sp = self._pitcher_features(game.get("away_pp_id"), game_date)

        venue_id = game.get("venue_id")
        park = self.venue(int(venue_id)).factor(self.league_rpg) if venue_id else 1.0

        home_net_fast = home.rs_fast - home.ra_fast
        away_net_fast = away.rs_fast - away.ra_fast
        home_net_slow = home.rs_slow - home.ra_slow
        away_net_slow = away.rs_slow - away.ra_slow

        try:
            from datetime import date as date_cls, timedelta

            floor3 = (date_cls.fromisoformat(game_date) - timedelta(days=3)).isoformat()
        except ValueError:
            floor3 = game_date
        home_pen_recent = home.bullpen_outs_since(floor3)
        away_pen_recent = away.bullpen_outs_since(floor3)

        # crude expected run environment: offense vs starter quality + park
        exp_total_env = park * (
            (home.rs_slow + away.rs_slow) / 2.0
            + 0.35 * ((home_sp["fip_blend"] - LEAGUE_FIP) + (away_sp["fip_blend"] - LEAGUE_FIP))
            + self.league_rpg
        )

        home_rest = self._rest_days(home, game_date)
        away_rest = self._rest_days(away, game_date)
        home_g7 = self._games_last7(home, game_date)
        away_g7 = self._games_last7(away, game_date)
        season_frac = min(float(home.games_played) / 162.0, 1.0)
        home_travel = _travel_km(home.last_venue_id, venue_id)
        away_travel = _travel_km(away.last_venue_id, venue_id)
        elev_kft = (
            VENUE_COORDS.get(int(venue_id), (0.0, 0.0, 0.0))[2] / 1000.0
            if venue_id
            else 0.0
        )
        series_num = self._series_game_num(home, int(game["away_id"]))
        elo_diff = home.elo - away.elo
        sp_fip_blend_diff = home_sp["fip_blend"] - away_sp["fip_blend"]
        sp_hr_bf_diff = home_sp["hr_bf"] - away_sp["hr_bf"]

        # timezone shifts from previous venue → tonight
        game_tz = _tz_hours(int(venue_id) if venue_id else None)
        home_prev_tz = (
            _tz_hours(home.last_venue_id) if home.last_venue_id else game_tz
        )
        away_prev_tz = (
            _tz_hours(away.last_venue_id) if away.last_venue_id else game_tz
        )
        tz_shift_home = game_tz - home_prev_tz
        tz_shift_away = game_tz - away_prev_tz

        # SP rest / debut heuristics (require known starter)
        home_sp_known = home_sp["known"]
        away_sp_known = away_sp["known"]
        home_sp_short = (
            1.0 if home_sp_known >= 1.0 and home_sp["days_rest"] <= 3.0 else 0.0
        )
        away_sp_short = (
            1.0 if away_sp_known >= 1.0 and away_sp["days_rest"] <= 3.0 else 0.0
        )
        home_sp_long = (
            1.0 if home_sp_known >= 1.0 and home_sp["days_rest"] >= 7.0 else 0.0
        )
        away_sp_long = (
            1.0 if away_sp_known >= 1.0 and away_sp["days_rest"] >= 7.0 else 0.0
        )
        home_sp_soft = (
            1.0 if home_sp_known >= 1.0 and home_sp["starts"] < 3.0 else 0.0
        )
        away_sp_soft = (
            1.0 if away_sp_known >= 1.0 and away_sp["starts"] < 3.0 else 0.0
        )
        home_sp_exp_short = (
            1.0 if home_sp_known >= 1.0 and home_sp["last_outs"] < 12.0 else 0.0
        )
        away_sp_exp_short = (
            1.0 if away_sp_known >= 1.0 and away_sp["last_outs"] < 12.0 else 0.0
        )

        home_bp_fried = 1.0 if home_pen_recent >= 9.0 else 0.0
        away_bp_fried = 1.0 if away_pen_recent >= 9.0 else 0.0
        home_bp_last1 = self._bp_outs_last1(home, game_date)
        away_bp_last1 = self._bp_outs_last1(away, game_date)

        # Market: never invent open=close when opens are missing
        open_h = _optional_float(game.get("home_open_ml"))
        open_a = _optional_float(game.get("away_open_ml"))
        close_h = _optional_float(game.get("home_close_ml"))
        if close_h is None:
            close_h = _optional_float(game.get("home_ml"))
        close_a = _optional_float(game.get("away_close_ml"))
        if close_a is None:
            close_a = _optional_float(game.get("away_ml"))
        open_spread = _optional_float(
            game.get("home_open_spread")
            if game.get("home_open_spread") is not None
            else game.get("open_spread")
        )
        close_spread = _optional_float(
            game.get("home_close_spread")
            if game.get("home_close_spread") is not None
            else game.get("home_spread")
        )
        open_total = _optional_float(game.get("open_total"))
        close_total = _optional_float(
            game.get("close_total")
            if game.get("close_total") is not None
            else game.get("total_line")
        )

        has_open_line = 0.0
        has_steam = 0.0
        ml_steam_pp = 0.0
        spread_move = 0.0
        total_move = 0.0
        has_total = 0.0
        mkt_home_prob = 0.5
        has_market = 0.0
        mkt_home_spread = 0.0
        has_spread = 0.0

        has_open_ml = open_h is not None and open_a is not None
        if has_open_ml:
            has_open_line = 1.0
        if open_spread is not None or open_total is not None:
            has_open_line = 1.0

        if has_open_ml and close_h is not None and close_a is not None:
            open_ph = _amer_implied(open_h)
            open_pa = _amer_implied(open_a)
            close_ph = _amer_implied(close_h)
            close_pa = _amer_implied(close_a)
            open_tot = open_ph + open_pa
            close_tot = close_ph + close_pa
            open_p = open_ph / open_tot if open_tot > 0 else 0.5
            close_p = close_ph / close_tot if close_tot > 0 else 0.5
            ml_steam_pp = (close_p - open_p) * 100.0
            has_steam = 1.0

        if open_spread is not None and close_spread is not None:
            spread_move = close_spread - open_spread
            has_steam = 1.0
            has_open_line = 1.0

        if open_total is not None and close_total is not None:
            total_move = close_total - open_total
            has_open_line = 1.0

        if close_total is not None:
            has_total = 1.0

        if close_h is not None and close_a is not None:
            ch = _amer_implied(close_h)
            ca = _amer_implied(close_a)
            tot = ch + ca
            mkt_home_prob = ch / tot if tot > 0 else 0.5
            has_market = 1.0

        if close_spread is not None:
            mkt_home_spread = close_spread
            has_spread = 1.0

        # Weather hooks (neutral defaults when has_weather != 1)
        has_weather_raw = game.get("has_weather")
        try:
            has_weather = (
                1.0
                if has_weather_raw is True or float(has_weather_raw or 0) == 1.0
                else 0.0
            )
        except (TypeError, ValueError):
            has_weather = 0.0
        is_roof_raw = game.get("is_roof")
        try:
            is_roof = (
                1.0
                if is_roof_raw is True or float(is_roof_raw or 0) == 1.0
                else 0.0
            )
        except (TypeError, ValueError):
            is_roof = 0.0

        if has_weather >= 1.0:
            temp_c = _optional_float(game.get("weather_temp_c"))
            if temp_c is None:
                temp_c = _optional_float(game.get("temp_c"))
            temp_c = 22.0 if temp_c is None else temp_c
            humidity = _optional_float(game.get("weather_humidity"))
            if humidity is None:
                humidity = _optional_float(game.get("humidity"))
            humidity = 50.0 if humidity is None else humidity
            precip_mm = _optional_float(game.get("weather_precip_mm"))
            if precip_mm is None:
                precip_mm = _optional_float(game.get("precip_mm"))
            precip_mm = 0.0 if precip_mm is None else precip_mm
            wind_kph = _optional_float(game.get("weather_wind_kph"))
            if wind_kph is None:
                wind_kph = _optional_float(game.get("wind_kph"))
            wind_kph = 0.0 if wind_kph is None else wind_kph
            cold_game = 1.0 if temp_c < 10.0 else 0.0
            hot_game = 1.0 if temp_c > 32.0 else 0.0
            windy_game = 1.0 if wind_kph > 25.0 and is_roof < 1.0 else 0.0
        else:
            temp_c = 22.0
            humidity = 50.0
            precip_mm = 0.0
            wind_kph = 0.0
            cold_game = 0.0
            hot_game = 0.0
            windy_game = 0.0

        coors_extreme = 1.0 if elev_kft >= 4.5 else 0.0

        try:
            month = int(str(game_date)[5:7])
        except (TypeError, ValueError):
            month = 0
        early_season = 1.0 if season_frac < 0.15 else 0.0
        late_season = 1.0 if season_frac > 0.85 else 0.0
        april_noise = 1.0 if month == 4 else 0.0

        sp_known_both = (home_sp_known + away_sp_known) / 2.0

        # Umpire rolling bias
        hp_raw = game.get("hp_ump_id")
        try:
            hp_ump_id = int(hp_raw) if hp_raw not in (None, "", 0, "0") else None
        except (TypeError, ValueError):
            hp_ump_id = None
        has_ump = 0.0
        ump_home_win_rate = 0.5
        ump_rpg = self.league_rpg * 2.0
        ump_games = 0.0
        if hp_ump_id is not None:
            ump_st = self.ump(hp_ump_id)
            has_ump = 1.0
            ump_home_win_rate = float(ump_st.home_win_rate)
            ump_rpg = float(ump_st.rpg)
            ump_games = float(ump_st.games)

        # IL burden (attached externally; defaults = unknown → has_il=0)
        home_il = _optional_float(game.get("home_il_count"))
        away_il = _optional_float(game.get("away_il_count"))
        home_il_p = _optional_float(game.get("home_il_pitchers"))
        away_il_p = _optional_float(game.get("away_il_pitchers"))
        has_il_raw = game.get("has_il")
        try:
            has_il = (
                1.0
                if has_il_raw is True or float(has_il_raw or 0) == 1.0
                else 0.0
            )
        except (TypeError, ValueError):
            has_il = 0.0
        if has_il < 1.0:
            home_il_count = 0.0
            away_il_count = 0.0
            home_il_pitchers = 0.0
            away_il_pitchers = 0.0
        else:
            home_il_count = 0.0 if home_il is None else home_il
            away_il_count = 0.0 if away_il is None else away_il
            home_il_pitchers = 0.0 if home_il_p is None else home_il_p
            away_il_pitchers = 0.0 if away_il_p is None else away_il_p

        # Statcast priors (prev season)
        home_sp_xera = _optional_float(game.get("home_sp_prev_xera"))
        away_sp_xera = _optional_float(game.get("away_sp_prev_xera"))
        has_sc_sp = 0.0
        if home_sp_xera is not None or away_sp_xera is not None:
            has_sc_sp = 1.0
        home_sp_prev_xera = 4.20 if home_sp_xera is None else home_sp_xera
        away_sp_prev_xera = 4.20 if away_sp_xera is None else away_sp_xera

        home_xw = _optional_float(game.get("home_prev_xwoba"))
        away_xw = _optional_float(game.get("away_prev_xwoba"))
        has_sc_off = 0.0
        if home_xw is not None or away_xw is not None:
            has_sc_off = 1.0
        home_prev_xwoba = 0.320 if home_xw is None else home_xw
        away_prev_xwoba = 0.320 if away_xw is None else away_xw

        # Market heuristics
        steam_to_fav = 0.0
        reverse_line_move = 0.0
        chalk_steam = 0.0
        if has_steam >= 1.0 and has_market >= 1.0:
            # Positive ml_steam_pp = home win% rose into close
            fav_is_home = mkt_home_prob >= 0.5
            if fav_is_home and ml_steam_pp > 0:
                steam_to_fav = 1.0
            elif (not fav_is_home) and ml_steam_pp < 0:
                steam_to_fav = 1.0
            # Reverse: steam toward underdog
            if fav_is_home and ml_steam_pp < -0.5:
                reverse_line_move = 1.0
            elif (not fav_is_home) and ml_steam_pp > 0.5:
                reverse_line_move = 1.0
            chalk_steam = 1.0 if abs(ml_steam_pp) >= 2.0 and steam_to_fav >= 1.0 else 0.0

        home_no_closer = 1.0 if home_bp_fried >= 1.0 and home_bp_last1 >= 6.0 else 0.0
        away_no_closer = 1.0 if away_bp_fried >= 1.0 and away_bp_last1 >= 6.0 else 0.0

        features: dict[str, float] = {
            "elo_diff": elo_diff,
            "net_fast_diff": home_net_fast - away_net_fast,
            "net_slow_diff": home_net_slow - away_net_slow,
            "rs_fast_diff": home.rs_fast - away.rs_fast,
            "ra_fast_diff": home.ra_fast - away.ra_fast,
            "obp_diff": home.obp - away.obp,
            "slg_diff": home.slg - away.slg,
            "hr_pg_diff": home.hr_pg - away.hr_pg,
            "bb_rate_diff": home.bb_rate - away.bb_rate,
            "so_rate_diff": home.so_rate - away.so_rate,
            "bullpen_ra9_diff": home.bullpen_ra9 - away.bullpen_ra9,
            "bullpen_k9_diff": home.bullpen_k9 - away.bullpen_k9,
            "win_ewma_diff": home.win_ewma - away.win_ewma,
            "home_split_win_ewma": home.home_win_ewma,
            "away_split_win_ewma": away.away_win_ewma,
            "prev_win_pct_diff": home.prev_win_pct - away.prev_win_pct,
            "prev_run_diff_pg_diff": home.prev_run_diff_pg - away.prev_run_diff_pg,
            "home_rest_days": home_rest,
            "away_rest_days": away_rest,
            "home_games_last7": home_g7,
            "away_games_last7": away_g7,
            "home_games_played": float(home.games_played),
            "away_games_played": float(away.games_played),
            "season_frac": season_frac,
            "is_night": 1.0 if str(game.get("day_night") or "") == "night" else 0.0,
            "is_double_header": 0.0 if str(game.get("double_header") or "N") == "N" else 1.0,
            "park_factor": park,
            "league_rpg": self.league_rpg,
            "sp_fip_blend_diff": sp_fip_blend_diff,
            "sp_fip_form_diff": home_sp["fip_form"] - away_sp["fip_form"],
            "sp_k_pct_diff": home_sp["k_pct"] - away_sp["k_pct"],
            "sp_bb_pct_diff": home_sp["bb_pct"] - away_sp["bb_pct"],
            "sp_ip_season_diff": home_sp["ip_season"] - away_sp["ip_season"],
            "sp_days_rest_diff": home_sp["days_rest"] - away_sp["days_rest"],
            "home_sp_fip_blend": home_sp["fip_blend"],
            "away_sp_fip_blend": away_sp["fip_blend"],
            "home_sp_k_pct": home_sp["k_pct"],
            "away_sp_k_pct": away_sp["k_pct"],
            "home_sp_is_lhp": home_sp["is_lhp"],
            "away_sp_is_lhp": away_sp["is_lhp"],
            "home_sp_known": home_sp_known,
            "away_sp_known": away_sp_known,
            "home_sp_days_rest": home_sp["days_rest"],
            "away_sp_days_rest": away_sp["days_rest"],
            "sp_kbb_diff": (home_sp["k_pct"] - home_sp["bb_pct"])
            - (away_sp["k_pct"] - away_sp["bb_pct"]),
            "home_sp_k_x_away_so": home_sp["k_pct"] * away.so_rate,
            "away_sp_k_x_home_so": away_sp["k_pct"] * home.so_rate,
            "home_sp_starts": home_sp["starts"],
            "away_sp_starts": away_sp["starts"],
            "home_bullpen_outs_last3": home_pen_recent,
            "away_bullpen_outs_last3": away_pen_recent,
            "exp_total_env": exp_total_env,
            "season_win_pct_diff": home.season_win_pct() - away.season_win_pct(),
            "pythag_diff": home.pythag() - away.pythag(),
            "home_travel_km": home_travel,
            "away_travel_km": away_travel,
            "venue_elevation_kft": elev_kft,
            "series_game_num": series_num,
            "adj_net_diff": (home.adj_rs - home.adj_ra) - (away.adj_rs - away.adj_ra),
            "adj_rs_diff": home.adj_rs - away.adj_rs,
            "adj_ra_diff": home.adj_ra - away.adj_ra,
            "sp_hr_bf_diff": sp_hr_bf_diff,
            "sp_whip_bf_diff": home_sp["whip_bf"] - away_sp["whip_bf"],
            "sp_outs_per_start_diff": home_sp["outs_start"] - away_sp["outs_start"],
            "home_sp_outs_per_start": home_sp["outs_start"],
            "away_sp_outs_per_start": away_sp["outs_start"],
            "iso_diff": home.iso - away.iso,
            "xbh_rate_diff": home.xbh_rate - away.xbh_rate,
            "elo_momentum_5_diff": home.elo_momentum(5) - away.elo_momentum(5),
            "home_bullpen_outs_last5": home.bullpen_outs_last_n(5),
            "away_bullpen_outs_last5": away.bullpen_outs_last_n(5),
            "sp_fip_trend_diff": home_sp["fip_trend"] - away_sp["fip_trend"],
            "sp_last_outs_diff": home_sp["last_outs"] - away_sp["last_outs"],
            # Offense SO/OBP when facing opposite-side starter (LHP flag × rates)
            "platoon_k_matchup": (
                away_sp["is_lhp"] * home.so_rate - home_sp["is_lhp"] * away.so_rate
            ),
            "platoon_obp_matchup": (
                away_sp["is_lhp"] * home.obp - home_sp["is_lhp"] * away.obp
            ),
            # market
            "has_open_line": has_open_line,
            "has_steam": has_steam,
            "ml_steam_pp": ml_steam_pp,
            "spread_move": spread_move,
            "total_move": total_move,
            "has_total": has_total,
            "mkt_home_prob": mkt_home_prob,
            "has_market": has_market,
            "mkt_home_spread": mkt_home_spread,
            "has_spread": has_spread,
            # offense extras
            "ba_diff": home.ba - away.ba,
            "ops_diff": home.ops - away.ops,
            "pythag_luck_diff": home.pythag_luck() - away.pythag_luck(),
            # form / streaks
            "win_streak_diff": home.win_streak() - away.win_streak(),
            "loss_streak_diff": home.loss_streak() - away.loss_streak(),
            "form10_net_diff": home.form10_net() - away.form10_net(),
            "rs_l10_diff": (
                _mean_or(home.recent_rs, LEAGUE_RPG_DEFAULT)
                - _mean_or(away.recent_rs, LEAGUE_RPG_DEFAULT)
            ),
            "ra_l10_diff": (
                _mean_or(home.recent_ra, LEAGUE_RPG_DEFAULT)
                - _mean_or(away.recent_ra, LEAGUE_RPG_DEFAULT)
            ),
            # travel / schedule
            "travel_diff": home_travel - away_travel,
            "tz_shift_home": tz_shift_home,
            "tz_shift_away": tz_shift_away,
            "tz_diff": tz_shift_home - tz_shift_away,
            "home_stand_len": float(home.home_stand_len),
            "away_road_trip_len": float(away.road_trip_len),
            "cross_country": 1.0 if max(home_travel, away_travel) > 2500.0 else 0.0,
            "is_series_opener": 1.0 if series_num == 1.0 else 0.0,
            "is_rubber_game": 1.0 if series_num >= 3.0 else 0.0,
            "rest_diff": home_rest - away_rest,
            "games_last7_diff": home_g7 - away_g7,
            # SP heuristics
            "home_sp_short_rest": home_sp_short,
            "away_sp_short_rest": away_sp_short,
            "home_sp_long_rest": home_sp_long,
            "away_sp_long_rest": away_sp_long,
            "home_sp_soft_debut": home_sp_soft,
            "away_sp_soft_debut": away_sp_soft,
            "sp_short_rest_diff": home_sp_short - away_sp_short,
            "home_sp_expected_short": home_sp_exp_short,
            "away_sp_expected_short": away_sp_exp_short,
            # bullpen
            "home_bp_fried": home_bp_fried,
            "away_bp_fried": away_bp_fried,
            "home_bp_outs_last1": home_bp_last1,
            "away_bp_outs_last1": away_bp_last1,
            "bp_fried_diff": home_bp_fried - away_bp_fried,
            "bp_fried_x_home_sp_short": home_bp_fried * home_sp_short,
            # park / weather
            "temp_c": float(temp_c),
            "humidity": float(humidity),
            "precip_mm": float(precip_mm),
            "wind_kph": float(wind_kph),
            "has_weather": has_weather,
            "is_roof": is_roof,
            "cold_game": cold_game,
            "hot_game": hot_game,
            "windy_game": windy_game,
            "coors_extreme": coors_extreme,
            # cross-terms / seasonality
            "sp_fip_x_park_diff": sp_fip_blend_diff * park,
            "elo_x_season_frac": elo_diff * season_frac,
            "steam_x_sp_known": ml_steam_pp * sp_known_both,
            "hr_env_x_sp_hr_diff": park * sp_hr_bf_diff,
            "early_season": early_season,
            "late_season": late_season,
            "april_noise": april_noise,
            # umpire
            "has_ump": has_ump,
            "ump_home_win_rate": ump_home_win_rate,
            "ump_rpg": ump_rpg,
            "ump_games": ump_games,
            # IL
            "home_il_count": home_il_count,
            "away_il_count": away_il_count,
            "il_diff": home_il_count - away_il_count,
            "home_il_pitchers": home_il_pitchers,
            "away_il_pitchers": away_il_pitchers,
            "has_il": has_il,
            # Statcast priors
            "home_sp_prev_xera": home_sp_prev_xera,
            "away_sp_prev_xera": away_sp_prev_xera,
            "sp_xera_diff": home_sp_prev_xera - away_sp_prev_xera,
            "has_statcast_sp": has_sc_sp,
            "home_prev_xwoba": home_prev_xwoba,
            "away_prev_xwoba": away_prev_xwoba,
            "xwoba_diff": home_prev_xwoba - away_prev_xwoba,
            "has_statcast_offense": has_sc_off,
            # heuristics
            "steam_to_fav": steam_to_fav,
            "reverse_line_move": reverse_line_move,
            "chalk_steam": chalk_steam,
            "home_no_closer_proxy": home_no_closer,
            "away_no_closer_proxy": away_no_closer,
        }
        # Guarantee exact FEATURE_COLUMNS coverage and finite floats
        return {col: float(features[col]) for col in FEATURE_COLUMNS}

    @staticmethod
    def _series_game_num(home: TeamState, away_id: int) -> float:
        count = 0
        for opp in reversed(home.recent_opponents):
            if opp == away_id:
                count += 1
            else:
                break
        return float(min(count + 1, 5))

    # -- state updates ----------------------------------------------------------
    def _update_elo(self, home: TeamState, away: TeamState, home_score: int, away_score: int) -> None:
        expected_home = 1.0 / (1.0 + 10 ** ((away.elo - home.elo - ELO_HOME_ADV) / 400.0))
        actual_home = 1.0 if home_score > away_score else 0.0
        margin = abs(home_score - away_score)
        # Clamp like NBA/NHL/NFL: underdog wins must not drive denominator ≤ 0
        # (sign-flip / explode near winner_elo_diff ≈ -2200).
        winner_elo_diff = (home.elo - away.elo) if home_score > away_score else (away.elo - home.elo)
        mov_mult = math.log(margin + 1.0) * (
            2.2 / (0.001 * max(winner_elo_diff, 0.0) + 2.2)
        )
        delta = ELO_K * mov_mult * (actual_home - expected_home)
        home.elo += delta
        away.elo -= delta
        home.elo_deltas.append(delta)
        away.elo_deltas.append(-delta)
        for team in (home, away):
            if len(team.elo_deltas) > 8:
                team.elo_deltas.pop(0)

    def update_after_game(
        self,
        game: dict[str, Any],
        *,
        home_hit_log: dict[str, Any] | None = None,
        away_hit_log: dict[str, Any] | None = None,
        home_bullpen: dict[str, float] | None = None,
        away_bullpen: dict[str, float] | None = None,
    ) -> None:
        home_score = game.get("home_score")
        away_score = game.get("away_score")
        if home_score is None or away_score is None or home_score == away_score:
            return
        home_score = int(home_score)
        away_score = int(away_score)
        home = self.team(int(game["home_id"]))
        away = self.team(int(game["away_id"]))
        game_date = str(game["date"])

        self._update_elo(home, away, home_score, away_score)

        # opponent-adjusted performance targets (baselines captured pre-update)
        ALPHA_ADJ = 0.08
        home_adj_rs_t = float(home_score) - away.ra_slow
        home_adj_ra_t = float(away_score) - away.rs_slow
        away_adj_rs_t = float(away_score) - home.ra_slow
        away_adj_ra_t = float(home_score) - home.rs_slow
        home.adj_rs = _ewma(home.adj_rs, home_adj_rs_t, ALPHA_ADJ)
        home.adj_ra = _ewma(home.adj_ra, home_adj_ra_t, ALPHA_ADJ)
        away.adj_rs = _ewma(away.adj_rs, away_adj_rs_t, ALPHA_ADJ)
        away.adj_ra = _ewma(away.adj_ra, away_adj_ra_t, ALPHA_ADJ)

        home_won = 1.0 if home_score > away_score else 0.0
        for team, scored, allowed, won, is_home in (
            (home, home_score, away_score, home_won, True),
            (away, away_score, home_score, 1.0 - home_won, False),
        ):
            team.rs_fast = _ewma(team.rs_fast, float(scored), ALPHA_RUNS_FAST)
            team.rs_slow = _ewma(team.rs_slow, float(scored), ALPHA_RUNS_SLOW)
            team.ra_fast = _ewma(team.ra_fast, float(allowed), ALPHA_RUNS_FAST)
            team.ra_slow = _ewma(team.ra_slow, float(allowed), ALPHA_RUNS_SLOW)
            team.win_ewma = _ewma(team.win_ewma, won, ALPHA_WIN)
            if is_home:
                team.home_win_ewma = _ewma(team.home_win_ewma, won, ALPHA_WIN)
                team.home_stand_len += 1
                team.road_trip_len = 0
            else:
                team.away_win_ewma = _ewma(team.away_win_ewma, won, ALPHA_WIN)
                team.road_trip_len += 1
                team.home_stand_len = 0
            team.last_game_date = game_date
            team.recent_dates.append(game_date)
            if len(team.recent_dates) > 10:
                team.recent_dates.pop(0)
            team.recent_results.append(1 if won >= 0.5 else 0)
            if len(team.recent_results) > 10:
                team.recent_results.pop(0)
            team.recent_rs.append(float(scored))
            if len(team.recent_rs) > 10:
                team.recent_rs.pop(0)
            team.recent_ra.append(float(allowed))
            if len(team.recent_ra) > 10:
                team.recent_ra.pop(0)
            team.games_played += 1
            team.season_rs += scored
            team.season_ra += allowed
            if won >= 0.5:
                team.season_wins += 1
            else:
                team.season_losses += 1

        venue_id_raw = game.get("venue_id")
        venue_id_int = int(venue_id_raw) if venue_id_raw else None
        home.last_venue_id = venue_id_int
        away.last_venue_id = venue_id_int
        home.recent_opponents.append(int(game["away_id"]))
        away.recent_opponents.append(int(game["home_id"]))
        for team in (home, away):
            if len(team.recent_opponents) > 6:
                team.recent_opponents.pop(0)

        hp_raw = game.get("hp_ump_id")
        try:
            hp_ump_id = int(hp_raw) if hp_raw not in (None, "", 0, "0") else None
        except (TypeError, ValueError):
            hp_ump_id = None
        if hp_ump_id is not None:
            self.ump(hp_ump_id).observe(
                1.0 if home_score > away_score else 0.0,
                float(home_score + away_score),
            )

        for team, hit_log in ((home, home_hit_log), (away, away_hit_log)):
            if not hit_log:
                continue
            pa = float(hit_log.get("pa") or 0)
            ab = float(hit_log.get("ab") or 0)
            if pa >= 10:
                # Do not use truthiness — 0.000 OBP/SLG is a real (hitless) game.
                obp_raw = hit_log.get("obp")
                slg_raw = hit_log.get("slg")
                obp_val = float(obp_raw) if obp_raw is not None else LEAGUE_OBP
                slg_val = float(slg_raw) if slg_raw is not None else LEAGUE_SLG
                team.obp = _ewma(team.obp, obp_val, ALPHA_BATTING)
                team.slg = _ewma(team.slg, slg_val, ALPHA_BATTING)
                team.ops = _ewma(team.ops, obp_val + slg_val, ALPHA_BATTING)
                team.hr_pg = _ewma(team.hr_pg, float(hit_log.get("hr") or 0), ALPHA_BATTING)
                team.bb_rate = _ewma(team.bb_rate, float(hit_log.get("bb") or 0) / pa, ALPHA_BATTING)
                team.so_rate = _ewma(team.so_rate, float(hit_log.get("so") or 0) / pa, ALPHA_BATTING)
            if ab >= 10:
                ba = float(hit_log.get("h") or 0) / ab
                team.ba = _ewma(team.ba, ba, ALPHA_BATTING)
                slg_raw = hit_log.get("slg")
                slg = float(slg_raw) if slg_raw is not None else LEAGUE_SLG
                team.iso = _ewma(team.iso, max(slg - ba, -0.05), ALPHA_BATTING)
                xbh = (
                    float(hit_log.get("d2") or 0)
                    + float(hit_log.get("d3") or 0)
                    + float(hit_log.get("hr") or 0)
                )
                team.xbh_rate = _ewma(team.xbh_rate, xbh / ab, ALPHA_BATTING)

        for team, pen in ((home, home_bullpen), (away, away_bullpen)):
            if not pen:
                continue
            outs = float(pen.get("outs") or 0.0)
            team.bullpen_recent.append([game_date, outs])
            if len(team.bullpen_recent) > 8:
                team.bullpen_recent.pop(0)
            if outs >= 3:
                ip = outs / 3.0
                ra9 = 9.0 * float(pen.get("r") or 0.0) / ip
                k9 = 9.0 * float(pen.get("so") or 0.0) / ip
                team.bullpen_ra9 = _ewma(team.bullpen_ra9, min(ra9, 20.0), ALPHA_BULLPEN)
                team.bullpen_k9 = _ewma(team.bullpen_k9, min(k9, 20.0), ALPHA_BULLPEN)

        venue_id = game.get("venue_id")
        if venue_id:
            self.venue(int(venue_id)).add_game(float(home_score + away_score))
        self.league_rpg = _ewma(
            self.league_rpg, (home_score + away_score) / 2.0, ALPHA_LEAGUE
        )

    def apply_pitcher_appearance(self, pitcher_id: int, hand: str, log: dict[str, Any]) -> None:
        state = self.pitcher(int(pitcher_id), hand=hand)
        state.hand = hand or state.hand
        state.add_appearance(log)

    # -- serialization ------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "season": self.season,
            "league_rpg": self.league_rpg,
            "teams": {str(k): v.to_dict() for k, v in self.teams.items()},
            "pitchers": {str(k): v.to_dict() for k, v in self.pitchers.items()},
            "venues": {str(k): v.to_dict() for k, v in self.venues.items()},
            "umps": {str(k): v.to_dict() for k, v in self.umps.items()},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MlbFeatureEngine":
        engine = cls()
        engine.season = payload.get("season")
        league_rpg = payload.get("league_rpg")
        engine.league_rpg = (
            float(league_rpg) if league_rpg is not None else LEAGUE_RPG_DEFAULT
        )
        engine.teams = {
            int(k): TeamState.from_dict(v) for k, v in (payload.get("teams") or {}).items()
        }
        engine.pitchers = {
            int(k): PitcherState.from_dict(v) for k, v in (payload.get("pitchers") or {}).items()
        }
        engine.venues = {
            int(k): VenueState.from_dict(v) for k, v in (payload.get("venues") or {}).items()
        }
        engine.umps = {
            int(k): UmpState.from_dict(v) for k, v in (payload.get("umps") or {}).items()
        }
        return engine
