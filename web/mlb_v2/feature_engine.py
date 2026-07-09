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

    def bullpen_outs_since(self, floor_date: str) -> float:
        return float(sum(outs for d, outs in self.bullpen_recent if d >= floor_date))

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


class MlbFeatureEngine:
    def __init__(self) -> None:
        self.teams: dict[int, TeamState] = {}
        self.pitchers: dict[int, PitcherState] = {}
        self.venues: dict[int, VenueState] = {}
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
        try:
            from datetime import date, timedelta

            current = date.fromisoformat(game_date)
            floor = (current - timedelta(days=7)).isoformat()
            return float(sum(1 for d in team.recent_dates if d >= floor))
        except ValueError:
            return 6.0

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
        return {
            "fip_blend": state.fip_blend(),
            "fip_form": state.fip_form,
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
        }

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

        return {
            "elo_diff": home.elo - away.elo,
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
            "home_rest_days": self._rest_days(home, game_date),
            "away_rest_days": self._rest_days(away, game_date),
            "home_games_last7": self._games_last7(home, game_date),
            "away_games_last7": self._games_last7(away, game_date),
            "home_games_played": float(home.games_played),
            "away_games_played": float(away.games_played),
            "season_frac": min(float(home.games_played) / 162.0, 1.0),
            "is_night": 1.0 if str(game.get("day_night") or "") == "night" else 0.0,
            "is_double_header": 0.0 if str(game.get("double_header") or "N") == "N" else 1.0,
            "park_factor": park,
            "league_rpg": self.league_rpg,
            "sp_fip_blend_diff": home_sp["fip_blend"] - away_sp["fip_blend"],
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
            "home_sp_known": home_sp["known"],
            "away_sp_known": away_sp["known"],
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
            "home_travel_km": _travel_km(home.last_venue_id, venue_id),
            "away_travel_km": _travel_km(away.last_venue_id, venue_id),
            "venue_elevation_kft": (
                VENUE_COORDS.get(int(venue_id), (0.0, 0.0, 0.0))[2] / 1000.0
                if venue_id
                else 0.0
            ),
            "series_game_num": self._series_game_num(home, int(game["away_id"])),
            "adj_net_diff": (home.adj_rs - home.adj_ra) - (away.adj_rs - away.adj_ra),
            "adj_rs_diff": home.adj_rs - away.adj_rs,
            "adj_ra_diff": home.adj_ra - away.adj_ra,
            "sp_hr_bf_diff": home_sp["hr_bf"] - away_sp["hr_bf"],
            "sp_whip_bf_diff": home_sp["whip_bf"] - away_sp["whip_bf"],
            "sp_outs_per_start_diff": home_sp["outs_start"] - away_sp["outs_start"],
            "home_sp_outs_per_start": home_sp["outs_start"],
            "away_sp_outs_per_start": away_sp["outs_start"],
        }

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
        winner_elo_diff = (home.elo - away.elo) if home_score > away_score else (away.elo - home.elo)
        mov_mult = math.log(margin + 1.0) * (2.2 / (0.001 * winner_elo_diff + 2.2))
        delta = ELO_K * mov_mult * (actual_home - expected_home)
        home.elo += delta
        away.elo -= delta

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
            else:
                team.away_win_ewma = _ewma(team.away_win_ewma, won, ALPHA_WIN)
            team.last_game_date = game_date
            team.recent_dates.append(game_date)
            if len(team.recent_dates) > 10:
                team.recent_dates.pop(0)
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

        for team, hit_log in ((home, home_hit_log), (away, away_hit_log)):
            if not hit_log:
                continue
            pa = float(hit_log.get("pa") or 0)
            if pa >= 10:
                team.obp = _ewma(team.obp, float(hit_log.get("obp") or LEAGUE_OBP), ALPHA_BATTING)
                team.slg = _ewma(team.slg, float(hit_log.get("slg") or LEAGUE_SLG), ALPHA_BATTING)
                team.hr_pg = _ewma(team.hr_pg, float(hit_log.get("hr") or 0), ALPHA_BATTING)
                team.bb_rate = _ewma(team.bb_rate, float(hit_log.get("bb") or 0) / pa, ALPHA_BATTING)
                team.so_rate = _ewma(team.so_rate, float(hit_log.get("so") or 0) / pa, ALPHA_BATTING)

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
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MlbFeatureEngine":
        engine = cls()
        engine.season = payload.get("season")
        engine.league_rpg = float(payload.get("league_rpg") or LEAGUE_RPG_DEFAULT)
        engine.teams = {
            int(k): TeamState.from_dict(v) for k, v in (payload.get("teams") or {}).items()
        }
        engine.pitchers = {
            int(k): PitcherState.from_dict(v) for k, v in (payload.get("pitchers") or {}).items()
        }
        engine.venues = {
            int(k): VenueState.from_dict(v) for k, v in (payload.get("venues") or {}).items()
        }
        return engine
