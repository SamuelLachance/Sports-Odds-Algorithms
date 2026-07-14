"""Walk-forward NFL feature engine shared by offline training and live inference.

Consumes NFL games chronologically. For each game it emits a feature row using
only state accumulated strictly before that date, then folds the result into
state. State is JSON-serializable so training can snapshot end-of-season state
and the live path replays only the current season.

Game dict schema:
  date (ISO), season, home / away (lowercase abbr), home_score, away_score,
  optional: week, game_type, location, div_game, roof, surface, temp, wind,
  weekday, home_rest, away_rest, home_qb_id, away_qb_id, home_coach, away_coach,
  home_epa_off / home_epa_def / home_sr_off / home_sr_def / home_explosive_off /
  home_pass_epa_off (and away_*), madden_ovr_diff / madden_known.

Injury note: same-day ESPN injury HTML is intentionally NOT trained (leak /
unstable). Use leak-free QB-change / backup proxies instead; live still applies
an availability nudge in blend_service.

Elo defaults: CFB-style K=20 with NFL HFA=48 and POINTS_PER_ELO=25 so
elo_diff is consistent with elo_to_prob(z=401.62). QB/coach Elo updated
only after games (leak-free).
"""

from __future__ import annotations

import math
from datetime import date as date_cls
from typing import Any

from web.nfl_v2.team_geo import altitude, team_tz, timezone_diff, travel_km

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
ALPHA_EPA = 0.20

LEAGUE_PPG = 22.5
DEFAULT_EPA = 0.0
DEFAULT_SR = 0.45
DEFAULT_EXPLOSIVE = 0.10
DEFAULT_SACK = 0.06
DEFAULT_QB_HIT = 0.10
DEFAULT_SNAP = 0.55
DEFAULT_OL_STARTERS = 5.0
REF_ELO_K = 12.0
CLOSE_GAME_MARGIN = 8.0
BLOWOUT_MARGIN = 17.0
DEFAULT_MARGIN_VOL = 14.0
MARGIN_HIST_LEN = 8
SEASON_GAMES_NOMINAL = 17.0
DEFAULT_TO = 0.02
DEFAULT_TEMP = 70.0
DEFAULT_WIND = 0.0
DEFAULT_REST = 7.0
BYE_REST_DAYS = 13.0
MADDEN_MIN_SEASON = 2025  # madden_nfl_2026.csv maps to 2025+ NFL seasons

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
    "wind_x_outdoor",
    "short_week_x_rest",
    "cold_game",
    "high_wind",
    "temp_x_outdoor",
    "tz_diff",
    "westbound_short_week",
    # matchup history
    "h2h_home_win_rate",
    "h2h_margin_ewma",
    # QB / coach / referee
    "qb_elo_diff",
    "qb_games_diff",
    "qb_win_ewma_diff",
    "home_qb_known",
    "away_qb_known",
    "home_qb_streak",
    "away_qb_streak",
    "qb_change_diff",
    "qb_games_out_proxy_diff",
    "coach_elo_diff",
    "coach_games_diff",
    "coach_win_ewma_diff",
    "home_coach_known",
    "away_coach_known",
    "home_coach_streak",
    "away_coach_streak",
    "ref_known",
    "ref_home_bias",
    # PBP efficiency (leak-free prior EWMA)
    "epa_off_diff",
    "epa_def_diff",
    "sr_off_diff",
    "sr_def_diff",
    "explosive_diff",
    "pass_epa_diff",
    "rush_epa_diff",
    "pass_rush_epa_gap_diff",
    "sack_rate_off_diff",
    "sack_rate_def_diff",
    "qb_hit_rate_off_diff",
    "qb_hit_rate_def_diff",
    "early_down_epa_diff",
    "third_down_sr_diff",
    "redzone_epa_diff",
    "epa_off_vs_def",
    "epa_x_season_frac",
    "pass_epa_x_wind",
    # Snap continuity
    "wr1_snap_share_diff",
    "ol_starter_share_diff",
    "skill_snap_share_diff",
    # Injury burden (as-of weekly nflverse / snapshots when present)
    "injury_burden_diff",
    "ol_out_diff",
    "skill_out_diff",
    "injury_known",
    # Opening-line steam (0 when opens missing)
    "has_open_line",
    "has_steam",
    "spread_move",
    "ml_steam_pp",
    "has_market",
    "mkt_home_prob",
    "has_spread",
    "mkt_home_spread",
    "has_total",
    "total_line",
    "model_total_vs_line",
    "total_move",
    # Madden prior (known seasons only)
    "madden_ovr_diff",
    "madden_known",
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
    # PBP depth / defensive micro
    "explosive_def_diff",
    "pass_epa_def_diff",
    "rush_epa_def_diff",
    "to_rate_off_diff",
    "to_rate_def_diff",
    "sr_gap_diff",
    # unit matchup cross-terms
    "home_pass_vs_away_pass_def",
    "away_pass_vs_home_pass_def",
    "home_rush_vs_away_rush_def",
    "away_rush_vs_home_rush_def",
    "qb_elo_x_pass_def",
    "qb_elo_x_wind",
    "qb_elo_x_short_week",
    "ol_snap_x_sack_allowed",
    "epa_x_div_game",
    "elo_x_is_playoff",
    "elo_x_primetime",
    "injury_x_ol_out",
    # schedule / travel depth
    "home_travel_km",
    "away_travel_km",
    "travel_diff",
    "bye_week_home",
    "bye_week_away",
    "home_stand_len",
    "away_road_trip",
    "home_altitude",
    "altitude_diff",
    "west_coast_early",
    "east_at_west_night",
    "tz_debt_away",
    # heuristics
    "early_season",
    "luck_diff",
    "big_fav_spread",
    "home_fav_elo",
    "dog_elo_gap",
    "rematch_flag",
    "primetime",
    "thanksgiving",
    "international",
    "qb_first_start_home",
    "qb_first_start_away",
    "backup_qb_elo_diff",
    "coach_change_diff",
)


def nfl_season_of(day: date_cls) -> int:
    """NFL season year: Sep–Dec belong to that year; Jan–Feb playoffs to prior."""
    return day.year if day.month >= 3 else day.year - 1


def _parse_date(value: str) -> date_cls | None:
    try:
        return date_cls.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _amer_implied(ml: float) -> float:
    if not math.isfinite(ml) or abs(ml) < 100:
        return 0.5
    return (-ml) / (-ml + 100.0) if ml < 0 else 100.0 / (ml + 100.0)


def _safe_float(value: Any, default: float) -> float:
    if value is None or isinstance(value, bool):
        return default
    try:
        if isinstance(value, float) and math.isnan(value):
            return default
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return parsed


def _is_missing_id(value: Any) -> bool:
    if value is None:
        return True
    text = str(value).strip().lower()
    return text in ("", "nan", "none", "null")


def _coerce_flag(value: Any) -> bool | None:
    """Parse explicit neutral/flag fields; reject stringy ``\"false\"`` / ``\"0\"``."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "t"}:
        return True
    if text in {"0", "false", "no", "n", "f"}:
        return False
    return None


def infer_neutral_site(game: dict[str, Any]) -> bool:
    flagged = _coerce_flag(game.get("neutral_site"))
    if flagged is not None:
        return flagged
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


_MADDEN_CACHE: dict[str, float] | None = None


def _load_madden_team_ovr(top_n: int = 8) -> dict[str, float]:
    global _MADDEN_CACHE
    if _MADDEN_CACHE is not None:
        return _MADDEN_CACHE
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent.parent / "data" / "ratings" / "madden_nfl_2026.csv"
    out: dict[str, float] = {}
    if not path.is_file():
        _MADDEN_CACHE = out
        return out
    try:
        import pandas as pd

        frame = pd.read_csv(path)
        if "team_abbr" not in frame.columns or "overall" not in frame.columns:
            _MADDEN_CACHE = out
            return out
        frame = frame.copy()
        frame["team_abbr"] = frame["team_abbr"].astype(str).str.lower().str.strip()
        frame["overall"] = pd.to_numeric(frame["overall"], errors="coerce")
        frame = frame.dropna(subset=["overall"])
        for team, group in frame.groupby("team_abbr"):
            top = group["overall"].nlargest(top_n)
            if len(top):
                out[str(team)] = float(top.mean())
    except Exception:  # noqa: BLE001
        out = {}
    _MADDEN_CACHE = out
    return out


def _madden_ovr_diff(home: str, away: str) -> tuple[float, float]:
    ratings = _load_madden_team_ovr()
    home_k = str(home or "").lower().strip()
    away_k = str(away or "").lower().strip()
    # Madden uses LAR→LA style; tolerate common aliases.
    aliases = {"lar": "la", "wsh": "was", "jac": "jax"}
    home_k = aliases.get(home_k, home_k)
    away_k = aliases.get(away_k, away_k)
    if home_k not in ratings or away_k not in ratings:
        return 0.0, 0.0
    return float(ratings[home_k] - ratings[away_k]), 1.0


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
        "epa_off", "epa_def", "sr_off", "sr_def", "explosive_off", "explosive_def",
        "pass_epa_off", "rush_epa_off", "pass_epa_def", "rush_epa_def",
        "sack_rate_off", "sack_rate_def", "qb_hit_rate_off", "qb_hit_rate_def",
        "early_down_epa_off", "third_down_sr_off", "redzone_epa_off",
        "to_rate_off", "to_rate_def",
        "wr1_snap_share", "ol_starter_share", "skill_snap_share",
        "qb_last_start_games",
        "last_venue", "home_stand", "road_trip",
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
        self.epa_off = DEFAULT_EPA
        self.epa_def = DEFAULT_EPA
        self.sr_off = DEFAULT_SR
        self.sr_def = DEFAULT_SR
        self.explosive_off = DEFAULT_EXPLOSIVE
        self.explosive_def = DEFAULT_EXPLOSIVE
        self.pass_epa_off = DEFAULT_EPA
        self.rush_epa_off = DEFAULT_EPA
        self.pass_epa_def = DEFAULT_EPA
        self.rush_epa_def = DEFAULT_EPA
        self.sack_rate_off = DEFAULT_SACK
        self.sack_rate_def = DEFAULT_SACK
        self.qb_hit_rate_off = DEFAULT_QB_HIT
        self.qb_hit_rate_def = DEFAULT_QB_HIT
        self.early_down_epa_off = DEFAULT_EPA
        self.third_down_sr_off = DEFAULT_SR
        self.redzone_epa_off = DEFAULT_EPA
        self.to_rate_off = DEFAULT_TO
        self.to_rate_def = DEFAULT_TO
        self.wr1_snap_share = DEFAULT_SNAP
        self.ol_starter_share = DEFAULT_SNAP
        self.skill_snap_share = DEFAULT_SNAP
        self.qb_last_start_games: dict[str, int] = {}
        self.last_venue = ""
        self.home_stand = 0
        self.road_trip = 0

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

    def coach_streak_for(self, coach_name: str) -> float:
        if _is_missing_id(coach_name) or not self.current_coach:
            return 0.0
        if str(coach_name).strip().lower() != str(self.current_coach).strip().lower():
            return 0.0
        return float(self.coach_streak)

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
            "epa_off": self.epa_off,
            "epa_def": self.epa_def,
            "sr_off": self.sr_off,
            "sr_def": self.sr_def,
            "explosive_off": self.explosive_off,
            "explosive_def": self.explosive_def,
            "pass_epa_off": self.pass_epa_off,
            "rush_epa_off": self.rush_epa_off,
            "pass_epa_def": self.pass_epa_def,
            "rush_epa_def": self.rush_epa_def,
            "sack_rate_off": self.sack_rate_off,
            "sack_rate_def": self.sack_rate_def,
            "qb_hit_rate_off": self.qb_hit_rate_off,
            "qb_hit_rate_def": self.qb_hit_rate_def,
            "early_down_epa_off": self.early_down_epa_off,
            "third_down_sr_off": self.third_down_sr_off,
            "redzone_epa_off": self.redzone_epa_off,
            "to_rate_off": self.to_rate_off,
            "to_rate_def": self.to_rate_def,
            "wr1_snap_share": self.wr1_snap_share,
            "ol_starter_share": self.ol_starter_share,
            "skill_snap_share": self.skill_snap_share,
            "qb_last_start_games": dict(self.qb_last_start_games),
            "last_venue": self.last_venue,
            "home_stand": self.home_stand,
            "road_trip": self.road_trip,
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
            elif key == "qb_last_start_games":
                state.qb_last_start_games = {
                    str(k): int(v) for k, v in dict(value or {}).items()
                }
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
        self.refs: dict[str, EntityElo] = {}
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

    def ref(self, key: str) -> EntityElo:
        key = str(key or "").strip().lower()
        state = self.refs.get(key)
        if state is None:
            state = EntityElo(key)
            self.refs[key] = state
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

        # Leak-free QB change / games-out proxies (vs last starter on this team).
        home_qb_change = 0.0
        away_qb_change = 0.0
        if home_qb_known and home.current_qb_id and home.current_qb_id != home_qb_id:
            home_qb_change = 1.0
        if away_qb_known and away.current_qb_id and away.current_qb_id != away_qb_id:
            away_qb_change = 1.0
        home_qb_out = 0.0
        away_qb_out = 0.0
        if home_qb_known:
            last_gp = home.qb_last_start_games.get(home_qb_id)
            if last_gp is not None:
                home_qb_out = float(max(home.games_played - last_gp, 0))
            elif home.current_qb_id and home.current_qb_id != home_qb_id:
                home_qb_out = 4.0
        if away_qb_known:
            last_gp = away.qb_last_start_games.get(away_qb_id)
            if last_gp is not None:
                away_qb_out = float(max(away.games_played - last_gp, 0))
            elif away.current_qb_id and away.current_qb_id != away_qb_id:
                away_qb_out = 4.0

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

        madden_known = 0.0
        madden_diff = 0.0
        if "madden_ovr_diff" in game and game.get("madden_ovr_diff") is not None:
            madden_diff = _safe_float(game.get("madden_ovr_diff"), 0.0)
            madden_known = _safe_float(game.get("madden_known"), 1.0 if season >= MADDEN_MIN_SEASON else 0.0)
        elif season >= MADDEN_MIN_SEASON:
            madden_diff, madden_known = _madden_ovr_diff(str(game["home"]), str(game["away"]))

        home_key = str(game["home"]).lower()
        away_key = str(game["away"]).lower()
        venue = home_key
        home_prev = home.last_venue or home_key
        away_prev = away.last_venue or away_key
        home_travel = travel_km(home_prev, venue) / 1000.0
        away_travel = travel_km(away_prev, venue) / 1000.0
        home_alt = altitude(home_key)
        away_home_alt = altitude(away_key)
        tz_shift = timezone_diff(home_key, away_key)
        tz_debt = abs(tz_shift) * (1.0 if away_rest <= 7.0 else 0.5)
        gametime = str(game.get("gametime") or "").strip()
        hour = 13
        if ":" in gametime:
            try:
                hour = int(gametime.split(":")[0])
            except ValueError:
                hour = 13
        primetime = 1.0 if (
            weekday.startswith("mon") or weekday.startswith("thu") or hour >= 20
        ) else 0.0
        thanksgiving = 1.0 if weekday.startswith("thu") and int(week) in (12, 13) else 0.0
        loc = str(game.get("location") or "").lower()
        stadium = str(game.get("stadium") or "").lower()
        international = 1.0 if any(
            token in loc or token in stadium
            for token in ("london", "mexico", "munich", "frankfurt", "azteca", "tottenham")
        ) else 0.0
        west_coast_early = 1.0 if (
            team_tz(away_key) <= -7.5 and hour <= 14 and tz_shift > 1.5
        ) else 0.0
        east_at_west_night = 1.0 if (
            team_tz(away_key) >= -6.0 and team_tz(home_key) <= -7.5 and hour >= 20
        ) else 0.0
        exp_total = (home.pf_fast + home.pa_fast + away.pf_fast + away.pa_fast) / 2.0
        qb_elo_edge = (home_qb_elo - away_qb_elo) / 100.0
        rematch = 1.0 if h2h_total >= 1 else 0.0

        def _backup_qb_elo(team_state: TeamState, starter_id: str) -> float:
            best = LEAGUE_ELO
            found = False
            for qid, qent in self.qbs.items():
                if starter_id and qid == starter_id:
                    continue
                if qid not in team_state.qb_last_start_games:
                    continue
                found = True
                if qent.elo > best:
                    best = qent.elo
            return best if found else LEAGUE_ELO

        home_backup = _backup_qb_elo(home, home_qb_id if home_qb_known else "")
        away_backup = _backup_qb_elo(away, away_qb_id if away_qb_known else "")
        backup_qb_diff = (home_backup - away_backup) / 100.0
        coach_change_home = (
            1.0
            if home_coach_known and home.current_coach
            and home.current_coach != home_coach_name.lower()
            else 0.0
        )
        coach_change_away = (
            1.0
            if away_coach_known and away.current_coach
            and away.current_coach != away_coach_name.lower()
            else 0.0
        )
        coach_change_diff = coach_change_home - coach_change_away

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
            "exp_total_env": exp_total,
            "elo_x_season_frac": elo_diff * season_frac,
            "wind_x_outdoor": wind * roof_outdoor,
            "short_week_x_rest": short_week * (home_rest - away_rest),
            "cold_game": 1.0 if (roof_outdoor > 0.5 and temp < 32.0) else 0.0,
            "high_wind": 1.0 if (roof_outdoor > 0.5 and wind >= 15.0) else 0.0,
            "temp_x_outdoor": (temp - DEFAULT_TEMP) * roof_outdoor,
            "tz_diff": timezone_diff(str(game["home"]), str(game["away"])),
            "westbound_short_week": (
                1.0
                if short_week > 0.5 and timezone_diff(str(game["home"]), str(game["away"])) > 1.5
                else 0.0
            ),
            "h2h_home_win_rate": h2h_rate,
            "h2h_margin_ewma": home.h2h_margin.get(away.key, 0.0),
            "qb_elo_diff": home_qb_elo - away_qb_elo,
            "qb_games_diff": home_qb_games - away_qb_games,
            "qb_win_ewma_diff": home_qb_win - away_qb_win,
            "home_qb_known": home_qb_known,
            "away_qb_known": away_qb_known,
            "home_qb_streak": home.qb_streak_for(home_qb_id),
            "away_qb_streak": away.qb_streak_for(away_qb_id),
            "qb_change_diff": home_qb_change - away_qb_change,
            "qb_games_out_proxy_diff": home_qb_out - away_qb_out,
            "coach_elo_diff": home_coach_elo - away_coach_elo,
            "coach_games_diff": home_coach_games - away_coach_games,
            "coach_win_ewma_diff": home_coach_win - away_coach_win,
            "home_coach_known": home_coach_known,
            "away_coach_known": away_coach_known,
            "home_coach_streak": home.coach_streak_for(home_coach_name),
            "away_coach_streak": away.coach_streak_for(away_coach_name),
            "ref_known": 0.0,
            "ref_home_bias": 0.0,
            "epa_off_diff": home.epa_off - away.epa_off,
            "epa_def_diff": home.epa_def - away.epa_def,
            "sr_off_diff": home.sr_off - away.sr_off,
            "sr_def_diff": home.sr_def - away.sr_def,
            "explosive_diff": home.explosive_off - away.explosive_off,
            "pass_epa_diff": home.pass_epa_off - away.pass_epa_off,
            "rush_epa_diff": home.rush_epa_off - away.rush_epa_off,
            "pass_rush_epa_gap_diff": (
                (home.pass_epa_off - home.rush_epa_off) - (away.pass_epa_off - away.rush_epa_off)
            ),
            "sack_rate_off_diff": home.sack_rate_off - away.sack_rate_off,
            "sack_rate_def_diff": home.sack_rate_def - away.sack_rate_def,
            "qb_hit_rate_off_diff": home.qb_hit_rate_off - away.qb_hit_rate_off,
            "qb_hit_rate_def_diff": home.qb_hit_rate_def - away.qb_hit_rate_def,
            "early_down_epa_diff": home.early_down_epa_off - away.early_down_epa_off,
            "third_down_sr_diff": home.third_down_sr_off - away.third_down_sr_off,
            "redzone_epa_diff": home.redzone_epa_off - away.redzone_epa_off,
            "epa_off_vs_def": (home.epa_off - away.epa_def) - (away.epa_off - home.epa_def),
            "epa_x_season_frac": (home.epa_off - away.epa_off) * season_frac,
            "pass_epa_x_wind": (home.pass_epa_off - away.pass_epa_off) * wind * roof_outdoor,
            "wr1_snap_share_diff": home.wr1_snap_share - away.wr1_snap_share,
            "ol_starter_share_diff": home.ol_starter_share - away.ol_starter_share,
            "skill_snap_share_diff": home.skill_snap_share - away.skill_snap_share,
            "injury_burden_diff": (
                _safe_float(game.get("home_out"), 0.0) - _safe_float(game.get("away_out"), 0.0)
            ),
            "ol_out_diff": (
                _safe_float(game.get("home_ol_out"), 0.0) - _safe_float(game.get("away_ol_out"), 0.0)
            ),
            "skill_out_diff": (
                _safe_float(game.get("home_skill_out"), 0.0)
                - _safe_float(game.get("away_skill_out"), 0.0)
            ),
            "injury_known": _safe_float(game.get("injury_known"), 0.0),
            "has_open_line": 0.0,
            "has_steam": 0.0,
            "spread_move": 0.0,
            "ml_steam_pp": 0.0,
            "has_market": 0.0,
            "mkt_home_prob": 0.5,
            "has_spread": 0.0,
            "mkt_home_spread": 0.0,
            "has_total": 0.0,
            "total_line": exp_total,
            "model_total_vs_line": 0.0,
            "total_move": 0.0,
            "madden_ovr_diff": madden_diff,
            "madden_known": madden_known,
            "close_win_ewma_diff": home.close_win_ewma - away.close_win_ewma,
            "blowout_net_ewma_diff": home.blowout_net_ewma - away.blowout_net_ewma,
            "blowout_rate_diff": home.blowout_rate() - away.blowout_rate(),
            "margin_vol_diff": home.margin_volatility() - away.margin_volatility(),
            "scoring_vol_diff": home.scoring_volatility() - away.scoring_volatility(),
            "one_score_rate_diff": home.one_score_rate() - away.one_score_rate(),
            "home_off_vs_away_def": home.pf_fast - away.pa_fast,
            "away_off_vs_home_def": away.pf_fast - home.pa_fast,
            "net_x_season_frac": (home.margin_pg() - away.margin_pg()) * season_frac,
            "explosive_def_diff": home.explosive_def - away.explosive_def,
            "pass_epa_def_diff": home.pass_epa_def - away.pass_epa_def,
            "rush_epa_def_diff": home.rush_epa_def - away.rush_epa_def,
            "to_rate_off_diff": home.to_rate_off - away.to_rate_off,
            "to_rate_def_diff": home.to_rate_def - away.to_rate_def,
            "sr_gap_diff": (
                (home.sr_off - home.sr_def) - (away.sr_off - away.sr_def)
            ),
            "home_pass_vs_away_pass_def": home.pass_epa_off - away.pass_epa_def,
            "away_pass_vs_home_pass_def": away.pass_epa_off - home.pass_epa_def,
            "home_rush_vs_away_rush_def": home.rush_epa_off - away.rush_epa_def,
            "away_rush_vs_home_rush_def": away.rush_epa_off - home.rush_epa_def,
            "qb_elo_x_pass_def": qb_elo_edge * (away.pass_epa_def - home.pass_epa_def),
            "qb_elo_x_wind": qb_elo_edge * wind * roof_outdoor,
            "qb_elo_x_short_week": qb_elo_edge * short_week,
            "ol_snap_x_sack_allowed": (
                home.ol_starter_share * away.sack_rate_def
                - away.ol_starter_share * home.sack_rate_def
            ),
            "epa_x_div_game": (home.epa_off - away.epa_off) * div_game,
            "elo_x_is_playoff": elo_diff * is_playoff,
            "elo_x_primetime": elo_diff * primetime,
            "injury_x_ol_out": (
                (_safe_float(game.get("home_out"), 0.0) - _safe_float(game.get("away_out"), 0.0))
                * (
                    _safe_float(game.get("home_ol_out"), 0.0)
                    - _safe_float(game.get("away_ol_out"), 0.0)
                )
            ),
            "home_travel_km": home_travel,
            "away_travel_km": away_travel,
            "travel_diff": home_travel - away_travel,
            "bye_week_home": 1.0 if home_rest >= BYE_REST_DAYS else 0.0,
            "bye_week_away": 1.0 if away_rest >= BYE_REST_DAYS else 0.0,
            "home_stand_len": float(home.home_stand) if home.last_venue == str(game["home"]).lower() else 0.0,
            "away_road_trip": float(away.road_trip) if away.last_venue and away.last_venue != str(game["away"]).lower() else 0.0,
            "home_altitude": home_alt,
            "altitude_diff": home_alt - away_home_alt,
            "west_coast_early": west_coast_early,
            "east_at_west_night": east_at_west_night,
            "tz_debt_away": tz_debt,
            "early_season": 1.0 if min(home.games_played, away.games_played) < 4 else 0.0,
            "luck_diff": (home.pythag() - home.win_pct()) - (away.pythag() - away.win_pct()),
            "big_fav_spread": 1.0 if abs(_safe_float(game.get("home_close_spread"), 0.0)) >= 10.0 else 0.0,
            "home_fav_elo": 1.0 if elo_diff > 0 else 0.0,
            "dog_elo_gap": float(max(-elo_diff, 0.0)),
            "rematch_flag": rematch,
            "primetime": primetime,
            "thanksgiving": thanksgiving,
            "international": international,
            "qb_first_start_home": 1.0 if home_qb_known and home_qb_games <= 0 else 0.0,
            "qb_first_start_away": 1.0 if away_qb_known and away_qb_games <= 0 else 0.0,
            "backup_qb_elo_diff": backup_qb_diff,
            "coach_change_diff": coach_change_diff,
        }
        # Referee home-bias Elo (prior to this game)
        ref_name = str(game.get("referee") or "").strip()
        if not _is_missing_id(ref_name):
            ref_ent = self.ref(ref_name)
            features["ref_known"] = 1.0
            features["ref_home_bias"] = (ref_ent.elo - LEAGUE_ELO) / 100.0

        # Opening-line steam when opens are present (never invent open=close)
        open_spread = game.get("home_open_spread")
        close_spread = game.get("home_close_spread")
        open_hml = game.get("home_open_ml")
        close_hml = game.get("home_close_ml")
        open_aml = game.get("away_open_ml")
        close_aml = game.get("away_close_ml")
        open_total = game.get("open_total")
        close_total = game.get("close_total") or game.get("total_line")
        has_spread_open = open_spread is not None and close_spread is not None
        has_ml_open = (
            open_hml is not None
            and close_hml is not None
            and open_aml is not None
            and close_aml is not None
        )
        if has_spread_open or has_ml_open:
            features["has_open_line"] = 1.0
        if has_spread_open:
            try:
                features["spread_move"] = float(open_spread) - float(close_spread)
                features["has_steam"] = 1.0
            except (TypeError, ValueError):
                features["spread_move"] = 0.0
        if has_ml_open:
            try:
                features["ml_steam_pp"] = (
                    _amer_implied(float(close_hml)) - _amer_implied(float(open_hml))
                ) * 100.0
                features["has_steam"] = 1.0
                features["has_open_line"] = 1.0
            except (TypeError, ValueError):
                features["ml_steam_pp"] = 0.0
        if close_hml is not None and close_aml is not None:
            try:
                features["has_market"] = 1.0
                ch = _amer_implied(float(close_hml))
                ca = _amer_implied(float(close_aml))
                tot = ch + ca
                features["mkt_home_prob"] = ch / tot if tot > 0 else 0.5
            except (TypeError, ValueError):
                pass
        if close_spread is not None:
            try:
                features["has_spread"] = 1.0
                features["mkt_home_spread"] = float(close_spread)
            except (TypeError, ValueError):
                pass
        if close_total is not None:
            try:
                features["has_total"] = 1.0
                features["total_line"] = float(close_total)
                features["model_total_vs_line"] = exp_total - float(close_total)
            except (TypeError, ValueError):
                pass
        if open_total is not None and close_total is not None:
            try:
                features["total_move"] = float(close_total) - float(open_total)
                features["has_open_line"] = 1.0
            except (TypeError, ValueError):
                pass
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

        # Bilateral W/L (mirrors NBA/NHL): reverse fixtures must not fall back to 0.5.
        if not tied:
            home_rec = home.h2h.setdefault(away.key, [0, 0])
            away_rec = away.h2h.setdefault(home.key, [0, 0])
            if home_win:
                home_rec[0] += 1
                away_rec[1] += 1
            else:
                home_rec[1] += 1
                away_rec[0] += 1
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
            home.qb_last_start_games[home_qb_id] = home.games_played
        if not _is_missing_id(away_qb_id):
            if away.current_qb_id == away_qb_id:
                away.qb_streak += 1
            else:
                away.current_qb_id = away_qb_id
                away.qb_streak = 1
            away.qb_last_start_games[away_qb_id] = away.games_played

        # PBP EPA — post-game only (features already emitted from prior EWMA).
        def _ewma_epa(old: float, new: float | None) -> float:
            if new is None or (isinstance(new, float) and not math.isfinite(new)):
                return old
            return ALPHA_EPA * float(new) + (1.0 - ALPHA_EPA) * old

        for team, prefix in ((home, "home"), (away, "away")):
            team.epa_off = _ewma_epa(team.epa_off, game.get(f"{prefix}_epa_off"))
            team.epa_def = _ewma_epa(team.epa_def, game.get(f"{prefix}_epa_def"))
            team.sr_off = _ewma_epa(team.sr_off, game.get(f"{prefix}_sr_off"))
            team.sr_def = _ewma_epa(team.sr_def, game.get(f"{prefix}_sr_def"))
            team.explosive_off = _ewma_epa(team.explosive_off, game.get(f"{prefix}_explosive_off"))
            team.explosive_def = _ewma_epa(team.explosive_def, game.get(f"{prefix}_explosive_def"))
            team.pass_epa_off = _ewma_epa(team.pass_epa_off, game.get(f"{prefix}_pass_epa_off"))
            team.rush_epa_off = _ewma_epa(team.rush_epa_off, game.get(f"{prefix}_rush_epa_off"))
            team.pass_epa_def = _ewma_epa(team.pass_epa_def, game.get(f"{prefix}_pass_epa_def"))
            team.rush_epa_def = _ewma_epa(team.rush_epa_def, game.get(f"{prefix}_rush_epa_def"))
            team.sack_rate_off = _ewma_epa(team.sack_rate_off, game.get(f"{prefix}_sack_rate_off"))
            team.sack_rate_def = _ewma_epa(team.sack_rate_def, game.get(f"{prefix}_sack_rate_def"))
            team.qb_hit_rate_off = _ewma_epa(team.qb_hit_rate_off, game.get(f"{prefix}_qb_hit_rate_off"))
            team.qb_hit_rate_def = _ewma_epa(team.qb_hit_rate_def, game.get(f"{prefix}_qb_hit_rate_def"))
            team.early_down_epa_off = _ewma_epa(
                team.early_down_epa_off, game.get(f"{prefix}_early_down_epa_off")
            )
            team.third_down_sr_off = _ewma_epa(
                team.third_down_sr_off, game.get(f"{prefix}_third_down_sr_off")
            )
            team.redzone_epa_off = _ewma_epa(
                team.redzone_epa_off, game.get(f"{prefix}_redzone_epa_off")
            )
            team.to_rate_off = _ewma_epa(team.to_rate_off, game.get(f"{prefix}_to_rate_off"))
            team.to_rate_def = _ewma_epa(team.to_rate_def, game.get(f"{prefix}_to_rate_def"))
            team.wr1_snap_share = _ewma_epa(team.wr1_snap_share, game.get(f"{prefix}_wr1_snap_share"))
            team.ol_starter_share = _ewma_epa(
                team.ol_starter_share, game.get(f"{prefix}_ol_starter_share")
            )
            team.skill_snap_share = _ewma_epa(
                team.skill_snap_share, game.get(f"{prefix}_skill_snap_share")
            )

        # Home stand / road trip after this game.
        venue_key = str(game["home"]).lower()
        home.road_trip = 0
        away.road_trip = (
            (away.road_trip if away.last_venue and away.last_venue != str(game["away"]).lower() else 0)
            + 1
        )
        away.home_stand = 0
        home.home_stand = (
            (home.home_stand if home.last_venue == venue_key else 0) + 1
        )
        home.last_venue = venue_key
        away.last_venue = venue_key

        # Referee Elo — home-team perspective after the game
        ref_name = str(game.get("referee") or "").strip()
        if not _is_missing_id(ref_name) and not tied:
            ref_ent = self.ref(ref_name)
            # Edge near 0 so K updates track home win frequency under this crew.
            self._update_entity(
                ref_ent, won=home_win, tied=False, edge=0.0, k=REF_ELO_K
            )

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
            "refs": {key: ref.to_dict() for key, ref in self.refs.items()},
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
        for key, ref_payload in dict(payload.get("refs") or {}).items():
            if "key" not in ref_payload:
                ref_payload = dict(ref_payload)
                ref_payload["key"] = key
            engine.refs[str(key)] = EntityElo.from_dict(ref_payload)
        return engine
