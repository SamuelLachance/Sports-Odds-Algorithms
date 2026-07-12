"""Walk-forward WNBA feature engine shared by offline training and live inference.

Consumes games chronologically (1997+). For each game it can produce a feature
row using only state accumulated strictly before that date, then folds the
result into state. State is JSON-serializable so training can snapshot
end-of-season state and the live path replays only the current season.

Game dict schema (merged from data.wnba.com results + ESPN boxes):
  date (ISO), season, season_type (2 reg / 3 post), home, away (franchise keys),
  home_score, away_score, neutral_site (optional),
  home_box / away_box (optional dicts: fgm fga tpm tpa ftm fta orb drb tov ast,
  plus optional players: [[athlete_id, minutes, fga, ast, tov, pf, plus_minus], ...]
  and optional dnp_ids: [athlete_id, ...]).
"""

from __future__ import annotations

import math
from datetime import date as date_cls
from typing import Any

from web.v2_schedule_utils import count_games_in_last_n_days
from web.wnba_v2.arenas import market_altitude_m, market_coords

LEAGUE_ELO = 1500.0
ELO_K = 24.0
ELO_HOME_ADV = 70.0
ELO_SEASON_CARRYOVER = 0.70
ELO_PER_POINT = 28.0  # ~70 Elo home edge ~ 2.5 pts

ALPHA_FAST = 0.22
ALPHA_SLOW = 0.08
ALPHA_WIN = 0.12
ALPHA_LEAGUE = 0.02
ALPHA_SHOOT = 0.10
ALPHA_H2H_MARGIN = 0.30
ALPHA_PLAYER = 0.20
PLAYER_ALPHA_FAST = 0.30
PLAYER_ALPHA_SLOW = 0.12

LEAGUE_PPG = 78.0
LEAGUE_PACE = 78.0
LEAGUE_ORTG = 100.0
LEAGUE_EFG = 0.47
LEAGUE_TOV_RATE = 0.155
LEAGUE_ORB_PCT = 0.28
LEAGUE_FT_RATE = 0.22
LEAGUE_TPA_RATE = 0.25
LEAGUE_TP_PCT = 0.335
LEAGUE_FT_PCT = 0.77

# player-box priors (minutes-only caches still update share / depth / HHI)
LEAGUE_TOP1_MIN_SHARE = 0.18
LEAGUE_TOP3_MIN_SHARE = 0.48
LEAGUE_TOP1_USAGE = 0.20
LEAGUE_TOP3_USAGE = 0.50
LEAGUE_HIGH_MIN_AST_TOV = 1.5
LEAGUE_HIGH_MIN_FOUL36 = 3.6
LEAGUE_STAR_MIN = 30.0
LEAGUE_BENCH_PM = 0.0
LEAGUE_DNP_STAR_RATE = 0.05
LEAGUE_ROTATION_DEPTH = 8.0
LEAGUE_MIN_HHI = 0.14
LEAGUE_BENCH_MIN_SHARE = 0.30
HIGH_MIN_FLOOR = 18.0
BENCH_TOP_N = 5

SEASON_GAMES_NOMINAL = 40.0
CLOSE_MARGIN = 5.0
BLOWOUT_MARGIN = 15.0
DEFAULT_MARGIN_VOL = 11.0
HOME_ADV_POINTS = 2.5

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
    # v2.1 additions
    "net_rtg_trend_diff",
    "ortg_trend_diff",
    "drtg_trend_diff",
    "efg_trend_diff",
    "elo_mom5_diff",
    "blowout_rate_diff",
    "home_stand_len",
    "away_road_trip_len",
    "home_tz_shift",
    "tp_pct_diff",
    "top8_continuity_diff",
    # availability / richer player-rotation proxies (prior boxes only)
    "star_avail_diff",
    "top1_min_share_diff",
    "top3_min_share_diff",
    "top1_usage_diff",
    "top3_usage_diff",
    "high_min_ast_tov_diff",
    "high_min_foul_rate_diff",
    "star_min_gap_diff",
    "bench_pm_diff",
    "dnp_star_rate_diff",
    "rotation_depth_diff",
    "min_hhi_diff",
    "bench_min_share_diff",
    "h2h_margin_ewma",
    "adj_margin_ewma_diff",
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


def _std(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / n)


def _top_players(minutes: dict[str, float], count: int, floor: float) -> list[str]:
    ranked = sorted(minutes.items(), key=lambda kv: (-kv[1], kv[0]))
    return [pid for pid, mins in ranked[:count] if mins >= floor]


def _player_id(row: Any) -> str | None:
    if isinstance(row, (list, tuple)) and row:
        pid = str(row[0] or "")
        return pid or None
    if isinstance(row, str) and row:
        return row
    return None


def _parse_player_row(row: Any) -> tuple[str, float, float, float, float, float, float] | None:
    """Normalize legacy [id, min] or rich [id, min, fga, ast, tov, pf, +/-] rows."""
    if not isinstance(row, (list, tuple)) or len(row) < 2:
        return None
    pid = str(row[0] or "")
    try:
        mins = float(row[1])
    except (TypeError, ValueError):
        return None
    if not pid or mins <= 0:
        return None

    def _at(idx: int) -> float:
        if idx >= len(row):
            return 0.0
        try:
            return float(row[idx])
        except (TypeError, ValueError):
            return 0.0

    return pid, mins, _at(2), _at(3), _at(4), _at(5), _at(6)


def _player_rotation_metrics(
    rows: list[Any],
    *,
    dnp_ids: list[str] | None = None,
    season_minutes: dict[str, float] | None = None,
    games_played: int = 0,
) -> dict[str, float]:
    """Instantaneous rotation/usage metrics from one prior box (leak-free inputs)."""
    parsed = []
    for row in rows:
        item = _parse_player_row(row)
        if item is not None:
            parsed.append(item)
    if not parsed:
        return {}
    parsed.sort(key=lambda r: -r[1])
    total_min = sum(r[1] for r in parsed)
    total_fga = sum(r[2] for r in parsed)
    if total_min <= 0:
        return {}

    top1 = parsed[0]
    top3 = parsed[:3]
    top1_min_share = top1[1] / total_min
    top3_min_share = sum(r[1] for r in top3) / total_min
    if total_fga > 0:
        top1_usage = top1[2] / total_fga
        top3_usage = sum(r[2] for r in top3) / total_fga
    else:
        # minutes-only caches: usage proxy = minutes share
        top1_usage = top1_min_share
        top3_usage = top3_min_share

    high = [r for r in parsed if r[1] >= HIGH_MIN_FLOOR] or parsed[:5]
    high_ast = sum(r[3] for r in high)
    high_tov = sum(r[4] for r in high)
    high_pf = sum(r[5] for r in high)
    high_min = sum(r[1] for r in high)
    high_min_ast_tov = high_ast / max(high_tov, 0.5)
    high_min_foul_rate = (high_pf / high_min) * 36.0 if high_min > 0 else LEAGUE_HIGH_MIN_FOUL36

    bench = parsed[BENCH_TOP_N:]
    bench_min = sum(r[1] for r in bench)
    bench_min_share = bench_min / total_min
    # only trust plus/minus when any non-zero (rich boxes); else prior stays
    has_pm = any(abs(r[6]) > 1e-9 for r in parsed)
    bench_pm = sum(r[6] for r in bench) if has_pm else LEAGUE_BENCH_PM

    shares = [r[1] / total_min for r in parsed]
    min_hhi = sum(s * s for s in shares)
    rotation_depth = float(sum(1 for r in parsed if r[1] >= 10.0))

    season_minutes = season_minutes or {}
    played_ids = {r[0] for r in parsed}
    dnp_set = set(dnp_ids or [])
    ranked_season = sorted(season_minutes, key=lambda k: -season_minutes[k])
    top_season = ranked_season[:8]
    if top_season:
        absent = sum(
            1 for pid in top_season if pid not in played_ids or pid in dnp_set
        )
        dnp_star_rate = absent / len(top_season)
    else:
        dnp_star_rate = LEAGUE_DNP_STAR_RATE

    star_ids = ranked_season[:3]
    if star_ids and games_played > 0:
        last_mins = []
        season_mpg = []
        for pid in star_ids:
            matched = next((r[1] for r in parsed if r[0] == pid), 0.0)
            last_mins.append(matched)
            season_mpg.append(season_minutes.get(pid, 0.0) / max(games_played, 1))
        star_min_last = sum(last_mins) / len(last_mins)
        star_min_season_avg = sum(season_mpg) / len(season_mpg)
    else:
        star_min_last = LEAGUE_STAR_MIN
        star_min_season_avg = LEAGUE_STAR_MIN

    return {
        "top1_min_share": top1_min_share,
        "top3_min_share": top3_min_share,
        "top1_usage": top1_usage,
        "top3_usage": top3_usage,
        "high_min_ast_tov": high_min_ast_tov,
        "high_min_foul_rate": high_min_foul_rate,
        "star_min_last": star_min_last,
        "star_min_season_avg": star_min_season_avg,
        "bench_pm": bench_pm,
        "dnp_star_rate": dnp_star_rate,
        "rotation_depth": rotation_depth,
        "min_hhi": min_hhi,
        "bench_min_share": bench_min_share,
        "has_pm": 1.0 if has_pm else 0.0,
        "has_usage": 1.0 if total_fga > 0 else 0.0,
    }


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
        # v2.1 state
        "efg_for_slow", "tpa_rate", "tp_pct", "ft_pct",
        "elo_hist", "close_win_ewma", "blowout_ewma", "recent_margins",
        "venue_streak", "adj_margin_ewma", "last_altitude",
        "player_min_fast", "player_min_slow", "last_players", "h2h_margin",
        # rotation / usage EWMAs (NBA-style; prior boxes only)
        "prev_player_ids", "season_minutes",
        "top1_min_share", "top3_min_share", "top1_usage", "top3_usage",
        "high_min_ast_tov", "high_min_foul_rate",
        "star_min_ewma", "star_min_season_avg", "bench_pm",
        "dnp_star_rate", "rotation_depth", "min_hhi", "bench_min_share",
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
        self.efg_for_slow = LEAGUE_EFG
        self.tpa_rate = LEAGUE_TPA_RATE
        self.tp_pct = LEAGUE_TP_PCT
        self.ft_pct = LEAGUE_FT_PCT
        self.elo_hist: list[float] = []
        self.close_win_ewma = 0.5
        self.blowout_ewma = 0.0
        self.recent_margins: list[float] = []
        self.venue_streak = 0
        self.adj_margin_ewma = 0.0
        self.last_altitude: float | None = None
        self.player_min_fast: dict[str, float] = {}
        self.player_min_slow: dict[str, float] = {}
        self.last_players: list[Any] = []
        self.h2h_margin: dict[str, float] = {}
        self.prev_player_ids: list[str] = []
        self.season_minutes: dict[str, float] = {}
        self.top1_min_share = LEAGUE_TOP1_MIN_SHARE
        self.top3_min_share = LEAGUE_TOP3_MIN_SHARE
        self.top1_usage = LEAGUE_TOP1_USAGE
        self.top3_usage = LEAGUE_TOP3_USAGE
        self.high_min_ast_tov = LEAGUE_HIGH_MIN_AST_TOV
        self.high_min_foul_rate = LEAGUE_HIGH_MIN_FOUL36
        self.star_min_ewma = LEAGUE_STAR_MIN
        self.star_min_season_avg = LEAGUE_STAR_MIN
        self.bench_pm = LEAGUE_BENCH_PM
        self.dnp_star_rate = LEAGUE_DNP_STAR_RATE
        self.rotation_depth = LEAGUE_ROTATION_DEPTH
        self.min_hhi = LEAGUE_MIN_HHI
        self.bench_min_share = LEAGUE_BENCH_MIN_SHARE

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
            self.adj_margin_ewma *= ELO_SEASON_CARRYOVER
        self.wins = 0
        self.losses = 0
        self.points_for = 0.0
        self.points_against = 0.0
        self.sos_elo_sum = 0.0
        self.last_game_date = ""
        self.recent_dates = []
        # Drop last venue so opening-night travel/tz do not inherit prior season.
        self.last_market = None
        self.streak = 0
        self.games_played = 0
        self.season_seen = season
        self.elo_hist = []
        self.recent_margins = []
        self.venue_streak = 0
        # offseason roster churn: fade minutes credit, forget last lineup
        self.player_min_fast = {
            pid: mins * 0.5 for pid, mins in self.player_min_fast.items() if mins * 0.5 >= 1.0
        }
        self.player_min_slow = {
            pid: mins * 0.5 for pid, mins in self.player_min_slow.items() if mins * 0.5 >= 1.0
        }
        self.last_players = []
        self.prev_player_ids = []
        self.season_minutes = {}
        # keep EWMA rotation priors across seasons (gentle carry); reset DNP noise
        self.dnp_star_rate = LEAGUE_DNP_STAR_RATE

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
        # Floor at 0: inverted/out-of-order dates must not invent negative rest
        # (which also falsely trips B2B via rest <= 1).
        return float(min(max((game_date - prior).days, 0), 10))

    def games_in_last7(self, game_date: date_cls) -> int:
        return count_games_in_last_n_days(self.recent_dates, game_date, days=7)

    def is_3in4(self, game_date: date_cls) -> bool:
        return count_games_in_last_n_days(self.recent_dates, game_date, days=3) >= 2

    def elo_momentum5(self) -> float:
        if not self.elo_hist:
            return 0.0
        anchor = self.elo_hist[-6] if len(self.elo_hist) >= 6 else self.elo_hist[0]
        return self.elo - anchor

    def margin_volatility(self) -> float:
        if len(self.recent_margins) < 4:
            return DEFAULT_MARGIN_VOL
        return _std(self.recent_margins)

    def _last_player_ids(self) -> set[str]:
        ids: set[str] = set()
        for row in self.last_players:
            pid = _player_id(row)
            if pid:
                ids.add(pid)
        return ids

    def top8_continuity(self) -> float:
        """Share of the recent-minutes core that appeared in the last game."""
        if not self.last_players:
            return 1.0
        pool = _top_players(self.player_min_fast, 8, 3.0)
        if len(pool) < 5:
            return 1.0
        present = self._last_player_ids()
        return sum(1 for pid in pool if pid in present) / len(pool)

    def star_availability(self) -> float:
        """Share of the slow-decay top-3 stars that appeared in the last game."""
        if not self.last_players:
            return 1.0
        pool = _top_players(self.player_min_slow, 3, 8.0)
        if not pool:
            return 1.0
        present = self._last_player_ids()
        return sum(1 for pid in pool if pid in present) / len(pool)

    def star_min_gap(self) -> float:
        """Recent star minutes EWMA minus season-average MPG of those stars."""
        return self.star_min_ewma - self.star_min_season_avg

    def apply_player_metrics(self, metrics: dict[str, float]) -> None:
        """Fold one game's rotation metrics into team EWMAs (call after features)."""
        if not metrics:
            return
        a = ALPHA_PLAYER

        def _ew(attr: str, key: str) -> None:
            cur = getattr(self, attr)
            setattr(self, attr, cur + a * (float(metrics[key]) - cur))

        for attr, key in (
            ("top1_min_share", "top1_min_share"),
            ("top3_min_share", "top3_min_share"),
            ("rotation_depth", "rotation_depth"),
            ("min_hhi", "min_hhi"),
            ("bench_min_share", "bench_min_share"),
            ("dnp_star_rate", "dnp_star_rate"),
            ("star_min_ewma", "star_min_last"),
            ("star_min_season_avg", "star_min_season_avg"),
        ):
            if key in metrics:
                _ew(attr, key)
        if metrics.get("has_usage", 0.0) > 0:
            _ew("top1_usage", "top1_usage")
            _ew("top3_usage", "top3_usage")
            _ew("high_min_ast_tov", "high_min_ast_tov")
            _ew("high_min_foul_rate", "high_min_foul_rate")
        else:
            # minutes-only: still update usage proxies from minute shares
            _ew("top1_usage", "top1_usage")
            _ew("top3_usage", "top3_usage")
        if metrics.get("has_pm", 0.0) > 0:
            _ew("bench_pm", "bench_pm")

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
            "efg_for_slow": self.efg_for_slow,
            "tpa_rate": self.tpa_rate,
            "tp_pct": self.tp_pct,
            "ft_pct": self.ft_pct,
            "elo_hist": list(self.elo_hist[-8:]),
            "close_win_ewma": self.close_win_ewma,
            "blowout_ewma": self.blowout_ewma,
            "recent_margins": list(self.recent_margins[-10:]),
            "venue_streak": self.venue_streak,
            "adj_margin_ewma": self.adj_margin_ewma,
            "last_altitude": self.last_altitude,
            "player_min_fast": dict(self.player_min_fast),
            "player_min_slow": dict(self.player_min_slow),
            "last_players": [
                list(row) if isinstance(row, (list, tuple)) else row
                for row in self.last_players
            ],
            "h2h_margin": dict(self.h2h_margin),
            "prev_player_ids": [str(pid) for pid in self.prev_player_ids],
            "season_minutes": {
                pid: round(mins, 1)
                for pid, mins in sorted(
                    self.season_minutes.items(), key=lambda kv: -kv[1]
                )[:15]
            },
            "top1_min_share": self.top1_min_share,
            "top3_min_share": self.top3_min_share,
            "top1_usage": self.top1_usage,
            "top3_usage": self.top3_usage,
            "high_min_ast_tov": self.high_min_ast_tov,
            "high_min_foul_rate": self.high_min_foul_rate,
            "star_min_ewma": self.star_min_ewma,
            "star_min_season_avg": self.star_min_season_avg,
            "bench_pm": self.bench_pm,
            "dnp_star_rate": self.dnp_star_rate,
            "rotation_depth": self.rotation_depth,
            "min_hhi": self.min_hhi,
            "bench_min_share": self.bench_min_share,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TeamState":
        """Restore state; fields absent from old snapshots keep their defaults."""
        state = cls(str(payload.get("franchise") or ""))
        for key, value in payload.items():
            if key == "recent_dates":
                state.recent_dates = [str(v) for v in value]
            elif key == "h2h":
                state.h2h = {str(k): [int(x) for x in v] for k, v in dict(value).items()}
            elif key == "last_market":
                state.last_market = (float(value[0]), float(value[1])) if value else None
            elif key == "elo_hist":
                state.elo_hist = [float(v) for v in value]
            elif key == "recent_margins":
                state.recent_margins = [float(v) for v in value]
            elif key == "last_players":
                # accept legacy list[str] or rich list[list]
                restored: list[Any] = []
                for row in value or []:
                    if isinstance(row, (list, tuple)):
                        restored.append(list(row))
                    else:
                        restored.append(str(row))
                state.last_players = restored
            elif key == "prev_player_ids":
                state.prev_player_ids = [str(v) for v in value]
            elif key == "season_minutes":
                state.season_minutes = {
                    str(k): float(v) for k, v in dict(value).items()
                }
            elif key in ("player_min_fast", "player_min_slow"):
                setattr(state, key, {str(k): float(v) for k, v in dict(value).items()})
            elif key == "h2h_margin":
                state.h2h_margin = {str(k): float(v) for k, v in dict(value).items()}
            elif key == "last_altitude":
                state.last_altitude = float(value) if value is not None else None
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
        home_tz_shift = 0.0
        away_tz_shift = 0.0
        if venue:
            home_from = home.last_market or home_market
            away_from = away.last_market or away_market
            if home_from:
                home_travel = _haversine_km(home_from, venue)
                home_tz_shift = (venue[1] - home_from[1]) / 15.0
            if away_from:
                away_travel = _haversine_km(away_from, venue)
                away_tz_shift = (venue[1] - away_from[1]) / 15.0

        venue_alt = market_altitude_m(home.franchise, season) or 0.0
        away_alt_from = away.last_altitude
        if away_alt_from is None:
            away_alt_from = market_altitude_m(away.franchise, season) or venue_alt

        h2h_record = home.h2h.get(away.franchise) or [0, 0]
        h2h_total = h2h_record[0] + h2h_record[1]
        h2h_rate = h2h_record[0] / h2h_total if h2h_total else 0.5

        season_frac = min(home.games_played, away.games_played) / SEASON_GAMES_NOMINAL
        elo_diff = home.elo - away.elo + (0.0 if neutral else ELO_HOME_ADV)
        rest_diff = home_rest - away_rest

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
            "rest_diff": rest_diff,
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
            "net_rtg_trend_diff": (
                (home.ortg_fast - home.drtg_fast) - (home.ortg_slow - home.drtg_slow)
            )
            - ((away.ortg_fast - away.drtg_fast) - (away.ortg_slow - away.drtg_slow)),
            "ortg_trend_diff": (home.ortg_fast - home.ortg_slow)
            - (away.ortg_fast - away.ortg_slow),
            "drtg_trend_diff": (home.drtg_fast - home.drtg_slow)
            - (away.drtg_fast - away.drtg_slow),
            "efg_trend_diff": (home.efg_for - home.efg_for_slow)
            - (away.efg_for - away.efg_for_slow),
            "elo_mom5_diff": home.elo_momentum5() - away.elo_momentum5(),
            "blowout_rate_diff": home.blowout_ewma - away.blowout_ewma,
            "home_stand_len": home_stand,
            "away_road_trip_len": road_trip,
            "home_tz_shift": home_tz_shift,
            "tp_pct_diff": home.tp_pct - away.tp_pct,
            "top8_continuity_diff": home.top8_continuity() - away.top8_continuity(),
            "star_avail_diff": home.star_availability() - away.star_availability(),
            "top1_min_share_diff": home.top1_min_share - away.top1_min_share,
            "top3_min_share_diff": home.top3_min_share - away.top3_min_share,
            "top1_usage_diff": home.top1_usage - away.top1_usage,
            "top3_usage_diff": home.top3_usage - away.top3_usage,
            "high_min_ast_tov_diff": home.high_min_ast_tov - away.high_min_ast_tov,
            "high_min_foul_rate_diff": home.high_min_foul_rate - away.high_min_foul_rate,
            "star_min_gap_diff": home.star_min_gap() - away.star_min_gap(),
            "bench_pm_diff": home.bench_pm - away.bench_pm,
            "dnp_star_rate_diff": home.dnp_star_rate - away.dnp_star_rate,
            "rotation_depth_diff": home.rotation_depth - away.rotation_depth,
            "min_hhi_diff": home.min_hhi - away.min_hhi,
            "bench_min_share_diff": home.bench_min_share - away.bench_min_share,
            "h2h_margin_ewma": home.h2h_margin.get(away.franchise, 0.0),
            "adj_margin_ewma_diff": home.adj_margin_ewma - away.adj_margin_ewma,
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
        signed_margin = home_score - away_score
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
        for team in (home, away):
            team.elo_hist.append(team.elo)
            if len(team.elo_hist) > 8:
                team.elo_hist = team.elo_hist[-8:]

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
                        team.efg_for_slow += ALPHA_SLOW * (efg - team.efg_for_slow)
                        team.ftr_for += ALPHA_FAST * (
                            float(box.get("fta") or 0.0) / fga - team.ftr_for
                        )
                        team.tpa_rate += ALPHA_FAST * (
                            float(box.get("tpa") or 0.0) / fga - team.tpa_rate
                        )
                    tpa = float(box.get("tpa") or 0.0)
                    if tpa >= 5:
                        team.tp_pct += ALPHA_SHOOT * (
                            float(box.get("tpm") or 0.0) / tpa - team.tp_pct
                        )
                    fta = float(box.get("fta") or 0.0)
                    if fta >= 5:
                        team.ft_pct += ALPHA_SHOOT * (
                            float(box.get("ftm") or 0.0) / fta - team.ft_pct
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
            if margin <= CLOSE_MARGIN:
                team.close_win_ewma += ALPHA_WIN * (result - team.close_win_ewma)
            blowout_signal = 0.0
            if margin >= BLOWOUT_MARGIN:
                blowout_signal = 1.0 if won else -1.0
            team.blowout_ewma += ALPHA_WIN * (blowout_signal - team.blowout_ewma)
            team_margin = signed_margin if was_home else -signed_margin
            team.recent_margins.append(team_margin)
            if len(team.recent_margins) > 10:
                team.recent_margins = team.recent_margins[-10:]
            hca = 0.0 if neutral else (HOME_ADV_POINTS if was_home else -HOME_ADV_POINTS)
            adj_margin = team_margin - hca + (opp_pre_elo - LEAGUE_ELO) / ELO_PER_POINT
            team.adj_margin_ewma += ALPHA_WIN * (adj_margin - team.adj_margin_ewma)
            team.sos_elo_sum += opp_pre_elo
            team.last_game_date = str(game.get("date") or "")
            team.recent_dates.append(str(game.get("date") or ""))
            if len(team.recent_dates) > 12:
                team.recent_dates = team.recent_dates[-12:]
            team.games_played += 1

        if neutral:
            home.venue_streak = 0
            away.venue_streak = 0
        else:
            home.venue_streak = home.venue_streak + 1 if home.venue_streak > 0 else 1
            away.venue_streak = away.venue_streak - 1 if away.venue_streak < 0 else -1

        home.points_for += home_score
        home.points_against += away_score
        away.points_for += away_score
        away.points_against += home_score

        venue = market_coords(home.franchise, season)
        if venue:
            home.last_market = venue
            away.last_market = venue
        venue_alt = market_altitude_m(home.franchise, season)
        if venue_alt is not None:
            home.last_altitude = venue_alt
            away.last_altitude = venue_alt

        # player availability / rotation bookkeeping (skipped when player rows
        # absent, leaving the previous rotation state in place)
        for team, box in ((home, home_box), (away, away_box)):
            if not isinstance(box, dict):
                continue
            rows = box.get("players")
            if not isinstance(rows, list) or not rows:
                continue
            prior_gp = max(team.games_played - 1, 0)
            metrics = _player_rotation_metrics(
                rows,
                dnp_ids=list(box.get("dnp_ids") or []),
                season_minutes=dict(team.season_minutes),
                games_played=prior_gp,
            )
            players: list[list[Any]] = []
            minutes_map: dict[str, float] = {}
            for row in rows:
                parsed = _parse_player_row(row)
                if parsed is None:
                    continue
                players.append(list(parsed))
                minutes_map[str(parsed[0])] = float(parsed[1])
            if not players:
                continue
            for store, alpha in (
                (team.player_min_fast, PLAYER_ALPHA_FAST),
                (team.player_min_slow, PLAYER_ALPHA_SLOW),
            ):
                for pid in list(store):
                    store[pid] *= 1.0 - alpha
                for pid, mins in minutes_map.items():
                    store[pid] = store.get(pid, 0.0) + alpha * mins
                for pid in [p for p, v in store.items() if v < 0.5]:
                    del store[pid]
            team.prev_player_ids = [
                pid
                for pid in (_player_id(row) for row in team.last_players)
                if pid
            ]
            team.last_players = players
            for parsed in players:
                pid, mins = str(parsed[0]), float(parsed[1])
                team.season_minutes[pid] = team.season_minutes.get(pid, 0.0) + mins
            team.apply_player_metrics(metrics)

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

        # head-to-head margin EWMA (signed, from each side's perspective)
        prior_h = home.h2h_margin.get(away.franchise, 0.0)
        home.h2h_margin[away.franchise] = prior_h + ALPHA_H2H_MARGIN * (signed_margin - prior_h)
        prior_a = away.h2h_margin.get(home.franchise, 0.0)
        away.h2h_margin[home.franchise] = prior_a + ALPHA_H2H_MARGIN * (-signed_margin - prior_a)

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
        league_ppg = payload.get("league_ppg")
        engine.league_ppg = float(league_ppg) if league_ppg is not None else LEAGUE_PPG
        league_pace = payload.get("league_pace")
        engine.league_pace = float(league_pace) if league_pace is not None else LEAGUE_PACE
        for key, team_payload in dict(payload.get("teams") or {}).items():
            engine.teams[str(key)] = TeamState.from_dict(team_payload)
        return engine
