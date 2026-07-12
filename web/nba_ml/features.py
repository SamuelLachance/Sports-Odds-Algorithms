"""Leak-free, point-in-time NBA feature engineering.

Every feature for a game is computed using only information available *before*
tip-off. State (Elo, rolling form, last-game date/venue) is updated only after a
game's features and targets are emitted, so there is no lookahead.

The same builder is used for training (iterating the historical odds table) and
for live inference (``FeatureState.features_for`` on the current state).
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from web.nba_ml.config import (
    ELO_BASE,
    ELO_HOME_ADV,
    ELO_K,
    ELO_SEASON_REGRESS,
    FEATURE_COLUMNS,
    FORM_WINDOWS,
)

# Approximate arena coordinates (lat, lon) and IANA-style UTC offset (hours,
# standard time) per ESPN team abbreviation. Used for travel + time-zone features.
TEAM_GEO: dict[str, tuple[float, float, float]] = {
    "atl": (33.757, -84.396, -5),
    "bos": (42.366, -71.062, -5),
    "bkn": (40.683, -73.975, -5),
    "cha": (35.225, -80.839, -5),
    "chi": (41.881, -87.674, -6),
    "cle": (41.496, -81.688, -5),
    "dal": (32.790, -96.810, -6),
    "den": (39.749, -105.008, -7),
    "det": (42.341, -83.055, -5),
    "gs": (37.768, -122.388, -8),
    "hou": (29.751, -95.362, -6),
    "ind": (39.764, -86.155, -5),
    "lac": (34.043, -118.267, -8),
    "lal": (34.043, -118.267, -8),
    "mem": (35.138, -90.051, -6),
    "mia": (25.781, -80.187, -5),
    "mil": (43.045, -87.917, -6),
    "min": (44.979, -93.276, -6),
    "no": (29.949, -90.082, -6),
    "ny": (40.751, -73.993, -5),
    "okc": (35.463, -97.515, -6),
    "orl": (28.539, -81.384, -5),
    "phi": (39.901, -75.172, -5),
    "phx": (33.446, -112.071, -7),
    "por": (45.532, -122.667, -8),
    "sac": (38.580, -121.500, -8),
    "sa": (29.427, -98.437, -6),
    "tor": (43.643, -79.379, -5),
    "uta": (40.768, -111.901, -7),
    "wsh": (38.898, -77.021, -5),
}

# Common alias fixups so odds/game keys resolve to the geo table.
GEO_ALIASES = {
    "gsw": "gs",
    "nop": "no",
    "nyk": "ny",
    "sas": "sa",
    "pho": "phx",
    "was": "wsh",
    "uth": "uta",
    "utah": "uta",  # ESPN / closing-odds key (espn_client maps UTAH → utah)
    "bro": "bkn",
    "njn": "bkn",
    "cho": "cha",
}


def _geo(team: str) -> tuple[float, float, float] | None:
    key = team.lower()
    key = GEO_ALIASES.get(key, key)
    return TEAM_GEO.get(key)


def _haversine_km(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(h))


def american_to_prob(odds: float | None) -> float | None:
    if odds is None:
        return None
    # bool is a subclass of int; False→0.0 must not invent EVEN (+100 → 50%).
    if isinstance(odds, bool):
        return None
    try:
        odds = float(odds)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(odds):
        return None
    # ESPN EVEN sometimes arrives as numeric 0 — treat as +100.
    if odds == 0:
        odds = 100.0
    # Invalid American magnitudes must not pollute market features.
    if abs(odds) < 100:
        return None
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return -odds / (-odds + 100.0)


def devig_home_prob(home_ml: float | None, away_ml: float | None) -> float | None:
    home_p = american_to_prob(home_ml)
    away_p = american_to_prob(away_ml)
    if home_p is None or away_p is None:
        return None
    total = home_p + away_p
    if total <= 0:
        return None
    return home_p / total


def coerce_market_spread(home_spread: Any) -> float | None:
    """Finite NBA handicap, else None (reject bool pick'em / juice-as-spread)."""
    if home_spread is None or isinstance(home_spread, bool):
        return None
    try:
        value = float(home_spread)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or abs(value) >= 100.0:
        return None
    return value


def spread_to_home_prob(home_spread: float | None) -> float | None:
    """Rough NBA spread -> home win prob (logistic, ~0.28 pts of sigma scale)."""
    coerced = coerce_market_spread(home_spread)
    if coerced is None:
        return None
    return 1.0 / (1.0 + math.exp(coerced / 4.05))


def _season_of(game_day: date) -> int:
    return game_day.year if game_day.month >= 9 else game_day.year - 1


@dataclass
class _TeamState:
    elo: float = ELO_BASE
    games: int = 0
    pts_for: float = 0.0
    pts_against: float = 0.0
    season_games: int = 0
    margins: deque = field(default_factory=lambda: deque(maxlen=max(FORM_WINDOWS)))
    results: deque = field(default_factory=lambda: deque(maxlen=max(FORM_WINDOWS)))
    last_date: date | None = None
    prev_dates: deque = field(default_factory=lambda: deque(maxlen=4))
    last_venue: str | None = None


@dataclass
class FeatureState:
    """Mutable, leak-free NBA state that produces feature vectors on demand."""

    teams: dict[str, _TeamState] = field(default_factory=lambda: defaultdict(_TeamState))
    current_season: int | None = None

    def _maybe_new_season(self, game_day: date) -> None:
        season = _season_of(game_day)
        if self.current_season is None:
            self.current_season = season
            return
        if season != self.current_season:
            for state in self.teams.values():
                state.elo = ELO_BASE + (state.elo - ELO_BASE) * (1.0 - ELO_SEASON_REGRESS)
                state.season_games = 0
                state.pts_for = 0.0
                state.pts_against = 0.0
                state.margins.clear()
                state.results.clear()
            self.current_season = season

    def _rest_features(self, state: _TeamState, game_day: date) -> tuple[float, int, int]:
        if state.last_date is None:
            return 3.0, 0, 0
        delta = (game_day - state.last_date).days
        # Inverted / future last_date must not invent B2B (rest=0 → b2b=1).
        if delta < 0:
            return 3.0, 0, 0
        rest = min(delta, 10)
        b2b = 1 if rest <= 1 else 0
        # 3-in-4: current game is the 3rd within a 4-calendar-day window, i.e.
        # at least two prior games fell within the last 3 days.
        recent_dates = list(state.prev_dates) + [state.last_date]
        within = [d for d in recent_dates if 0 <= (game_day - d).days <= 3]
        three_in_four = 1 if len(within) >= 2 else 0
        return float(rest), b2b, three_in_four

    def _form(self, state: _TeamState, window: int) -> float:
        if not state.margins:
            return 0.0
        vals = list(state.margins)[-window:]
        return sum(vals) / len(vals)

    def _winrate(self, state: _TeamState, window: int) -> float:
        if not state.results:
            return 0.5
        vals = list(state.results)[-window:]
        return sum(vals) / len(vals)

    def _off_def(self, state: _TeamState) -> tuple[float, float, float]:
        if state.season_games == 0:
            return 112.0, 112.0, 224.0
        off = state.pts_for / state.season_games
        deff = state.pts_against / state.season_games
        pace = off + deff
        return off, deff, pace

    def features_for(
        self,
        home: str,
        away: str,
        game_day: date,
        *,
        market_spread: float | None = None,
        market_home_ml: float | None = None,
        market_away_ml: float | None = None,
    ) -> dict[str, float]:
        self._maybe_new_season(game_day)
        home_state = self.teams[home]
        away_state = self.teams[away]

        elo_diff = (home_state.elo + ELO_HOME_ADV) - away_state.elo
        elo_win_prob = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))

        h_off, h_def, h_pace = self._off_def(home_state)
        a_off, a_def, a_pace = self._off_def(away_state)

        h_rest, h_b2b, h_three = self._rest_features(home_state, game_day)
        a_rest, a_b2b, a_three = self._rest_features(away_state, game_day)

        home_geo = _geo(home)
        travel_home = 0.0
        travel_away = 0.0
        tz_home = 0.0
        tz_away = 0.0
        if home_geo:
            if home_state.last_venue and _geo(home_state.last_venue):
                travel_home = _haversine_km(_geo(home_state.last_venue), home_geo)
                tz_home = home_geo[2] - _geo(home_state.last_venue)[2]
            if away_state.last_venue and _geo(away_state.last_venue):
                travel_away = _haversine_km(_geo(away_state.last_venue), home_geo)
                tz_away = home_geo[2] - _geo(away_state.last_venue)[2]
            elif away_geo := _geo(away):
                travel_away = _haversine_km(away_geo, home_geo)
                tz_away = home_geo[2] - away_geo[2]

        devig = devig_home_prob(market_home_ml, market_away_ml)
        if devig is None:
            devig = spread_to_home_prob(market_spread)

        season_progress = min(
            (home_state.season_games + away_state.season_games) / 164.0, 1.0
        )
        spread_feat = coerce_market_spread(market_spread)

        return {
            "elo_diff": elo_diff,
            "elo_win_prob": elo_win_prob,
            "off_eff_diff": h_off - a_off,
            "def_eff_diff": h_def - a_def,
            "net_eff_diff": (h_off - h_def) - (a_off - a_def),
            "pace_sum": (h_pace + a_pace) / 2.0,
            "form5_margin_diff": self._form(home_state, 5) - self._form(away_state, 5),
            "form10_margin_diff": self._form(home_state, 10) - self._form(away_state, 10),
            "form20_margin_diff": self._form(home_state, 20) - self._form(away_state, 20),
            "form10_winrate_diff": self._winrate(home_state, 10) - self._winrate(away_state, 10),
            "rest_diff": h_rest - a_rest,
            "home_b2b": float(h_b2b),
            "away_b2b": float(a_b2b),
            "home_three_in_four": float(h_three),
            "away_three_in_four": float(a_three),
            "travel_home_km": travel_home,
            "travel_away_km": travel_away,
            "tz_shift_diff": tz_home - tz_away,
            "season_progress": season_progress,
            "market_spread": (
                float(spread_feat) if spread_feat is not None else float("nan")
            ),
            "market_devig_home_prob": float(devig) if devig is not None else float("nan"),
        }

    def update(self, home: str, away: str, game_day: date, home_score: int, away_score: int) -> None:
        # Live replay (_replayed_state) only calls update(); without a season
        # reset here, Elo/form/season_games skew vs training features_for path.
        self._maybe_new_season(game_day)
        home_state = self.teams[home]
        away_state = self.teams[away]

        margin = home_score - away_score
        home_win = 1.0 if margin > 0 else 0.0

        # Elo update with margin-of-victory multiplier.
        elo_diff = (home_state.elo + ELO_HOME_ADV) - away_state.elo
        expected_home = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
        mov_mult = math.log(abs(margin) + 1.0) * (
            2.2 / (abs(elo_diff) * 0.001 + 2.2)
        )
        delta = ELO_K * mov_mult * (home_win - expected_home)
        home_state.elo += delta
        away_state.elo -= delta

        for state, pf, pa, m, res, venue in (
            (home_state, home_score, away_score, margin, home_win, home),
            (away_state, away_score, home_score, -margin, 1.0 - home_win, home),
        ):
            state.games += 1
            state.season_games += 1
            state.pts_for += pf
            state.pts_against += pa
            state.margins.append(m)
            state.results.append(res)
            if state.last_date is not None:
                state.prev_dates.append(state.last_date)
            state.last_date = game_day
            state.last_venue = venue


def parse_game_date(raw: str) -> date:
    return datetime.strptime(str(raw)[:10], "%Y-%m-%d").date()


def feature_columns() -> tuple[str, ...]:
    return FEATURE_COLUMNS
