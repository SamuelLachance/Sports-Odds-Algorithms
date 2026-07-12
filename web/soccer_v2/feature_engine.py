"""Walk-forward soccer feature engine shared by offline training and live.

Consumes football-data.co.uk match rows chronologically per COUNTRY POOL
(top flight + second division share one Elo/state pool so promoted teams
carry ratings). For each top-flight match it can emit a feature row using
only state accumulated strictly before that date, then folds the result in.

State is JSON-serializable: training snapshots end-of-season state and the
live path replays only the current season on top.
"""

from __future__ import annotations

import math
from datetime import date as date_cls
from typing import Any

LEAGUE_ELO = 1500.0
ELO_HOME_ADV = 60.0
ELO_K = 20.0
ELO_SEASON_CARRYOVER = 0.80

ALPHA_FAST = 0.16
ALPHA_SLOW = 0.045
ALPHA_SPLIT = 0.10
ALPHA_LEAGUE = 0.02

# league-average priors (goals per team per match)
PRIOR_GPG = 1.40
PRIOR_SHOTS = 12.5
PRIOR_SOT = 4.4
PRIOR_CORNERS = 5.2
PRIOR_HOME_EDGE = 0.38  # home minus away goals league-wide

# goals per shot-on-target (league level, era-stable ~0.30)
SOT_CONVERSION = 0.30

FEATURE_COLUMNS: tuple[str, ...] = (
    "elo_diff",
    "elo_abs_diff",
    "elo_sum_centered",
    "home_adv_ewma",
    "gf_fast_diff",
    "ga_fast_diff",
    "gf_slow_diff",
    "ga_slow_diff",
    "home_attack_split",
    "home_defence_split",
    "away_attack_split",
    "away_defence_split",
    "att_residual_diff",
    "def_residual_diff",
    "sot_for_diff",
    "sot_against_diff",
    "shots_for_diff",
    "shots_against_diff",
    "sot_xg_for_diff",
    "sot_xg_against_diff",
    "home_sot_split",
    "away_sot_split",
    "finish_luck_diff",
    "concede_luck_diff",
    "corners_for_diff",
    "points_fast_diff",
    "points_slow_diff",
    "win_rate_diff",
    "draw_rate_sum",
    "exp_total_goals",
    "exp_goal_gap",
    "dc_home_prob",
    "dc_draw_prob",
    "dc_away_prob",
    "home_rest_days",
    "away_rest_days",
    "rest_diff",
    "home_congestion",
    "away_congestion",
    "home_promoted",
    "away_promoted",
    "home_tier1_games",
    "away_tier1_games",
    "home_games_played",
    "away_games_played",
    "h2h_home_points",
    "h2h_goal_diff",
    "season_frac",
    "league_gpg",
    "league_epl",
    "league_bundesliga",
    "league_laliga",
    "league_seriea",
    "league_ligue1",
)

DC_RHO = -0.10
_MAX_GOALS = 8


def _dixon_coles_tau(
    i: int,
    j: int,
    exp_home: float,
    exp_away: float,
    rho: float,
) -> float:
    """Low-score dependency factor; clamped to ≥ 0 for large λ."""
    if i == 0 and j == 0:
        return max(0.0, 1.0 - exp_home * exp_away * rho)
    if i == 0 and j == 1:
        return max(0.0, 1.0 + exp_home * rho)
    if i == 1 and j == 0:
        return max(0.0, 1.0 + exp_away * rho)
    if i == 1 and j == 1:
        return max(0.0, 1.0 - rho)
    return 1.0


def poisson_1x2(
    exp_home: float,
    exp_away: float,
    *,
    rho: float = DC_RHO,
) -> tuple[float, float, float]:
    """Dixon-Coles-corrected Poisson 1X2 probabilities."""
    home_p = draw_p = away_p = 0.0
    ph = [
        math.exp(-exp_home) * exp_home**k / math.factorial(k)
        for k in range(_MAX_GOALS + 1)
    ]
    pa = [
        math.exp(-exp_away) * exp_away**k / math.factorial(k)
        for k in range(_MAX_GOALS + 1)
    ]
    for i in range(_MAX_GOALS + 1):
        for j in range(_MAX_GOALS + 1):
            p = ph[i] * pa[j] * _dixon_coles_tau(i, j, exp_home, exp_away, rho)
            if i > j:
                home_p += p
            elif i == j:
                draw_p += p
            else:
                away_p += p
    total = home_p + draw_p + away_p
    if total <= 0:
        return 0.44, 0.26, 0.30
    return home_p / total, draw_p / total, away_p / total


_LEAGUE_ONEHOT = {
    "epl": "league_epl",
    "bundesliga": "league_bundesliga",
    "laliga": "league_laliga",
    "seriea": "league_seriea",
    "ligue1": "league_ligue1",
}


def _parse_date(value: str) -> date_cls | None:
    try:
        return date_cls.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _ewma(current: float, value: float, alpha: float) -> float:
    return (1.0 - alpha) * current + alpha * value


class TeamState:
    """Per-team rolling state within one country pool."""

    __slots__ = (
        "elo",
        "gf_fast",
        "ga_fast",
        "gf_slow",
        "ga_slow",
        "gf_home",
        "ga_home",
        "gf_away",
        "ga_away",
        "sot_for",
        "sot_against",
        "shots_for",
        "shots_against",
        "corners_for",
        "corners_against",
        "finish_luck",
        "concede_luck",
        "points_fast",
        "points_slow",
        "win_rate",
        "draw_rate",
        "last_date",
        "recent_dates",
        "games_played",
        "tier1_games",
        "tier",
        "prev_tier",
        "season_seen",
    )

    def __init__(self) -> None:
        self.elo = LEAGUE_ELO
        self.gf_fast = self.gf_slow = PRIOR_GPG
        self.ga_fast = self.ga_slow = PRIOR_GPG
        self.gf_home = self.gf_away = PRIOR_GPG
        self.ga_home = self.ga_away = PRIOR_GPG
        self.sot_for = self.sot_against = PRIOR_SOT
        self.shots_for = self.shots_against = PRIOR_SHOTS
        self.corners_for = self.corners_against = PRIOR_CORNERS
        self.finish_luck = 0.0
        self.concede_luck = 0.0
        self.points_fast = self.points_slow = 1.35
        self.win_rate = 0.36
        self.draw_rate = 0.26
        self.last_date = ""
        self.recent_dates: list[str] = []
        self.games_played = 0
        self.tier1_games = 0
        self.tier = 1
        self.prev_tier = 1
        self.season_seen = -1

    def rest_days(self, on_date: str, default: float = 7.0) -> float:
        if not self.last_date:
            return default
        d0 = _parse_date(self.last_date)
        d1 = _parse_date(on_date)
        if d0 is None or d1 is None:
            return default
        # Floor at 0: same-day / inverted dates must not invent a rest day.
        return float(min(max((d1 - d0).days, 0), 21))

    def congestion(self, on_date: str) -> float:
        """Matches in the previous 21 days."""
        d1 = _parse_date(on_date)
        if d1 is None:
            return 0.0
        count = 0
        for raw in self.recent_dates:
            d0 = _parse_date(raw)
            if d0 is not None and 0 < (d1 - d0).days <= 21:
                count += 1
        return float(count)

    def to_dict(self) -> dict[str, Any]:
        return {slot: getattr(self, slot) for slot in self.__slots__}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TeamState":
        state = cls()
        for slot in cls.__slots__:
            if slot in payload:
                setattr(state, slot, payload[slot])
        return state


class SoccerFeatureEngine:
    """One country pool (e.g. England E0+E1) of walk-forward team state."""

    def __init__(self, country: str) -> None:
        self.country = country
        self.teams: dict[str, TeamState] = {}
        self.league_gpg = PRIOR_GPG  # per-team goals per match (tier 1)
        self.home_adv = PRIOR_HOME_EDGE  # home minus away goals (tier 1)
        self.h2h: dict[str, list[tuple[str, int, int]]] = {}
        self.current_season = -1

    # -- state access -----------------------------------------------------
    def team(self, name: str) -> TeamState:
        state = self.teams.get(name)
        if state is None:
            state = TeamState()
            self.teams[name] = state
        return state

    def _h2h_key(self, home: str, away: str) -> str:
        return f"{home}|{away}" if home < away else f"{away}|{home}"

    def _h2h_features(self, home: str, away: str) -> tuple[float, float]:
        """Home team's points-per-game and goal diff over last 5 meetings."""
        history = self.h2h.get(self._h2h_key(home, away)) or []
        if not history:
            return 1.35, 0.0
        points = 0.0
        gd = 0.0
        recent = history[-5:]
        for first_team, first_goals, second_goals in recent:
            # stored oriented to alphabetical first team
            if first_team == home:
                mine, theirs = first_goals, second_goals
            else:
                mine, theirs = second_goals, first_goals
            gd += mine - theirs
            points += 3.0 if mine > theirs else (1.0 if mine == theirs else 0.0)
        n = len(recent)
        return points / n, gd / n

    # -- season rollover ---------------------------------------------------
    def start_season(self, season: int) -> None:
        if season == self.current_season:
            return
        self.current_season = season
        for state in self.teams.values():
            state.elo = LEAGUE_ELO + ELO_SEASON_CARRYOVER * (state.elo - LEAGUE_ELO)
            state.prev_tier = state.tier
            state.season_seen = -1  # tier updated on first match of new season

    # -- features ----------------------------------------------------------
    def features_for_match(
        self,
        row: dict[str, Any],
        league: str,
    ) -> dict[str, float]:
        home = self.team(row["home"])
        away = self.team(row["away"])
        on_date = str(row["date"])

        home_rest = home.rest_days(on_date)
        away_rest = away.rest_days(on_date)

        # expected goals from split attack/defence vs league average
        lg = max(self.league_gpg, 0.6)
        exp_home = max(
            0.2,
            (home.gf_home / lg) * (away.ga_away / lg) * lg + self.home_adv / 2.0,
        )
        exp_away = max(
            0.15,
            (away.gf_away / lg) * (home.ga_home / lg) * lg - self.home_adv / 2.0,
        )

        h2h_points, h2h_gd = self._h2h_features(row["home"], row["away"])

        month = _parse_date(on_date)
        season_frac = 0.0
        if month is not None:
            # season runs Aug..May
            offset = (month.month - 8) % 12
            season_frac = min(offset / 10.0, 1.0)

        sot_xg_home = home.sot_for * SOT_CONVERSION
        sot_xg_away = away.sot_for * SOT_CONVERSION
        sot_xga_home = home.sot_against * SOT_CONVERSION
        sot_xga_away = away.sot_against * SOT_CONVERSION

        dc_home, dc_draw, dc_away = poisson_1x2(exp_home, exp_away)

        features: dict[str, float] = {
            "elo_diff": (home.elo + ELO_HOME_ADV) - away.elo,
            "elo_abs_diff": abs(home.elo - away.elo),
            "elo_sum_centered": (home.elo + away.elo) / 2.0 - LEAGUE_ELO,
            "home_adv_ewma": self.home_adv,
            "gf_fast_diff": home.gf_fast - away.gf_fast,
            "ga_fast_diff": home.ga_fast - away.ga_fast,
            "gf_slow_diff": home.gf_slow - away.gf_slow,
            "ga_slow_diff": home.ga_slow - away.ga_slow,
            "home_attack_split": home.gf_home / lg,
            "home_defence_split": home.ga_home / lg,
            "away_attack_split": away.gf_away / lg,
            "away_defence_split": away.ga_away / lg,
            "att_residual_diff": (sot_xg_home - home.gf_slow)
            - (sot_xg_away - away.gf_slow),
            "def_residual_diff": (sot_xga_home - home.ga_slow)
            - (sot_xga_away - away.ga_slow),
            "sot_for_diff": home.sot_for - away.sot_for,
            "sot_against_diff": home.sot_against - away.sot_against,
            "shots_for_diff": home.shots_for - away.shots_for,
            "shots_against_diff": home.shots_against - away.shots_against,
            "sot_xg_for_diff": sot_xg_home - sot_xg_away,
            "sot_xg_against_diff": sot_xga_home - sot_xga_away,
            "home_sot_split": home.sot_for - home.sot_against,
            "away_sot_split": away.sot_for - away.sot_against,
            "finish_luck_diff": home.finish_luck - away.finish_luck,
            "concede_luck_diff": home.concede_luck - away.concede_luck,
            "corners_for_diff": home.corners_for - away.corners_for,
            "points_fast_diff": home.points_fast - away.points_fast,
            "points_slow_diff": home.points_slow - away.points_slow,
            "win_rate_diff": home.win_rate - away.win_rate,
            "draw_rate_sum": home.draw_rate + away.draw_rate,
            "exp_total_goals": exp_home + exp_away,
            "exp_goal_gap": exp_home - exp_away,
            "dc_home_prob": dc_home,
            "dc_draw_prob": dc_draw,
            "dc_away_prob": dc_away,
            "home_rest_days": home_rest,
            "away_rest_days": away_rest,
            "rest_diff": home_rest - away_rest,
            "home_congestion": home.congestion(on_date),
            "away_congestion": away.congestion(on_date),
            "home_promoted": 1.0 if home.prev_tier > 1 else 0.0,
            "away_promoted": 1.0 if away.prev_tier > 1 else 0.0,
            "home_tier1_games": float(min(home.tier1_games, 400)),
            "away_tier1_games": float(min(away.tier1_games, 400)),
            "home_games_played": float(min(home.games_played, 600)),
            "away_games_played": float(min(away.games_played, 600)),
            "h2h_home_points": h2h_points,
            "h2h_goal_diff": h2h_gd,
            "season_frac": season_frac,
            "league_gpg": self.league_gpg,
        }
        for lg_key, column in _LEAGUE_ONEHOT.items():
            features[column] = 1.0 if league == lg_key else 0.0
        return features

    # -- update ------------------------------------------------------------
    def update_after_match(self, row: dict[str, Any], *, tier: int) -> None:
        home = self.team(row["home"])
        away = self.team(row["away"])
        hg = int(row["home_goals"])
        ag = int(row["away_goals"])
        on_date = str(row["date"])
        season = int(row.get("season") or 0)

        for team in (home, away):
            if team.season_seen != season:
                # keep last season's tier for the promoted/relegated flag;
                # brand-new teams start in the tier they debut in
                team.prev_tier = team.tier if team.games_played > 0 else tier
                team.tier = tier
                team.season_seen = season

        # Elo with goal-diff multiplier (FiveThirtyEight-style, signed
        # winner-advantage autocorrelation guard)
        home_edge = (home.elo + ELO_HOME_ADV) - away.elo
        exp_home = 1.0 / (1.0 + 10.0 ** (-home_edge / 400.0))
        result = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        margin = abs(hg - ag)
        if result == 1.0:
            winner_edge = home_edge
        elif result == 0.0:
            winner_edge = -home_edge
        else:
            winner_edge = 0.0
        multiplier = math.log(max(margin, 1) + 1.0) * (
            2.2 / (0.001 * max(winner_edge, 0.0) + 2.2)
        )
        delta = ELO_K * multiplier * (result - exp_home)
        home.elo += delta
        away.elo -= delta

        # attack/defence + shot EWMAs
        for team, gf, ga, is_home in ((home, hg, ag, True), (away, ag, hg, False)):
            team.gf_fast = _ewma(team.gf_fast, gf, ALPHA_FAST)
            team.ga_fast = _ewma(team.ga_fast, ga, ALPHA_FAST)
            team.gf_slow = _ewma(team.gf_slow, gf, ALPHA_SLOW)
            team.ga_slow = _ewma(team.ga_slow, ga, ALPHA_SLOW)
            if is_home:
                team.gf_home = _ewma(team.gf_home, gf, ALPHA_SPLIT)
                team.ga_home = _ewma(team.ga_home, ga, ALPHA_SPLIT)
            else:
                team.gf_away = _ewma(team.gf_away, gf, ALPHA_SPLIT)
                team.ga_away = _ewma(team.ga_away, ga, ALPHA_SPLIT)
            points = 3.0 if (gf > ga) else (1.0 if gf == ga else 0.0)
            team.points_fast = _ewma(team.points_fast, points, ALPHA_FAST)
            team.points_slow = _ewma(team.points_slow, points, ALPHA_SLOW)
            team.win_rate = _ewma(team.win_rate, 1.0 if gf > ga else 0.0, ALPHA_SLOW)
            team.draw_rate = _ewma(team.draw_rate, 1.0 if gf == ga else 0.0, ALPHA_SLOW)
            team.games_played += 1
            if tier == 1:
                team.tier1_games += 1
            team.last_date = on_date
            team.recent_dates.append(on_date)
            if len(team.recent_dates) > 14:
                team.recent_dates = team.recent_dates[-14:]

        hs, as_ = row.get("home_shots"), row.get("away_shots")
        hst, ast = row.get("home_sot"), row.get("away_sot")
        hc, ac = row.get("home_corners"), row.get("away_corners")
        if hs is not None and as_ is not None:
            home.shots_for = _ewma(home.shots_for, float(hs), ALPHA_FAST)
            home.shots_against = _ewma(home.shots_against, float(as_), ALPHA_FAST)
            away.shots_for = _ewma(away.shots_for, float(as_), ALPHA_FAST)
            away.shots_against = _ewma(away.shots_against, float(hs), ALPHA_FAST)
        if hst is not None and ast is not None:
            home.sot_for = _ewma(home.sot_for, float(hst), ALPHA_FAST)
            home.sot_against = _ewma(home.sot_against, float(ast), ALPHA_FAST)
            away.sot_for = _ewma(away.sot_for, float(ast), ALPHA_FAST)
            away.sot_against = _ewma(away.sot_against, float(hst), ALPHA_FAST)
            home.finish_luck = _ewma(
                home.finish_luck, hg - SOT_CONVERSION * float(hst), ALPHA_SLOW
            )
            away.finish_luck = _ewma(
                away.finish_luck, ag - SOT_CONVERSION * float(ast), ALPHA_SLOW
            )
            home.concede_luck = _ewma(
                home.concede_luck, ag - SOT_CONVERSION * float(ast), ALPHA_SLOW
            )
            away.concede_luck = _ewma(
                away.concede_luck, hg - SOT_CONVERSION * float(hst), ALPHA_SLOW
            )
        if hc is not None and ac is not None:
            home.corners_for = _ewma(home.corners_for, float(hc), ALPHA_FAST)
            home.corners_against = _ewma(home.corners_against, float(ac), ALPHA_FAST)
            away.corners_for = _ewma(away.corners_for, float(ac), ALPHA_FAST)
            away.corners_against = _ewma(away.corners_against, float(hc), ALPHA_FAST)

        # league environment (tier-1 only)
        if tier == 1:
            self.league_gpg = _ewma(self.league_gpg, (hg + ag) / 2.0, ALPHA_LEAGUE)
            self.home_adv = _ewma(self.home_adv, float(hg - ag), ALPHA_LEAGUE)

        # head-to-head log (top flight only, oriented alphabetically)
        if tier == 1:
            key = self._h2h_key(row["home"], row["away"])
            first = row["home"] if row["home"] < row["away"] else row["away"]
            log = self.h2h.setdefault(key, [])
            if first == row["home"]:
                log.append((first, hg, ag))
            else:
                log.append((first, ag, hg))
            if len(log) > 10:
                del log[:-10]

    # -- serialization -------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "country": self.country,
            "current_season": self.current_season,
            "league_gpg": self.league_gpg,
            "home_adv": self.home_adv,
            "teams": {name: state.to_dict() for name, state in self.teams.items()},
            "h2h": {key: [list(item) for item in log] for key, log in self.h2h.items()},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SoccerFeatureEngine":
        engine = cls(str(payload.get("country") or ""))
        engine.current_season = int(payload.get("current_season") or -1)
        league_gpg = payload.get("league_gpg")
        engine.league_gpg = float(league_gpg) if league_gpg is not None else PRIOR_GPG
        home_adv = payload.get("home_adv")
        engine.home_adv = float(home_adv) if home_adv is not None else PRIOR_HOME_EDGE
        engine.teams = {
            name: TeamState.from_dict(state)
            for name, state in (payload.get("teams") or {}).items()
        }
        engine.h2h = {
            key: [(str(a), int(b), int(c)) for a, b, c in log]
            for key, log in (payload.get("h2h") or {}).items()
        }
        return engine
