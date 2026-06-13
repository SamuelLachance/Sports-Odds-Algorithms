"""World Cup 2026 forward simulation — unified 3-layer model + Poisson scores + Monte Carlo."""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from dataclasses import replace
from random import Random
from typing import Any, Callable

from web.bet_advisor import soccer_threeway_probs
from web.espn_client import ScheduledGame
from web.live_data import resolve_team
from web.soccer_pred_model import _dc_score_matrix
from web.wc_groups import is_placeholder_team, normalize_team_name
from web.world_cup_service import (
    ROUND_ORDER,
    _compute_group_standings,
    _compute_third_place_ranking,
)

ROUND_WINNER_RE = re.compile(
    r"^(Round of 32|Round of 16|Quarterfinal|Semifinal)\s+(\d+)\s+Winner$",
    re.IGNORECASE,
)
GROUP_WINNER_RE = re.compile(r"^Group\s+([A-L])\s+Winner$", re.IGNORECASE)
GROUP_2ND_RE = re.compile(r"^Group\s+([A-L])\s+2nd\s+Place$", re.IGNORECASE)
THIRD_PLACE_RE = re.compile(r"^Third Place Group\s+(.+)$", re.IGNORECASE)

ROUND_SLUG_BY_LABEL = {
    "round of 32": "round-of-32",
    "round of 16": "round-of-16",
    "quarterfinal": "quarterfinals",
    "semifinal": "semifinals",
}

def _default_mc_iterations() -> int:
    if "WC_MC_ITERATIONS" in os.environ:
        return int(os.environ["WC_MC_ITERATIONS"])
    if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
        return 25
    return 500


DEFAULT_MC_ITERATIONS = _default_mc_iterations()
DEFAULT_MC_SEED = int(os.environ.get("WC_MC_SEED", "42"))
WC_AVG_TOTAL_GOALS = 2.65
THREE_LAYER_WEIGHT = 1.0 / 3.0

# Knockout bracket feed map (child slot -> parent round/slots)
BRACKET_FEEDS: dict[str, dict[int, tuple[tuple[str, int], tuple[str, int]]]] = {
    "round-of-16": {
        1: (("round-of-32", 3), ("round-of-32", 1)),
        2: (("round-of-32", 5), ("round-of-32", 2)),
        3: (("round-of-32", 6), ("round-of-32", 4)),
        4: (("round-of-32", 8), ("round-of-32", 7)),
        5: (("round-of-32", 12), ("round-of-32", 11)),
        6: (("round-of-32", 10), ("round-of-32", 9)),
        7: (("round-of-32", 16), ("round-of-32", 14)),
        8: (("round-of-32", 15), ("round-of-32", 13)),
    },
    "quarterfinals": {
        1: (("round-of-16", 2), ("round-of-16", 1)),
        2: (("round-of-16", 4), ("round-of-16", 3)),
        3: (("round-of-16", 6), ("round-of-16", 5)),
        4: (("round-of-16", 8), ("round-of-16", 7)),
    },
    "semifinals": {
        1: (("quarterfinals", 2), ("quarterfinals", 1)),
        2: (("quarterfinals", 4), ("quarterfinals", 3)),
    },
    "final": {
        1: (("semifinals", 2), ("semifinals", 1)),
    },
}


def _layer_threeway_probs(model: dict[str, Any], league: str = "worldcup") -> tuple[float, float, float] | None:
    """Return home/draw/away % for one model layer payload."""
    if not model:
        return None
    if all(k in model for k in ("home_win_probability", "draw_probability", "away_win_probability")):
        return (
            float(model["home_win_probability"]),
            float(model["draw_probability"]),
            float(model["away_win_probability"]),
        )
    total = model.get("total_score")
    if total is not None:
        return soccer_threeway_probs(float(total), league)
    home_win = model.get("home_win_probability")
    if home_win is not None:
        h, d, a = soccer_threeway_probs(float(home_win), league)
        return h, d, a
    return None


def extract_unified_probs(
    prediction: dict[str, Any] | None,
    *,
    league: str = "worldcup",
) -> dict[str, float]:
    """
    Unified 3-layer 1X2 probabilities.

    Prefer explicit blended threeway on the unified model; otherwise blend legacy,
    power, and soccer_pred layer probabilities with equal weights.
    """
    if not prediction:
        return {"home": 33.33, "draw": 33.33, "away": 33.34}

    model = prediction.get("model") or prediction
    if model.get("threeway"):
        return {
            "home": float(model.get("home_win_probability", 33.33)),
            "draw": float(model.get("draw_probability", 33.33)),
            "away": float(model.get("away_win_probability", 33.34)),
        }

    layers: list[tuple[float, float, float]] = []
    legacy_tw = _layer_threeway_probs(model.get("legacy_threeway") or {}, league)
    if legacy_tw:
        layers.append(legacy_tw)
    elif model.get("legacy"):
        legacy_probs = _layer_threeway_probs(model["legacy"], league)
        if legacy_probs:
            layers.append(legacy_probs)

    power_tw = _layer_threeway_probs(model.get("power_threeway") or {}, league)
    if power_tw:
        layers.append(power_tw)
    elif model.get("power"):
        power_probs = _layer_threeway_probs(model["power"], league)
        if power_probs:
            layers.append(power_probs)

    soccer = model.get("soccer_pred") or {}
    soccer_probs = _layer_threeway_probs(soccer, league)
    if soccer_probs:
        layers.append(soccer_probs)

    if len(layers) >= 2:
        home = sum(layer[0] for layer in layers) / len(layers)
        draw = sum(layer[1] for layer in layers) / len(layers)
        away = sum(layer[2] for layer in layers) / len(layers)
        total = home + draw + away
        if total > 0:
            scale = 100.0 / total
            return {
                "home": round(home * scale, 2),
                "draw": round(draw * scale, 2),
                "away": round(away * scale, 2),
            }

    home_p = float(model.get("blended_home_win_probability") or model.get("win_probability") or 50.0)
    h, d, a = soccer_threeway_probs(home_p, league)
    return {"home": h, "draw": d, "away": a}


def _threeway_to_lambdas(
    home_pct: float,
    draw_pct: float,
    away_pct: float,
    *,
    total_goals: float = WC_AVG_TOTAL_GOALS,
) -> tuple[float, float]:
    """Implied Poisson rates from 1X2 probabilities."""
    h = max(home_pct, 0.1) / 100.0
    d = max(draw_pct, 0.1) / 100.0
    a = max(away_pct, 0.1) / 100.0
    non_draw = max(h + a, 0.05)
    home_share = h / non_draw
    expected_total = total_goals * (1.0 - d * 0.12)
    lam_h = max(0.35, expected_total * (0.42 + home_share * 0.58))
    lam_a = max(0.35, expected_total - lam_h * (1.0 - d * 0.5))
    return lam_h, lam_a


def extract_unified_lambdas(
    prediction: dict[str, Any] | None,
    *,
    league: str = "worldcup",
) -> tuple[float, float]:
    """Blend expected goals from legacy, power, and soccer_pred layers (⅓ each)."""
    if not prediction:
        return WC_AVG_TOTAL_GOALS * 0.52, WC_AVG_TOTAL_GOALS * 0.48

    model = prediction.get("model") or prediction
    lambdas: list[tuple[float, float]] = []

    soccer = model.get("soccer_pred") or {}
    if soccer.get("expected_home_goals") is not None and soccer.get("expected_away_goals") is not None:
        lambdas.append(
            (float(soccer["expected_home_goals"]), float(soccer["expected_away_goals"]))
        )

    legacy_probs = _layer_threeway_probs(model.get("legacy_threeway") or model.get("legacy") or {}, league)
    if legacy_probs:
        lambdas.append(_threeway_to_lambdas(*legacy_probs))

    power_probs = _layer_threeway_probs(model.get("power_threeway") or model.get("power") or {}, league)
    if power_probs:
        lambdas.append(_threeway_to_lambdas(*power_probs))

    if not lambdas and model.get("threeway"):
        probs = extract_unified_probs(prediction, league=league)
        lambdas.append(_threeway_to_lambdas(probs["home"], probs["draw"], probs["away"]))

    if not lambdas:
        probs = extract_unified_probs(prediction, league=league)
        return _threeway_to_lambdas(probs["home"], probs["draw"], probs["away"])

    lam_h = sum(item[0] for item in lambdas) / len(lambdas)
    lam_a = sum(item[1] for item in lambdas) / len(lambdas)
    return round(lam_h, 3), round(lam_a, 3)


def extract_model_probs(prediction: dict[str, Any] | None) -> dict[str, float]:
    """Backward-compatible alias for unified probabilities."""
    return extract_unified_probs(prediction)


def pick_outcome(probs: dict[str, float], *, allow_draw: bool = True) -> str:
    """Deterministic outcome: highest unified probability."""
    candidates: list[tuple[str, float]] = [
        ("home", probs.get("home", 0.0)),
        ("draw", probs.get("draw", 0.0)),
        ("away", probs.get("away", 0.0)),
    ]
    if not allow_draw:
        candidates = [c for c in candidates if c[0] != "draw"]
    candidates.sort(key=lambda item: (-item[1], item[0]))
    return candidates[0][0]


def sample_outcome(
    probs: dict[str, float],
    rng: Random,
    *,
    allow_draw: bool = True,
) -> str:
    """Stochastic 1X2 outcome from unified probabilities."""
    weights = {
        "home": probs.get("home", 0.0),
        "draw": probs.get("draw", 0.0) if allow_draw else 0.0,
        "away": probs.get("away", 0.0),
    }
    total = sum(weights.values())
    if total <= 0:
        return pick_outcome(probs, allow_draw=allow_draw)
    roll = rng.random() * total
    cumulative = 0.0
    for side, weight in weights.items():
        cumulative += weight
        if roll <= cumulative:
            return side
    return "home"


def pick_knockout_outcome(probs: dict[str, float]) -> str:
    """Knockout deterministic — no draws."""
    return pick_outcome(probs, allow_draw=False)


def sample_knockout_outcome(probs: dict[str, float], rng: Random) -> str:
    return sample_outcome(probs, rng, allow_draw=False)


def _sample_from_score_matrix(
    matrix: list[list[float]],
    rng: Random,
) -> tuple[int, int]:
    flat: list[tuple[int, int, float]] = []
    for i, row in enumerate(matrix):
        for j, weight in enumerate(row):
            if weight > 0:
                flat.append((i, j, weight))
    total = sum(item[2] for item in flat)
    if total <= 0:
        return 1, 0
    roll = rng.random() * total
    cumulative = 0.0
    for home_goals, away_goals, weight in flat:
        cumulative += weight
        if roll <= cumulative:
            return home_goals, away_goals
    return flat[-1][0], flat[-1][1]


def _poisson_sample(lam: float, rng: Random) -> int:
    """Knuth algorithm for Poisson draws."""
    lam = max(lam, 1e-9)
    limit = math.exp(-lam)
    k = 0
    product = 1.0
    while product > limit:
        k += 1
        product *= rng.random()
    return k - 1


def sample_poisson_score(
    lam_h: float,
    lam_a: float,
    rng: Random,
    *,
    allow_draw: bool,
    knockout: bool,
) -> tuple[int, int, str]:
    """
    Sample integer score from Dixon–Coles Poisson matrix built on unified xG.

    Knockout ties go to extra time (Poisson bump) then penalties by xG if still tied.
    """
    matrix = _dc_score_matrix(max(lam_h, 0.2), max(lam_a, 0.2))
    home_score, away_score = _sample_from_score_matrix(matrix, rng)

    if knockout and home_score == away_score:
        for _ in range(4):
            home_score += _poisson_sample(max(lam_h * 0.22, 0.15), rng)
            away_score += _poisson_sample(max(lam_a * 0.22, 0.15), rng)
            if home_score != away_score:
                break
        if home_score == away_score:
            if lam_h > lam_a:
                home_score += 1
            elif lam_a > lam_h:
                away_score += 1
            else:
                home_score += 1 if rng.random() >= 0.5 else 0
                away_score += 0 if home_score > away_score else 1

    if home_score > away_score:
        outcome = "home"
    elif away_score > home_score:
        outcome = "away"
    else:
        outcome = "draw" if allow_draw else ("home" if lam_h >= lam_a else "away")
        if not allow_draw and outcome == "draw":
            home_score += 1

    return home_score, away_score, outcome


def derive_scores(
    outcome: str,
    prediction: dict[str, Any] | None,
    *,
    allow_draw: bool,
    rng: Random | None = None,
    knockout: bool = False,
) -> tuple[int, int]:
    """Poisson score from unified expected goals (deterministic if rng is None)."""
    lam_h, lam_a = extract_unified_lambdas(prediction)
    if rng is not None:
        home_score, away_score, sampled = sample_poisson_score(
            lam_h, lam_a, rng, allow_draw=allow_draw, knockout=knockout
        )
        if sampled == outcome or (outcome == "draw" and sampled == "draw"):
            return home_score, away_score
        # Align score with chosen outcome while keeping Poisson flavour
        if outcome == "home":
            away_score = min(away_score, home_score - 1) if home_score > 0 else 0
            if home_score <= away_score:
                home_score = max(away_score + 1, int(round(lam_h)))
        elif outcome == "away":
            home_score = min(home_score, away_score - 1) if away_score > 0 else 0
            if away_score <= home_score:
                away_score = max(home_score + 1, int(round(lam_a)))
        else:
            high = max(int(round(lam_h)), int(round(lam_a)), 1)
            home_score = away_score = high
        return home_score, away_score

    # Deterministic fallback: Poisson mode-ish integers from lambdas
    home_score = max(0, int(round(lam_h)))
    away_score = max(0, int(round(lam_a)))
    if outcome == "home" and home_score <= away_score:
        home_score = away_score + 1
    elif outcome == "away" and away_score <= home_score:
        away_score = home_score + 1
    elif outcome == "draw":
        high = max(home_score, away_score, 1)
        home_score = away_score = high
    return home_score, away_score


def _team_abbr(name: str) -> str:
    canonical = normalize_team_name(name)
    resolved = resolve_team("worldcup", canonical[:3].upper(), canonical)
    if resolved:
        return resolved[0].upper()
    slug = canonical.lower().replace(" ", "-").replace("'", "")
    resolved = resolve_team("worldcup", slug[:3].upper(), canonical)
    if resolved:
        return resolved[0].upper()
    return slug[:3].upper() or "TBD"


def build_scheduled_game(
    template: ScheduledGame | None,
    away_name: str,
    home_name: str,
) -> ScheduledGame | None:
    if not template:
        return None
    return replace(
        template,
        away_name=away_name,
        home_name=home_name,
        away_abbr=_team_abbr(away_name),
        home_abbr=_team_abbr(home_name),
    )


def resolve_placeholder(
    name: str,
    *,
    group_standings: dict[str, list[dict[str, Any]]],
    third_place_ranking: list[dict[str, Any]],
    round_winners: dict[str, dict[int, str]],
    used_third_place_teams: set[str],
) -> str | None:
    if not name or not is_placeholder_team(name):
        return name

    group_match = GROUP_WINNER_RE.match(name.strip())
    if group_match:
        gid = group_match.group(1).upper()
        rows = group_standings.get(gid) or []
        return rows[0]["team"] if rows else None

    second_match = GROUP_2ND_RE.match(name.strip())
    if second_match:
        gid = second_match.group(1).upper()
        rows = group_standings.get(gid) or []
        return rows[1]["team"] if len(rows) > 1 else None

    third_match = THIRD_PLACE_RE.match(name.strip())
    if third_match:
        letters_raw = third_match.group(1).strip()
        allowed = {letter.strip().upper() for letter in letters_raw.split("/") if letter.strip()}
        for row in third_place_ranking:
            if not row.get("third_place_qualified"):
                continue
            team = row["team"]
            if team in used_third_place_teams:
                continue
            if row.get("group", "").upper() in allowed:
                used_third_place_teams.add(team)
                return team
        return None

    round_match = ROUND_WINNER_RE.match(name.strip())
    if round_match:
        label = round_match.group(1).lower()
        index = int(round_match.group(2))
        slug = ROUND_SLUG_BY_LABEL.get(label)
        if slug:
            return (round_winners.get(slug) or {}).get(index)
        return None

    if "semifinal" in name.lower() and "loser" in name.lower():
        semi_match = re.search(r"Semifinal\s+(\d+)", name, re.IGNORECASE)
        if semi_match:
            idx = int(semi_match.group(1))
            return (round_winners.get("semifinal_losers") or {}).get(idx)
    return None


def _hub_prediction(pred: dict[str, Any] | None) -> dict[str, Any] | None:
    if not pred:
        return None
    if "model" in pred:
        return pred
    return {"model": pred}


class _PredictionCache:
    def __init__(
        self,
        base: dict[str, Any],
        predict_fn: Callable[[ScheduledGame | None], dict[str, Any] | None] | None,
    ) -> None:
        self._base = base
        self._predict_fn = predict_fn
        self._pair_cache: dict[tuple[str, str], dict[str, Any]] = {}

    def get(
        self,
        event_id: str,
        match: dict[str, Any],
        away_name: str,
        home_name: str,
    ) -> dict[str, Any] | None:
        if event_id in self._base:
            return _hub_prediction(self._base[event_id])

        if not self._predict_fn or not away_name or not home_name:
            return None
        if is_placeholder_team(away_name) or is_placeholder_team(home_name):
            return None

        key = tuple(sorted([normalize_team_name(away_name), normalize_team_name(home_name)]))
        if key not in self._pair_cache:
            scheduled = build_scheduled_game(match.get("scheduled_game"), away_name, home_name)
            raw = self._predict_fn(scheduled)
            if raw and "error" not in raw:
                self._pair_cache[key] = {
                    "matchup": raw.get("matchup"),
                    "market": raw.get("market"),
                    "model": raw.get("model"),
                    "top_pick": raw.get("top_pick"),
                    "recommendations": raw.get("recommendations"),
                }
        return _hub_prediction(self._pair_cache.get(key))


def _simulate_match(
    match: dict[str, Any],
    prediction: dict[str, Any] | None,
    *,
    allow_draw: bool,
    away_resolved: str,
    home_resolved: str,
    resolved_from_placeholder: bool,
    rng: Random | None,
    bracket_slot: int | None = None,
) -> dict[str, Any]:
    probs = extract_unified_probs(prediction)
    lam_h, lam_a = extract_unified_lambdas(prediction)

    if rng is None:
        outcome = pick_outcome(probs, allow_draw=allow_draw) if allow_draw else pick_knockout_outcome(probs)
    elif allow_draw:
        outcome = sample_outcome(probs, rng, allow_draw=True)
    else:
        outcome = sample_knockout_outcome(probs, rng)

    knockout = not allow_draw
    home_score, away_score = derive_scores(
        outcome,
        prediction,
        allow_draw=allow_draw,
        rng=rng,
        knockout=knockout,
    )

    if knockout and home_score == away_score:
        if lam_h >= lam_a:
            home_score += 1
        else:
            away_score += 1
        outcome = "home" if home_score > away_score else "away"

    winner_side = outcome
    if outcome == "home":
        winner_name = home_resolved
    elif outcome == "away":
        winner_name = away_resolved
    else:
        winner_name = None

    return {
        "event_id": match.get("event_id"),
        "name": match.get("name"),
        "start_time": match.get("start_time"),
        "round_slug": match.get("round_slug"),
        "round_label": match.get("round_label"),
        "group": match.get("group"),
        "venue": match.get("venue"),
        "bracket_slot": bracket_slot,
        "completed": True,
        "away": {
            "name": away_resolved,
            "abbr": match.get("away", {}).get("abbr"),
            "score": away_score,
            "winner": outcome == "away",
        },
        "home": {
            "name": home_resolved,
            "abbr": match.get("home", {}).get("abbr"),
            "score": home_score,
            "winner": outcome == "home",
        },
        "scoreline": f"{away_score}–{home_score}",
        "winner": winner_side,
        "winner_name": winner_name,
        "outcome": outcome,
        "model_probs": {
            "home": round(probs["home"], 1),
            "draw": round(probs.get("draw", 0.0), 1),
            "away": round(probs["away"], 1),
        },
        "expected_goals": {"home": lam_h, "away": lam_a},
        "resolved_from_placeholder": resolved_from_placeholder,
    }


def _run_tournament_once(
    matches: list[dict[str, Any]],
    cache: _PredictionCache,
    *,
    rng: Random | None,
) -> dict[str, Any]:
    group_matches = sorted(
        [m for m in matches if m.get("round_slug") == "group-stage"],
        key=lambda m: m.get("start_time") or "",
    )
    simulated_group: list[dict[str, Any]] = []

    for match in group_matches:
        pred = cache.get(
            match["event_id"],
            match,
            match["away"]["name"],
            match["home"]["name"],
        )
        simulated_group.append(
            _simulate_match(
                match,
                pred,
                allow_draw=True,
                away_resolved=match["away"]["name"],
                home_resolved=match["home"]["name"],
                resolved_from_placeholder=False,
                rng=rng,
            )
        )

    group_standings = _compute_group_standings(simulated_group)
    third_place_ranking = _compute_third_place_ranking(group_standings)

    round_winners: dict[str, dict[int, str]] = {slug: {} for slug, _ in ROUND_ORDER}
    round_winners["semifinal_losers"] = {}
    used_third_place_teams: set[str] = set()
    simulated_knockout: list[dict[str, Any]] = []

    knockout_slugs = [slug for slug, _ in ROUND_ORDER if slug != "group-stage"]
    for round_slug in knockout_slugs:
        round_matches = sorted(
            [m for m in matches if m.get("round_slug") == round_slug],
            key=lambda m: m.get("start_time") or "",
        )
        for idx, match in enumerate(round_matches, start=1):
            away_raw = match["away"]["name"]
            home_raw = match["home"]["name"]
            away_ph = is_placeholder_team(away_raw)
            home_ph = is_placeholder_team(home_raw)

            away_name = (
                resolve_placeholder(
                    away_raw,
                    group_standings=group_standings,
                    third_place_ranking=third_place_ranking,
                    round_winners=round_winners,
                    used_third_place_teams=used_third_place_teams,
                )
                if away_ph
                else away_raw
            )
            home_name = (
                resolve_placeholder(
                    home_raw,
                    group_standings=group_standings,
                    third_place_ranking=third_place_ranking,
                    round_winners=round_winners,
                    used_third_place_teams=used_third_place_teams,
                )
                if home_ph
                else home_raw
            )

            if not away_name or not home_name:
                away_name = away_name or away_raw
                home_name = home_name or home_raw

            pred = cache.get(match["event_id"], match, away_name, home_name)
            sim = _simulate_match(
                match,
                pred,
                allow_draw=False,
                away_resolved=away_name,
                home_resolved=home_name,
                resolved_from_placeholder=away_ph or home_ph,
                rng=rng,
                bracket_slot=idx,
            )
            simulated_knockout.append(sim)

            if round_slug == "semifinals":
                loser = (
                    sim["away"]["name"]
                    if sim.get("winner") == "home"
                    else sim["home"]["name"]
                )
                round_winners["semifinal_losers"][idx] = loser

            winner = sim.get("winner_name")
            if winner and round_slug in round_winners:
                round_winners[round_slug][idx] = winner

    all_simulated = simulated_group + simulated_knockout
    final = next((m for m in simulated_knockout if m.get("round_slug") == "final"), None)
    third_match = next(
        (m for m in simulated_knockout if m.get("round_slug") == "3rd-place-match"),
        None,
    )

    champion = final.get("winner_name") if final else None
    runner_up = None
    if final:
        runner_up = (
            final["away"]["name"]
            if final.get("winner") == "home"
            else final["home"]["name"]
        )
    third_place = third_match.get("winner_name") if third_match else None

    return {
        "champion": champion,
        "runner_up": runner_up,
        "third_place": third_place,
        "final_scoreline": final.get("scoreline") if final else None,
        "matches": all_simulated,
        "simulated_group": simulated_group,
        "simulated_knockout": simulated_knockout,
        "group_standings": group_standings,
        "third_place_ranking": third_place_ranking,
        "round_winners": round_winners,
    }


def _match_node(sim: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_id": sim.get("event_id"),
        "round_slug": sim.get("round_slug"),
        "round_label": sim.get("round_label"),
        "bracket_slot": sim.get("bracket_slot"),
        "away": sim.get("away"),
        "home": sim.get("home"),
        "scoreline": sim.get("scoreline"),
        "winner": sim.get("winner_name"),
        "model_probs": sim.get("model_probs"),
        "expected_goals": sim.get("expected_goals"),
    }


def build_knockout_bracket_tree(simulated_knockout: list[dict[str, Any]]) -> dict[str, Any]:
    """Build nested knockout bracket tree for UI (R32 → Final)."""
    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for sim in simulated_knockout:
        slug = sim.get("round_slug")
        slot = sim.get("bracket_slot")
        if slug and slot:
            by_key[(slug, slot)] = sim

    def node_for(slug: str, slot: int) -> dict[str, Any]:
        sim = by_key.get((slug, slot))
        if not sim:
            return {"round_slug": slug, "bracket_slot": slot, "placeholder": True}

        feeds = BRACKET_FEEDS.get(slug, {}).get(slot)
        children: list[dict[str, Any]] = []
        if feeds:
            (away_round, away_slot), (home_round, home_slot) = feeds
            children = [
                node_for(away_round, away_slot),
                node_for(home_round, home_slot),
            ]

        payload = _match_node(sim)
        payload["children"] = children
        return payload

    final_node = node_for("final", 1) if ("final", 1) in by_key else None
    third_node = _match_node(by_key["3rd-place-match", 1]) if ("3rd-place-match", 1) in by_key else None

    columns: list[dict[str, Any]] = []
    for slug, label in ROUND_ORDER:
        if slug == "group-stage":
            continue
        slots = sorted(
            [sim for sim in simulated_knockout if sim.get("round_slug") == slug],
            key=lambda item: item.get("bracket_slot") or 0,
        )
        columns.append(
            {
                "slug": slug,
                "label": label,
                "matches": [_match_node(sim) for sim in slots],
            }
        )

    return {
        "final": final_node,
        "third_place_match": third_node,
        "columns": columns,
        "root": final_node,
    }


def _probability_table(counter: Counter[str], iterations: int, limit: int = 12) -> list[dict[str, Any]]:
    rows = []
    for team, count in counter.most_common(limit):
        rows.append(
            {
                "team": team,
                "count": count,
                "probability": round(100.0 * count / max(iterations, 1), 1),
            }
        )
    return rows


def simulate_tournament(
    matches: list[dict[str, Any]],
    predictions: dict[str, Any],
    predict_fn: Callable[[ScheduledGame | None], dict[str, Any] | None] | None = None,
    *,
    mc_iterations: int | None = None,
    mc_seed: int | None = None,
) -> dict[str, Any]:
    """
    Monte Carlo tournament simulation using unified 3-layer probabilities and
    Poisson/Dixon–Coles scores. Returns MC probabilities plus one representative
    full 104-match path (mode-champion iteration).
    """
    iterations = mc_iterations if mc_iterations is not None else DEFAULT_MC_ITERATIONS
    seed = mc_seed if mc_seed is not None else DEFAULT_MC_SEED
    cache = _PredictionCache(predictions, predict_fn)

    champion_counter: Counter[str] = Counter()
    runner_up_counter: Counter[str] = Counter()
    final_counter: Counter[str] = Counter()
    first_run_by_champion: dict[str, dict[str, Any]] = {}

    representative: dict[str, Any] | None = None
    mode_champion: str | None = None

    for i in range(iterations):
        rng = Random(seed + i)
        run = _run_tournament_once(matches, cache, rng=rng)
        champion = run.get("champion")
        if champion:
            champion_counter[champion] += 1
            first_run_by_champion.setdefault(champion, run)
            if run.get("runner_up"):
                runner_up_counter[run["runner_up"]] += 1
            final_counter[champion] += 1
            final_counter[run.get("runner_up") or ""] += 1

    if champion_counter:
        mode_champion = champion_counter.most_common(1)[0][0]
        representative = first_run_by_champion.get(mode_champion)

    if representative is None:
        representative = _run_tournament_once(matches, cache, rng=Random(seed))

    all_simulated = representative["matches"]
    by_round: dict[str, list[dict[str, Any]]] = {slug: [] for slug, _ in ROUND_ORDER}
    for sim in all_simulated:
        slug = sim.get("round_slug") or "group-stage"
        if slug in by_round:
            by_round[slug].append(sim)

    bracket_tree = build_knockout_bracket_tree(representative["simulated_knockout"])

    return {
        "method": "unified_3_layer_monte_carlo_poisson",
        "champion": representative.get("champion"),
        "runner_up": representative.get("runner_up"),
        "third_place": representative.get("third_place"),
        "final_scoreline": representative.get("final_scoreline"),
        "monte_carlo": {
            "iterations": iterations,
            "seed": seed,
            "mode_champion": mode_champion,
            "champion_probability": _probability_table(champion_counter, iterations),
            "runner_up_probability": _probability_table(runner_up_counter, iterations, limit=8),
            "reach_final_probability": _probability_table(final_counter, iterations, limit=16),
        },
        "summary": {
            "total_simulated": len(all_simulated),
            "group_stage": len(representative["simulated_group"]),
            "knockout": len(representative["simulated_knockout"]),
            "mc_iterations": iterations,
        },
        "simulated_standings": representative["group_standings"],
        "third_place_ranking": representative["third_place_ranking"],
        "matches": all_simulated,
        "rounds": [
            {"slug": slug, "label": label, "matches": by_round.get(slug, [])}
            for slug, label in ROUND_ORDER
        ],
        "bracket_tree": bracket_tree,
    }
