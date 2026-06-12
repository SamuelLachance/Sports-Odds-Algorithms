"""Forward tournament simulation for FIFA World Cup 2026 using unified model predictions."""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any, Callable

from web.espn_client import ScheduledGame
from web.live_data import resolve_team
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


def extract_model_probs(prediction: dict[str, Any] | None) -> dict[str, float]:
    """Return home/draw/away probabilities from a hub prediction payload."""
    if not prediction:
        return {"home": 33.33, "draw": 33.33, "away": 33.33}

    model = prediction.get("model") or {}
    if model.get("threeway"):
        return {
            "home": float(model.get("home_win_probability", 33.33)),
            "draw": float(model.get("draw_probability", 33.33)),
            "away": float(model.get("away_win_probability", 33.33)),
        }

    home_p = float(
        model.get("blended_home_win_probability")
        or model.get("win_probability")
        or 50.0
    )
    away_p = max(0.0, 100.0 - home_p)
    return {"home": home_p, "draw": 0.0, "away": away_p}


def pick_outcome(probs: dict[str, float], *, allow_draw: bool = True) -> str:
    """Deterministic outcome: highest probability wins."""
    candidates: list[tuple[str, float]] = [
        ("home", probs.get("home", 0.0)),
        ("draw", probs.get("draw", 0.0)),
        ("away", probs.get("away", 0.0)),
    ]
    if not allow_draw:
        candidates = [c for c in candidates if c[0] != "draw"]
    candidates.sort(key=lambda item: (-item[1], item[0]))
    return candidates[0][0]


def pick_knockout_outcome(probs: dict[str, float]) -> str:
    """Knockout: no draws — tiebreak draw prob toward home/away."""
    best = max(probs.get("home", 0.0), probs.get("draw", 0.0), probs.get("away", 0.0))
    if probs.get("draw", 0.0) >= best - 1e-9:
        home_p = probs.get("home", 0.0)
        away_p = probs.get("away", 0.0)
        if home_p > away_p:
            return "home"
        if away_p > home_p:
            return "away"
        return "home"
    return pick_outcome(probs, allow_draw=False)


def derive_scores(
    outcome: str,
    prediction: dict[str, Any] | None,
    *,
    allow_draw: bool,
) -> tuple[int, int]:
    """Build integer scores from expected goals or simple fallbacks."""
    model = (prediction or {}).get("model") or {}
    soccer = model.get("soccer_pred") or {}
    exp_h = soccer.get("expected_home_goals")
    exp_a = soccer.get("expected_away_goals")

    if exp_h is not None and exp_a is not None:
        home_score = max(0, int(round(float(exp_h))))
        away_score = max(0, int(round(float(exp_a))))
        if outcome == "home" and home_score <= away_score:
            home_score = away_score + 1
        elif outcome == "away" and away_score <= home_score:
            away_score = home_score + 1
        elif outcome == "draw":
            high = max(home_score, away_score, 1)
            home_score = away_score = high
        return home_score, away_score

    if outcome == "home":
        return (2, 1) if allow_draw else (1, 0)
    if outcome == "away":
        return (0, 1) if allow_draw else (0, 1)
    return (1, 1)


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
    """Clone ESPN scheduled game with resolved team names for knockout prediction."""
    if not template:
        return None
    away_abbr = _team_abbr(away_name)
    home_abbr = _team_abbr(home_name)
    return replace(
        template,
        away_name=away_name,
        home_name=home_name,
        away_abbr=away_abbr,
        home_abbr=home_abbr,
    )


def resolve_placeholder(
    name: str,
    *,
    group_standings: dict[str, list[dict[str, Any]]],
    third_place_ranking: list[dict[str, Any]],
    round_winners: dict[str, dict[int, str]],
    used_third_place_teams: set[str],
) -> str | None:
    """Map ESPN placeholder label to simulated team name."""
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

    return None


def _hub_prediction(pred: dict[str, Any] | None) -> dict[str, Any] | None:
    if not pred:
        return None
    if "model" in pred:
        return pred
    return {"model": pred}


def _simulate_match(
    match: dict[str, Any],
    prediction: dict[str, Any] | None,
    *,
    allow_draw: bool,
    away_resolved: str,
    home_resolved: str,
    resolved_from_placeholder: bool,
) -> dict[str, Any]:
    probs = extract_model_probs(prediction)
    outcome = pick_outcome(probs, allow_draw=allow_draw) if allow_draw else pick_knockout_outcome(probs)
    home_score, away_score = derive_scores(outcome, prediction, allow_draw=allow_draw)

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
        "resolved_from_placeholder": resolved_from_placeholder,
    }


def simulate_tournament(
    matches: list[dict[str, Any]],
    predictions: dict[str, Any],
    predict_fn: Callable[[ScheduledGame | None], dict[str, Any] | None] | None = None,
) -> dict[str, Any]:
    """
    Simulate all 104 matches forward using model predictions.

    Group stage uses precomputed predictions; knockout resolves placeholders then
    calls predict_fn when needed.
    """
    group_matches = sorted(
        [m for m in matches if m.get("round_slug") == "group-stage"],
        key=lambda m: m.get("start_time") or "",
    )
    simulated_group: list[dict[str, Any]] = []
    runtime_predictions = dict(predictions)

    for match in group_matches:
        event_id = match["event_id"]
        pred = _hub_prediction(runtime_predictions.get(event_id))
        if pred is None and predict_fn:
            pred = _hub_prediction(predict_fn(match.get("scheduled_game")))
            if pred and "error" not in pred:
                runtime_predictions[event_id] = {
                    "matchup": pred.get("matchup"),
                    "market": pred.get("market"),
                    "model": pred.get("model"),
                    "top_pick": pred.get("top_pick"),
                    "recommendations": pred.get("recommendations"),
                }

        sim = _simulate_match(
            match,
            pred,
            allow_draw=True,
            away_resolved=match["away"]["name"],
            home_resolved=match["home"]["name"],
            resolved_from_placeholder=False,
        )
        simulated_group.append(sim)

    group_standings = _compute_group_standings(simulated_group)
    third_place_ranking = _compute_third_place_ranking(group_standings)

    round_winners: dict[str, dict[int, str]] = {slug: {} for slug, _ in ROUND_ORDER}
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

            event_id = match["event_id"]
            pred = _hub_prediction(runtime_predictions.get(event_id))
            if pred is None and predict_fn and away_name and home_name:
                scheduled = build_scheduled_game(
                    match.get("scheduled_game"),
                    away_name,
                    home_name,
                )
                raw_pred = predict_fn(scheduled)
                if raw_pred and "error" not in raw_pred:
                    pred = _hub_prediction(raw_pred)
                    runtime_predictions[event_id] = {
                        "matchup": raw_pred.get("matchup"),
                        "market": raw_pred.get("market"),
                        "model": raw_pred.get("model"),
                        "top_pick": raw_pred.get("top_pick"),
                        "recommendations": raw_pred.get("recommendations"),
                    }

            sim = _simulate_match(
                match,
                pred,
                allow_draw=False,
                away_resolved=away_name,
                home_resolved=home_name,
                resolved_from_placeholder=away_ph or home_ph,
            )
            simulated_knockout.append(sim)

            winner = sim.get("winner_name")
            if winner and round_slug in round_winners:
                round_winners[round_slug][idx] = winner

    all_simulated = simulated_group + simulated_knockout
    by_round: dict[str, list[dict[str, Any]]] = {slug: [] for slug, _ in ROUND_ORDER}
    for sim in all_simulated:
        slug = sim.get("round_slug") or "group-stage"
        if slug in by_round:
            by_round[slug].append(sim)

    final = next((m for m in simulated_knockout if m.get("round_slug") == "final"), None)
    third_match = next(
        (m for m in simulated_knockout if m.get("round_slug") == "3rd-place-match"),
        None,
    )
    semis = [m for m in simulated_knockout if m.get("round_slug") == "semifinals"]

    champion = final.get("winner_name") if final else None
    runner_up = None
    if final:
        if final.get("winner") == "home":
            runner_up = final["away"]["name"]
        elif final.get("winner") == "away":
            runner_up = final["home"]["name"]
    third_place = third_match.get("winner_name") if third_match else None
    final_scoreline = final.get("scoreline") if final else None

    return {
        "method": "unified_3_layer_deterministic",
        "champion": champion,
        "runner_up": runner_up,
        "third_place": third_place,
        "final_scoreline": final_scoreline,
        "summary": {
            "total_simulated": len(all_simulated),
            "group_stage": len(simulated_group),
            "knockout": len(simulated_knockout),
        },
        "simulated_standings": group_standings,
        "third_place_ranking": third_place_ranking,
        "matches": all_simulated,
        "rounds": [
            {"slug": slug, "label": label, "matches": by_round.get(slug, [])}
            for slug, label in ROUND_ORDER
        ],
        "semifinal_losers": [
            m["away"]["name"]
            if m.get("winner") == "home"
            else m["home"]["name"]
            for m in semis
            if m.get("winner") in {"home", "away"}
        ],
    }
