"""Log-loss calibrated stacking for soccer 1X2 (home / draw / away).

Tunes layer weights and temperature from walk-forward history without sklearn.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
META_WEIGHTS_PATH = PROJECT_ROOT / "data" / "soccer_meta_weights.json"

OUTCOME_HOME = 0
OUTCOME_DRAW = 1
OUTCOME_AWAY = 2

DEFAULT_STAT_WEIGHTS: tuple[float, float, float] = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
DEFAULT_BLEND_WEIGHTS: dict[str, float] = {
    "legacy": 0.05,
    "power": 0.15,
    "soccer_pred": 0.80,
}
DEFAULT_TEMPERATURE = 1.0

Threeway = tuple[float, float, float]


def outcome_from_score(home_goals: int, away_goals: int) -> int:
    if home_goals > away_goals:
        return OUTCOME_HOME
    if home_goals < away_goals:
        return OUTCOME_AWAY
    return OUTCOME_DRAW


def normalize_threeway(home: float, draw: float, away: float) -> Threeway:
    total = home + draw + away
    if total <= 0:
        return 33.33, 33.33, 33.34
    scale = 100.0 / total
    return home * scale, draw * scale, away * scale


def multiclass_log_loss(home: float, draw: float, away: float, outcome: int) -> float:
    probs = (home / 100.0, draw / 100.0, away / 100.0)
    prob = max(probs[outcome], 1e-9)
    return -math.log(prob)


def blend_weighted_threeway(
    layers: list[Threeway],
    weights: list[float],
) -> Threeway:
    weight_sum = sum(weights) or 1.0
    home = sum(weight * layer[0] for weight, layer in zip(weights, layers)) / weight_sum
    draw = sum(weight * layer[1] for weight, layer in zip(weights, layers)) / weight_sum
    away = sum(weight * layer[2] for weight, layer in zip(weights, layers)) / weight_sum
    return normalize_threeway(home, draw, away)


def temperature_scale(home: float, draw: float, away: float, temperature: float) -> Threeway:
    if temperature <= 0:
        temperature = 1.0
    logits = [math.log(max(prob / 100.0, 1e-9)) for prob in (home, draw, away)]
    scaled = [logit / temperature for logit in logits]
    peak = max(scaled)
    exponentials = [math.exp(value - peak) for value in scaled]
    total = sum(exponentials) or 1.0
    return tuple(100.0 * value / total for value in exponentials)


def _coerce_stat_weights(raw: Any) -> tuple[float, float, float]:
    if isinstance(raw, dict):
        elo = float(raw.get("elo", DEFAULT_STAT_WEIGHTS[0]))
        pi = float(raw.get("pi", DEFAULT_STAT_WEIGHTS[1]))
        dc = float(raw.get("dc", DEFAULT_STAT_WEIGHTS[2]))
    elif isinstance(raw, (list, tuple)) and len(raw) == 3:
        elo, pi, dc = (float(raw[0]), float(raw[1]), float(raw[2]))
    else:
        return DEFAULT_STAT_WEIGHTS
    total = elo + pi + dc
    if total <= 0:
        return DEFAULT_STAT_WEIGHTS
    return elo / total, pi / total, dc / total


def _coerce_blend_weights(raw: Any) -> dict[str, float]:
    if not isinstance(raw, dict):
        return dict(DEFAULT_BLEND_WEIGHTS)
    merged = dict(DEFAULT_BLEND_WEIGHTS)
    for key in ("legacy", "power", "soccer_pred"):
        if key in raw:
            merged[key] = float(raw[key])
    total = sum(merged.values())
    if total <= 0:
        return dict(DEFAULT_BLEND_WEIGHTS)
    return {key: value / total for key, value in merged.items()}


@lru_cache(maxsize=1)
def load_soccer_meta_config() -> dict[str, Any]:
    if not META_WEIGHTS_PATH.is_file():
        return {"default": _default_league_entry()}
    try:
        payload = json.loads(META_WEIGHTS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"default": _default_league_entry()}
    if "default" not in payload:
        payload["default"] = _default_league_entry()
    return payload


def _default_league_entry() -> dict[str, Any]:
    return {
        "stat": {
            "elo": DEFAULT_STAT_WEIGHTS[0],
            "pi": DEFAULT_STAT_WEIGHTS[1],
            "dc": DEFAULT_STAT_WEIGHTS[2],
        },
        "blend": dict(DEFAULT_BLEND_WEIGHTS),
        "temperature": DEFAULT_TEMPERATURE,
    }


def get_league_meta_config(league: str) -> dict[str, Any]:
    config = load_soccer_meta_config()
    league = league.lower()
    entry = config.get(league) or config.get("default") or _default_league_entry()
    return {
        "stat_weights": _coerce_stat_weights(entry.get("stat")),
        "blend_weights": _coerce_blend_weights(entry.get("blend")),
        "temperature": float(entry.get("temperature", DEFAULT_TEMPERATURE)),
    }


def apply_threeway_calibration(
    home: float,
    draw: float,
    away: float,
    league: str,
) -> Threeway:
    config = get_league_meta_config(league)
    calibrated = temperature_scale(home, draw, away, config["temperature"])
    return tuple(round(value, 2) for value in calibrated)


def stack_soccer_blend_layers(
    *,
    legacy: Threeway | None,
    power: Threeway | None,
    soccer_pred: Threeway | None,
    league: str,
) -> Threeway | None:
    config = get_league_meta_config(league)
    weights = config["blend_weights"]
    layers: list[Threeway] = []
    layer_weights: list[float] = []
    for key, triple in (
        ("legacy", legacy),
        ("power", power),
        ("soccer_pred", soccer_pred),
    ):
        if triple is None:
            continue
        layers.append(triple)
        layer_weights.append(weights.get(key, 0.0))
    if not layers:
        return None
    if sum(layer_weights) <= 0:
        layer_weights = [1.0 / len(layers)] * len(layers)
    blended = blend_weighted_threeway(layers, layer_weights)
    return blended


def fit_stat_weights_grid(
    samples: list[tuple[Threeway, Threeway, Threeway, int]],
) -> tuple[float, float, float]:
    """Grid-search Elo / Pi / Dixon-Coles blend weights to minimize log-loss."""
    if not samples:
        return DEFAULT_STAT_WEIGHTS

    best_loss = float("inf")
    best_weights = DEFAULT_STAT_WEIGHTS
    for elo_step in range(0, 11):
        for pi_step in range(0, 11):
            dc_step = 10 - elo_step - pi_step
            if dc_step < 0:
                continue
            weights = (elo_step / 10.0, pi_step / 10.0, dc_step / 10.0)
            if weights[2] < 0.05 and pi_step > 0:
                continue
            total_loss = 0.0
            for elo, pi, dc, outcome in samples:
                home, draw, away = blend_weighted_threeway(
                    [elo, pi, dc], list(weights)
                )
                total_loss += multiclass_log_loss(home, draw, away, outcome)
            avg_loss = total_loss / len(samples)
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_weights = weights
    return best_weights


def fit_temperature_grid(
    samples: list[tuple[float, float, float, int]],
) -> float:
    if not samples:
        return DEFAULT_TEMPERATURE

    best_loss = float("inf")
    best_temperature = DEFAULT_TEMPERATURE
    for step in range(7, 16):
        temperature = step / 10.0
        total_loss = 0.0
        for home, draw, away, outcome in samples:
            calibrated = temperature_scale(home, draw, away, temperature)
            total_loss += multiclass_log_loss(*calibrated, outcome)
        avg_loss = total_loss / len(samples)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_temperature = temperature
    return best_temperature


def fit_blend_weights_grid(
    samples: list[tuple[Threeway | None, Threeway | None, Threeway, int]],
) -> dict[str, float]:
    """Tune legacy / power / soccer_pred blend on walk-forward samples."""
    if not samples:
        return dict(DEFAULT_BLEND_WEIGHTS)

    best_loss = float("inf")
    best_weights = dict(DEFAULT_BLEND_WEIGHTS)
    for legacy_step in range(0, 4):
        for power_step in range(0, 6):
            soccer_step = 10 - legacy_step - power_step
            if soccer_step < 4:
                continue
            weights = {
                "legacy": legacy_step / 10.0,
                "power": power_step / 10.0,
                "soccer_pred": soccer_step / 10.0,
            }
            total_loss = 0.0
            count = 0
            for legacy, power, soccer, outcome in samples:
                layers: list[Threeway] = []
                layer_weights: list[float] = []
                for key, triple in (
                    ("legacy", legacy),
                    ("power", power),
                    ("soccer_pred", soccer),
                ):
                    if triple is None:
                        continue
                    layers.append(triple)
                    layer_weights.append(weights[key])
                if not layers:
                    continue
                home, draw, away = blend_weighted_threeway(layers, layer_weights)
                total_loss += multiclass_log_loss(home, draw, away, outcome)
                count += 1
            if count == 0:
                continue
            avg_loss = total_loss / count
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_weights = weights
    total = sum(best_weights.values()) or 1.0
    return {key: value / total for key, value in best_weights.items()}
