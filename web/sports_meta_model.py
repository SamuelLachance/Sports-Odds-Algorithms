"""Log-loss calibrated stacking for binary home-win predictions (all non-soccer sports)."""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
META_WEIGHTS_PATH = PROJECT_ROOT / "data" / "sports_meta_weights.json"

DEFAULT_BLEND_WEIGHTS: dict[str, float] = {
    "legacy": 0.05,
    "power": 0.15,
    "sport_pred": 0.80,
}
DEFAULT_TWO_LAYER_WEIGHTS: dict[str, float] = {
    "legacy": 0.35,
    "power": 0.65,
}
DEFAULT_TEMPERATURE = 1.0


def binary_log_loss(home_prob: float, home_won: bool) -> float:
    prob = min(max(home_prob / 100.0, 1e-9), 1.0 - 1e-9)
    return -(math.log(prob) if home_won else math.log(1.0 - prob))


def binary_temperature_scale(home_prob: float, temperature: float) -> float:
    if temperature <= 0:
        temperature = 1.0
    prob = min(max(home_prob / 100.0, 1e-9), 1.0 - 1e-9)
    logit = math.log(prob / (1.0 - prob))
    scaled = logit / temperature
    calibrated = 1.0 / (1.0 + math.exp(-scaled))
    return min(max(calibrated * 100.0, 0.01), 99.99)


def _coerce_blend_weights(raw: Any, *, two_layer: bool = False) -> dict[str, float]:
    defaults = DEFAULT_TWO_LAYER_WEIGHTS if two_layer else DEFAULT_BLEND_WEIGHTS
    if not isinstance(raw, dict):
        return dict(defaults)
    merged = dict(defaults)
    for key in defaults:
        if key in raw:
            merged[key] = float(raw[key])
    if "sport_pred" in raw and not two_layer:
        merged["sport_pred"] = float(raw["sport_pred"])
    total = sum(merged.values())
    if total <= 0:
        return dict(defaults)
    return {key: value / total for key, value in merged.items()}


@lru_cache(maxsize=1)
def load_sports_meta_config() -> dict[str, Any]:
    if not META_WEIGHTS_PATH.is_file():
        return {"default": _default_entry()}
    try:
        payload = json.loads(META_WEIGHTS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"default": _default_entry()}
    if "default" not in payload:
        payload["default"] = _default_entry()
    return payload


def _default_entry(*, two_layer: bool = False) -> dict[str, Any]:
    return {
        "blend": dict(DEFAULT_TWO_LAYER_WEIGHTS if two_layer else DEFAULT_BLEND_WEIGHTS),
        "temperature": DEFAULT_TEMPERATURE,
        "two_layer": two_layer,
    }


def get_sports_meta_config(league: str) -> dict[str, Any]:
    config = load_sports_meta_config()
    league = league.lower()
    entry = config.get(league) or config.get("default") or _default_entry()
    two_layer = bool(entry.get("two_layer"))
    return {
        "blend_weights": _coerce_blend_weights(entry.get("blend"), two_layer=two_layer),
        "temperature": float(entry.get("temperature", DEFAULT_TEMPERATURE)),
        "two_layer": two_layer,
    }


def apply_binary_calibration(home_prob: float, league: str) -> float:
    config = get_sports_meta_config(league)
    calibrated = binary_temperature_scale(home_prob, config["temperature"])
    return round(calibrated, 2)


def stack_binary_blend_layers(
    *,
    legacy_home: float | None,
    power_home: float | None,
    sport_home: float | None,
    league: str,
    sport_key: str | None = None,
) -> float | None:
    config = get_sports_meta_config(league)
    weights = config["blend_weights"]
    values: list[float] = []
    layer_weights: list[float] = []

    for key, value in (
        ("legacy", legacy_home),
        ("power", power_home),
        ("sport_pred", sport_home),
    ):
        if value is None:
            continue
        if key == "sport_pred" and weights.get("sport_pred", 0.0) <= 0:
            continue
        weight = weights.get(key, 0.0)
        if weight <= 0:
            continue
        values.append(float(value))
        layer_weights.append(weight)

    if not values:
        return None
    if sum(layer_weights) <= 0:
        layer_weights = [1.0 / len(values)] * len(values)
    blended = sum(weight * value for weight, value in zip(layer_weights, values))
    blended /= sum(layer_weights)
    return min(max(blended, 0.0), 100.0)


def apply_binary_calibration_to_blend(result: dict[str, Any], league: str) -> dict[str, Any]:
    home_prob = result.get("blended_home_win_probability")
    if home_prob is None and result.get("home_win_probability") is not None:
        home_prob = result["home_win_probability"]
    if home_prob is None:
        return result

    calibrated = apply_binary_calibration(float(home_prob), league)
    from web.blend_service import home_win_prob_to_total_score

    total, win_prob = home_win_prob_to_total_score(calibrated)
    updated = dict(result)
    updated["blended_home_win_probability"] = calibrated
    updated["total_score"] = round(total, 2)
    updated["win_probability"] = round(win_prob, 2)
    updated["favorite_side"] = "home" if total <= 0 else "away"
    return updated


def fit_binary_blend_weights_grid(
    samples: list[tuple[float | None, float | None, float | None, bool]],
    *,
    two_layer: bool = False,
    max_sport_step: int = 10,
) -> dict[str, float]:
    if not samples:
        return dict(DEFAULT_TWO_LAYER_WEIGHTS if two_layer else DEFAULT_BLEND_WEIGHTS)

    best_loss = float("inf")
    best_weights = dict(DEFAULT_TWO_LAYER_WEIGHTS if two_layer else DEFAULT_BLEND_WEIGHTS)

    legacy_range = range(0, 4) if two_layer else range(0, 3)
    power_range = range(0, 11) if two_layer else range(0, 6)

    min_sport_step = 0 if max_sport_step == 0 else 3
    for legacy_step in legacy_range:
        for power_step in power_range:
            if two_layer:
                sport_step = 10 - legacy_step - power_step
                if sport_step != 0:
                    continue
                weights = {
                    "legacy": legacy_step / 10.0,
                    "power": power_step / 10.0,
                }
            else:
                sport_step = 10 - legacy_step - power_step
                if sport_step < min_sport_step or sport_step > max_sport_step:
                    continue
                weights = {
                    "legacy": legacy_step / 10.0,
                    "power": power_step / 10.0,
                    "sport_pred": sport_step / 10.0,
                }

            total_loss = 0.0
            count = 0
            for legacy, power, sport, home_won in samples:
                values: list[float] = []
                wts: list[float] = []
                for key, val in (
                    ("legacy", legacy),
                    ("power", power),
                    ("sport_pred", sport),
                ):
                    if val is None:
                        continue
                    weight = weights.get(key, 0.0)
                    if weight <= 0:
                        continue
                    values.append(float(val))
                    wts.append(weight)
                if not values:
                    continue
                blended = sum(w * v for w, v in zip(wts, values)) / sum(wts)
                total_loss += binary_log_loss(blended, home_won)
                count += 1

            if count == 0:
                continue
            avg_loss = total_loss / count
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_weights = dict(weights)

    total = sum(best_weights.values()) or 1.0
    return {key: value / total for key, value in best_weights.items()}


def fit_binary_temperature_grid(
    samples: list[tuple[float, bool]],
    *,
    min_step: int = 7,
    max_step: int = 15,
) -> float:
    if not samples:
        return DEFAULT_TEMPERATURE

    best_loss = float("inf")
    best_temperature = DEFAULT_TEMPERATURE
    for step in range(min_step, max_step + 1):
        temperature = step / 10.0
        total_loss = 0.0
        for home_prob, home_won in samples:
            calibrated = binary_temperature_scale(home_prob, temperature)
            total_loss += binary_log_loss(calibrated, home_won)
        avg_loss = total_loss / len(samples)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_temperature = temperature
    return best_temperature
