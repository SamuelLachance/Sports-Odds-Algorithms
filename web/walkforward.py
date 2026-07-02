"""Shared walk-forward index selection for tuning and ensemble datasets."""

from __future__ import annotations

from web.ensemble_ml.config import MIN_TRAIN_ROWS


def calibration_indices(
    total: int,
    *,
    warmup: int | None = None,
    max_calibration_games: int | None = None,
    target_rows: int | None = None,
) -> range:
    """Indices for point-in-time calibration using as much history as configured.

  When ``max_calibration_games`` and ``target_rows`` are None, every game after
  ``warmup`` is used (full publicly available walk-forward history).
    """
    warmup = warmup if warmup is not None else MIN_TRAIN_ROWS // 2
    if max_calibration_games is None:
        start = warmup
    else:
        start = max(warmup, total - int(max_calibration_games))
    span = total - start
    if target_rows is None or span <= int(target_rows):
        step = 1
    else:
        step = max(1, span // int(target_rows))
    return range(start, total, step)
