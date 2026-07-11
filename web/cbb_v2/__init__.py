"""CBB GradientBoost v2 prediction stack.

Walk-forward features from ESPN completed games only (point-in-time safe).
Torvik season-end ratings are intentionally excluded from training; live
inference may still fall back to Torvik via blend_service when artifacts
are missing. Offline scripts: scripts/build_cbb_training_table.py,
scripts/train_cbb_model.py, scripts/fetch_cbb_odds.py.
"""
