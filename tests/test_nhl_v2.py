"""NHL v2 live artifact loading defensive tests."""

from __future__ import annotations

import gzip
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_load_artifacts_returns_none_on_corrupt_json(tmp_path, monkeypatch) -> None:
    """Corrupt on-disk JSON must not raise — live layer returns None instead."""
    import web.nhl_v2.live as live

    model_dir = tmp_path / "nhl_v2"
    model_dir.mkdir()
    for name in ("model_clf.json", "model_lr.json", "calibrator.json", "metadata.json"):
        (model_dir / name).write_text("{not-json", encoding="utf-8")
    with gzip.open(model_dir / "state_2024.json.gz", "wt", encoding="utf-8") as handle:
        handle.write("{}")

    monkeypatch.setattr(live, "MODEL_DIR", model_dir)
    live._load_artifacts.cache_clear()
    assert live.artifacts_available() is True
    assert live._load_artifacts() is None


def test_load_snapshot_state_returns_none_on_bad_gzip(tmp_path, monkeypatch) -> None:
    import web.nhl_v2.live as live

    model_dir = tmp_path / "nhl_v2"
    model_dir.mkdir()
    bad = model_dir / "state_2024.json.gz"
    bad.write_bytes(b"not-gzip-data")
    art = {"snapshots": {2024: bad}}
    assert live._load_snapshot_state(art, 2025) is None


def test_fetch_stats_bundle_flags_stale_on_network_failure(tmp_path, monkeypatch) -> None:
    import json
    import time

    import web.nhl_v2.live as live

    monkeypatch.setattr(live, "LIVE_CACHE_DIR", tmp_path)
    cache = tmp_path / "stats_2024.json"
    payload = {"team_games": [{"id": 1}], "goalies": []}
    cache.write_text(json.dumps(payload), encoding="utf-8")
    # Age the file past the fresh TTL but within the soft-fail window.
    aged = time.time() - (live.STATS_TTL_SECONDS + 60)
    import os

    os.utime(cache, (aged, aged))

    def _boom(*_a, **_k):
        raise OSError("offline")

    monkeypatch.setattr(live, "fetch_team_games", _boom)
    monkeypatch.setattr(live, "fetch_goalie_games", _boom)
    bundle, stale = live._fetch_stats_bundle(2024)
    assert stale is True
    assert bundle == payload


def test_blend_nhl_survives_hockey_pred_crash() -> None:
    """NHL blend must fall back to Algo V1 when hockey pred raises."""
    from web.blend_service import blend_predictions
    import web.blend_service as blend_module

    original = blend_module.run_hockey_pred_model

    def _boom(*_a, **_k):
        raise RuntimeError("simulated nhl v2 failure")

    try:
        blend_module.run_hockey_pred_model = _boom
        result = blend_predictions(
            legacy_total_score=-4.5,
            legacy_win_probability=58.0,
            league="nhl",
            cutoff_date="1-15-2026",
            home_abbr="tor",
            away_abbr="bos",
            home_moneyline=-140,
            away_moneyline=120,
        )
    finally:
        blend_module.run_hockey_pred_model = original

    assert result["blend_mode"] == "algo_v1"
    assert result["algorithm"] == "Algo_V1"
    assert result.get("hockey_pred") is None


def test_infer_nhl_game_type_playoff_window() -> None:
    from web.nhl_v2.live import infer_nhl_game_type

    assert infer_nhl_game_type("2026-01-15") == 2
    assert infer_nhl_game_type("2026-04-10") == 2
    assert infer_nhl_game_type("2026-04-15") == 3
    assert infer_nhl_game_type("2026-05-20") == 3
    assert infer_nhl_game_type("2026-06-15") == 3
    assert infer_nhl_game_type("not-a-date") == 2


def test_get_live_context_fails_closed_on_missing_gap_season(monkeypatch) -> None:
    """Hard-missing intermediate seasons must not silently skip Elo/form state."""
    from unittest.mock import MagicMock

    import web.nhl_v2.live as live

    live.get_live_context.cache_clear()
    art = {"snapshots": {2023: MagicMock()}}
    monkeypatch.setattr(live, "_load_artifacts", lambda: art)
    monkeypatch.setattr(
        live,
        "_load_snapshot_state",
        lambda _art, _season: (2023, {"teams": {}}),
    )
    monkeypatch.setattr(
        live.NhlFeatureEngine,
        "from_dict",
        classmethod(lambda cls, _payload: MagicMock()),
    )
    monkeypatch.setattr(live, "nhl_season_for_date", lambda _d: 2025)

    calls: list[int] = []

    def fake_bundle(season: int):
        calls.append(season)
        if season == 2024:
            return None, False
        return {"team_games": [], "goalies": []}, False

    monkeypatch.setattr(live, "_fetch_stats_bundle", fake_bundle)
    assert live.get_live_context("2025-01-15") is None
    assert 2024 in calls


def test_get_live_context_fails_closed_on_empty_gap_games(monkeypatch) -> None:
    """Empty intermediate-season game bundles must fail closed (not silent Elo skip)."""
    from unittest.mock import MagicMock

    import web.nhl_v2.live as live

    live.get_live_context.cache_clear()
    art = {"snapshots": {2023: MagicMock()}}
    monkeypatch.setattr(live, "_load_artifacts", lambda: art)
    monkeypatch.setattr(
        live,
        "_load_snapshot_state",
        lambda _art, _season: (2023, {"teams": {}}),
    )
    monkeypatch.setattr(
        live.NhlFeatureEngine,
        "from_dict",
        classmethod(lambda cls, _payload: MagicMock()),
    )
    monkeypatch.setattr(live, "nhl_season_for_date", lambda _d: 2025)
    monkeypatch.setattr(
        live,
        "_fetch_stats_bundle",
        lambda season: ({"team_games": [], "goalies": []}, False),
    )
    monkeypatch.setattr(live, "_fetch_moneypuck_slices", lambda seasons: ({}, False))
    assert live.get_live_context("2025-01-15") is None


def test_get_live_context_fails_closed_on_empty_gap_moneypuck(monkeypatch) -> None:
    """Gap seasons with Stats games but empty MoneyPuck must fail closed."""
    from unittest.mock import MagicMock

    import web.nhl_v2.live as live

    live.get_live_context.cache_clear()
    art = {"snapshots": {2023: MagicMock()}}
    monkeypatch.setattr(live, "_load_artifacts", lambda: art)
    monkeypatch.setattr(
        live,
        "_load_snapshot_state",
        lambda _art, _season: (2023, {"teams": {}}),
    )
    monkeypatch.setattr(
        live.NhlFeatureEngine,
        "from_dict",
        classmethod(lambda cls, _payload: MagicMock()),
    )
    monkeypatch.setattr(live, "nhl_season_for_date", lambda _d: 2025)

    monkeypatch.setattr(
        live,
        "_fetch_stats_bundle",
        lambda season: (
            {"team_games": [{"gameId": 1, "teamAbbrev": "TOR"}], "goalies": []},
            False,
        ),
    )
    monkeypatch.setattr(
        live,
        "_fetch_moneypuck_slices",
        lambda seasons: ({}, False),
    )
    monkeypatch.setattr(
        live,
        "build_game_index",
        lambda _rows: {1: {"home": "TOR", "away": "MTL"}},
    )
    assert live.get_live_context("2025-01-15") is None
