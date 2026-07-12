"""Tests for scripts/check_v2_data.py completeness gating."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _load_check_v2_data():
    path = PROJECT_ROOT / "scripts" / "check_v2_data.py"
    spec = importlib.util.spec_from_file_location("check_v2_data", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hollow_metadata_is_not_ok(tmp_path, monkeypatch) -> None:
    check = _load_check_v2_data()
    models = tmp_path / "models"
    league_dir = models / "nba_v2"
    league_dir.mkdir(parents=True)
    (league_dir / "metadata.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(check, "MODELS_ROOT", models)

    row = check.load_league("nba")
    assert row["ok"] is False
    assert "incomplete metadata" in row["error"]
    assert "oos_model_logloss" in row["error"]


def test_complete_metadata_is_ok(tmp_path, monkeypatch) -> None:
    check = _load_check_v2_data()
    models = tmp_path / "models"
    league_dir = models / "soccer_v2"
    league_dir.mkdir(parents=True)
    (league_dir / "metadata.json").write_text(
        json.dumps(
            {
                "oos_model_logloss": 0.95,
                "train_rows": 1000,
                "created_at": "2026-07-01T00:00:00+00:00",
                "algorithm": "gb",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(check, "MODELS_ROOT", models)

    row = check.load_league("soccer")
    assert row["ok"] is True
    assert row["oos_model_logloss"] == 0.95
    assert "oos_model_acc" not in row  # soccer often omits this key
