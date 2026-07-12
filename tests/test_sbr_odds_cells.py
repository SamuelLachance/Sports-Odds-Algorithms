"""SBR closing-line cell parsers must keep EVEN/pick'em (not falsy-or drop)."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.nba_ml.dataset import _spread_juice  # noqa: E402
from web.sbr_odds import (  # noqa: E402
    _american_ml_cell,
    _parse_optional_float,
    _parse_optional_int,
    _spread_juice_cell,
    _spread_line_cell,
)


def test_sbr_american_ml_keeps_even_zero_as_plus_100() -> None:
    assert _american_ml_cell("0") == 100
    assert _american_ml_cell("-110") == -110
    assert _american_ml_cell("") is None
    assert _american_ml_cell("NL") is None


def test_sbr_spread_line_keeps_pickem_zero() -> None:
    assert _spread_line_cell("0") == 0.0
    assert _spread_line_cell("pk") == 0.0
    assert _spread_line_cell("PK") == 0.0
    assert _spread_line_cell("-3.5") == -3.5
    assert _spread_line_cell("") is None


def test_sbr_spread_juice_maps_even_zero() -> None:
    assert _spread_juice_cell("0") == 100
    assert _spread_juice_cell("") is None
    assert _spread_juice_cell("-105") == -105


def test_nba_ml_dataset_spread_juice_maps_even_zero() -> None:
    assert _spread_juice(0) == 100.0
    assert _spread_juice(None) == -110.0
    assert _spread_juice(-105) == -105.0


def test_sbr_html_and_archive_do_not_invent_spread_juice() -> None:
    """Fail closed: missing juice must stay None, not fake -110."""
    from unittest.mock import patch

    from web.sbr_odds import _rows_from_html_table, fetch_sbr_archive_rows

    # Header + subheader + away + home (pairwise starts at body index 1).
    cells = lambda *vals: "".join(f"<td>{v}</td>" for v in vals)
    html = (
        "<table>"
        f"<tr>{cells(*([f'h{i}' for i in range(12)]))}</tr>"
        f"<tr>{cells(*(['sub'] * 12))}</tr>"
        f"<tr>{cells('1015', '', '', 'Boston', '', '', '', '', '', '-3.5', '-3.5', '-110')}</tr>"
        f"<tr>{cells('1015', '', '', 'New York', '', '', '', '', '', '3.5', '3.5', '0')}</tr>"
        "</table>"
    )
    with patch("web.sbr_odds._translate_name", side_effect=lambda _s, name, _t: name):
        with patch(
            "web.sbr_odds.normalize_team_key",
            side_effect=lambda _s, name: name.lower()[:3],
        ):
            rows = _rows_from_html_table("nba", 2023, html, {})
    assert len(rows) == 1
    assert rows[0]["home_spread_odds"] is None
    assert rows[0]["away_spread_odds"] is None
    assert rows[0]["home_close_ml"] == 100  # EVEN 0 → +100

    archive_payload = (
        '[{"date":"20220115","home_team":"bos","away_team":"nyk",'
        '"home_close_ml":0,"away_close_ml":-110,'
        '"home_close_spread":0,"away_close_spread":0}]'
    )
    with patch("web.sbr_odds._fetch_text", return_value=archive_payload):
        with patch(
            "web.sbr_odds.normalize_team_key",
            side_effect=lambda _s, name: str(name).lower()[:3],
        ):
            with patch.dict(
                "web.sbr_odds.SBR_ARCHIVE_URLS", {"nba": "http://example"}, clear=False
            ):
                archive = fetch_sbr_archive_rows("nba")
    assert len(archive) == 1
    assert archive[0]["home_close_ml"] == 100
    assert archive[0]["home_close_spread"] == 0.0
    assert archive[0]["home_spread_odds"] is None
    assert archive[0]["away_spread_odds"] is None


def test_sbr_archive_parsers_keep_even_ml_and_pickem_spread() -> None:
    """Archive JSON path must match HTML scrapers: EVEN ML → +100, pick'em → 0.0."""
    assert _parse_optional_int(0) == 100
    assert _parse_optional_int("0") == 100
    assert _parse_optional_int(-110) == -110
    assert _parse_optional_int("") is None
    assert _parse_optional_int(None) is None

    assert _parse_optional_float(0) == 0.0
    assert _parse_optional_float("0") == 0.0
    assert _parse_optional_float(-3.5) == -3.5
    assert _parse_optional_float("") is None
    assert _parse_optional_float(None) is None
