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
    assert _american_ml_cell("EVEN") == 100
    assert _american_ml_cell("even") == 100
    assert _american_ml_cell("-110") == -110
    assert _american_ml_cell("") is None
    assert _american_ml_cell("NL") is None


def test_sbr_american_parsers_reject_invalid_magnitude() -> None:
    """|odds| < 100 (except EVEN/0) must not enter closing tables."""
    assert _american_ml_cell("50") is None
    assert _american_ml_cell("-50") is None
    assert _spread_juice_cell("75") is None
    assert _spread_juice_cell("-75") is None
    assert _parse_optional_int(50) is None
    assert _parse_optional_int(-50) is None
    assert _parse_optional_int("EVEN") == 100


def test_nhl_covid_datestr_uses_calendar_2021() -> None:
    """2020-21 NHL COVID slate was played in 2021; start=1 must not stamp 2020."""
    from web.sbr_odds import _make_datestr

    # Callers pass season+1 (calendar year) when covid=True / start=1.
    assert _make_datestr("0115", 2021, start=1, yr_end=12) == "2021-01-15"
    assert _make_datestr("0210", 2021, start=1, yr_end=12) == "2021-02-10"
    assert _make_datestr("0515", 2021, start=1, yr_end=12) == "2021-05-15"
    # Regular NHL season still rolls Aug→next calendar year.
    assert _make_datestr("1015", 2023, start=8, yr_end=12) == "2023-10-15"
    assert _make_datestr("0115", 2023, start=8, yr_end=12) == "2024-01-15"


def test_nhl_covid_html_rows_stamp_2021_calendar() -> None:
    from unittest.mock import patch

    from web.sbr_odds import _rows_from_nhl_html

    cells = lambda *vals: "".join(f"<td>{v}</td>" for v in vals)
    # Need >= 12 cells for NHL juice column (index 11).
    html = (
        "<table>"
        f"<tr>{cells(*([f'h{i}' for i in range(12)]))}</tr>"
        f"<tr>{cells(*(['sub'] * 12))}</tr>"
        f"<tr>{cells('0115', '', '', 'Boston', '', '', '', '', '', '-110', '1.5', '-110')}</tr>"
        f"<tr>{cells('0115', '', '', 'New York', '', '', '', '', '', '100', '-1.5', '-110')}</tr>"
        "</table>"
    )
    with patch("web.sbr_odds._translate_name", side_effect=lambda _s, name, _t: name):
        with patch(
            "web.sbr_odds.normalize_team_key",
            side_effect=lambda _s, name: name.lower()[:3],
        ):
            rows = _rows_from_nhl_html("nhl", 2020, html, {}, covid=True)
    assert len(rows) == 1
    assert rows[0]["date"] == "2021-01-15"


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
    import math

    assert _spread_juice(0) == 100.0
    assert math.isnan(_spread_juice(None))
    assert _spread_juice(-105) == -105.0


def test_sbr_html_emits_opposite_sign_spreads() -> None:
    """Signed open/close cells must not write same-sign home/away closes."""
    from unittest.mock import patch

    from web.sbr_odds import _rows_from_html_table

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
    assert rows[0]["home_close_spread"] == 3.5
    assert rows[0]["away_close_spread"] == -3.5


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


def test_nhl_html_repairs_same_sign_puck_lines() -> None:
    """NHL HTML must mirror same-sign closes like NBA/NFL, not write both negative."""
    from unittest.mock import patch

    from web.sbr_odds import _rows_from_nhl_html

    cells = lambda *vals: "".join(f"<td>{v}</td>" for v in vals)
    # NHL layout: date, …, name@3, …, ML@9, spread@10, juice@11
    html = (
        "<table>"
        f"<tr>{cells(*([f'h{i}' for i in range(12)]))}</tr>"
        f"<tr>{cells(*(['sub'] * 12))}</tr>"
        f"<tr>{cells('1015', '', '', 'Boston', '', '', '', '', '', '-150', '-1.5', '-110')}</tr>"
        f"<tr>{cells('1015', '', '', 'Montreal', '', '', '', '', '', '130', '-1.5', '-110')}</tr>"
        "</table>"
    )
    with patch("web.sbr_odds._translate_name", side_effect=lambda _s, name, _t: name):
        with patch(
            "web.sbr_odds.normalize_team_key",
            side_effect=lambda _s, name: name.lower()[:3],
        ):
            rows = _rows_from_nhl_html("nhl", 2023, html, {})
    assert len(rows) == 1
    assert rows[0]["home_close_spread"] == -1.5
    assert rows[0]["away_close_spread"] == 1.5


def test_mlb_xlsx_repairs_same_sign_run_lines() -> None:
    """MLB xlsx must mirror same-sign closes like NHL HTML, not write both negative."""
    from unittest.mock import patch

    from web.sbr_odds import fetch_sbr_season_rows

    # Header + subheader + away + home; MLB uses cols 16=ML, 17=spread, 18=juice.
    # fetch_sbr_season_rows skips sheet_rows[0] then starts pairs at body index 1.
    header = [f"h{i}" for i in range(19)]
    sub = [f"s{i}" for i in range(19)]
    away = [""] * 19
    home = [""] * 19
    away[0], home[0] = "0415", "0415"
    away[3], home[3] = "Boston", "New York"
    away[16], home[16] = "130", "-150"
    away[17], home[17] = "-1.5", "-1.5"  # same-sign bug from SBR
    away[18], home[18] = "-110", "-110"
    sheet = [header, sub, away, home]

    with patch("web.sbr_odds._fetch_bytes", return_value=b"xlsx"):
        with patch("web.sbr_odds._xlsx_rows", return_value=sheet):
            with patch("web.sbr_odds._translate_name", side_effect=lambda _s, name, _t: name):
                with patch(
                    "web.sbr_odds.normalize_team_key",
                    side_effect=lambda _s, name: name.lower()[:3],
                ):
                    rows = fetch_sbr_season_rows("mlb", 2024, {})
    assert len(rows) == 1
    assert rows[0]["home_close_spread"] == -1.5
    assert rows[0]["away_close_spread"] == 1.5


def test_sbr_html_equal_opens_uses_ml_favorite() -> None:
    """SBR often duplicates the favorite line on both rows; ML breaks the tie."""
    from unittest.mock import patch

    from web.sbr_odds import _rows_from_html_table

    cells = lambda *vals: "".join(f"<td>{v}</td>" for v in vals)
    html = (
        "<table>"
        f"<tr>{cells(*([f'h{i}' for i in range(12)]))}</tr>"
        f"<tr>{cells(*(['sub'] * 12))}</tr>"
        f"<tr>{cells('1015', '', '', 'Boston', '', '', '', '', '', '-5.5', '-6', '-200')}</tr>"
        f"<tr>{cells('1015', '', '', 'New York', '', '', '', '', '', '-5.5', '-6', '170')}</tr>"
        "</table>"
    )
    with patch("web.sbr_odds._translate_name", side_effect=lambda _s, name, _t: name):
        with patch(
            "web.sbr_odds.normalize_team_key",
            side_effect=lambda _s, name: name.lower()[:3],
        ):
            rows = _rows_from_html_table("nba", 2023, html, {})
    assert len(rows) == 1
    # Away ML favorite → home gets the plus side.
    assert rows[0]["home_close_spread"] == 6.0
    assert rows[0]["away_close_spread"] == -6.0


def test_sbr_html_open_branch_mirrors_sparse_close() -> None:
    """Open-based favorite path must mirror a missing close side."""
    from unittest.mock import patch

    from web.sbr_odds import _rows_from_html_table

    cells = lambda *vals: "".join(f"<td>{v}</td>" for v in vals)
    html = (
        "<table>"
        f"<tr>{cells(*([f'h{i}' for i in range(12)]))}</tr>"
        f"<tr>{cells(*(['sub'] * 12))}</tr>"
        f"<tr>{cells('1015', '', '', 'Boston', '', '', '', '', '', '-5.5', '', '-200')}</tr>"
        f"<tr>{cells('1015', '', '', 'New York', '', '', '', '', '', '5.5', '5.5', '170')}</tr>"
        "</table>"
    )
    with patch("web.sbr_odds._translate_name", side_effect=lambda _s, name, _t: name):
        with patch(
            "web.sbr_odds.normalize_team_key",
            side_effect=lambda _s, name: name.lower()[:3],
        ):
            rows = _rows_from_html_table("nba", 2023, html, {})
    assert len(rows) == 1
    assert rows[0]["home_close_spread"] == 5.5
    assert rows[0]["away_close_spread"] == -5.5
