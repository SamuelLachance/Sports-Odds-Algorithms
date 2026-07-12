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


def test_mlb_datestr_uses_calendar_season_year() -> None:
    """MLB season year == calendar year; early months must not roll to season+1."""
    from web.sbr_odds import _make_datestr

    assert _make_datestr("0215", 2024, start=1, yr_end=12) == "2024-02-15"
    assert _make_datestr("0415", 2024, start=1, yr_end=12) == "2024-04-15"
    assert _make_datestr("1101", 2024, start=1, yr_end=12) == "2024-11-01"


def test_xlsx_rows_honors_sparse_column_refs() -> None:
    """OpenXML omits empty cells; column ``r`` must pad so Q → index 16."""
    import io
    import zipfile

    from web.sbr_odds import _xlsx_rows

    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    ss = f"""<?xml version="1.0"?>
<sst xmlns="{ns}" count="2" uniqueCount="2">
  <si><t>0415</t></si>
  <si><t>Boston</t></si>
</sst>"""
    sheet = f"""<?xml version="1.0"?>
<worksheet xmlns="{ns}">
  <sheetData>
    <row r="1">
      <c r="A1" t="s"><v>0</v></c>
      <c r="D1" t="s"><v>1</v></c>
      <c r="Q1"><v>-150</v></c>
    </row>
  </sheetData>
</worksheet>"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("xl/sharedStrings.xml", ss)
        zf.writestr("xl/worksheets/sheet1.xml", sheet)
        zf.writestr("[Content_Types].xml", "<Types></Types>")
    rows = _xlsx_rows(buf.getvalue())
    assert len(rows) == 1
    assert rows[0][0] == "0415"
    assert rows[0][3] == "Boston"
    assert rows[0][16] == "-150"



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


def test_sbr_html_open_favorite_flip_follows_close_ml() -> None:
    """When the favorite flips after open, close ML must win over open chalk.

    Open home chalk + same-sign close dump previously mirrored from open and
    skipped ``_repair_same_sign_spreads``, keeping the wrong side as favorite.
    """
    from unittest.mock import patch

    from web.sbr_odds import _rows_from_html_table

    cells = lambda *vals: "".join(f"<td>{v}</td>" for v in vals)
    # Open: home −7 / away +7. Close dump: both −3.5. Close ML: away −150.
    html = (
        "<table>"
        f"<tr>{cells(*([f'h{i}' for i in range(12)]))}</tr>"
        f"<tr>{cells(*(['sub'] * 12))}</tr>"
        f"<tr>{cells('1015', '', '', 'Boston', '', '', '', '', '', '7', '-3.5', '-150')}</tr>"
        f"<tr>{cells('1015', '', '', 'Montreal', '', '', '', '', '', '-7', '-3.5', '130')}</tr>"
        "</table>"
    )
    with patch("web.sbr_odds._translate_name", side_effect=lambda _s, name, _t: name):
        with patch(
            "web.sbr_odds.normalize_team_key",
            side_effect=lambda _s, name: name.lower()[:3],
        ):
            rows = _rows_from_html_table("nba", 2023, html, {})
    assert len(rows) == 1
    assert rows[0]["away_close_ml"] == -150
    assert rows[0]["home_close_ml"] == 130
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
    # Home is ML favorite (−150); both rows dump −1.5.
    html = (
        "<table>"
        f"<tr>{cells(*([f'h{i}' for i in range(12)]))}</tr>"
        f"<tr>{cells(*(['sub'] * 12))}</tr>"
        f"<tr>{cells('1015', '', '', 'Boston', '', '', '', '', '', '130', '-1.5', '-110')}</tr>"
        f"<tr>{cells('1015', '', '', 'Montreal', '', '', '', '', '', '-150', '-1.5', '-110')}</tr>"
        "</table>"
    )
    with patch("web.sbr_odds._translate_name", side_effect=lambda _s, name, _t: name):
        with patch(
            "web.sbr_odds.normalize_team_key",
            side_effect=lambda _s, name: name.lower()[:3],
        ):
            rows = _rows_from_nhl_html("nhl", 2023, html, {})
    assert len(rows) == 1
    assert rows[0]["home_close_ml"] == -150
    assert rows[0]["away_close_ml"] == 130
    assert rows[0]["home_close_spread"] == -1.5
    assert rows[0]["away_close_spread"] == 1.5


def test_nhl_html_same_sign_uses_away_ml_favorite() -> None:
    """Same-sign puck dump must assign −line to the away ML favorite, not always home."""
    from unittest.mock import patch

    from web.sbr_odds import _rows_from_nhl_html

    cells = lambda *vals: "".join(f"<td>{v}</td>" for v in vals)
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
    assert rows[0]["away_close_ml"] == -150
    assert rows[0]["home_close_ml"] == 130
    assert rows[0]["home_close_spread"] == 1.5
    assert rows[0]["away_close_spread"] == -1.5


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


def test_mlb_xlsx_same_sign_uses_away_ml_favorite() -> None:
    """Same-sign run-line dump must assign −line to the away ML favorite."""
    from unittest.mock import patch

    from web.sbr_odds import fetch_sbr_season_rows

    header = [f"h{i}" for i in range(19)]
    sub = [f"s{i}" for i in range(19)]
    away = [""] * 19
    home = [""] * 19
    away[0], home[0] = "0415", "0415"
    away[3], home[3] = "Boston", "New York"
    away[16], home[16] = "-150", "130"  # away chalk
    away[17], home[17] = "-1.5", "-1.5"
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
    assert rows[0]["away_close_ml"] == -150
    assert rows[0]["home_close_spread"] == 1.5
    assert rows[0]["away_close_spread"] == -1.5


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


def test_repair_same_sign_handles_pk_mismatch_and_asymmetric_magnitude() -> None:
    from web.sbr_odds import _repair_same_sign_spreads

    # One-sided pick'em: non-zero dump sign identifies the favorite when MLs tie.
    assert _repair_same_sign_spreads(0.0, -3.5, home_ml=-110, away_ml=-110) == (
        3.5,
        -3.5,
    )
    assert _repair_same_sign_spreads(-3.5, 0.0, home_ml=-150, away_ml=130) == (
        -3.5,
        3.5,
    )
    # Same-sign dump with unequal abs → prefer the larger magnitude; ML picks favorite.
    assert _repair_same_sign_spreads(-7.0, -3.5, home_ml=-150, away_ml=130) == (
        -7.0,
        7.0,
    )
    assert _repair_same_sign_spreads(-7.0, -3.5, home_ml=130, away_ml=-150) == (
        7.0,
        -7.0,
    )
    # Both PK / already opposite stay unchanged.
    assert _repair_same_sign_spreads(0.0, 0.0) == (0.0, 0.0)
    assert _repair_same_sign_spreads(-3.5, 3.5) == (-3.5, 3.5)


def test_dedupe_keeps_conflicting_mlb_doubleheader_rows() -> None:
    """Conflicting same-day rows must both survive so closing_odds_db fails closed."""
    from web.sbr_odds import _dedupe_odds_rows

    rows = [
        {
            "date": "2024-07-04",
            "home_key": "nyy",
            "away_key": "bos",
            "home_close_ml": -150,
            "away_close_ml": 130,
            "home_close_spread": -1.5,
            "away_close_spread": 1.5,
            "home_spread_odds": -110,
            "away_spread_odds": -110,
            "source": "sbr-online",
        },
        {
            "date": "2024-07-04",
            "home_key": "nyy",
            "away_key": "bos",
            "home_close_ml": -120,
            "away_close_ml": 100,
            "home_close_spread": -1.5,
            "away_close_spread": 1.5,
            "home_spread_odds": -115,
            "away_spread_odds": -105,
            "source": "sbr-online",
        },
    ]
    out = _dedupe_odds_rows(rows)
    assert len(out) == 2


def test_dedupe_prefers_online_over_archive_identical_key() -> None:
    from web.sbr_odds import _dedupe_odds_rows

    rows = [
        {
            "date": "2024-07-04",
            "home_key": "nyy",
            "away_key": "bos",
            "home_close_ml": -150,
            "away_close_ml": 130,
            "home_close_spread": -1.5,
            "away_close_spread": 1.5,
            "home_spread_odds": None,
            "away_spread_odds": None,
            "source": "sbr-archive",
        },
        {
            "date": "2024-07-04",
            "home_key": "nyy",
            "away_key": "bos",
            "home_close_ml": -150,
            "away_close_ml": 130,
            "home_close_spread": -1.5,
            "away_close_spread": 1.5,
            "home_spread_odds": -110,
            "away_spread_odds": -110,
            "source": "sbr-online",
        },
    ]
    out = _dedupe_odds_rows(rows)
    assert len(out) == 1
    assert out[0]["source"] == "sbr-online"
    assert out[0]["home_spread_odds"] == -110


def test_archive_rows_repair_same_sign_spreads() -> None:
    """Archive JSON path must mirror same-sign dumps like HTML/xlsx scrapers."""
    from unittest.mock import patch

    from web.sbr_odds import fetch_sbr_archive_rows

    payload = [
        {
            "date": "2019-01-15",
            "home_team": "Boston",
            "away_team": "New York",
            "home_close_ml": -150,
            "away_close_ml": 130,
            "home_close_spread": -1.5,
            "away_close_spread": -1.5,
        }
    ]
    with patch("web.sbr_odds._fetch_text", return_value=__import__("json").dumps(payload)):
        with patch(
            "web.sbr_odds.normalize_team_key",
            side_effect=lambda _s, name: name.lower()[:3],
        ):
            rows = fetch_sbr_archive_rows("nhl")
    assert len(rows) == 1
    assert rows[0]["home_close_spread"] == -1.5
    assert rows[0]["away_close_spread"] == 1.5
