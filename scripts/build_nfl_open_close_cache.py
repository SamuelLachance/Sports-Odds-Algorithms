"""Extract NFL open/close lines from market-movement DuckDB + merge into nfl.csv."""
from __future__ import annotations

import csv
from pathlib import Path

import duckdb
import pandas as pd

from web.nflverse_odds import rows_from_nflverse_csv
from web.sbr_odds import fetch_sbr_archive_rows, nfl_team_key

ROOT = Path(__file__).resolve().parents[1]
DUCK = ROOT / "data" / "supplemental" / "nfl-opens" / "nfl_odds.duckdb"
OUT_OPENS = ROOT / "data" / "supplemental" / "nfl-opens" / "opens_2025.csv"
NFL_CSV = ROOT / "data" / "supplemental" / "closing-odds" / "nfl.csv"
NFLVERSE = ROOT / "data" / "supplemental" / "closing-odds" / "nflverse_games.csv"

# Prefer sharp-leaning books when averaging open/close.
PREFERRED_BOOKS = ("pinnacle", "circa", "betonlineag", "lowvig", "draftkings", "fanduel")


def _extract_2025_opens() -> pd.DataFrame:
    con = duckdb.connect(str(DUCK), read_only=True)
    # One row per game/market/outcome: average open/close across preferred books,
    # falling back to all books.
    spreads = con.execute(
        """
        WITH ranked AS (
          SELECT *,
            CASE
              WHEN lower(sportsbook) IN ('pinnacle','circa','betonlineag','lowvig') THEN 1
              WHEN lower(sportsbook) IN ('draftkings','fanduel','betmgm') THEN 2
              ELSE 3
            END AS pref
          FROM fct_game_summary
          WHERE market_type = 'spreads'
        ),
        best AS (
          SELECT event_id, home_team, away_team, game_start_time, outcome,
                 min(pref) AS best_pref
          FROM ranked
          GROUP BY 1,2,3,4,5
        )
        SELECT r.event_id, r.home_team, r.away_team, r.game_start_time,
               r.outcome,
               avg(r.opening_line) AS open_line,
               avg(r.closing_line) AS close_line
        FROM ranked r
        JOIN best b
          ON r.event_id = b.event_id
         AND r.outcome = b.outcome
         AND r.pref = b.best_pref
        GROUP BY 1,2,3,4,5
        """
    ).df()
    totals = con.execute(
        """
        WITH ranked AS (
          SELECT *,
            CASE
              WHEN lower(sportsbook) IN ('pinnacle','circa','betonlineag','lowvig') THEN 1
              WHEN lower(sportsbook) IN ('draftkings','fanduel','betmgm') THEN 2
              ELSE 3
            END AS pref
          FROM fct_game_summary
          WHERE market_type = 'totals' AND lower(outcome) = 'over'
        ),
        best AS (
          SELECT event_id, min(pref) AS best_pref
          FROM ranked
          GROUP BY 1
        )
        SELECT r.event_id,
               avg(r.opening_line) AS open_total,
               avg(r.closing_line) AS close_total
        FROM ranked r
        JOIN best b ON r.event_id = b.event_id AND r.pref = b.best_pref
        GROUP BY 1
        """
    ).df()
    con.close()

    # Pivot home/away spreads to one row per game.
    rows: list[dict] = []
    for event_id, grp in spreads.groupby("event_id"):
        home_name = str(grp["home_team"].iloc[0])
        away_name = str(grp["away_team"].iloc[0])
        home_key = nfl_team_key(home_name)
        away_key = nfl_team_key(away_name)
        if not home_key or not away_key:
            continue
        # Odds API commence times are UTC; NFL slate keys use local US game day.
        start = pd.to_datetime(grp["game_start_time"].iloc[0], utc=True)
        day = start.tz_convert("America/New_York").strftime("%Y-%m-%d")
        home_open = home_close = away_open = away_close = None
        for _, r in grp.iterrows():
            outcome = str(r["outcome"])
            if outcome == home_name:
                home_open = float(r["open_line"]) if pd.notna(r["open_line"]) else None
                home_close = float(r["close_line"]) if pd.notna(r["close_line"]) else None
            elif outcome == away_name:
                away_open = float(r["open_line"]) if pd.notna(r["open_line"]) else None
                away_close = float(r["close_line"]) if pd.notna(r["close_line"]) else None
        tot = totals.loc[totals["event_id"] == event_id]
        open_total = float(tot["open_total"].iloc[0]) if len(tot) else None
        close_total = float(tot["close_total"].iloc[0]) if len(tot) else None
        rows.append(
            {
                "date": day,
                "home_key": home_key,
                "away_key": away_key,
                "home_open_spread": home_open,
                "away_open_spread": away_open,
                "home_close_spread": home_close,
                "away_close_spread": away_close,
                "open_total": open_total,
                "close_total": close_total,
                "source": "oddsapi-snapshots-2025",
            }
        )
    frame = pd.DataFrame(rows)
    OUT_OPENS.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT_OPENS, index=False)
    print(f"wrote {len(frame)} open/close rows -> {OUT_OPENS}")
    return frame


def _row_key(row: dict) -> tuple[str, str, str]:
    return (
        str(row.get("date") or "")[:10],
        str(row.get("home_key") or "").lower(),
        str(row.get("away_key") or "").lower(),
    )


def _merge_cache(opens_2025: pd.DataFrame) -> int:
    # Start from SBR archive (open+close through early 2022).
    rows = list(fetch_sbr_archive_rows("nfl"))
    by_key: dict[tuple[str, str, str], dict] = {_row_key(r): dict(r) for r in rows}

    # Overlay nflverse closes (full history) without inventing opens.
    if NFLVERSE.is_file():
        nv = rows_from_nflverse_csv(NFLVERSE)
        for r in nv:
            key = _row_key(r)
            cur = by_key.get(key, {
                "date": key[0],
                "home_key": key[1],
                "away_key": key[2],
                "source": "nflverse",
            })
            for field in (
                "home_close_ml",
                "away_close_ml",
                "home_close_spread",
                "away_close_spread",
                "home_spread_odds",
                "away_spread_odds",
            ):
                val = r.get(field)
                if val is None or val == "":
                    continue
                cur[field] = val
            # total_line from nflverse if present on source file
            cur.setdefault("source", r.get("source") or "nflverse")
            if cur.get("source") == "sbr-archive":
                pass
            elif "sbr" not in str(cur.get("source", "")):
                cur["source"] = "nflverse" if "open" not in str(cur.get("source", "")) else cur["source"]
            by_key[key] = cur

        # Attach total_line from raw nflverse CSV.
        games = pd.read_csv(NFLVERSE)
        for _, g in games.iterrows():
            home = nfl_team_key(g.get("home_team", ""))
            away = nfl_team_key(g.get("away_team", ""))
            day = str(g.get("gameday") or "")[:10]
            if not home or not away or not day:
                continue
            key = (day, home, away)
            cur = by_key.get(key)
            if cur is None:
                continue
            total = g.get("total_line")
            if pd.notna(total) and cur.get("close_total") in (None, ""):
                cur["close_total"] = float(total)

    # Overlay 2025 Odds API snapshot opens/closes.
    for _, r in opens_2025.iterrows():
        key = (str(r["date"])[:10], str(r["home_key"]).lower(), str(r["away_key"]).lower())
        cur = by_key.get(key, {
            "date": key[0],
            "home_key": key[1],
            "away_key": key[2],
        })
        for field in (
            "home_open_spread",
            "away_open_spread",
            "home_close_spread",
            "away_close_spread",
            "open_total",
            "close_total",
        ):
            val = r.get(field)
            if val is None or (isinstance(val, float) and pd.isna(val)):
                continue
            cur[field] = float(val)
        cur["source"] = "oddsapi-snapshots-2025"
        by_key[key] = cur

    fieldnames = [
        "date",
        "home_key",
        "away_key",
        "home_close_ml",
        "away_close_ml",
        "home_close_spread",
        "away_close_spread",
        "home_open_ml",
        "away_open_ml",
        "home_open_spread",
        "away_open_spread",
        "home_spread_odds",
        "away_spread_odds",
        "open_total",
        "close_total",
        "source",
    ]
    NFL_CSV.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(by_key.values(), key=lambda r: (r["date"], r["home_key"], r["away_key"]))
    with NFL_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in ordered:
            writer.writerow({k: row.get(k) for k in fieldnames})
    print(f"wrote {len(ordered)} rows -> {NFL_CSV}")
    return len(ordered)


def main() -> None:
    opens = _extract_2025_opens()
    n = _merge_cache(opens)
    # coverage summary
    df = pd.read_csv(NFL_CSV)
    df["year"] = df["date"].astype(str).str[:4]
    print(
        df.groupby("year")
        .agg(
            n=("date", "size"),
            open_spread=("home_open_spread", lambda s: s.notna().mean()),
            close_spread=("home_close_spread", lambda s: s.notna().mean()),
        )
        .tail(16)
        .round(3)
        .to_string()
    )
    print("total", n)


if __name__ == "__main__":
    main()
