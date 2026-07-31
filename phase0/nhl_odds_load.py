"""Download SportsbookReviewsOnline NHL moneyline archives -> data/odds_nhl.csv.

Mirrors phase0/odds_load.py (MLB). The NHL archive is NOT published as .xlsx like
the MLB one -- the season pages under /scoresoddsarchives/nhl-odds-<slug> render
the whole season as a single HTML <table>, two rows per game (V then H) with
American Open and Close moneylines. We pair them, map SBR team names to the NHL
API codes used by data/nhl_games.csv, convert to decimal odds, and record the
final result.

This is EVALUATION data only (opening/closing lines) -- the model itself never
sees it; the market-blind guard is about model INPUTS, not backtests.

Source: sportsbookreviewsonline.com (free archive; NHL 2007-08 .. 2022-23; the
game spine only starts 2010-11 so that is where this loader starts).

Row layout differs by era but the first ten cells are stable:
  Date Rot VH Team 1st 2nd 3rd Final Open Close  [PuckLine PLprice] OpenOU ..
Only indices 0,2,3,7,8,9 are used.

Usage:  python -X utf8 phase0/nhl_odds_load.py
"""

from __future__ import annotations

import csv
import html
import os
import re
import urllib.request

BASE = "https://www.sportsbookreviewsonline.com/scoresoddsarchives/nhl-odds-{}"
OUT = "data/odds_nhl.csv"

# season slug -> (season_id, year of the Aug-Dec half, year of the Jan-Jul half)
SEASONS = [
    ("2010-11", 20102011, 2010, 2011),
    ("2011-12", 20112012, 2011, 2012),
    ("2012-13", 20122013, 2012, 2013),
    ("2013-14", 20132014, 2013, 2014),
    ("2014-15", 20142015, 2014, 2015),
    ("2015-16", 20152016, 2015, 2016),
    ("2016-17", 20162017, 2016, 2017),
    ("2017-18", 20172018, 2017, 2018),
    ("2018-19", 20182019, 2018, 2019),
    ("2019-20", 20192020, 2019, 2020),
    ("2021",    20202021, 2020, 2021),   # covid season, played Jan-Jul 2021
    ("2021-22", 20212022, 2021, 2022),
    ("2022-23", 20222023, 2022, 2023),
]

# SBR renders team names without spaces in most seasons and with spaces in a few
# rows; NAMEKEY strips everything non-alphanumeric and lowercases, so one entry
# covers both. Franchise moves are kept as SEPARATE keys because the spine uses
# the code that was current at the time (ATL->WPG 2011, PHX->ARI 2014).
SBR2NHL = {
    "anaheim": "ANA", "arizona": "ARI", "phoenix": "PHX", "atlanta": "ATL",
    "boston": "BOS", "buffalo": "BUF", "calgary": "CGY", "carolina": "CAR",
    "chicago": "CHI", "colorado": "COL", "columbus": "CBJ", "dallas": "DAL",
    "detroit": "DET", "edmonton": "EDM", "florida": "FLA", "losangeles": "LAK",
    "minnesota": "MIN", "montreal": "MTL", "nashville": "NSH",
    "newjersey": "NJD", "nyislanders": "NYI", "nyrangers": "NYR",
    "ottawa": "OTT", "philadelphia": "PHI", "pittsburgh": "PIT",
    "sanjose": "SJS", "stlouis": "STL", "tampabay": "TBL", "toronto": "TOR",
    "vancouver": "VAN", "vegas": "VGK", "washington": "WSH",
    "winnipeg": "WPG", "winnipegjets": "WPG",
    "seattle": "SEA", "seattlekraken": "SEA", "utah": "UTA",
}


def namekey(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def american_to_decimal(ml):
    try:
        ml = float(str(ml).replace("+", "").strip())
    except (TypeError, ValueError):
        return None
    if ml >= 100:
        return 1.0 + ml / 100.0
    if ml <= -100:
        return 1.0 + 100.0 / (-ml)
    return None                     # -99..99 is not a valid American price


def fetch(slug: str) -> str:
    req = urllib.request.Request(BASE.format(slug),
                                 headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=120).read().decode("utf-8", "replace")


def table_rows(page: str) -> list[list[str]]:
    out = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", page, re.S):
        tds = [html.unescape(re.sub("<.*?>", "", t)).strip()
               for t in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)]
        if len(tds) >= 10 and tds[0] != "Date":
            out.append(tds)
    return out


def load_season(slug: str, y_fall: int, y_spring: int) -> tuple[list[dict], dict]:
    rows = table_rows(fetch(slug))
    out = []
    bad = {"unpaired": 0, "bad_date": 0, "unknown_team": 0,
           "no_price": 0, "tie_or_no_final": 0}
    for i in range(0, len(rows) - 1, 2):
        v, h = rows[i], rows[i + 1]
        if v[2].strip().upper() != "V" or h[2].strip().upper() != "H":
            bad["unpaired"] += 1
            continue
        try:
            d = int(v[0])
            mm, dd = d // 100, d % 100
            assert 1 <= mm <= 12 and 1 <= dd <= 31
            yr = y_fall if mm >= 8 else y_spring
            date = f"{yr}-{mm:02d}-{dd:02d}"
        except (ValueError, AssertionError):
            bad["bad_date"] += 1
            continue
        away = SBR2NHL.get(namekey(v[3]))
        home = SBR2NHL.get(namekey(h[3]))
        if not away or not home:
            bad["unknown_team"] += 1
            continue
        ao, ho = american_to_decimal(v[8]), american_to_decimal(h[8])
        ac, hc = american_to_decimal(v[9]), american_to_decimal(h[9])
        try:
            af, hf = float(v[7]), float(h[7])
        except (TypeError, ValueError):
            bad["tie_or_no_final"] += 1
            continue
        if af == hf:
            bad["tie_or_no_final"] += 1
            continue
        if None in (ac, hc):
            bad["no_price"] += 1
            continue
        out.append({
            "date": date, "away": away, "home": home,
            "away_open": round(ao, 4) if ao else "",
            "home_open": round(ho, 4) if ho else "",
            "away_close": round(ac, 4), "home_close": round(hc, 4),
            "home_win": 1 if hf > af else 0,
        })
    return out, bad


def main():
    allrows = []
    for slug, _sid, yf, ys in SEASONS:
        try:
            r, bad = load_season(slug, yf, ys)
            allrows += r
            drop = ", ".join(f"{k}={v}" for k, v in bad.items() if v)
            print(f"{slug}: {len(r)} games" + (f"  [dropped {drop}]" if drop else ""),
                  flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"{slug}: skip ({type(e).__name__} {str(e)[:60]})", flush=True)
    allrows.sort(key=lambda r: (r["date"], r["home"], r["away"]))
    cols = ["date", "away", "home", "away_open", "home_open",
            "away_close", "home_close", "home_win"]
    tmp = OUT + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(allrows)
    os.replace(tmp, OUT)
    print(f"wrote {OUT}: {len(allrows)} games")
    import statistics
    orr = [1 / g["home_close"] + 1 / g["away_close"] for g in allrows]
    hw = statistics.mean(g["home_win"] for g in allrows)
    print(f"  mean closing overround: {statistics.mean(orr):.4f} "
          f"(vig ~{(statistics.mean(orr) - 1) * 100:.1f}%)")
    print(f"  home win rate: {hw:.4f}")


if __name__ == "__main__":
    main()
