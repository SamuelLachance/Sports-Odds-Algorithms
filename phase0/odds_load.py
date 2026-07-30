"""Download SportsbookReviewsOnline MLB moneyline archives -> data/odds_mlb.csv.

Two rows per game (V then H) with American-odds Open and Close moneylines. We pair
them, map SBR team codes to Retrosheet, convert to decimal odds, and record the
final result. This is EVALUATION data only (opening/closing lines) -- the model
itself never sees it; the market-blind guard is about model INPUTS, not backtests.
Source: sportsbookreviewsonline.com (free archive; 2010-2021 available).
"""

from __future__ import annotations

import csv
import io
import urllib.request

import pandas as pd

BASE = ("https://www.sportsbookreviewsonline.com/wp-content/uploads/"
        "sportsbookreviewsonline_com_737/mlb-odds-{}.xlsx")
OUT = "data/odds_mlb.csv"
YEARS = range(2010, 2022)

# SBR team code -> Retrosheet team code (covers naming variants across seasons).
SBR2RETRO = {
    "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BOS": "BOS", "BRS": "BOS",
    "CHC": "CHN", "CUB": "CHN", "CIN": "CIN", "CLE": "CLE", "COL": "COL",
    "CWS": "CHA", "DET": "DET", "HOU": "HOU", "KAN": "KCA", "LAA": "ANA",
    "LAD": "LAN", "LOS": "LAN", "MIA": "MIA", "FLA": "FLO", "MIL": "MIL",
    "MIN": "MIN", "NYM": "NYN", "NYY": "NYA", "OAK": "OAK", "PHI": "PHI",
    "PIT": "PIT", "SDG": "SDN", "SEA": "SEA", "SFG": "SFN", "SFO": "SFN",
    "STL": "SLN", "TAM": "TBA", "TB": "TBA", "TEX": "TEX", "TOR": "TOR",
    "WAS": "WAS", "ANA": "ANA", "KC": "KCA", "SD": "SDN", "SF": "SFN",
}


def american_to_decimal(ml):
    try:
        ml = float(ml)
    except (TypeError, ValueError):
        return None
    if ml >= 100:
        return 1.0 + ml / 100.0
    if ml <= -100:
        return 1.0 + 100.0 / (-ml)
    return None                     # -99..99 is not a valid American price


def load_year(yr: int) -> list[dict]:
    raw = urllib.request.urlopen(
        urllib.request.Request(BASE.format(yr), headers={"User-Agent": "Mozilla/5.0"}),
        timeout=60).read()
    df = pd.read_excel(io.BytesIO(raw))
    out = []
    rows = df.to_dict("records")
    for i in range(0, len(rows) - 1, 2):
        v, h = rows[i], rows[i + 1]
        if str(v.get("VH")).strip() != "V" or str(h.get("VH")).strip() != "H":
            continue                # not a clean V,H pair (skip malformed)
        try:
            d = int(v["Date"])
            date = f"{yr}-{d // 100:02d}-{d % 100:02d}"
        except (TypeError, ValueError):
            continue
        away = SBR2RETRO.get(str(v.get("Team")).strip().upper())
        home = SBR2RETRO.get(str(h.get("Team")).strip().upper())
        if not away or not home:
            continue
        ao, ho = american_to_decimal(v.get("Open")), american_to_decimal(h.get("Open"))
        ac, hc = american_to_decimal(v.get("Close")), american_to_decimal(h.get("Close"))
        try:
            af, hf = float(v.get("Final")), float(h.get("Final"))
        except (TypeError, ValueError):
            continue
        if af == hf or None in (ao, ho, ac, hc):
            continue                # tie/incomplete or missing a price
        out.append({
            "date": date, "away": away, "home": home,
            "away_open": round(ao, 4), "home_open": round(ho, 4),
            "away_close": round(ac, 4), "home_close": round(hc, 4),
            "home_win": 1 if hf > af else 0,
        })
    return out


def main():
    allrows = []
    for yr in YEARS:
        try:
            r = load_year(yr)
            allrows += r
            print(f"{yr}: {len(r)} games", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"{yr}: skip ({type(e).__name__} {str(e)[:40]})", flush=True)
    cols = ["date", "away", "home", "away_open", "home_open",
            "away_close", "home_close", "home_win"]
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(allrows)
    print(f"wrote {OUT}: {len(allrows)} games")
    # sanity: closing overround + favorite/home base rates
    import statistics
    orr = [1 / g["home_close"] + 1 / g["away_close"] for g in allrows]
    hw = statistics.mean(g["home_win"] for g in allrows)
    print(f"  mean closing overround: {statistics.mean(orr):.4f} (vig ~{(statistics.mean(orr)-1)*100:.1f}%)")
    print(f"  home win rate: {hw:.4f}")


if __name__ == "__main__":
    main()
