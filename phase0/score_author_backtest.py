"""Score the AUTHOR'S OWN published predictions, not a reimplementation.

mlb/analyze/mlb_Algo_V2_7-1-2009_10-1-2015_backtest_odds.csv contains the model's
actual output ("Algo proj") for ~7,500 games. Joining it to Retrosheet results gives
the most faithful possible backtest: it removes any doubt about whether my
transcription of algo.py is correct.

"Algo proj" is a SIGNED percentage: positive favours the away team, negative the home
team, magnitude = the favoured side's win percentage. Values exceed 100% in the
source data, which is a bug in the original; those rows are reported separately and
the probability is clipped, which is the most charitable reading available.
"""

from __future__ import annotations

import csv
import glob
from datetime import datetime

import numpy as np

JQ = r"C:\Users\Admin\AppData\Local\Temp\jq"
BACKTEST = JQ + r"\mlb\analyze\mlb_Algo_V2_7-1-2009_10-1-2015_backtest_odds.csv"
TEAMS = JQ + r"\mlb\mlb_teams.txt"
EPS = 1e-15


def load_slug_map() -> dict[str, str]:
    """slug -> retrosheet-ish abbrev, from the repo's own team file."""
    m = {}
    with open(TEAMS, encoding="latin-1") as fh:
        for line in fh:
            if "|" not in line:
                continue
            abbr, slug = (p.strip() for p in line.split("|", 1))
            m[slug] = abbr
    return m


# Retrosheet codes differ from the repo's abbreviations for several clubs.
RETRO = {
    "ari": "ARI", "atl": "ATL", "bal": "BAL", "bos": "BOS", "chc": "CHN",
    "chw": "CHA", "cin": "CIN", "cle": "CLE", "col": "COL", "det": "DET",
    "hou": "HOU", "kc": "KCA", "laa": "ANA", "lad": "LAN", "mia": "MIA",
    "mil": "MIL", "min": "MIN", "nym": "NYN", "nyy": "NYA", "oak": "OAK",
    "phi": "PHI", "pit": "PIT", "sd": "SDN", "sea": "SEA", "sf": "SFN",
    "stl": "SLN", "tb": "TBA", "tex": "TEX", "tor": "TOR", "wsh": "WAS",
    "fla": "FLO", "was": "WAS",
}

F_DATE, F_VIS, F_HOME, F_VIS_R, F_HOME_R = 0, 3, 6, 9, 10


def load_results() -> dict[tuple, float]:
    out = {}
    for path in sorted(glob.glob("data/retrosheet/gl*.txt")):
        with open(path, newline="", encoding="latin-1") as fh:
            for r in csv.reader(fh):
                if len(r) < 11:
                    continue
                try:
                    vr, hr = int(r[F_VIS_R]), int(r[F_HOME_R])
                except ValueError:
                    continue
                if vr == hr:
                    continue
                out[(r[F_DATE], r[F_VIS], r[F_HOME])] = 1.0 if hr > vr else 0.0
    return out


def main() -> None:
    slug = load_slug_map()
    results = load_results()
    print(f"retrosheet results loaded: {len(results):,}")

    rows = list(csv.reader(open(BACKTEST, encoding="latin-1")))[1:]
    print(f"author backtest rows: {len(rows):,}")

    p_home, ys = [], []
    matched = unmatched = bad = over100 = 0

    for r in rows:
        if len(r) < 4 or not r[3].strip():
            continue
        try:
            proj = float(r[3].replace("%", "").strip())
        except ValueError:
            bad += 1
            continue
        try:
            d = datetime.strptime(r[0].strip(), "%m-%d-%Y")
        except ValueError:
            bad += 1
            continue
        key_date = d.strftime("%Y%m%d")
        a_abbr = RETRO.get(slug.get(r[1].strip(), ""), None)
        h_abbr = RETRO.get(slug.get(r[2].strip(), ""), None)
        if not a_abbr or not h_abbr:
            unmatched += 1
            continue
        y = results.get((key_date, a_abbr, h_abbr))
        if y is None:
            unmatched += 1
            continue

        if abs(proj) > 100:
            over100 += 1
        # positive favours AWAY -> P(home) = 1 - p_fav ; negative favours HOME
        p_fav = min(abs(proj), 99.9) / 100.0
        ph = (1.0 - p_fav) if proj > 0 else p_fav
        p_home.append(ph)
        ys.append(y)
        matched += 1

    p = np.array(p_home)
    y = np.array(ys)
    print(f"matched: {matched:,}   unmatched: {unmatched:,}   unparseable: {bad:,}")
    print(f"rows with |proj| > 100% (impossible): {over100:,}")
    if not matched:
        print("no overlap — cannot score")
        return

    def ll(pp):
        pp = np.clip(pp, EPS, 1 - EPS)
        return float(-(y * np.log(pp) + (1 - y) * np.log(1 - pp)).mean())

    print(f"\nscored n={len(y):,}  home win rate={y.mean():.4f}")
    print(f"{'model':<42}{'log loss':>10}{'Brier':>9}{'acc':>8}")
    print("-" * 69)
    print(f"{'AUTHOR published Algo proj':<42}{ll(p):>10.5f}"
          f"{float(np.mean((p-y)**2)):>9.5f}{float(np.mean((p>0.5)==(y>0.5))):>8.4f}")
    hc = np.full_like(y, y.mean())
    print(f"{'constant home rate':<42}{ll(hc):>10.5f}"
          f"{float(np.mean((hc-y)**2)):>9.5f}{float(np.mean((hc>0.5)==(y>0.5))):>8.4f}")
    c5 = np.full_like(y, 0.5)
    print(f"{'constant 0.5':<42}{ll(c5):>10.5f}{0.25:>9.5f}{'':>8}")

    # How extreme are the published probabilities?
    print(f"\npublished P(home): mean={p.mean():.4f} sd={p.std():.4f} "
          f"min={p.min():.3f} max={p.max():.3f}")
    for lo, hi in [(0.0, 0.35), (0.35, 0.45), (0.45, 0.55), (0.55, 0.65), (0.65, 1.01)]:
        m = (p >= lo) & (p < hi)
        if m.sum():
            print(f"  P(home) in [{lo:.2f},{hi:.2f}): n={m.sum():>5}  "
                  f"actual home win rate={y[m].mean():.4f}  (model says ~{p[m].mean():.4f})")


if __name__ == "__main__":
    main()
