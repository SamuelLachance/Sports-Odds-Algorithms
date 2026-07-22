"""Per-PA TrueSkill ratings for batters (offense) and pitchers (run prevention),
and the per-game lineup-vs-starter feature that layers onto the game model.

Every plate appearance is a 1v1 match: batter vs pitcher, one binary outcome
(reached base = batter wins, retired = pitcher wins). Both live on one latent
scale; P(on-base) = Phi((mu_batter - mu_pitcher)/c). The batter-vs-pitcher base
rate is asymmetric (~0.331), but that is a constant that cancels in the per-game
DIFFERENTIAL we feed the model, so no offset is needed — only the relative
spread of ratings matters, and that is what separates good hitters/pitchers.

Ratings are walked in strict game-date order. For each game the feature is read
from ratings as they stand BEFORE that game's PAs are applied (leak-safe), then
the game's PAs update the ratings.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict

MU0, SIG0 = 25.0, 25.0 / 3.0
BETA = SIG0 / 2.0
TAU = SIG0 / 100.0
_SQRT2PI = math.sqrt(2 * math.pi)

F_DATE, F_HOME, F_VIS, F_VIS_SP, F_HOME_SP = 0, 6, 3, 101, 103


def _pdf(x): return math.exp(-0.5 * x * x) / _SQRT2PI
def _cdf(x): return 0.5 * (1 + math.erf(x / math.sqrt(2)))


class TrueSkill1v1:
    """Closed-form two-player TrueSkill. Winner/loser passed each update."""

    def __init__(self, beta=BETA, tau=TAU):
        self.beta, self.tau = beta, tau
        self.mu = defaultdict(lambda: MU0)
        self.var = defaultdict(lambda: SIG0 * SIG0)
        self.n = defaultdict(int)

    def update(self, winner, loser):
        # skill drift before absorbing evidence
        vw = self.var[winner] + self.tau ** 2
        vl = self.var[loser] + self.tau ** 2
        c2 = 2 * self.beta ** 2 + vw + vl
        c = math.sqrt(c2)
        t = (self.mu[winner] - self.mu[loser]) / c
        denom = _cdf(t)
        v = _pdf(t) / denom if denom > 1e-12 else -t
        w = v * (v + t)
        w = min(max(w, 0.0), 1.0)
        self.mu[winner] += vw / c * v
        self.mu[loser] -= vl / c * v
        self.var[winner] = vw * (1 - vw / c2 * w)
        self.var[loser] = vl * (1 - vl / c2 * w)
        self.n[winner] += 1
        self.n[loser] += 1


def load_pa_by_game():
    games = {}
    for r in csv.reader(open("data/retro_events/pa.csv")):
        if r[0] == "game_id":
            continue
        gid = r[0]
        g = games.get(gid)
        if g is None:
            g = games[gid] = {"date": r[1], "season": int(r[2]), "pa": []}
        g["pa"].append((r[4], r[5], int(r[6])))   # batter, pitcher, on_base
    return games


def load_lineups():
    ln = defaultdict(lambda: {0: [], 1: []})
    for r in csv.reader(open("data/retro_events/lineups.csv")):
        if r[0] == "game_id":
            continue
        ln[r[0]][int(r[1])].append(r[3])          # side -> [batters]
    return ln


def load_starters():
    """game_id -> (home_sp, vis_sp) from the game logs."""
    import glob
    out = {}
    for path in sorted(glob.glob("data/retrosheet/gl*.txt")):
        for r in csv.reader(open(path, encoding="latin-1")):
            if len(r) < 105:
                continue
            gid = r[F_HOME] + r[F_DATE] + r[1]
            out[gid] = (r[F_HOME_SP].strip(), r[F_VIS_SP].strip())
    return out


def build_feature(out_path="data/retro_events/ts_feature.csv"):
    games = load_pa_by_game()
    lineups = load_lineups()
    starters = load_starters()
    order = sorted(games, key=lambda g: (games[g]["date"], g))
    print(f"games with PAs: {len(games):,}", flush=True)

    ts = TrueSkill1v1()
    rows = []
    for gid in order:
        g = games[gid]
        ln = lineups.get(gid)
        sp = starters.get(gid)
        if ln and sp:
            home_bat = [ts.mu[b] for b in ln[1]]
            away_bat = [ts.mu[b] for b in ln[0]]
            if home_bat and away_bat and sp[0] and sp[1]:
                home_off = sum(home_bat) / len(home_bat)
                away_off = sum(away_bat) / len(away_bat)
                home_def = ts.mu[sp[0]]     # home starting pitcher defense
                away_def = ts.mu[sp[1]]     # away starting pitcher
                # home offensive advantage: home bats vs away pitcher, minus away bats vs home pitcher
                edge = (home_off - away_def) - (away_off - home_def)
                rows.append([gid, g["date"], g["season"], round(edge, 4),
                             round(home_off - away_def, 4), round(away_off - home_def, 4)])
        # apply this game's PAs
        for batter, pitcher, on_base in g["pa"]:
            if on_base:
                ts.update(batter, pitcher)      # batter wins
            else:
                ts.update(pitcher, batter)      # pitcher wins

    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["game_id", "date", "season", "ts_edge", "home_matchup", "away_matchup"])
        w.writerows(rows)
    print(f"wrote {out_path}: {len(rows):,} games with a TrueSkill feature", flush=True)

    # face validity: top batters (offense) and pitchers (defense) with enough PAs
    import glob
    names = {}
    for path in glob.glob("data/retrosheet/gl*.txt"):
        for r in csv.reader(open(path, encoding="latin-1")):
            if len(r) >= 105:
                if r[101].strip():
                    names[r[101].strip()] = r[102]
                if r[103].strip():
                    names[r[103].strip()] = r[104]
    bat = sorted([(p, ts.mu[p]) for p in ts.mu if ts.n[p] > 1500], key=lambda x: -x[1])[:8]
    pit = sorted([(p, ts.mu[p]) for p in ts.mu if ts.n[p] > 1500], key=lambda x: -x[1])
    print("top offense ratings:", ", ".join(f"{p}={m:.1f}" for p, m in bat))
    return ts


if __name__ == "__main__":
    build_feature()
