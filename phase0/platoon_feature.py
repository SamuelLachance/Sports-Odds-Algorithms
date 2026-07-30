"""Per-game platoon feature: how much a lineup's handedness helps against the
opposing starter, BEYOND its overall quality.

Our TrueSkill/FIP ratings are handedness-agnostic. This isolates the extra bit:
for each batter, the gap between his on-base rate vs the starter's hand and his
overall on-base rate (both leak-safe EWMA, as-of the game). Averaged over the
lineup and differenced home-vs-away, that is the pure platoon signal orthogonal
to the lineup-quality feature the model already carries.

Switch hitters bat opposite the pitcher, so their "vs-hand" split is taken from
the side they would actually hit from.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict

from trueskill_pa import load_lineups, load_pa_by_game, load_starters

DECAY = 0.994          # ~a season and a half of PAs of memory
PRIOR = 60.0           # empirical-Bayes prior weight (PA) toward the player's own overall


def load_hand():
    h = json.load(open("data/retro_events/handedness.json"))
    return h["bat"], h["throw"]


def eff_side(bat: str, pit_hand: str) -> str:
    """Effective batter side; switch hitters bat opposite the pitcher."""
    if bat == "B":
        return "L" if pit_hand == "R" else "R"
    return bat if bat in ("L", "R") else "R"


class OB:
    """Per-batter on-base EWMA: overall and split by opposing pitcher hand."""

    def __init__(self):
        self.all = defaultdict(lambda: [0.0, 0.0])           # id -> [ob, n] decayed
        self.vs = {"L": defaultdict(lambda: [0.0, 0.0]),
                   "R": defaultdict(lambda: [0.0, 0.0])}

    def rate_all(self, b):
        s = self.all[b]
        return s[0] / s[1] if s[1] > 0 else None

    def adj(self, b, hand):
        """on-base vs `hand` minus overall, empirical-Bayes shrunk to 0."""
        base = self.rate_all(b)
        if base is None:
            return 0.0
        s = self.vs[hand][b]
        if s[1] <= 0:
            return 0.0
        vs = s[0] / s[1]
        return (s[1] / (s[1] + PRIOR)) * (vs - base)

    def update(self, b, hand, on_base):
        for store in (self.all[b], self.vs[hand][b]):
            store[0] = DECAY * store[0] + on_base
            store[1] = DECAY * store[1] + 1.0


def build(out="data/retro_events/platoon_feature.csv"):
    games = load_pa_by_game()
    lineups = load_lineups()
    starters = load_starters()
    bat_hand, throw_hand = load_hand()
    order = sorted(games, key=lambda g: (games[g]["date"], g))

    ob = OB()
    rows = []
    for gid in order:
        g = games[gid]
        ln = lineups.get(gid)
        sp = starters.get(gid)   # (home_sp, vis_sp)
        if ln and sp and sp[0] and sp[1]:
            h_hand = throw_hand.get(sp[0])   # home starter throws
            a_hand = throw_hand.get(sp[1])   # away starter throws
            home_b, away_b = ln[1], ln[0]
            if h_hand and a_hand and home_b and away_b:
                # home lineup faces the AWAY starter (hand a_hand)
                home_p = sum(ob.adj(b, a_hand) for b in home_b) / len(home_b)
                away_p = sum(ob.adj(b, h_hand) for b in away_b) / len(away_b)
                rows.append([gid, g["date"], g["season"], round(home_p - away_p, 5)])
        # update from this game's PAs
        for batter, pitcher, on_base in g["pa"]:
            ph = throw_hand.get(pitcher)
            if ph:
                ob.update(batter, eff_side(bat_hand.get(batter, "R"), ph), on_base)

    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["game_id", "date", "season", "platoon_edge"])
        w.writerows(rows)
    import statistics
    e = [r[3] for r in rows]
    print(f"wrote {out}: {len(rows):,} games | platoon_edge mean {statistics.mean(e):+.5f} "
          f"sd {statistics.pstdev(e):.5f}")


if __name__ == "__main__":
    build()
