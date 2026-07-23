"""Refresh the board's live EV edges + rebuild the site (fast, no model rebuild).

Run on a short cadence between full refreshes: pulls current odds, recomputes the
>=15%-EV value edges against the frozen opening consensus, and rebuilds index.html so
the badge appears/updates/disappears near-live. Needs ODDS_API_KEY in the environment.
"""

from __future__ import annotations

from market import edges, nfl_edges
from mlbwp_site import build_site


def main() -> int:
    n = edges.attach_and_save()
    print(f"[value] MLB: {n} games with >=15% EV vs opening consensus")
    m = nfl_edges.attach_and_save()
    print(f"[value] NFL: {m} games with >=15% EV vs opening consensus (7-day window)")
    build_site.build()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
