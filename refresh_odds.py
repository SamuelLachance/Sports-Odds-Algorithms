"""Refresh the board's live EV edges + rebuild the site (fast, no model rebuild).

Run on a short cadence between full refreshes: pulls current odds, recomputes the
>=20%-EV value edges against the frozen opening consensus, and rebuilds index.html so
the badge appears/updates/disappears near-live. Needs ODDS_API_KEY in the environment.
"""

from __future__ import annotations

from market import edges
from mlbwp_site import build_site


def main() -> int:
    n = edges.attach_and_save()
    print(f"[value] {n} games with >=20% EV vs opening consensus")
    build_site.build()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
