"""Refresh the board's live EV edges + rebuild the site (fast, no model rebuild).

Run on a short cadence between full refreshes: pulls current odds, recomputes the
>=20%-EV value edges against the frozen opening consensus, and rebuilds index.html so
the badge appears/updates/disappears near-live. Needs ODDS_API_KEY in the environment.

After each league's attach_and_save, market/edge_ledger.py reads that league's freshly
written payload and accrues the LIVE EDGE LEDGER (data/edge_ledger.json): it records
every edge that fired (first record wins), rolls each recorded side's consensus price
forward while the game is strictly pre-game, and freezes CLV + result once it starts.
That ledger is the evidence stream for the CLV promotion gate; without it the badges
were recomputed and thrown away every cycle. It costs no extra Odds-API request — the
edge layers write the current both-sides consensus onto every pre-game card ("mkt")
from the fetch the badge already made, and the ledger rolls the close off that rather
than off the badge (the badge re-gates on the CURRENT model probability, so it can
vanish mid-week and would otherwise freeze a stale price as the "close"). No-ops
cleanly when a payload is missing or no edges fired (offseason).
"""

from __future__ import annotations

from market import edge_ledger, edges, nfl_edges, nhl_edges
from mlbwp_site import build_site


def main() -> int:
    n = edges.attach_and_save()
    print(f"[value] MLB: {n} games with >=20% EV vs opening consensus")
    m = nfl_edges.attach_and_save()
    print(f"[value] NFL: {m} games with >=20% EV vs opening consensus (7-day window)")
    k = nhl_edges.attach_and_save()
    print(f"[value] NHL: {k} games with >=20% EV vs opening consensus (7-day window)")

    for st in edge_ledger.update_all():
        print(f"[ledger] {st['league']}: {st['recorded']} recorded, "
              f"{st['touched']} touched, {st['closed']} CLV-graded, "
              f"{st['settled']} settled"
              + ("" if st["payload"] else " (no payload -> no-op)"))
    rep = edge_ledger.report()
    ov = rep["overall"]
    print(f"[ledger] total {ov['n_recorded']} recorded / {ov['n_clv_graded']} "
          f"CLV-graded / {ov['n_settled']} settled "
          f"({ov['n_clv_censored']} censored, {ov['n_stale_unsettled']} closed-unsettled)")

    build_site.build()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
