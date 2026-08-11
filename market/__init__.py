"""Standalone market-comparison layer (odds + EV edge).

Deliberately OUTSIDE the mlbwp model package: mlbwp is provably market-blind (the
CI guard forbids any mlbwp file from importing this), so all odds handling lives
here and only ever compares the market against the model's already-computed output.
"""

# ---- policy constants, defined ONCE ----
# The EV badge threshold was previously re-typed in edges.py, nfl_edges.py and
# nhl_edges.py. Three copies of one policy number is three chances for the
# leagues to silently disagree while the site copy claims a single figure — the
# same drift class already fixed for the tier rule and the lean threshold.
# documents/pick_policy.md governs what the badge means; this governs when it fires.
#
# 0.08, not 0.20 (2026-08-11, from documents/bet_logic_verdict_2026_08_10.md):
# the walk-forward audit showed the model's CLV first covers the opening vig at
# ~+8% EV (net +0.06 pts — break-even, NOT profit), while >=20% is the worst
# winner's-curse cell in the menu (+10.66 pts of claimed-vs-delivered win rate):
# a higher gate selects precisely the games the model is most wrong about. The
# ledger therefore tracks the >=8% gate and nothing else — it accrues the CLV
# evidence at the only threshold with an open question left, and the badge
# remains a disagreement indicator, never a profit claim.
EV_THRESHOLD = 0.08
