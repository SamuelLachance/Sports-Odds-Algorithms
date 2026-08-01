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
EV_THRESHOLD = 0.20
