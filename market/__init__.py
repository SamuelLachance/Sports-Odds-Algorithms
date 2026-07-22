"""Standalone market-comparison layer (odds + EV edge).

Deliberately OUTSIDE the mlbwp model package: mlbwp is provably market-blind (the
CI guard forbids any mlbwp file from importing this), so all odds handling lives
here and only ever compares the market against the model's already-computed output.
"""
