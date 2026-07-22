"""mlbwp — a market-independent MLB win-probability model.

Team Elo plus a FIP-based starting-pitcher adjustment. No odds, no line movement,
no market-derived input anywhere in this package — that boundary is enforced by
tests/test_market_independence.py, not merely intended.

The model and its measured edge over a plain team rating are documented in
mlbwp/artifacts/metrics.json.
"""

MODEL_NAME = "fip-pitcher-elo"
MODEL_VERSION = "1.0"
