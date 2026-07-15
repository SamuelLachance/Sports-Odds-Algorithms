"""Hubáček-style betting system (IJF 2019, archived in documents/).

Implements the paper's three ingredients on top of the hybrid_v2 pipeline:
decorrelated learning objective (objective.py), Markowitz max-Sharpe bet
allocation per round (portfolio.py), and round-based profit evaluation with
confidence thresholding + quadrant telemetry (eval.py).
"""
