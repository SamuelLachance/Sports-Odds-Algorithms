"""The two NFL in-season guards, extracted so they can be TESTED.

Both fire for the first time on the day the 2026 season starts — the worst
possible moment to discover either is wrong — and both were previously inline
expressions in `nfl_season_serve.py` with no coverage.

1. `apply_2026_boundary` — the serve applies a manual season-boundary transform
   to every feature walk (Elo regression, EPA decay, QB new-season, ...). Those
   transforms assume the walk ended on a 2025 game. Once a 2026 final lands in
   the spine, the walk's OWN in-loop season-change trigger fires instead, and
   applying the manual transform on top would DOUBLE-REGRESS every rating. The
   failure is silent: no crash, just quietly wrong probabilities all season.

2. `check_v7_npy` — column 7 of the feature matrix is loaded from a frozen
   .npy whose writer only exists in `nfl_v7_feature_gen.py`. It must have one
   row per game in the spine. The instant a 2026 final is appended the lengths
   diverge, and the serve must stop with an ACTIONABLE message rather than an
   opaque shape error — the dry run showed a bare assert here is useless at 6am
   on a game day.

Kept dependency-free (no numpy) so the guards are testable in isolation.
"""
from __future__ import annotations

V7_GEN = "python phase0/nfl_v7_feature_gen.py"


def apply_2026_boundary(last_walked_season: int, serve_season: int = 2026) -> bool:
    """True while the manual boundary transform is still the right thing to do.

    The walk has not yet crossed into the serve season, so the transform has not
    already been applied by the walk itself.
    """
    return last_walked_season < serve_season


def v7_npy_error(n_npy: int, n_games: int, gen_cmd: str = V7_GEN) -> str | None:
    """None when the frozen feature column matches the spine, else the message.

    Both directions are named explicitly because they mean different things and
    have different fixes: SHORTER means new finals landed and the column must be
    regenerated; LONGER means the spine went backwards, which is a data problem,
    not a regeneration problem.
    """
    if n_npy == n_games:
        return None
    cause = ("(new finals landed in data/nfl_games.csv)" if n_npy < n_games
             else "(npy has MORE rows than the spine — was data/nfl_games.csv "
                  "rolled back?)")
    return (f"data/nfl_v7_feature.npy is stale: {n_npy} rows vs {n_games} games "
            f"in the spine {cause}. Regenerate it: {gen_cmd}")
