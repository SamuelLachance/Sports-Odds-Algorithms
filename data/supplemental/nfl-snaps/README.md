# NFL snap shares

Built by `scripts/build_nfl_snap_shares.py` from nflverse snap counts.

Columns: `wr1_snap_share`, `skill_snap_share`, `ol_starter_share`, `ol_starters`.
Attached onto games by `web/nfl_v2/snaps.py` (features use prior-game EWMA).
