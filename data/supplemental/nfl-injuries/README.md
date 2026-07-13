# NFL injury snapshots

Daily ESPN injury dumps from `scripts/snapshot_espn_injuries.py`.

- `YYYY-MM-DD.json` — full detail
- `YYYY-MM-DD.csv` — flat team burden (`out`, `ol_out`, `skill_out`)

Joined point-in-time (strictly before game day) via `web/nfl_v2/injuries.py`.
Historical coverage starts only after snapshots exist — run daily in season.
