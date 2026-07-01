"""Tune SharpBaseball sport-layer meta weights for all baseball leagues."""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tune_sports_meta import _cutoff_today, tune_league  # noqa: E402
from web.league_profiles import LEAGUE_PROFILES  # noqa: E402
from web.sports_meta_model import META_WEIGHTS_PATH, load_sports_meta_config  # noqa: E402

BASEBALL_LEAGUES = tuple(
    key for key, profile in LEAGUE_PROFILES.items() if profile["category"] == "baseball"
)


def main() -> int:
    cutoff = _cutoff_today()
    payload = dict(load_sports_meta_config())
    payload.setdefault("_meta", {})

    for league in BASEBALL_LEAGUES:
        print(f"Tuning {league} (SharpBaseball)...", flush=True)
        entry = tune_league(league, cutoff)
        if entry:
            payload[league] = entry
            print(
                f"  games={entry.get('data_depth', {}).get('games_loaded')} "
                f"log_loss={entry.get('log_loss_tuned')} "
                f"blend={entry.get('blend')}",
                flush=True,
            )
        else:
            print(f"  skipped (insufficient data)", flush=True)

    payload["_meta"]["baseball_sharp_tuned_at"] = date.today().isoformat()
    META_WEIGHTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {META_WEIGHTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
