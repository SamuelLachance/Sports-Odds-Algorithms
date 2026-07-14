"""Prune data/models to the best OOS log-loss win-prob model per league.

Default is dry-run. Pass --apply to delete.

Decision rule:
  - Compare XGB vs shipped hybrid (and NFL Revolution via nfl_v2/hybrid_meta)
  - Winner = lowest oos_model_logloss
  - Never delete XGB live-serving files from *_v2/ (state, margin, score heads)
  - Keep bet_policy.json / hubacek_search_summary.json

Usage:
  python scripts/prune_models_keep_best_ll.py
  python scripts/prune_models_keep_best_ll.py --apply
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS = PROJECT_ROOT / "data" / "models"

LEAGUES = ("nba", "wnba", "nfl", "cfb", "cbb", "nhl", "mlb", "soccer")

# Never delete these from *_v2/ even when hybrid wins.
LIVE_XGB_KEEP_PREFIXES = (
    "model_clf",
    "model_lr",
    "model_margin",
    "model_score_",
    "model_goals_",
    "calibrator",
    "state_",
    "metadata.json",
    "bet_policy.json",
    "hubacek_search_summary.json",
)

HYBRID_KEEP_WHEN_WINNER = (
    "hybrid_meta.json",
    "model_clf_hybrid.cbm",
    "model_margin_hybrid.cbm",
    "model_ats_residual.cbm",
    "calibrator_hybrid.json",
)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _hybrid_ll(meta: dict[str, Any] | None) -> float | None:
    if not meta:
        return None
    if meta.get("oos_model_logloss") is not None:
        try:
            return float(meta["oos_model_logloss"])
        except (TypeError, ValueError):
            pass
    summary = meta.get("oos_summary") or {}
    if summary.get("oos_model_logloss") is not None:
        try:
            return float(summary["oos_model_logloss"])
        except (TypeError, ValueError):
            pass
    return None


def _xgb_ll(meta: dict[str, Any] | None, *, hybrid_shipped: bool) -> float | None:
    if not meta:
        return None
    # Prefer explicit baseline when hybrid overwrote oos_model_logloss.
    if hybrid_shipped and meta.get("baseline_xgb_logloss") is not None:
        try:
            return float(meta["baseline_xgb_logloss"])
        except (TypeError, ValueError):
            pass
    if meta.get("oos_model_logloss") is not None and not hybrid_shipped:
        try:
            return float(meta["oos_model_logloss"])
        except (TypeError, ValueError):
            pass
    if meta.get("baseline_xgb_logloss") is not None:
        try:
            return float(meta["baseline_xgb_logloss"])
        except (TypeError, ValueError):
            pass
    # Fallback: metadata may only have hybrid LL after ship — use that only
    # when no hybrid_meta exists (treat as XGB-reported).
    if not hybrid_shipped and meta.get("oos_model_logloss") is not None:
        try:
            return float(meta["oos_model_logloss"])
        except (TypeError, ValueError):
            pass
    return None


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)


def resolve_winner(league: str) -> dict[str, Any]:
    v2 = MODELS / f"{league}_v2"
    hybrid_dir = MODELS / f"{league}_hybrid"
    meta = _load_json(v2 / "metadata.json") or {}
    hybrid_meta = _load_json(v2 / "hybrid_meta.json")
    hybrid_shipped = hybrid_meta is not None and (v2 / "model_clf_hybrid.cbm").is_file()
    best_cfg = _load_json(hybrid_dir / "best_config.json")

    # True XGB LL: prefer hybrid search baseline (pre-ship), then metadata.
    xgb: float | None = None
    for src in (
        (hybrid_meta or {}).get("oos_summary", {}).get("baseline_ll") if hybrid_meta else None,
        (hybrid_meta or {}).get("baseline_ll") if hybrid_meta else None,
        (best_cfg or {}).get("baseline_ll") if best_cfg else None,
        meta.get("baseline_xgb_logloss"),
        None if hybrid_meta else meta.get("oos_model_logloss"),
    ):
        if src is None:
            continue
        try:
            xgb = float(src)
            break
        except (TypeError, ValueError):
            continue

    hyb = _hybrid_ll(hybrid_meta)
    if hyb is None and best_cfg:
        hyb = _hybrid_ll(best_cfg)

    algo = str((hybrid_meta or {}).get("algorithm") or "")
    is_revolution = "revolution" in algo.lower() or (
        league == "nfl"
        and hyb is not None
        and (v2 / "model_ats_residual.cbm").is_file()
    )

    candidates: list[tuple[str, float]] = []
    if xgb is not None:
        candidates.append(("xgb", xgb))
    if hyb is not None:
        label = "revolution" if is_revolution else "hybrid"
        candidates.append((label, hyb))

    if not candidates:
        winner = "xgb"
        winner_ll = None
    else:
        # Lowest LL wins. On a tie, prefer shipped hybrid/revolution (what live serves).
        best_ll = min(c[1] for c in candidates)
        tied = [c for c in candidates if abs(c[1] - best_ll) < 1e-9]
        prefer = [c for c in tied if c[0] in ("hybrid", "revolution")]
        if prefer and (hybrid_shipped or hybrid_meta is not None):
            winner, winner_ll = prefer[0]
        else:
            winner, winner_ll = min(tied, key=lambda t: (0 if t[0] == "xgb" else 1, t[1]))
            # If hybrid exists in tied and is shipped, still prefer it.
            if prefer:
                winner, winner_ll = prefer[0]

    best_name = None
    if best_cfg and isinstance(best_cfg.get("config"), dict):
        best_name = best_cfg["config"].get("name")
    elif hybrid_meta and isinstance(hybrid_meta.get("config"), dict):
        best_name = hybrid_meta["config"].get("name")
    elif hybrid_meta and isinstance((hybrid_meta.get("oos_summary") or {}).get("config"), dict):
        best_name = hybrid_meta["oos_summary"]["config"].get("name")

    return {
        "league": league,
        "xgb_ll": xgb,
        "hybrid_ll": hyb,
        "winner": winner,
        "winner_ll": winner_ll,
        "best_config_name": best_name,
        "hybrid_shipped": hybrid_shipped,
        "is_revolution": is_revolution,
    }


def _path_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return 0


def plan_deletions() -> list[tuple[Path, str]]:
    """Return list of (path, reason) to delete."""
    deletes: list[tuple[Path, str]] = []

    for league in LEAGUES:
        info = resolve_winner(league)
        v2 = MODELS / f"{league}_v2"
        hybrid_dir = MODELS / f"{league}_hybrid"
        winner = info["winner"]
        best_name = info.get("best_config_name")
        safe = _safe_name(best_name) if best_name else None

        if winner == "xgb":
            # Delete entire hybrid search dir.
            if hybrid_dir.exists():
                deletes.append((hybrid_dir, f"{league}: XGB wins LL — drop hybrid search"))
            # Drop hybrid serving artifacts from v2 if present (worse / not winner).
            for name in HYBRID_KEEP_WHEN_WINNER:
                path = v2 / name
                if path.exists():
                    deletes.append((path, f"{league}: XGB wins — drop hybrid serving file"))
            # Hybrid OOS dump not needed for XGB leagues.
            for name in ("oos_predictions_hybrid.csv",):
                path = v2 / name
                if path.exists():
                    deletes.append((path, f"{league}: XGB wins — drop hybrid OOS"))
        else:
            # Hybrid/Revolution wins: prune hybrid search debris.
            if hybrid_dir.is_dir():
                keep_names = {
                    "best_config.json",
                    "focused_final.json",
                    "overnight_final.json",
                }
                if safe:
                    keep_names.add(f"oos_{safe}.csv")
                    keep_names.add(f"summary_{safe}.json")
                for path in hybrid_dir.iterdir():
                    if path.name in keep_names:
                        continue
                    # WNBA overnight: keep overnight_final; drop mut_r* etc.
                    if path.name in ("best_config.json", "focused_final.json", "overnight_final.json"):
                        continue
                    deletes.append(
                        (
                            path,
                            f"{league}: hybrid wins — drop non-best search artifact",
                        )
                    )
            # Never delete live XGB files from v2.
            # Optionally drop non-serving junk OOS if both exist? Keep oos_predictions*.csv
            # (gitignored) — plan doesn't require deleting them from hybrid winners.

        # NFL: drop standard nfl_hybrid always when revolution wins; also prune revolution dups.
        if league == "nfl" and winner in ("revolution", "hybrid"):
            nfl_hybrid = MODELS / "nfl_hybrid"
            if nfl_hybrid.exists() and (nfl_hybrid, f"{league}: Revolution/hybrid wins — drop nfl_hybrid") not in [
                (p, r) for p, r in deletes
            ]:
                # May already be queued if winner==hybrid via above; ensure full dir delete.
                if winner == "revolution" and nfl_hybrid.exists():
                    # Remove any file-level entries under nfl_hybrid, replace with dir.
                    deletes = [(p, r) for p, r in deletes if not str(p).startswith(str(nfl_hybrid))]
                    deletes.append((nfl_hybrid, "nfl: Revolution wins — drop nfl_hybrid search"))

            rev = MODELS / "nfl_revolution"
            if rev.is_dir():
                keep_rev = {"best_config.json", "leaderboard.json", "best_so_far.json"}
                for path in rev.iterdir():
                    if path.name in keep_rev:
                        continue
                    deletes.append(
                        (
                            path,
                            "nfl: drop revolution duplicate/prepared (shipped copy in nfl_v2)",
                        )
                    )

    # Non-live legacy
    for rel, reason in (
        ("ensemble", "legacy ensemble"),
        ("nba", "legacy non-v2 nba"),
        ("nfl_backtest", "non-live backtest"),
        ("cfb_backtest", "non-live backtest"),
        ("hybrid_all_leagues_summary.json", "regenerable orchestration log"),
        ("hybrid_before_after.json", "regenerable orchestration log"),
        ("hybrid_bet_strategy_summary.json", "regenerable orchestration log"),
    ):
        path = MODELS / rel
        if path.exists():
            deletes.append((path, reason))

    # Deduplicate paths (prefer directory-level if both file and parent listed).
    by_path: dict[Path, str] = {}
    for path, reason in deletes:
        by_path[path.resolve()] = reason
    # If a parent dir is deleted, drop children entries.
    paths = sorted(by_path.keys(), key=lambda p: len(p.parts))
    kept: dict[Path, str] = {}
    for path in paths:
        if any(parent in kept and kept_path.is_dir() for kept_path, parent in [(k, k) for k in kept] if False):
            pass
        skip = False
        for parent in kept:
            try:
                path.relative_to(parent)
                if parent != path and parent.is_dir():
                    skip = True
                    break
            except ValueError:
                continue
        if not skip:
            # Re-check: if we're adding a dir, remove children already in kept
            if path.is_dir() or (not path.exists() and path.suffix == ""):
                for child in list(kept):
                    try:
                        child.relative_to(path)
                        if child != path:
                            del kept[child]
                    except ValueError:
                        pass
            kept[path] = by_path[path]

    # Simpler dedupe: unique resolved paths
    out: list[tuple[Path, str]] = []
    seen: set[Path] = set()
    for path, reason in deletes:
        rp = path.resolve()
        if rp in seen:
            continue
        # Skip if any already-queued ancestor directory covers this path
        covered = False
        for prev, _ in out:
            if prev.is_dir() or prev.suffix == "":
                try:
                    rp.relative_to(prev.resolve())
                    if rp != prev.resolve():
                        covered = True
                        break
                except ValueError:
                    pass
        if covered:
            continue
        # If adding a directory, drop previously queued children
        out = [
            (p, r)
            for p, r in out
            if not (
                (path.is_dir() or path.suffix == "")
                and _is_relative_to(p.resolve(), rp)
                and p.resolve() != rp
            )
        ]
        seen = {p.resolve() for p, _ in out}
        if rp in seen:
            continue
        out.append((path, reason))
        seen.add(rp)
    return out


def _is_relative_to(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _is_protected_v2_file(path: Path) -> bool:
    """Refuse deleting required live XGB artifacts under *_v2/."""
    parts = path.parts
    try:
        idx = parts.index("models")
    except ValueError:
        return False
    if idx + 1 >= len(parts):
        return False
    folder = parts[idx + 1]
    if not folder.endswith("_v2"):
        return False
    name = path.name
    if name in HYBRID_KEEP_WHEN_WINNER:
        return False  # hybrid files may be deleted when XGB wins
    for prefix in LIVE_XGB_KEEP_PREFIXES:
        if name == prefix or name.startswith(prefix):
            return True
    return False


def apply_deletes(items: list[tuple[Path, str]], *, dry_run: bool) -> int:
    total = 0
    for path, reason in items:
        if _is_protected_v2_file(path):
            print(f"SKIP protected live file: {path.relative_to(PROJECT_ROOT)} ({reason})", flush=True)
            continue
        size = _path_size(path) if path.exists() else 0
        rel = path.relative_to(PROJECT_ROOT) if path.is_absolute() else path
        action = "WOULD DELETE" if dry_run else "DELETE"
        print(f"{action} {rel}  ({size:,} bytes)  — {reason}", flush=True)
        if not dry_run and path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        total += size
    return total


def print_winners() -> None:
    print("=== Winners (lowest OOS logloss) ===", flush=True)
    for league in LEAGUES:
        info = resolve_winner(league)
        print(
            f"  {league:7s}  winner={info['winner']:10s}  "
            f"ll={info['winner_ll']!s:10s}  "
            f"xgb={info['xgb_ll']!s:10s}  hybrid={info['hybrid_ll']!s:10s}  "
            f"best_cfg={info['best_config_name']}",
            flush=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete files (default is dry-run)",
    )
    args = parser.parse_args()
    dry_run = not args.apply

    print_winners()
    print(flush=True)
    items = plan_deletions()
    print(f"=== {'Dry-run' if dry_run else 'Apply'} deletions ({len(items)} paths) ===", flush=True)
    saved = apply_deletes(items, dry_run=dry_run)
    print(flush=True)
    print(f"{'Would free' if dry_run else 'Freed'} ~{saved / (1024 * 1024):.1f} MB", flush=True)
    if dry_run:
        print("Re-run with --apply to delete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
