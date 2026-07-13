"""Sync data/pick_strategy.json from refreshed bet_policy.json artifacts."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PICK = PROJECT_ROOT / "data" / "pick_strategy.json"
MODELS = PROJECT_ROOT / "data" / "models"


def _roi(policy: dict) -> tuple[float | None, int | None, float | None]:
    for key in ("backtest_open", "backtest", "backtest_close"):
        bt = policy.get(key) or {}
        if "roi_pct" in bt:
            return (
                float(bt["roi_pct"]),
                int(bt.get("bets") or 0) or None,
                bt.get("worst_season_roi"),
            )
    return None, None, None


def _enable(policy: dict) -> bool:
    """Enable only when worst-season ROI > 0 on the preferred execution window."""
    for key in ("backtest_open", "backtest", "backtest_close"):
        bt = policy.get(key) or {}
        worst = bt.get("worst_season_roi")
        if worst is None:
            continue
        return float(worst) > 0
    return False


def apply_mlb_nhl(entry: dict, policy: dict) -> None:
    entry["bet_type"] = "moneyline"
    entry["min_market_gap_pp"] = float(policy.get("min_edge_pp") or entry.get("min_market_gap_pp") or 3.0)
    entry["min_ev_pct"] = float(policy.get("min_ev_pct") or 0.0)
    entry["ml_lo"] = int(policy.get("ml_lo") or -200)
    entry["ml_hi"] = int(policy.get("ml_hi") or 200)
    # Open-line policies are broad either-side books (no legacy dog/fav niche).
    entry.pop("allowed_sides", None)
    entry["fav_mode"] = "any"
    entry["min_win_confidence_pp"] = float(policy.get("min_win_confidence_pp") or 0.0)
    roi, bets, worst = _roi(policy)
    entry["enabled"] = _enable(policy)
    if roi is not None:
        entry["backtest_roi_pct"] = roi
    if bets is not None:
        entry["backtest_bets"] = bets
    close = policy.get("backtest_close") or {}
    if "roi_pct" in close:
        entry["backtest_close_roi_pct"] = close["roi_pct"]
    entry["note"] = (
        f"Hybrid OOS Hubáček refresh {date.today().isoformat()}: "
        f"gap>={entry['min_market_gap_pp']}pp EV>={entry['min_ev_pct']}% "
        f"ML[{entry['ml_lo']},{entry['ml_hi']}] open ROI {roi}% worst {worst}% n={bets}. "
        f"Close ROI {close.get('roi_pct')}% (morning prices sit between open and close). "
        f"{'ENABLED' if entry['enabled'] else 'DISABLED (worst-season ROI <= 0)'}."
    )


def apply_spread(entry: dict, policy: dict) -> None:
    entry["bet_type"] = "spread"
    entry["min_spread_cover_gap_pp"] = float(
        policy.get("min_cover_gap_pp") or policy.get("min_spread_cover_gap_pp") or 10.0
    )
    entry["min_spread_point_edge"] = float(
        policy.get("min_point_edge") or policy.get("min_spread_point_edge") or 0.0
    )
    if "min_ev_pct" in policy:
        entry["min_ev_pct"] = float(policy["min_ev_pct"])
    # Clear legacy side filters unless the new policy redefines them.
    if "allowed_sides" in policy:
        entry["allowed_sides"] = list(policy["allowed_sides"])
    else:
        entry.pop("allowed_sides", None)
    if "fav_mode" in policy:
        entry["fav_mode"] = policy["fav_mode"]
    else:
        entry.pop("fav_mode", None)
    bt = policy.get("backtest") or {}
    entry["enabled"] = float(bt.get("worst_season_roi") or -1) > 0
    if "roi_pct" in bt:
        entry["backtest_roi_pct"] = bt["roi_pct"]
    if "bets" in bt:
        entry["backtest_bets"] = bt["bets"]
    entry["note"] = (
        f"Hybrid OOS Hubáček refresh {date.today().isoformat()}: "
        f"cover gap>={entry['min_spread_cover_gap_pp']}pp point_edge>={entry['min_spread_point_edge']} "
        f"exec={policy.get('exec_price')} ROI {bt.get('roi_pct')}% worst {bt.get('worst_season_roi')}% "
        f"n={bt.get('bets')}. {'ENABLED' if entry['enabled'] else 'DISABLED'}."
    )


def apply_soccer(entry: dict, policy: dict) -> None:
    entry["bet_type"] = "soccer_1x2"
    entry["min_market_gap_pp"] = float(policy.get("min_gap_pp") or 4.0)
    entry["min_ev_pct"] = float(policy.get("min_ev_pct") or 2.0)
    entry["min_win_confidence_pp"] = float(policy.get("min_win_confidence_pp") or 0.0)
    outcomes = [str(x).lower() for x in (policy.get("outcomes") or ["home"])]
    entry["allowed_sides"] = outcomes
    bt = policy.get("backtest") or {}
    entry["enabled"] = float(bt.get("worst_season_roi") or -1) > 0
    if "roi_pct" in bt:
        entry["backtest_roi_pct"] = bt["roi_pct"]
    if "bets" in bt:
        entry["backtest_bets"] = bt["bets"]
    entry["note"] = (
        f"Hybrid OOS soccer refresh {date.today().isoformat()}: "
        f"gap>={entry['min_market_gap_pp']} EV>={entry['min_ev_pct']}% "
        f"outcomes={outcomes} exec={policy.get('exec_price')} "
        f"ROI {bt.get('roi_pct')}% worst {bt.get('worst_season_roi')}% n={bt.get('bets')}. "
        f"{'ENABLED' if entry['enabled'] else 'DISABLED'}."
    )


def _apply_sides(entry: dict, policy: dict) -> None:
    sides = str(policy.get("sides") or "").lower()
    if sides in {"home", "away"}:
        entry["allowed_sides"] = [sides]
        entry["fav_mode"] = "any"
    elif sides == "favorite":
        entry.pop("allowed_sides", None)
        entry["fav_mode"] = "favorite"
    elif sides == "dog":
        entry.pop("allowed_sides", None)
        entry["fav_mode"] = "dog"
    elif sides in {"either", "any", ""}:
        entry.pop("allowed_sides", None)
        entry["fav_mode"] = "any"


def apply_moneyline_generic(entry: dict, policy: dict) -> None:
    # NFL/CBB style policies sometimes stored differently
    if policy.get("bet_type") == "spread" or "min_cover_gap_pp" in policy:
        apply_spread(entry, policy)
        return
    entry["bet_type"] = "moneyline"
    entry["min_market_gap_pp"] = float(
        policy.get("min_edge_pp") or policy.get("min_market_gap_pp") or entry.get("min_market_gap_pp") or 3.0
    )
    if "min_ev_pct" in policy:
        entry["min_ev_pct"] = float(policy["min_ev_pct"])
    if "ml_lo" in policy:
        entry["ml_lo"] = int(policy["ml_lo"])
    if "ml_hi" in policy:
        entry["ml_hi"] = int(policy["ml_hi"])
    _apply_sides(entry, policy)
    entry["min_win_confidence_pp"] = float(policy.get("min_win_confidence_pp") or 0.0)
    bt = policy.get("backtest") or policy.get("backtest_open") or {}
    worst = bt.get("worst_season_roi")
    # Prefer explicit enabled flag when present (disabled snapshots).
    if "enabled" in policy:
        entry["enabled"] = bool(policy["enabled"]) and worst is not None and float(worst) > 0
    else:
        entry["enabled"] = worst is not None and float(worst) > 0
    if "roi_pct" in bt:
        entry["backtest_roi_pct"] = bt["roi_pct"]
    if "bets" in bt:
        entry["backtest_bets"] = bt["bets"]
    entry["note"] = (
        f"Hybrid OOS Hubáček refresh {date.today().isoformat()}: "
        f"gap>={entry['min_market_gap_pp']}pp EV>={entry.get('min_ev_pct', 0)}% "
        f"ML[{entry.get('ml_lo')},{entry.get('ml_hi')}] sides={policy.get('sides', 'any')} "
        f"exec={policy.get('exec_price', 'close')} ROI {bt.get('roi_pct')}% "
        f"worst {worst}% n={bt.get('bets')}. "
        f"{'ENABLED' if entry['enabled'] else 'DISABLED'}."
    )


def main() -> int:
    strategy = json.loads(PICK.read_text(encoding="utf-8"))
    strategy["generated_at"] = date.today().isoformat()
    strategy["policy"] = (
        "Official picks use Hubáček gates from walk-forward hybrid OOS bet backtests. "
        "Enable bar: positive worst-season ROI at the execution price we would actually take."
    )

    mapping = {
        "mlb": apply_mlb_nhl,
        "nhl": apply_mlb_nhl,
        "nba": apply_spread,
        "wnba": apply_spread,
        "nfl": apply_moneyline_generic,
        "cfb": apply_moneyline_generic,
        "cbb": apply_moneyline_generic,
    }

    summary = {}
    for league, apply in mapping.items():
        path = MODELS / f"{league}_v2" / "bet_policy.json"
        if not path.is_file():
            summary[league] = "missing bet_policy"
            continue
        policy = json.loads(path.read_text(encoding="utf-8"))
        entry = strategy.setdefault(league, {})
        apply(entry, policy)
        summary[league] = {
            "enabled": entry.get("enabled"),
            "bet_type": entry.get("bet_type"),
            "roi": entry.get("backtest_roi_pct"),
            "bets": entry.get("backtest_bets"),
        }

    # Soccer club leagues share soccer_v2 policy
    soccer_path = MODELS / "soccer_v2" / "bet_policy.json"
    if soccer_path.is_file():
        policy = json.loads(soccer_path.read_text(encoding="utf-8"))
        for league in ("epl", "laliga", "seriea", "bundesliga", "ligue1"):
            entry = strategy.setdefault(league, {})
            apply_soccer(entry, policy)
            summary[league] = {
                "enabled": entry.get("enabled"),
                "bet_type": entry.get("bet_type"),
                "roi": entry.get("backtest_roi_pct"),
                "bets": entry.get("backtest_bets"),
            }

    PICK.write_text(json.dumps(strategy, indent=2), encoding="utf-8")
    out = MODELS / "hybrid_bet_strategy_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {PICK}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
