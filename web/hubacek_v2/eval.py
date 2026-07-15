"""Round-based betting evaluation (Hubáček et al. 2019, §5-7).

Simulates the paper's protocol on a walk-forward OOS prediction frame:
rounds (one slate per round key), confidence thresholding |p̂-0.5| > φ,
unit budget per round allocated by a strategy (opt = max-Sharpe, unif),
profits settled against actual outcomes at a chosen price column.

Also reports the diagnostics the paper leans on: Pearson correlation with the
bookmaker, accuracy, and the quadrant split Consensus / Upset / Missed /
Spotted (profit lives in Spotted: bettor right while the book is wrong).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from web.hubacek_v2.portfolio import max_sharpe_bets, uniform_bets

EPS = 1e-9


def _devig_pair(home_ml: pd.Series, away_ml: pd.Series) -> pd.Series:
    def implied(ml: pd.Series) -> pd.Series:
        m = pd.to_numeric(ml, errors="coerce")
        return pd.Series(
            np.where(m < 0, -m / (-m + 100.0), 100.0 / (m + 100.0)),
            index=m.index,
        ).where(m.abs() >= 100)

    ph, pa = implied(home_ml), implied(away_ml)
    tot = ph + pa
    return (ph / tot).where(tot > 0.9)


def _decimal_odds_from_american(ml: pd.Series) -> pd.Series:
    m = pd.to_numeric(ml, errors="coerce")
    dec = pd.Series(
        np.where(m > 0, 1.0 + m / 100.0, 1.0 + 100.0 / (-m)),
        index=m.index,
    )
    return dec.where(m.abs() >= 100)


def quadrants(p_model: np.ndarray, p_book: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Consensus/Upset/Missed/Spotted proportions (%), paper Table 2."""
    model_right = (p_model > 0.5) == (y > 0.5)
    book_right = (p_book > 0.5) == (y > 0.5)
    n = max(len(y), 1)
    return {
        "consensus": round(100.0 * np.mean(model_right & book_right), 2),
        "upset": round(100.0 * np.mean(~model_right & ~book_right), 2),
        "missed": round(100.0 * np.mean(~model_right & book_right), 2),
        "spotted": round(100.0 * np.mean(model_right & ~book_right), 2),
        "n": int(n),
    }


def evaluate_rounds(
    oos: pd.DataFrame,
    *,
    prob_col: str = "model_prob",
    price: str = "close",
    phi: float = 0.0,
    strategy: str = "opt",
    round_key: str = "date",
    margin_haircut: float = 0.0,
) -> dict[str, Any] | None:
    """Simulate round-by-round betting on an OOS frame.

    ``price``: 'close' uses market_home_prob (or de-vigged close MLs) as the
    book probability and, when close ML columns exist, their actual decimal
    payouts; otherwise fair odds 1/q with ``margin_haircut`` (e.g. 0.025)
    applied as a synthetic vig. 'open' uses home_open_ml/away_open_ml rows.
    Both sides of every game are candidate outcomes (the portfolio keeps at
    most one).
    """
    frame = oos.copy()
    if round_key not in frame.columns:
        return None

    if price == "open" and {"home_open_ml", "away_open_ml"}.issubset(frame.columns):
        q_home = _devig_pair(frame["home_open_ml"], frame["away_open_ml"])
        o_home = _decimal_odds_from_american(frame["home_open_ml"])
        o_away = _decimal_odds_from_american(frame["away_open_ml"])
    else:
        if "market_home_prob" in frame.columns and frame["market_home_prob"].notna().any():
            q_home = pd.to_numeric(frame["market_home_prob"], errors="coerce")
        elif {"home_close_ml", "away_close_ml"}.issubset(frame.columns):
            q_home = _devig_pair(frame["home_close_ml"], frame["away_close_ml"])
        else:
            return None
        if {"home_close_ml", "away_close_ml"}.issubset(frame.columns) and (
            pd.to_numeric(frame["home_close_ml"], errors="coerce").notna().any()
        ):
            o_home = _decimal_odds_from_american(frame["home_close_ml"])
            o_away = _decimal_odds_from_american(frame["away_close_ml"])
        else:
            # Fair odds from the de-vigged book prob, minus a synthetic margin.
            o_home = (1.0 - margin_haircut) / q_home.clip(EPS, 1 - EPS)
            o_away = (1.0 - margin_haircut) / (1.0 - q_home).clip(EPS, 1 - EPS)

    frame["_q"] = q_home
    frame["_oh"] = o_home
    frame["_oa"] = o_away
    frame = frame.dropna(subset=[prob_col, "_q", "_oh", "_oa", "home_win"])
    if frame.empty:
        return None

    p = frame[prob_col].to_numpy(dtype=float)
    q = frame["_q"].to_numpy(dtype=float)
    y = frame["home_win"].to_numpy(dtype=float)

    result: dict[str, Any] = {
        "price": price,
        "phi": phi,
        "strategy": strategy,
        "n_games": int(len(frame)),
        "rho_model_book": round(float(np.corrcoef(p, q)[0, 1]), 4) if len(frame) > 2 else None,
        "accuracy": round(float(np.mean((p > 0.5) == (y > 0.5))), 4),
        "quadrants": quadrants(p, q, y),
    }

    profits: list[float] = []
    bets_placed = 0
    for _, day in frame.groupby(round_key, sort=True):
        ph = day[prob_col].to_numpy(dtype=float)
        yh = day["home_win"].to_numpy(dtype=float)
        oh = day["_oh"].to_numpy(dtype=float)
        oa = day["_oa"].to_numpy(dtype=float)
        gid = np.arange(len(day))
        # Candidate outcomes: home and away side of each game.
        probs = np.concatenate([ph, 1.0 - ph])
        odds = np.concatenate([oh, oa])
        wins = np.concatenate([yh, 1.0 - yh])
        gids = np.concatenate([gid, gid])
        keep = np.abs(probs - 0.5) > phi
        if not keep.any():
            profits.append(0.0)
            continue
        alloc_fn = max_sharpe_bets if strategy == "opt" else uniform_bets
        bets = np.zeros(len(probs))
        bets[keep] = alloc_fn(probs[keep], odds[keep], game_ids=gids[keep])
        if bets.sum() <= 0:
            profits.append(0.0)
            continue
        bets_placed += int(np.sum(bets > 0))
        payout = np.where(wins > 0.5, odds * bets, 0.0)
        profits.append(float(np.sum(payout) - np.sum(bets)))

    arr = np.asarray(profits, dtype=float)
    active = arr[arr != 0.0]
    result.update(
        {
            "n_rounds": int(len(arr)),
            "n_rounds_bet": int(len(active)),
            "n_bets": bets_placed,
            "mean_profit_per_round_pct": round(100.0 * float(arr.mean()), 3) if len(arr) else None,
            "std_profit_per_round_pct": round(100.0 * float(arr.std()), 3) if len(arr) else None,
            "sharpe_per_round": round(float(arr.mean() / arr.std()), 4) if len(arr) and arr.std() > 0 else None,
            "cumulative_profit_units": round(float(arr.sum()), 3),
        }
    )
    return result


def hubacek_report(
    oos: pd.DataFrame,
    *,
    prob_col: str = "model_prob",
    round_key: str = "date",
    phis: tuple[float, ...] = (0.0, 0.1, 0.15, 0.2),
    prices: tuple[str, ...] = ("close",),
) -> dict[str, Any]:
    """Grid of round evaluations across φ and price sources (opt + unif)."""
    out: dict[str, Any] = {"settings": []}
    for price in prices:
        for phi in phis:
            for strategy in ("opt", "unif"):
                res = evaluate_rounds(
                    oos,
                    prob_col=prob_col,
                    price=price,
                    phi=phi,
                    strategy=strategy,
                    round_key=round_key,
                )
                if res:
                    out["settings"].append(res)
    # Convenience: the best opt setting by Sharpe.
    opts = [s for s in out["settings"] if s["strategy"] == "opt" and s.get("sharpe_per_round") is not None]
    if opts:
        out["best_opt"] = max(opts, key=lambda s: s["sharpe_per_round"])
    return out
