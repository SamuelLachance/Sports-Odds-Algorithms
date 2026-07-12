"""Fourth-layer context adjustments grounded in market-bias research.

Pure math only in the core helpers (no network). Optional fetch helpers may
live elsewhere; this module consumes already-fetched headlines / odds.

Signals:
- Favorite-longshot bias (FLB) nudge from open-like moneylines
  (Harvard MLB thesis 2026: FLB concentrated in opening prices)
- Public sentiment / steam proxies from line movement when available
- News/injury keyword heuristics from ESPN-style headlines
- Sparse-sample EV caps for thin international tournaments
"""

from __future__ import annotations

import re
from typing import Any

# Max signed home-prob nudge from favorite-longshot bias (percentage points).
MAX_FLB_SHIFT_PP = 1.5

# Max signed home-prob nudge from headline keyword heuristics.
MAX_NEWS_SHIFT_PP = 2.0

# Max signed home-prob nudge from steam / line-movement proxy.
MAX_STEAM_SHIFT_PP = 1.0

# Combined context layer clamp (keeps A+ upgrades conservative).
MAX_TOTAL_CONTEXT_PP = 3.0

# American odds at/through which FLB is considered present at open-like prices.
FLB_FAVORITE_ML_THRESHOLD = -150
FLB_HEAVY_FAVORITE_ML = -250

# Leagues with thin samples (World Cup, friendlies, confederation tournaments).
SPARSE_SAMPLE_LEAGUES = frozenset(
    {
        "worldcup",
        "fifa_friendlies",
        "copa_america",
        "concacaf_wcq",
        "concacaf_gold",
        "concacaf_nations",
        "uefa_euro",
        "uefa_nations",
    }
)

# Soft EV caps (honest EV% after vig) when sample is thin.
_SPARSE_EV_CAP_STRICT = 18.0
_SPARSE_EV_CAP_MODERATE = 28.0
_SPARSE_EV_CAP_RELAXED = 40.0
_DEFAULT_EV_CAP_THIN = 55.0

# Prefer whole words / specific phrases — bare stems like "injur" or "out for"
# false-positive on "looking out for" and loose copy.
_INJURY_STRONG_KEYWORDS = (
    "injured",
    "sidelined",
    "questionable",
    "doubtful",
    "ruled out",
    "suspended",
    "suspension",
    "rest day",
    "sitting out",
    "will miss",
    "day-to-day",
    "day to day",
    "out indefinitely",
    "out for the season",
    "out for game",
)

# Alone these are too weak inside "injury report" headers; require no false alarm.
_INJURY_WEAK_KEYWORDS = (
    "injury",
    "injuries",
)

# Suppress weak injury hits (and pure "all clear" copy) when these appear.
_INJURY_FALSE_ALARM = (
    "no injury",
    "no injuries",
    "injury-free",
    "injury free",
    "injury report",
    "cleared to play",
    "returns from injury",
    "returned from injury",
    "avoided injury",
    "escaped injury",
)

# Avoid loose single tokens ("dominant", "rolling") that fire on non-form copy.
_HOT_POS_KEYWORDS = (
    "hot streak",
    "winning streak",
    "unbeaten",
    "in form",
    "on fire",
    "on a roll",
    "red hot",
)


def _clamp(value: float, lo: float, hi: float) -> float:
    return min(max(value, lo), hi)


def _american_to_implied_pct(american: int) -> float | None:
    """Implied win % from American odds; None for invalid |odds| < 100.

    Reject bools: ``False == 0`` must not become EVEN (50%).
    """
    if american is None or isinstance(american, bool):
        return None
    if american == 0:
        return 50.0
    if abs(american) < 100:
        return None
    if american < 0:
        return abs(american) / (abs(american) + 100.0) * 100.0
    return 100.0 / (american + 100.0) * 100.0


def favorite_longshot_adjustment(
    home_ml: int | None,
    away_ml: int | None,
    model_home_prob_pct: float,
) -> float:
    """Signed pp shift to home win probability toward the underdog when FLB is present.

    Opening-like heavy favorites are slightly overbet (favorite-longshot bias).
    When both the book and the model lean the same favorite at open-like prices,
    nudge a small amount toward the dog. Magnitude capped at ±1.5 pp.
    """
    if home_ml is None or away_ml is None:
        return 0.0
    # bool is a subclass of int; False must not become EVEN / favorite poison.
    if isinstance(home_ml, bool) or isinstance(away_ml, bool):
        return 0.0
    try:
        home_ml_i = int(home_ml)
        away_ml_i = int(away_ml)
    except (TypeError, ValueError):
        return 0.0

    if home_ml_i == away_ml_i:
        return 0.0

    home_is_fav = home_ml_i < away_ml_i
    fav_ml = home_ml_i if home_is_fav else away_ml_i
    if fav_ml > FLB_FAVORITE_ML_THRESHOLD:
        return 0.0

    model_home = float(model_home_prob_pct)
    model_favors_home = model_home >= 50.0
    if home_is_fav != model_favors_home:
        # Model already disagrees with the chalk — no FLB correction.
        return 0.0

    # Scale 0→1 from threshold to heavy favorite.
    span = abs(FLB_HEAVY_FAVORITE_ML - FLB_FAVORITE_ML_THRESHOLD)
    intensity = _clamp(abs(fav_ml - FLB_FAVORITE_ML_THRESHOLD) / max(span, 1.0), 0.0, 1.0)
    # Mild base even at the threshold so open-like -150 still gets a tiny nudge.
    magnitude = MAX_FLB_SHIFT_PP * (0.35 + 0.65 * intensity)

    # Toward underdog: if home is favorite, reduce home prob (negative shift).
    shift = -magnitude if home_is_fav else magnitude
    return round(_clamp(shift, -MAX_FLB_SHIFT_PP, MAX_FLB_SHIFT_PP), 3)


def steam_line_movement_shift(
    home_ml: int | None,
    away_ml: int | None,
    *,
    open_home_ml: int | None = None,
    open_away_ml: int | None = None,
) -> float:
    """Signed pp home-prob nudge from open→close moneyline steam (public/sharp proxy).

    Positive = steam toward home. Uses the average of home implied lift and away
    implied drop so pure vig reshuffles cancel. Capped at ±1.0 pp. Returns 0 when
    opens are missing.
    """
    if (
        home_ml is None
        or away_ml is None
        or open_home_ml is None
        or open_away_ml is None
    ):
        return 0.0
    # bool is a subclass of int; False→0 must not fabricate EVEN open/close steam.
    if any(
        isinstance(value, bool)
        for value in (home_ml, away_ml, open_home_ml, open_away_ml)
    ):
        return 0.0
    try:
        close_home = _american_to_implied_pct(int(home_ml))
        open_home = _american_to_implied_pct(int(open_home_ml))
        close_away = _american_to_implied_pct(int(away_ml))
        open_away = _american_to_implied_pct(int(open_away_ml))
    except (TypeError, ValueError):
        return 0.0
    if None in (close_home, open_home, close_away, open_away):
        return 0.0

    # Average home lift with away drop so vig reshuffles cancel out.
    move_pp = ((close_home - open_home) - (close_away - open_away)) / 2.0
    # Ignore noise under ~1.5 pp implied; scale gently above that.
    if abs(move_pp) < 1.5:
        return 0.0
    # 1.5→6 pp move maps into ~0.25→1.0 pp model nudge.
    scaled = _clamp(abs(move_pp) / 6.0, 0.0, 1.0) * MAX_STEAM_SHIFT_PP
    shift = scaled if move_pp > 0 else -scaled
    return round(_clamp(shift, -MAX_STEAM_SHIFT_PP, MAX_STEAM_SHIFT_PP), 3)


def _token_in_text(text: str, token: str) -> bool:
    """Whole-token match so short nicknames (Heat/Sun) do not hit substrings."""
    if not token:
        return False
    return re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", text) is not None


def _headline_mentions_side(headline: str, names: list[str]) -> bool:
    text = headline.lower()
    for name in names:
        token = (name or "").strip().lower()
        if len(token) >= 3 and _token_in_text(text, token):
            return True
    return False


def _keyword_hit(text: str, keywords: tuple[str, ...]) -> bool:
    """Phrase substring for multi-word keys; whole-token match for single words."""
    for key in keywords:
        if " " in key or "-" in key:
            if key in text:
                return True
        elif _token_in_text(text, key):
            return True
    return False


def news_sentiment_shift(
    headlines: list[str],
    home_names: list[str],
    away_names: list[str],
) -> float:
    """Keyword heuristic (±2 pp max) for injury/suspension/rest/hot-streak language.

    Negative keywords on a side hurt that side; hot-streak language helps.
    Returns signed pp shift to home win probability. Skips injury false alarms
    (e.g. "injury report", "no injuries", "cleared to play").
    """
    if not headlines:
        return 0.0
    home_names = [n for n in (home_names or []) if n]
    away_names = [n for n in (away_names or []) if n]
    if not home_names and not away_names:
        return 0.0

    home_score = 0.0
    away_score = 0.0
    for raw in headlines:
        if not raw or not isinstance(raw, str):
            continue
        text = raw.lower()
        home_hit = _headline_mentions_side(text, home_names)
        away_hit = _headline_mentions_side(text, away_names)
        if not home_hit and not away_hit:
            continue
        injury_alarm = _keyword_hit(text, _INJURY_FALSE_ALARM)
        strong_injury = _keyword_hit(text, _INJURY_STRONG_KEYWORDS)
        weak_injury = _keyword_hit(text, _INJURY_WEAK_KEYWORDS)
        # Strong availability verbs always count; weak "injury" alone is skipped
        # inside report/all-clear copy so headers do not nudge probabilities.
        injury = strong_injury or (weak_injury and not injury_alarm)
        hot = _keyword_hit(text, _HOT_POS_KEYWORDS)
        if injury:
            if home_hit and not away_hit:
                home_score -= 1.0
            elif away_hit and not home_hit:
                away_score -= 1.0
            elif home_hit and away_hit:
                # Ambiguous dual mention — skip.
                pass
        if hot:
            if home_hit and not away_hit:
                home_score += 0.75
            elif away_hit and not home_hit:
                away_score += 0.75

    # Net home advantage from relative sentiment.
    raw_shift = (home_score - away_score) * 0.85
    return round(_clamp(raw_shift, -MAX_NEWS_SHIFT_PP, MAX_NEWS_SHIFT_PP), 3)


def is_sparse_sample_league(league: str | None) -> bool:
    if not league:
        return False
    key = league.lower().strip()
    if key in SPARSE_SAMPLE_LEAGUES:
        return True
    return key.startswith("concacaf_")


def sparse_sample_ev_cap(
    league: str,
    games_played_proxy: int | None,
    ev_pct: float,
) -> float:
    """Cap absurd EV% when the league/sample is thin (e.g. World Cup 99% EV).

    International tournaments and friendlies get stricter caps. Regular leagues
    only soft-cap when ``games_played_proxy`` is explicitly very small.
    """
    try:
        ev = float(ev_pct)
    except (TypeError, ValueError):
        return 0.0

    sparse = is_sparse_sample_league(league)
    games = games_played_proxy
    if games is not None:
        try:
            games = int(games)
        except (TypeError, ValueError):
            games = None

    if sparse:
        if games is None or games < 8:
            cap = _SPARSE_EV_CAP_STRICT
        elif games < 16:
            cap = _SPARSE_EV_CAP_MODERATE
        else:
            cap = _SPARSE_EV_CAP_RELAXED
        return round(min(ev, cap), 4)

    if games is not None and games < 6:
        return round(min(ev, _DEFAULT_EV_CAP_THIN), 4)
    return round(ev, 4)


def _resolve_home_prob(blended: dict[str, Any]) -> float | None:
    if blended.get("blended_home_win_probability") is not None:
        return float(blended["blended_home_win_probability"])
    if blended.get("home_win_probability") is not None and blended.get("threeway"):
        return float(blended["home_win_probability"])
    if blended.get("home_win_probability") is not None and not blended.get("draw_probability"):
        return float(blended["home_win_probability"])
    if blended.get("total_score") is not None:
        total = float(blended["total_score"])
        if abs(total) < 1e-9:
            return 50.0
        win_prob = abs(total)
        return win_prob if total <= 0 else 100.0 - win_prob
    if blended.get("win_probability") is not None and blended.get("favorite_side"):
        win_prob = float(blended["win_probability"])
        return win_prob if blended["favorite_side"] == "home" else 100.0 - win_prob
    return None


def _games_played_proxy_from_blend(blended: dict[str, Any]) -> int | None:
    for key in (
        "hockey_pred",
        "basketball_pred",
        "baseball_pred",
        "soccer_pred",
        "football_pred",
        "mlb_pred",
    ):
        pred = blended.get(key)
        if not isinstance(pred, dict):
            continue
        home_g = pred.get("home_games")
        away_g = pred.get("away_games")
        if home_g is None and away_g is None:
            continue
        try:
            values = [int(v) for v in (home_g, away_g) if v is not None]
        except (TypeError, ValueError):
            continue
        if values:
            return min(values)
    return None


def games_played_proxy_from_blend(blended: dict[str, Any]) -> int | None:
    """Public wrapper: min team games from sport-pred layers when available."""
    ctx = blended.get("context_signals")
    if isinstance(ctx, dict) and ctx.get("games_played_proxy") is not None:
        try:
            return int(ctx["games_played_proxy"])
        except (TypeError, ValueError):
            pass
    return _games_played_proxy_from_blend(blended)


def apply_context_to_blend(
    blended: dict[str, Any],
    *,
    market: dict[str, Any] | None = None,
    headlines: list[str] | None = None,
    home_names: list[str] | None = None,
    away_names: list[str] | None = None,
    league: str | None = None,
) -> dict[str, Any]:
    """Apply context layer to a blend payload; returns updated dict.

    Stores ``pre_context_*`` copies, sets ``context_adjustment_pp``, and updates
    home/away (or binary) probabilities. Soccer three-way blends adjust home/away
    mass while holding draw fixed, then renormalize.
    """
    home_prob = _resolve_home_prob(blended)
    if home_prob is None:
        return blended

    market = market or {}
    home_ml = market.get("home_moneyline")
    away_ml = market.get("away_moneyline")
    # Prefer primary open keys; do not use `or` — American 0 (EVEN) is falsy.
    open_home = market.get("open_home_moneyline")
    if open_home is None:
        open_home = market.get("opening_home_ml")
    open_away = market.get("open_away_moneyline")
    if open_away is None:
        open_away = market.get("opening_away_ml")

    # FLB is an opening-price bias; only use opens when both sides are present.
    if open_home is not None and open_away is not None:
        flb_home, flb_away = open_home, open_away
    else:
        flb_home, flb_away = home_ml, away_ml
    flb = favorite_longshot_adjustment(flb_home, flb_away, home_prob)
    steam = steam_line_movement_shift(
        home_ml,
        away_ml,
        open_home_ml=open_home if open_home is not None else None,
        open_away_ml=open_away if open_away is not None else None,
    )
    news = news_sentiment_shift(
        list(headlines or []),
        list(home_names or []),
        list(away_names or []),
    )
    # Prefer steam already computed by sport models when present.
    # Floor magnitude only when signs agree — never flip ML-derived steam.
    opening_steam = blended.get("opening_steam")
    if isinstance(opening_steam, dict) and opening_steam.get("steam_signal"):
        direction = str(opening_steam.get("steam_direction") or "").lower()
        if direction == "home" and steam >= 0:
            steam = max(steam, 0.5)
        elif direction == "away" and steam <= 0:
            steam = min(steam, -0.5)

    total_shift = _clamp(flb + steam + news, -MAX_TOTAL_CONTEXT_PP, MAX_TOTAL_CONTEXT_PP)
    if abs(total_shift) < 1e-9:
        updated = dict(blended)
        updated["context_adjustment_pp"] = 0.0
        updated["context_signals"] = {
            "flb_pp": flb,
            "steam_pp": steam,
            "news_pp": news,
            "total_pp": 0.0,
        }
        return updated

    from web.blend_service import home_win_prob_to_total_score

    updated = dict(blended)
    updated["pre_context_home_win_probability"] = round(home_prob, 2)
    if updated.get("blended_home_win_probability") is not None:
        updated["pre_context_blended_home_win_probability"] = round(
            float(updated["blended_home_win_probability"]), 2
        )
    if updated.get("win_probability") is not None:
        updated["pre_context_win_probability"] = round(float(updated["win_probability"]), 2)
    if updated.get("total_score") is not None:
        updated["pre_context_total_score"] = round(float(updated["total_score"]), 2)

    if updated.get("threeway") and updated.get("draw_probability") is not None:
        draw_p = float(updated["draw_probability"])
        away_raw = updated.get("away_win_probability")
        if away_raw is None:
            away_p = 100.0 - home_prob - draw_p
        else:
            away_p = float(away_raw)
        updated["pre_context_away_win_probability"] = round(away_p, 2)
        updated["pre_context_draw_probability"] = round(draw_p, 2)
        # Hold draw fixed; move mass only between home and away within the
        # remaining share (do not renormalize draw after clamps).
        remaining = max(100.0 - draw_p, 0.0)
        lo = min(1.0, remaining)
        hi = max(lo, min(97.0, remaining - min(1.0, remaining)))
        new_home = _clamp(home_prob + total_shift, lo, hi)
        new_home = min(max(new_home, 0.0), remaining)
        new_away = remaining - new_home
        updated["home_win_probability"] = round(new_home, 2)
        updated["draw_probability"] = round(draw_p, 2)
        updated["away_win_probability"] = round(new_away, 2)
        if updated.get("blended_home_win_probability") is not None:
            updated["blended_home_win_probability"] = round(new_home, 2)
        # Keep nested 1X2 snapshot in step with the board (same as soccer_blend).
        if updated.get("blended_threeway"):
            updated["blended_threeway"] = {
                "home_win_probability": round(new_home, 2),
                "draw_probability": round(draw_p, 2),
                "away_win_probability": round(new_away, 2),
            }
        from web.soccer_blend import threeway_probs_to_total_score

        total_score, win_prob, favorite = threeway_probs_to_total_score(
            new_home, draw_p, new_away
        )
        updated["total_score"] = round(total_score, 2)
        updated["win_probability"] = round(win_prob, 2)
        updated["favorite_side"] = favorite
    else:
        adjusted = _clamp(home_prob + total_shift, 5.0, 95.0)
        total_score, win_prob = home_win_prob_to_total_score(adjusted)
        updated["blended_home_win_probability"] = round(adjusted, 2)
        updated["home_win_probability"] = round(adjusted, 2)
        if updated.get("away_win_probability") is not None:
            updated["away_win_probability"] = round(100.0 - adjusted, 2)
        updated["total_score"] = round(total_score, 2)
        updated["win_probability"] = round(win_prob, 2)
        updated["favorite_side"] = "home" if total_score <= 0 else "away"

    # Keep Honest-EV base (pre-decorrelation) and nested sport preds in step with
    # the board shift so Hubáček gates / EV do not silently disagree with display.
    if updated.get("pre_decorrelation_home_win_probability") is not None:
        pre = float(updated["pre_decorrelation_home_win_probability"])
        updated["pre_context_pre_decorrelation_home_win_probability"] = round(pre, 2)
        if updated.get("threeway") and updated.get("draw_probability") is not None:
            # Match threeway board clamps (non-draw mass), not binary 5–95.
            draw_for_pre = float(updated["draw_probability"])
            remaining_pre = max(100.0 - draw_for_pre, 0.0)
            lo_pre = min(1.0, remaining_pre)
            hi_pre = max(lo_pre, min(97.0, remaining_pre - min(1.0, remaining_pre)))
            new_pre = _clamp(pre + total_shift, lo_pre, hi_pre)
            new_pre = min(max(new_pre, 0.0), remaining_pre)
            updated["pre_decorrelation_home_win_probability"] = round(new_pre, 2)
        else:
            updated["pre_decorrelation_home_win_probability"] = round(
                _clamp(pre + total_shift, 5.0, 95.0), 2
            )
    for key in ("hockey_pred", "basketball_pred", "baseball_pred", "football_pred"):
        pred = updated.get(key)
        if not isinstance(pred, dict) or pred.get("home_win_probability") is None:
            continue
        nested = dict(pred)
        nested_home = float(nested["home_win_probability"])
        nested["home_win_probability"] = round(
            _clamp(nested_home + total_shift, 5.0, 95.0), 2
        )
        if nested.get("away_win_probability") is not None:
            nested["away_win_probability"] = round(
                100.0 - float(nested["home_win_probability"]), 2
            )
        if nested.get("pre_decorrelation_home_win_probability") is not None:
            nested_pre = float(nested["pre_decorrelation_home_win_probability"])
            nested["pre_decorrelation_home_win_probability"] = round(
                _clamp(nested_pre + total_shift, 5.0, 95.0), 2
            )
        updated[key] = nested

    # Soccer nested preds must follow the (possibly renormalized) 1X2 board.
    soccer_pred = updated.get("soccer_pred")
    if isinstance(soccer_pred, dict) and updated.get("threeway"):
        nested = dict(soccer_pred)
        if updated.get("home_win_probability") is not None:
            nested["home_win_probability"] = updated["home_win_probability"]
        if updated.get("draw_probability") is not None:
            nested["draw_probability"] = updated["draw_probability"]
        if updated.get("away_win_probability") is not None:
            nested["away_win_probability"] = updated["away_win_probability"]
        updated["soccer_pred"] = nested

    updated["context_adjustment_pp"] = round(total_shift, 3)
    updated["context_signals"] = {
        "flb_pp": flb,
        "steam_pp": steam,
        "news_pp": news,
        "total_pp": round(total_shift, 3),
        "league": league,
        "games_played_proxy": _games_played_proxy_from_blend(blended),
    }
    return updated
