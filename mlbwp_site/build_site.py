"""Generate the GLASSBOX SPA shell (site/index.html).

The page fetches ./data/board.json and ./data/db.json at runtime and renders four
views through a hash router: the predictions board, a per-game deep dive, the
standings, and the model's rankings. Rendering only; no model logic, no market
data. The data files are produced by mlbwp.predict_slate and mlbwp.db.
"""

from __future__ import annotations

import sys
from pathlib import Path

from mlbwp.pred_ledger import TIER_JS  # single source of truth for the tier rule
from mlbwp.predict_slate import LEAN_JS  # single source of truth for the lean threshold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase0"))
from nhl_contributions import CT_JS  # noqa: E402 — single source for the NHL ct keys

SITE = Path(__file__).resolve().parents[1] / "site"

_HERE = Path(__file__).resolve().parent
# The front end lives in per-area source files so leagues can be worked on
# independently; they are concatenated here into the single page. JS function
# declarations are hoisted, so file order only matters for top-level consts,
# which all live in core.js and record.js in their original order.
CSS_PARTS = ("core", "nhl", "nfl")
JS_PARTS = ("core", "nhl", "nfl", "mlb", "record")
CSS = "".join((_HERE / "css" / f"{p}.css").read_text(encoding="utf-8") for p in CSS_PARTS)
JS = "\n".join((_HERE / "js" / f"{p}.js").read_text(encoding="utf-8") for p in JS_PARTS)


# The tier rule is owned by mlbwp/pred_ledger.py so the Python stamp and the
# JS label can never drift (a drift there is silent and dishonest). JS is a
# RAW string -- it is full of ${...} template literals -- so this is an
# explicit substitution, not an f-string; the assert makes a missed or
# renamed placeholder a hard build failure instead of broken site JS.
assert "{TIER_JS}" in JS, "tier-rule placeholder missing from JS"
assert "{CT_JS}" in JS, "NHL contribution-key placeholder missing from JS"
assert "{LEAN_JS}" in JS, "lean-threshold placeholder missing from JS"
JS = JS.replace("{TIER_JS}", TIER_JS).replace("{LEAN_JS}", LEAN_JS).replace("{CT_JS}", CT_JS)

# A real document head: without the viewport meta, phones lay the page out at a
# 980px virtual width and every max-width @media rule in the CSS is dead; without
# a doctype the page renders in quirks mode. route() sets document.title per view.
SHELL = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GLASSBOX &middot; market-blind MLB, NFL &amp; NHL predictions</title>
<meta name="description" content="Market-blind model predictions for MLB, NFL and NHL games, with every pick graded in public by information tier. The models never see the odds.">
<meta name="color-scheme" content="dark light">
<meta name="theme-color" content="#0d0f13">
<style>{CSS}</style>
<script>try{{var t=localStorage.getItem('theme');document.documentElement.setAttribute('data-theme',t==='light'?'light':'dark');}}catch(e){{document.documentElement.setAttribute('data-theme','dark');}}</script>
</head>
<body>
<header><div class="wrap">
  <a class="brand" href="#/">GLASS<span class="b">BOX</span></a>
  <nav class="main">
    <a href="#/" data-v="board">Board</a>
    <a href="#/season" data-v="season" id="navseason" style="display:none">Season</a>
    <a href="#/record" data-v="record">Track record</a>
    <a href="#/standings" data-v="standings">Standings</a>
    <a href="#/teams" data-v="teams">Teams</a>
    <a href="#/players" data-v="players">Players</a>
  </nav>
  <span class="grow"></span>
  <span class="updated" id="updated"></span>
  <span class="acc" id="acc"></span>
  <button class="tog" id="tog" aria-label="Toggle theme">&#9681;</button>
</div></header>
<main><div class="wrap" id="view"><div class="loading">Loading predictions&hellip;</div></div></main>
<footer><div class="wrap">
  <b>Research only &mdash; not betting advice.</b> Market-blind models that never see the odds; every
  pick is locked before the game and graded in public on the <a href="#/record" id="ft-rec">track record</a>, split
  by information tier.
  <b>MLB</b>: team Elo + xFIP &amp; SIERA starting-pitcher ratings + season-to-date bullpen FIP +
  per-plate-appearance TrueSkill on-base ratings + lineup isolated-power + lineup baserunning.
  <b>NFL</b>: 14-feature blend around an 11v11 per-snap participation TrueSkill<span id="ft-nfl"></span>.
  <b>NHL</b>: tuned Elo + rest &amp; back-to-back + an expected-goals team rating (updated on shot
  quality, never on results); no lineup or goalie input, so every NHL forecast is published as an
  EARLY-tier lean, not a pick; skater cards show teammate-adjusted xG/60 (RAPM, display only).
  Data: <b>Retrosheet</b> (free of charge, copyrighted by Retrosheet,
  <a href="https://www.retrosheet.org">retrosheet.org</a>) and the MLB Stats API (individual,
  non-commercial use); NFL data from <b>nflverse</b> (community-maintained); NHL data from the
  NHL Stats API and shot xG from <b>MoneyPuck.com</b> (used with attribution, non-commercial research);
  NFL and NHL live scores from ESPN's public scoreboard (display only &mdash; never a model input,
  never used to grade).
  <br><span id="ft-fresh"></span>
</div></footer>
<script>{JS}</script>
</body>
</html>
"""


def build():
    body = SHELL.encode("ascii", "xmlcharrefreplace").decode("ascii")
    SITE.mkdir(exist_ok=True)
    (SITE / "index.html").write_text(body, encoding="utf-8")
    print(f"wrote {SITE/'index.html'} ({len(body):,} bytes)")


if __name__ == "__main__":
    build()
