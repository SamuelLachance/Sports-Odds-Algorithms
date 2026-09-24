# Pre-registration — NFL + NHL breakthrough program (2026-09-24)

Written before any screen runs. Nothing below may be edited after results exist;
amendments go in a new dated section at the bottom.

## Goal and honest framing

Samuel's bar: NFL and NHL models of the same professional grade as MLB. MLB
beats the vig but not vig-plus-the-book's-cut; the live EDGE tracker is at
break-even after 131 settled bets (70-61, -0.06u, ROI -0.04%).

Measured against each league's de-vigged closing line:

| league | model LL | close LL | gap |
|---|---|---|---|
| MLB | 0.6759 | 0.6736 | 0.0023 |
| NFL | 0.6192 | ~0.609 | ~0.010 |
| NHL | 0.6642 | ~0.655 | ~0.009 |

A "breakthrough" means closing a large share of the NFL/NHL gap with a
market-blind model. MLB closed most of its gap with DAY-OF PLAYER INFORMATION —
who starts on the mound, who is in the lineup, rated at the player level. The
working hypothesis is that NFL and NHL are further from the close because the
models carry less day-of player information, not because team ratings are badly
tuned. The NHL DEV sweep of 2026-08-11 (16 candidates, all null, harness
verified live) and the NFL ledger (75 rows) both say team-level history is
exhausted.

## Constitution (non-negotiable)

- **Market-blind.** Odds are never a model input or training signal. They may
  be used ONLY as an evaluation benchmark — including to locate where the model
  loses to the close (gap anatomy) on DEV seasons.
- **Locked splits.** NFL DEV 2006-2015 / TEST 2016+. NHL DEV 2010-11..2017-18
  (warm 2011-12) / TEST 2018-19..2025-26. Screens touch DEV only.
- **TEST looks are the lead engineer's only.** No agent computes a TEST-season
  metric, ever. Every TEST look gets a ledger row whether it ships or dies.
- **Walk-forward features.** Any player/team rating used as a feature for season
  s must be fit on data strictly before the game it describes. A rating fit on
  all seasons and then used on DEV games is a leak and voids the screen.

## Bar for a TEST look

A candidate earns one TEST look only if, on DEV, against the league's shipped
blend with everything else held fixed:

- **NFL:** paired improvement >= **+0.00150** nats with bootstrap 95% CI
  excluding zero.
- **NHL:** paired improvement >= **+0.00100** nats with bootstrap 95% CI
  excluding zero (same bar as the 2026-08-11 sweep).

AND it survives an adversarial leak audit (no future information in any input,
walk-forward ratings, identical game masks across arms), AND the screen's
harness reproduces a known effect of the shipped model (dropping its biggest
feature must cost roughly what the ledger says it costs) — so a null cannot be a
blind harness and a win cannot be a broken one.

At most three candidates per league go to TEST, best DEV first. Two candidates
within 0.0002 on DEV: the simpler wins.

## What is out of scope

Re-tuning shipped hyperparameters, re-opening ledger rows already closed without
new information, and anything the 2026-08-11 NHL sweep already screened
(score-adjusted/5v5/split xG, HFA drift, finishing, special teams beside 5v5,
fast/slow Elo, curvature, season-phase, OT rate, density, trips).

## Amendment 2026-09-24 (after the DEV round) — the NHL gap in the table above is wrong

The table's NHL row ("close ~0.655, gap ~0.009") came from a note in NHL ledger
row 7 with no recorded provenance. The project's own like-for-like measurement on
DEV (`data/nhl_floor.json`, SBR open/close archive `data/odds_nhl.csv`, 7,927 of
7,929 DEV games joined, 0 result mismatches) puts the de-vigged close at
**0.6707**, and the shipped model at 0.6724:

| league | like-for-like gap to the close | source |
|---|---|---|
| MLB | 0.0023 | TEST |
| NHL | **0.0017** (late DEV 2015-18: 0.0033) | DEV |
| NFL | **0.0117** | DEV, 2,531 games with a close |

The NHL model is level with the OPENING line on DEV (model - open -0.00016,
CI [-0.00217,+0.00177]); its whole gap is the open-to-close move, 94% of it on
nights a team starts a non-#1 goalie, the rest on top-minute skater absences
(`data/bt_nhl_anatomy*.json`). No TEST-period market comparison was computed, in
keeping with the floor programme's test_never_touched protocol.

Consequence for the program: NHL is already MLB-grade relative to the market; its
remaining headroom is day-of personnel information that becomes servable only with
confirmed goalies and lineups. NFL is the league genuinely far from the close.

## Results of the DEV round

| candidate | DEV gain | 95% CI | outcome |
|---|---|---|---|
| NFL movcore (MOV Elo replaces W/L core) | +0.00236 | [+0.00087,+0.00385] | cleared; **TEST +0.00053, CI [-0.00070,+0.00176], n.s. — ledger row 76, not shipped** |
| NFL schedadj (opponent-adjusted efficiency) | +0.00195 | [+0.00029,+0.00365] | failed its own pre-registered 2002-05 replication guard; no look |
| NFL clinchx (locked-seed late-season games) | +0.00189 | [-0.00044,+0.00428] | null; audit flagged post-game QB information in the shared baseline harness |
| NHL gmar (goalie inside the Elo, MLB-style) | +0.00064 | [-0.00024,+0.00154] | null, sound |
| NHL rapmel (walk-forward RAPM inside the ratings) | +0.00029 | [-0.00035,+0.00094] | null |
| NHL boxel | — | — | not run: needs ~9.4k NHL API box-score pulls |
