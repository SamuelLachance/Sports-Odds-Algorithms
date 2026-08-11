# NHL DEV screens, 2026-08-11 — sixteen candidates, no TEST look spent

Pre-registered in `documents/nhl_xg_measurement_prereg_2026_08_11.md`. The bar
was fixed before anything ran: **>= +0.00100 nats on pooled DEV with a bootstrap
95% CI excluding zero**, at most one candidate promoted to a TEST look.

Nothing cleared it. **No TEST look was taken and the ledger is unchanged at ten
rows.** The shipped NHL model stays at TEST 0.66418.

## Protocol

DEV only, leave-one-season-out over 2011-12..2017-18 (7 folds, 7,929 games): for
each DEV season the blend is fit on the other six and scored on the held-out one,
then the held-out predictions are pooled. Every candidate is a modification to
the shipped 4-feature blend (Elo logit + rest + b2b + xG) with everything else
held fixed, so a difference is attributable to the change and nothing else.

Scripts: `phase0/nhl_xg_adjust.py`, `phase0/nhl_xg_screen.py`,
`phase0/nhl_core_screen.py`, `phase0/nhl_form_screen.py`.

## Round 1 — is the xG rating badly MEASURED?

The xG rating is the model's biggest feature (+0.00316 on TEST). It is built on
raw xG margin, which is a known-biased measure of team strength: trailing teams
shoot more, leading teams sit back. `phase0/nhl_xg_adjust.py` reconstructs the
running score and strength state at every one of the shot file's shots and
rebuilds team xG four ways. Score-adjustment weights are fitted on DEV seasons
only and are correctly signed and monotone (all-situations 0.926 at -3 goals to
1.074 at +3; 5v5 0.913 to 1.087).

| candidate | DEV LL | vs RAW | 95% CI | verdict |
|---|---|---|---|---|
| RAW (ships) | 0.67239 | — | — | — |
| ADJ score-adjusted | 0.67245 | -0.00006 | [-0.00016,+0.00004] | n.s. |
| EV 5v5 only | 0.67226 | +0.00013 | [-0.00093,+0.00119] | n.s. |
| ADJEV both | 0.67230 | +0.00009 | [-0.00094,+0.00113] | n.s. |
| SPLIT off/def ratings | 0.67245 | -0.00005 | [-0.00056,+0.00044] | n.s. |
| SPLITADJ | 0.67246 | -0.00007 | [-0.00058,+0.00044] | n.s. |

**The harness is not blind and the candidates are not identical.** Dropping xG
entirely costs +0.00304 on this same DEV set (0.67543 -> 0.67239), matching the
+0.00316 the feature scored on TEST — so the screen sees an effect of the size
worth shipping. And 5v5-only correlates just +0.86 with raw (std 0.33 vs 0.49):
a genuinely different measurement that produces the same log-loss.

This is the pre-registered falsification condition, and it fired. **Channel
closed:** how the shots are weighted does not matter. The Elo core trains on
results, which embed score effects symmetrically, and the rating averages over
many games, so the +/-7-9% per-shot adjustment washes out.

## Round 2 — the rating CORE and its context

| candidate | DEV LL | vs shipped | 95% CI | verdict |
|---|---|---|---|---|
| SHIPPED | 0.67239 | — | — | — |
| HFA_DRIFT rolling home-ice | 0.67246 | -0.00006 | [-0.00031,+0.00018] | n.s. |
| FINISH goals-above-expected rating | 0.67234 | +0.00005 | [-0.00033,+0.00043] | n.s. |
| ST special teams beside 5v5 | 0.67216 | +0.00023 | [-0.00033,+0.00078] | n.s. |
| FAST_SLOW two Elo timescales | 0.67240 | -0.00000 | [-0.00035,+0.00035] | n.s. |
| ALL three | 0.67213 | +0.00027 | [-0.00046,+0.00099] | n.s. |

HFA_DRIFT is worth noting: the NFL model carries a rolling HFA and it helps
there. In the NHL it does nothing — a constant `ha=30` is sufficient across all
sixteen seasons.

## Round 3 — the model's FORM rather than its inputs

Thirteen nulls in, every one of them was another number bolted onto a
linear-in-logit blend. Round 3 changed the blend and left the inputs alone.

| candidate | DEV LL | vs shipped | 95% CI | verdict |
|---|---|---|---|---|
| SHIPPED | 0.67239 | — | — | — |
| NONLIN logit^2 + logit^3 | 0.67260 | -0.00021 | [-0.00043,+0.00000] | n.s. |
| EARLY logit x games-played | 0.67263 | -0.00024 | [-0.00046,-0.00002] | **SIG WORSE** |
| OT_RATE close-game tendency | 0.67237 | +0.00002 | [-0.00035,+0.00040] | n.s. |
| DENSITY games in last 5 days | 0.67259 | -0.00020 | [-0.00032,-0.00007] | **SIG WORSE** |
| TRIP homestand / road-trip length | 0.67250 | -0.00010 | [-0.00024,+0.00003] | n.s. |
| ALL | 0.67320 | -0.00080 | [-0.00132,-0.00029] | **SIG WORSE** |

NONLIN is the informative one. 23.0% of NHL games reach overtime, where the
result is far closer to a coin flip than team strength implies, so the tails
*should* be compressed relative to a linear-in-logit blend. Adding curvature
makes it slightly worse — the single fitted slope already absorbs it. The
functional form is right; it was not hiding anything.

## What this means

Twenty channels have now been tested against this model — ten TEST rows
(`data/nhl_test_ledger.csv`) and these sixteen DEV candidates. Everything after
the original four-feature blend has come back flat, and the nulls now cover
every family:

- **better team-quality signal** — xG measurement (round 1), finishing, special
  teams, multi-timescale ratings, Glicko-2 (ledger row 2)
- **player-level decomposition** — shift-TrueSkill, RAPM (row 8)
- **goaltending** — GSAx differential (row 6), starter-vs-baseline delta (row 9)
- **schedule and context** — rest and b2b ship; density, trips, HFA drift,
  season phase do not
- **functional form** — curvature, interactions

The one channel that has never come back flat is **who is actually dressed**:
ledger row 10 (trailing TOI of absent regulars) returned +0.00077, right-signed,
CI [-0.00009,+0.00164] — a hair from clearing. It was not shipped and gets no
second look on historical shift charts, whose 2025-26 coverage is 61.5% because
of an upstream NHL API gap. The deployable version needs in-season announced
lineups, which start accruing in October.

So the honest read: **0.66418 is at the limit of what team-level history
supports, and the remaining ~0.009 to the closing line is lineup and availability
information that does not exist in the historical record we have.** Further
feature search on this data is not a good use of TEST looks. The next real move
is the October lineup accrual, not another rating idea.
