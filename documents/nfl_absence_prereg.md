# Pre-registration — NFL absence-block repair screen

Written and committed 2026-07-31, **before any candidate was screened**. The git
timestamp on this file is the pre-registration proof.

## The defect

`phase0/nfl_headroom.py` measured (diagnostic, not a proposal) that the four
availability columns of the NFL X14 — `ol_absence`, `def_absence`,
`skill_absence`, `roster_quality` — **cost** log loss rather than earning it:

| window | cost of the block |
|---|---|
| joined DEV (2006-2015) | **+0.00132** [−0.0013, +0.0039] |
| 2013-2015 only | **+0.00417** |

Mechanism: the snap data these columns need starts in 2013, so they are
identically zero for 2006-2012 and live for only the final 2-3 DEV seasons. The
walk-forward blend therefore fits four coefficients on one or two seasons of
non-degenerate data. That is a textbook small-sample fitting failure, not a
statement about whether player availability matters in football.

Corroborating: the shipped walk-forward scores the 2013-15 window **worse**
(0.62400) than a single fit trained through 2012 (0.62090) — backwards unless
the late-arriving columns are hurting.

## Why this needs its own pre-registration

The Round 1/2 gates (+0.0020 for MLB/NFL new features) were written for features
that must earn their complexity. This is the opposite case: a **repair** of an
existing, already-shipped block that is currently *negative*. Choosing a bar for
it after seeing candidate results would be the exact sin the protocol prevents,
so the bar is fixed here, in advance.

## Pre-registered design

**Split.** NFL main DEV only (scored 2006-2015). TEST (2016+) is never touched.
Nothing here can be evaluated on TEST without a separate, single, recorded look.

**Baseline.** The shipped X14 walk-forward, DEV CV — reproduce it exactly before
screening anything and assert equality to the recorded 0.62296 within run jitter.

**Candidates** (all four screened; no post-hoc additions):
1. **Drop** — remove the four columns entirely.
2. **Pool** — collapse the three absence columns into one total-absence column
   (4 params -> 2), keeping `roster_quality` separate.
3. **Shrink** — keep all four but apply a much stronger ridge penalty to that
   block alone (the rest of the blend keeps its shipped C), sweeping the block
   penalty over a pre-declared grid.
4. **Gate** — zero the block's contribution until a pre-declared minimum number
   of live seasons is available to fit it, then let it switch on.

**Metric.** Walk-forward DEV log loss, the same objective the blend is fitted on.

**Method standard** (carried from Round 1, non-negotiable): every candidate is
reported RAW and **NESTED** — the winning variant re-selected without seeing the
fold it is scored on. Fold-to-fold noise is quantified and every gain is
expressed in units of it. The nested number is the one that counts.

## The bar, fixed in advance

- **ADOPT** if the nested gain is **>= +0.0010** in DEV log loss *and* positive in
  at least 7 of the 10 scored seasons. Rationale for a bar below the
  new-feature gate: this is a repair with **negative** complexity — candidates 1
  and 2 *remove* parameters — so it need not clear the bar a new feature must.
  It is set at half the new-feature gate, well above the measured fold noise,
  and above the +0.00057 residual-descent noise floor measured in the learning
  curve.
- **REJECT** otherwise, including the case where the block is merely
  "not harmful". No re-screening, no bar adjustment.
- If a candidate is adopted, it costs **one TEST look**, recorded in
  `data/nfl_test_ledger.csv` with the result whichever way it lands.

## Explicitly out of scope

Restoring the v7 participation column for DEV (impossible — nflverse
participation data begins 2016) and any change to the ratings engines. This
screen touches the blend's treatment of four availability columns and nothing
else.
