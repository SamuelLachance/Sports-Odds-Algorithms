# Pre-registration — NHL xG measurement quality (2026-08-11)

Written BEFORE any screen is run. No TEST look is taken under this document
without the DEV bar below being cleared first.

## Why this channel

Ten TEST looks are on the NHL ledger (`data/nhl_test_ledger.csv`). The four most
recent are flat nulls, and they are all the same kind of idea: bolt an extra
*player-level* signal onto the team model (goalie GSAx, per-shift TrueSkill,
RAPM, starting-goalie delta, scratch/absence). Rows 6 and 8 say why they failed
in the same words — the team xG rating already prices what those features carry,
and decomposing it to players reconstructs the same quantity with more noise.

So stop adding features beside the xG rating. The xG rating IS the model's
biggest single feature (+0.00316 on TEST, row 5, larger than rest and b2b
combined). It is currently built from **raw** xG margin, which is a known-biased
measurement of team strength:

1. **Score effects.** A trailing team shoots much more and a leading team sits
   back. Raw xG therefore partly measures *how the game went*, not how good the
   team is. This is the single best-documented bias in public hockey analytics
   and the reason score-adjustment is standard there.
2. **Strength state.** Power-play and penalty-kill xG are pooled into the same
   rating as 5v5. Special-teams results are lower-volume and much less
   repeatable season-to-season than 5v5, so pooling adds variance to the rating
   without adding much signal.
3. **One margin rating.** Offence and defence are collapsed into a single number.
   A team that generates 3.0 and allows 2.5 is rated identically to one that
   generates 2.0 and allows 1.5, though those are different teams facing
   different opponents.

None of this needs new data. `data/nhl_shots.csv` already carries per-shot xg,
goal, shooting team, period, seconds, and on-ice skater counts — enough to
reconstruct the running score at every shot and the strength state it was taken
in.

## Candidates

Each replaces the `xg_diff` feature in the shipped 4-feature blend. Everything
else (Elo core, rest, b2b, blend form) is held fixed.

| id | construction |
|----|--------------|
| RAW | shipped raw xG-margin rating — the thing to beat |
| ADJ | score-adjusted xG rating (weights estimated on DEV only) |
| EV | 5v5-only xG rating |
| ADJEV | score-adjusted, 5v5-only |
| SPLIT | separate xGF and xGA ratings, entered as two features |

## The bar (pre-registered)

Screened on **DEV only**, leave-one-season-out across DEV seasons
2011-12..2017-18: for each DEV season, fit the blend on the other DEV seasons and
score that season; pool the held-out predictions.

A candidate earns the single TEST look only if **both** hold against RAW on that
pooled DEV set:

- point estimate **>= +0.00100** nats, and
- bootstrap 95% CI on the paired per-game difference **excludes zero**.

Rationale for +0.00100: the raw xG feature is worth +0.00316 on TEST. A
*measurement* fix to it that is worth less than roughly a third of the feature
itself is not distinguishable from tuning noise at this sample size, and the
ledger already has four nulls from chasing smaller things.

**At most ONE candidate goes to TEST** — the best DEV performer that clears the
bar. If none clears it, no TEST look is taken and the channel is logged as
screened-and-dropped. If two are within 0.0002 of each other on DEV, the simpler
construction wins.

Score-adjustment weights are estimated on DEV seasons only and then applied
unchanged to every season, so no TEST information enters the feature definition.

## What would falsify the channel

If ADJ and ADJEV both come back flat on DEV, then score effects are already
absorbed by the Elo core (which trains on results, and results embed score
effects symmetrically), and xG *measurement* is not where the remaining NHL
headroom is. That is a real finding and closes the channel — it should be written
to the ledger as such rather than re-opened with a different weighting scheme.
