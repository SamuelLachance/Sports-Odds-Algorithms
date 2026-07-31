# Pick policy — what counts as a pick, and when we make one

Written 2026-07-31 after Samuel asked, correctly: *why are we publishing so many
picks in advance, and why do we pick at all before lineups are confirmed?*

## The problem this fixes

The board ran a 30-day horizon and the site treated every scheduled game as a
"pick". Measured on 2026-07-31: **420 published picks, of which 375 (89%) had no
confirmed starting pitcher** — some of them 30 days out, where the model has
nothing but team ratings. A 30-days-out MLB forecast and a 90-minutes-out
forecast with both starters and lineups confirmed are **different products**, and
publishing them under one label — then grading them into one accuracy number —
is dishonest in both directions: it overstates how much work is behind the early
ones and understates the informed ones.

There was also no *strategy*: the pick was simply "whichever side is above 50%",
so a 50.4% coin-flip was presented exactly like a 68% call.

## Information tiers (the honest unit)

Every forecast is stamped with the information available **at the moment it was
made**, and nothing is ever compared across tiers:

| tier | condition | what the model actually has |
|---|---|---|
| **CONFIRMED** | official lineups posted | starters + both batting orders + bullpen state |
| **PROJECTED** | probable starters announced (typically ≤2 days out) | starters + roster-average lineup |
| **EARLY** | no starter announced | team ratings only |

MLB serving already computes these distinctions (`pitcher_known`,
`lineup_source`, and the blend tier). This policy makes them first-class in the
product instead of an implementation detail.

## The rules

1. **A pick is only published as a pick at CONFIRMED or PROJECTED tier.** EARLY
   games are shown as *scheduled*, with the model's current lean available but
   explicitly labelled as pre-information and excluded from pick counts.
2. **Confidence is stated, never implied.** A forecast within 55/45 is labelled a
   *lean*, not a pick. Above that it is a pick. The probability is always shown;
   the label only tells the reader how much to weight it.
3. **The track record splits by tier.** Accuracy and log loss are reported per
   tier and never pooled into a single headline. Pooling would let a good
   CONFIRMED record hide behind a large volume of EARLY noise, or vice versa.
4. **The graded pick is the last pre-game forecast.** The ledger already updates
   while a game is strictly pre-game, so what gets graded is the most-informed
   version — not the 30-days-out guess. The tier stamp records which it was.
5. **We do not manufacture volume.** The number of picks on a given day is
   whatever that day's slate and information state produce. There is no target.

## What this does NOT change

The model, the probabilities, or any protocol. This is a presentation and
accounting policy: the same numbers, correctly labelled and correctly separated.
The edge layer's own gate (≥20% EV vs the opening price, unchanged) is a separate
question from whether a forecast is published as a pick.

## Honest note on the evidence

Whether the EARLY tier is *worth publishing at all* is an empirical question we
cannot answer yet: the prediction ledger began on 2026-07-31 and has 3 graded
rows. Once each tier has a real sample, its realized log loss against the
coin-flip baseline decides whether it stays. Until then the tiers are separated
and labelled, and no claim is made about them.
