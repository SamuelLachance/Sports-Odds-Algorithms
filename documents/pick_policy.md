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

## Implementation notes (added 2026-07-31, no rule changed)

These record how the rules above land on the actual code, including where the
implementation is weaker than the table implies. They are disclosures, not
amendments.

- **Other leagues are tiered by the same column.** The tier is decided by *what
  the model actually has*, not by which sport it is. The shipped NHL model is
  Elo + rest + back-to-back + an expected-goals **team** rating — no roster, no
  lineup, no goalie — so every NHL forecast is **EARLY**, and NHL is therefore
  not published as picks today. NFL is **PROJECTED** inside a week of kickoff,
  where the live model has depth charts and lineup power, and **EARLY** beyond
  that, where the number is the 20,000-run season simulation off team strength.
  A tier chip means the same thing on every league's page or it means nothing.
- **CONFIRMED is a start-time-selected subsample.** Rule 4 grades the last
  *pre-game* forecast, and the refresh runs every 4 hours. A game whose lineups
  post 2½ hours before a first pitch that falls before the next refresh is
  graded PROJECTED even though the lineups were public. Every label is true of
  the forecast it is attached to; the tier *populations* are not random samples
  of the slate, so the three tiers are not a controlled comparison.
- **CONFIRMED needs a lineup card, not nine names.** Serving accepts an official
  lineup at ≥5 batters per side and pads the rest with roster averages
  (unchanged — touching it would change probabilities). The UI says "official
  lineup card posted" rather than "both batting orders".
- **PROJECTED includes rotation-projected starters.** `pitcher_known` is true
  for a near-term rotation projection as well as an announced probable; the
  model conditions on a named arm either way, so both are PROJECTED. Those rows
  ship `sp_proj` and are marked *SP proj* wherever they are listed.

## Honest note on the evidence

Whether the EARLY tier is *worth publishing at all* is an empirical question we
cannot answer yet: the prediction ledger began on 2026-07-31 and has 3 graded
rows. Once each tier has a real sample, its realized log loss against the
coin-flip baseline decides whether it stays. Until then the tiers are separated
and labelled, and no claim is made about them.

## Addendum 2026-07-31 — the EV badge, per league

An EV badge is a **stronger** claim than a pick: it asserts the market is
mispriced by >=20%. It must therefore clear at least the same information bar,
and the bar differs by league because the leagues expose different information.

| league | information gate today | known deficit at badge time |
|---|---|---|
| **MLB** | **ENFORCED** — an EARLY card (no named starter) is skipped before it can be quoted or badged (`market/edges.py`, 4 tests). Latent when added: the odds feed reaches ~2 days, so no EARLY game was being badged. | none material |
| **NFL** | none needed *today*: there is no EARLY analogue — rosters and QBs are known a week out, so every pre-kickoff card is PROJECTED. | the badge window is 7 days but the **final injury report lands Friday**, so a badge placed early in the week is priced without it. A gate becomes possible when the 2026 injury feed lands (September) — pending, not forgotten. |
| **NHL** | none possible yet: the payload carries no information-state field because no goalie/lineup feed exists until the archive has volume (October). | **quantified and material.** `data/market_projection.json` measured that the closing line's information orthogonal to our features is worth **+0.00183** — *more than our entire 0.00171 gap to the market* — and the weight table points at goalie confirmation and scratches as its content. Starting goalies are confirmed on game-day morning, so a badge placed days ahead is priced **without the single thing we know the market has and we lack.** |

**Decision.** MLB is gated structurally. NFL and NHL are **disclosed rather than
gated**, because a window number I cannot justify from data would be a guess
dressed as a safeguard — and shortening the window reduces exposure without
removing it (we have no goalie feed at *any* horizon yet). The NHL badge carries
the deficit in its own tooltip, and both gates are wired the day their feeds
land: NFL injuries in September, NHL goalie confirmations in October.
