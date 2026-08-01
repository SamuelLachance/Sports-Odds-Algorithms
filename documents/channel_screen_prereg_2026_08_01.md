# Pre-registration — NFL environment / NHL xG-decomposition channel screens

Written and committed **while the screens were still running and before any
result was seen**. The git timestamp on this file is the pre-registration proof.
It exists because the alternative — reading ten results and then deciding what
counts as a win — is how a null becomes a "finding".

## What is being screened, and why these ten

Samuel asked for continued NHL and NFL model improvement. The prior claim in
`depth_program.md` is that the remaining gap is missing INFORMATION rather than
mis-specification. That claim is only credible if the obvious information
channels have actually been tried, so I checked which had not.

**NFL — six channels, all with data already in `data/nfl_games.csv`, none in the
README's screened list:**

| channel | column | DEV coverage |
|---|---|---|
| wind | `wind` | 1,957 / 2,670 (the outdoor games) |
| temperature | `temp` | 1,957 / 2,670 |
| roof type | `roof` | 2,670 / 2,670 (outdoors 1957, dome 415, closed 236, open 62) |
| surface | `surface` | 2,670 / 2,670 |
| divisional | `div_game` | 2,670 / 2,670 (974 divisional) |
| referee crew | `referee` | populated |

**NHL — four channels:**

- **time-varying home ice.** The model uses a FIXED Elo `ha=30` across 2011-2026
  while the per-season home win rate ranges 0.519-0.568. NFL already ships a
  time-varying HFA and it earned its place; this is that transplant.
- **shot quality vs volume.** `nhl_team_xg.csv` carries `xgf` and `shots_for`,
  so quality (xg per shot) and volume are separable. The shipped single xG
  rating collapses them.
- **finishing above expected** (`gf - xgf`).
- **goaltending above expected** (`xga - ga`) — the most theoretically motivated
  of the four, because the market-gap analysis says the closing line prices
  goalie information we have no column for.

## The bar, fixed in advance

A channel is **ADOPTED** only if all four hold:

1. **nested** DEV delta ≥ **+0.0010** (not the raw best cell);
2. the raw-to-nested gap is **under 60%** of the raw delta — a candidate that is
   mostly search optimism is rejected even if the nested figure clears (1);
3. it helps in **≥ 6 of 10** NFL DEV seasons, or **≥ 5 of 7** NHL DEV seasons;
4. it **survives an independent adversarial rebuild** whose author is instructed
   to refute it and defaults to rejection.

Anything below +0.0010 nested is **NULL** regardless of how it looks. Anything
below **1e-4** is not even discussed: these harnesses run through sklearn's
iterative solver and reproduce only to ~1e-4 (which is why the DEV asserts carry
a 5e-4 tolerance), so a delta under 1e-4 is solver noise.

## Specific traps I am pre-committing to treat as disqualifying

- **Referee** has many levels and few games each. A raw gain there is the single
  most likely overfit in the batch. Nested-only.
- **Finishing** and **goaltending** residuals are built from REALIZED goals,
  while the shipped xG rating deliberately updates on expected goals only,
  precisely to avoid re-importing outcome noise. If either shows a gain, the
  default explanation is outcome leakage until an independent rebuild says
  otherwise.
- **Weather** applies to 73% of games. A gain that lives entirely in how the
  missing 27% are encoded is an artefact of the imputation, not a weather
  effect.

## What happens next, either way

- **All null** → the strongest evidence yet for the missing-information thesis,
  because these are the channels a critic would name first. It gets written into
  `depth_program.md` as a round, not buried.
- **Something survives** → it is a DEV result only. Promotion still costs a TEST
  look, which is mine alone to spend, and gets a ledger row whether it ships or
  dies.

No TEST data was touched by any screen. The agents were instructed that the TEST
era is off-limits and that a null is the expected, valuable outcome.
