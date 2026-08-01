# Depth program — squeeze every component of every model to its limit

Opened 2026-07-31 on Samuel's directive: *improve every single aspect of each
model, step by step, market-blind, until the theoretical limit of prediction.*

## Why this exists

Past campaigns were **breadth**: bolt a NEW channel onto a frozen blend, screen
it, record a null (~60 nulls across three ledgers). What was never done is
**depth**: re-examining the components we already ship. Every league's model is
a linear logistic blend over features whose own hyperparameters were tuned once,
greedily, one at a time, and never revisited jointly. Interactions, nonlinear
forms, model class, calibration and shrinkage are all untouched.

Rules unchanged: market-blind (no odds in any feature), DEV screens only,
one TEST look per adopted change, ledger row either way.

## The axis grid (each cell = one screenable step)

| # | Axis | MLB | NFL | NHL |
|---|---|---|---|---|
| A | Core rating hyperparameters, **jointly** re-tuned (not greedily) | FIP/xFIP-Elo k, HFA, season regression | v7 β, τ, draw band, leverage weights | Elo k/ha/regress **with** xG-Elo K/HA/regress |
| B | Player-rating engine internals | per-PA TrueSkill β, τ, draw, priors | QB Kalman R, opp-quality k, σ-discount | (n/a — no shipped player feature) |
| C | Blend model class on the SAME features | logistic vs GBM vs stack | same | same |
| D | Feature functional form | splines/bins on continuous features | same | rest/B2B nonlinearity |
| E | Interactions | starter×bullpen, Elo×rest, park×power | QB×opponent-D, rest×travel | xG×Elo, rest×B2B |
| F | Regularization / shrinkage | blend C, per-feature priors | C, recency half-life | C, feature priors |
| G | Calibration | Platt vs isotonic, per-tier, drift | same | same |
| H | Training protocol | recency weighting, walk-forward cadence | half-life (currently 3) | warm-start window |
| I | Ensembling | tier-average, seed-average | same | same |

## Status

Round 1 launched 2026-07-31 (axes A, C, D, E, F per league; G-I in round 2).
Results land in `data/{mlb,nfl,nhl}_depth_dev.json` and are summarized here as
each round closes. A cell is CLOSED only when a DEV screen exists for it —
"we already tuned that once" is not a closure.

## Closure log

### Round 1 — 2026-07-31. Axes A-F × 3 leagues. **0 adoptions, 0 TEST looks spent.**

Every candidate was measured twice: the raw best-cell number, and a **nested**
estimate where the winning cell is re-chosen without seeing the fold it is
scored on. The gap between them is search optimism, and it was brutal.

| league | best candidate | raw | nested | optimism | pre-registered gate | verdict |
|---|---|---|---|---|---|---|
| MLB | TrueSkill read-time low-sample prior (n0≈200) | +0.00026 | **+0.00024** | 9% | +0.0020 | sub-gate, NOT adopted |
| MLB | A+B+F combined | +0.00065 | ≈+0.00051 | — | +0.0020 | sub-gate |
| NFL | v7 engine knobs, joint | +0.00497 | **+0.00092** | **82%** | 0.0020 | sub-gate, NOT adopted |
| NHL | 6-parameter joint core re-tune | +0.00020 | **−0.00007** | **130%** | +0.0005 | NULL |

**The premise of this round was half right.** For NFL, greedy one-at-a-time
re-tuning is worth *less than nothing* (nested −0.00062) while only the joint
search produces a positive honest estimate (+0.00092) — the mechanism is
visible: `lev_exp=0` is worse at the shipped anchor yet belongs to the joint
optimum, exactly the blind spot greedy tuning has. For NHL the premise is
simply **false**: greedy and joint differ by 0.6 SE, and 4,973 of 21,000 grid
points sit within 1 SE of the best. The shipped point was already on the
plateau.

**Three structural findings worth more than the numbers:**

1. **A rating's home-field parameter is a free parameter** when a logistic with
   an intercept sits downstream — the intercept absorbs a constant logit shift
   exactly. Same for any knob that merely scales the logit (`beta`, `k`). This
   is why decades of "Elo tuning" left nothing on the table here, and it means
   those knobs can only ever matter through their effect on the *online update*.
2. **MLB TrueSkill mu is read with no low-sample shrinkage** — a debut batter
   enters the lineup average at full weight. Adding `mu ← mu0 + (mu−mu0)·n/(n+n0)`
   at read time is SIG at every n0 from 25 to 400 with a smooth plateau. Real,
   mechanistically sound, and still 8× below the gate. Logged as a latent design
   gap, not adopted.
3. **The joint optima SIMPLIFY the engines.** NFL's best config turns *off* both
   leverage weighting and opponent-quality scaling. Complexity added in earlier
   campaigns is not paying rent.

**Closed as null across all three leagues:** model class (gradient boosting is
2.5-3 SE worse everywhere, train-CV gaps up to 0.016; stacks assign the GBM
negative weight), functional form (no curvature anywhere; MLB spline gains were
96% optimism), interactions (all null or SIG worse), regularization (blend C is
flat over 7 orders of magnitude — there is no variance to regularize).

**Protocol note.** The +0.0020 gate was pre-registered for NEW FEATURES, which
must earn their complexity. A zero-complexity correction like finding 2 arguably
deserves a different bar — but choosing that bar *after* seeing the result is
precisely the sin the protocol exists to prevent. If a lower bar for
zero-complexity corrections is ever wanted, it must be pre-registered in its own
document first, before any candidate is screened against it.

### Round 2 — 2026-07-31. Axes G-I + error map + floor. **0 adoptions, 0 TEST looks.**

| axis | MLB | NFL | NHL |
|---|---|---|---|
| G calibration | nested −0.000021 | null | null |
| H training cadence | nested −0.000013 | null | null |
| I ensembling | nested −0.000089 | null | null |

Largest raw gain anywhere: **+0.000008**, 250x below gate, and it inverts when
re-selected honestly. Three findings explain *why* these had to be null:

1. **Platt-on-a-rating is not calibration, it is a reparametrization** the blend
   already performs — verified algebraically (post-hoc Platt on the blend's own
   training predictions returns exactly (b0,b1)=(0,1) to 8 decimals). The round-1
   "free parameter" lesson, applied to the calibration axis.
2. **There is no calibration drift.** Per-season Platt slopes: Q=12.65 on 13 df
   (Q/df=0.97); intercepts Q/df=1.07. The season-to-season wobble is *exactly*
   sampling noise. The apparent cost of a "stale" curve decomposes into the
   oracle's own in-sample degrees of freedom (+0.000137 predicted vs +0.000234
   measured). No disease -> recency weighting (axis H) had nothing to cure, which
   is why it is monotonically harmful: half-life 0.5 seasons costs +0.000605,
   1-season windows +0.001252. **Expanding-window training is optimal.**
3. **Per-tier calibration IS load-bearing** — the one calibration choice that is
   not free. The Elo slope falls 1.31 -> 0.70 as features are added; pinning one
   shared curve costs 1.8 fold-sd. Correctly shipped.

### The error map (phase0/error_map.py, data/error_map.json)

Ranking slices by total excess loss produced a **noise ordering** — almost every
slice has *negative* excess (the model beats a slice-label oracle nearly
everywhere). Two better instruments were built: shortfall against the dimension's
own mean, and an in-slice recalibration likelihood ratio whose null expectation
is exactly 1.0 nat per slice at any n.

**What a PERFECT slice-conditional recalibration would buy, net of that null:**

| dimension | MLB | NFL | NHL |
|---|---|---|---|
| probability decile | −0.000039 | −0.000763 | +0.000666 |
| month / week | +0.000017 | +0.000113 | +0.000005 |
| favourite side | −0.000045 | −0.000178 | −0.000203 |
| rest | −0.000023 | −0.000917 | −0.000322 |
| team | −0.000006 | −0.000034 | −0.000228 |

**One cell in the entire map clears a gate, and it is in-sample on the model's own
output.** Conclusion: the models are not mis-specified anywhere we can find. The
common theme in the shortfall lists is early-season cold start (MLB team-games
0-9 cal_gap +0.028±0.011; NFL weeks 1-2; NHL team-games 5-14 +0.029±0.015) — a
real, cross-league, mechanistically sensible pattern that still nets ~0 once the
null is subtracted.

### The floor — how far is the theoretical limit?

Odds used as an evaluation benchmark only (sanctioned).

| league | n | our DEV | market | entropy floor | us -> market |
|---|---|---|---|---|---|
| MLB 2010-15 | 13,930 | 0.679233 | 0.676351 | 0.678391 | **+0.00288** [0.0015, 0.0041] |
| NFL 2006-15 | 2,531 | 0.622080 | 0.610363 | 0.606654 | **+0.01172** [0.0039, 0.0190] |
| NHL 2011-18 | 7,927 | 0.672423 | 0.670717 | 0.674176 | **+0.00171** [−0.0002, 0.0036] n.s. |

**NHL filled in 2026-07-31.** The NHL cell was blank only because no historical
closing line existed in the repo, not because none exists: SportsbookReviewsOnline
publishes NHL open/close moneylines back to 2007-08 as **inline HTML tables**, not
as the `.xlsx` workbooks the MLB loader consumes — which is why every filename
guess 404'd. `phase0/nhl_odds_load.py` → `data/odds_nhl.csv` (15,203 games,
2010-11..2022-23); `phase0/nhl_floor.py` → `data/nhl_floor.json` runs the same
`floor_block()` as the rows above. DEV join rate **0.9997** with **0 result
mismatches in 15,200 joined games**; devigged mean home prob 0.5476 vs realized
0.5494; closing overround 1.0331. Full sourcing and validation record in
`documents/nhl_odds_sources.md`. Odds are EVALUATION-only here, as in the rows
above.

Two things stand out. **(1) NHL is the one league whose gap to the close is not
significant** — +0.00171 with a CI straddling zero on n=7,927, versus MLB's
clearly-positive +0.00288 and NFL's +0.01172. On DEV, our NHL model is
statistically indistinguishable from the closing line. **(2) The NHL close is
mildly UNDER-confident on DEV** (recal slope 1.114 ± 0.068, z=+1.67 n.s.), which
makes mean H(p_mkt) = 0.674176 sit *above* both the market's realized loss and
our own. That is not "we beat the floor" — it is the concavity caveat already
recorded above showing its teeth: E[H(p_mkt)] is an **upper** bound on the
irreducible loss, and when the quoted line is a little too flat the bound is
loose enough to be exceeded. Read the NHL floor column as an upper bound only;
the honest NHL statement is the us→market gap, and it is ~0.

**The honest headline: the market-vs-floor comparison is nearly circular.** If the
close is calibrated then E[LL_mkt | p_mkt] = H(p_mkt) *identically* — the two
columns are one quantity measured twice, and both gaps are consistent with zero
(−0.0020±0.0014, +0.0037±0.0073). That is confirmation the market is calibrated,
NOT headroom. Also mean H(p_mkt) is an **upper bound** on the true irreducible
loss (the market's probability is a coarsening of the truth, H is concave), so the
real floor sits at or below these numbers and is not identified by the market
alone.

**So the answer to "how far from the theoretical limit": MLB ~0.0029, NFL ~0.0117,
NHL ~0.0017 (n.s.) in log loss** — and since the market sits essentially *at* the
floor, our gap to the market IS our gap to the limit.

### Correction — how much of NFL's gap was the DEV crippling? Only 12%.

Answered 2026-07-31 **without spending a TEST look**, because the number was
already recorded: the shipped NFL model card carries `test_log_loss` 0.61947 and
`close_log_loss` 0.60913 on TEST 2016-2025 (n=2,761) — a look spent and published
long ago, on the era where the v7 participation feature IS live.

| measurement | model | gap to close |
|---|---|---|
| DEV 2006-2015 | crippled (v7 identically zero, absence live 2 of 10 seasons) | +0.01172 |
| **TEST 2016-2025** | **full served model** | **+0.01034** |

So the availability asymmetry explains **0.00138 — about 12%** of the gap, not
the bulk. My earlier framing ("part of NFL's 4x gap is an availability
asymmetry, not football") was directionally right and quantitatively minor.
**NFL's full-model gap is still 3.6x MLB's and 6.0x NHL's.** The league really is
harder for us relative to its market, and the DEV caveat does not explain that
away. It remains a caveat for interpreting DEV-era NFL numbers, which is where
all screening happens.

### Where that leaves the program

Rounds 1-2 closed all 9 axes x 3 leagues with zero adoptions. Combined with ~60
prior channel nulls and the error map showing no exploitable residual structure,
the evidence is consistent: **the remaining gap is missing INFORMATION, not
mis-specification.** No re-parameterization, calibration, ensembling or model
class recovers it. Future rounds should be aimed at information the public record
does not contain (the in-season lineup/injury archives now accruing), not at
further optimization of the existing components.

---

## Round 3 — the park channel (MLB), 2026-08-01

The one classic MLB channel with no entry anywhere in this repo: ballpark.
`weather`, `umpire`, `temperature` and `altitude` appear in no document; `park`
appeared once, in prose. Park factors are not a feature and the spine carries no
venue column.

The error map's `team` dimension was NOT a test of it: it pools each franchise's
home and away games ("every game the team appears in, either side"), so a venue
effect — which by construction applies to only half of them — arrives diluted by
about a factor of two. Added a `home_venue` dimension (home games only, i.e. one
slice per ballpark) so the channel gets a sharp test rather than an inferred one.

**Result on MLB DEV (n=34,009, ll 0.676686, 32 ballparks):**

| test | outcome |
|---|---|
| ballparks with a fixable mis-scaling, Bonferroni α=0.00156 | **0 of 32** |
| smallest `recal_p` | CHN (Wrigley) 0.0290 — does not survive |
| total recalibration gain across all 32 parks | 36.45 nats vs **32.0 expected under the null** |
| **log-loss if every ballpark were perfectly recalibrated** | **0.000131** |

That last row is the number that closes it. Perfect per-park recalibration — an
in-sample ceiling that would itself be overfitting — is worth 0.00013 against a
gap to the closing line of 0.00288. **The park channel can account for at most
~4.5% of MLB's remaining gap, and realistically none of it.** Not adopted, and
not worth a data acquisition (venue, weather, umpire) to pursue.

### A Coors result that is not about Coors

The pooled `team` dimension has exactly one slice surviving Bonferroni: COL at
`recal_p` 0.0010. Coors Field is the textbook park effect, so the obvious read is
a park defect. It is not.

| slice | n | recal_a | recal_b | recal_p | excess/game | verdict |
|---|---|---|---|---|---|---|
| COL, home + away | 2,269 | +0.157 | 1.011 | **0.0010** | −0.00776 | SIG **better** |
| COL, home only (Coors) | 1,135 | +0.075 | 0.916 | 0.4947 | −0.00760 | n.s. |

At Coors itself the model is fine. The slope is ~1 in both, so the pooled result
is an intercept shift, not mis-scaling — and `excess_sig` reads "SIG better",
meaning the model already beats its own base rate on COL games. It is a
calibration offset on a slice we are good at, and since home is null it lives in
Colorado's road games — a property of the team, not the ballpark. Adding a
team-specific intercept for one of 32 slices would be textbook overfitting.

**Round 3 verdict: no adoption.** Consistent with Rounds 1-2 — the remaining gap
is missing information, and the ballpark is not the missing information.

### Round 3b — the venue channel in NFL and NHL, 2026-08-01

The MLB screen exposed a structural gap rather than a one-league oversight: all
three leagues share `_team_dim`, which pools home and away games, so none of
them had a sharp venue test. Added `home_venue` to NFL and NHL too. NFL gets a
proper one because its game records carry `neutral`, so London, Mexico and the
Super Bowl go to their own bucket instead of being charged to the nominal home
team's stadium. NHL's loaded spine drops the `neutral` column that
data/nhl_games.csv actually has, so its ~2-4 outdoor games a season are charged
to the nominal home arena — negligible, but stated.

| league | venue slices | significant excess | `recal_p` < Bonferroni | smallest p | ll ceiling if every venue were perfectly recalibrated |
|---|---|---|---|---|---|
| MLB | 32 | 15 | **0** | CHN 0.0290 | 0.000131 |
| NFL | 33 | 3 | **0** | NO 0.0084 | 0.006604 |
| NHL | 31 | 3 | **0** | MIN 0.0159 | 0.001709 |

**Zero venues survive multiple-comparison control in any league.** The channel is
null league-wide, not just in baseball.

NFL's ceiling (0.0066) looks large against its 0.01034 gap, but it is an
in-sample bound fitting 33 free parameters to ~2,500 games; its aggregate
recalibration gain is 50.6 nats against a null of 33, i.e. about 3 sd — which is
what 33 noisy slices produce. No individual stadium is close.

#### The neutral-site slice: checked, and it is noise

Worth writing down because it looks exactly like a real defect. NFL neutral-site
games score +0.09207/game excess with `recal_a` −0.927 — a large correction
pulling the home side DOWN, which is precisely the signature of a model applying
home advantage where none exists.

It is not that. **`mean_pred` on those games is 0.4942** — essentially even, so
home advantage IS already zeroed. A model that failed to zero it would predict
near 0.57. Both paths handle it: `nfl_season_serve.py` sets `hfa_pts = 0.0 if
s["neutral"]` and skips the travel/timezone terms, and the research harness's
`team_hfa()` guards with `if not g["neutral"]`.

The slice looks bad because those 28 games went 8-20 against the nominal home
team: against an expected 13.8 wins that is z = −2.2, p ≈ 0.03, and with 33
slices tested one such is expected. `excess_sig` is n.s. with a bootstrap CI of
[−0.055, +0.365]. **No action.** Recorded so the next person to notice the
−0.927 intercept does not spend a day on it.

### Round 3c — NFL broadcast slot, 2026-08-01

Same lens as the venue screens: `kick` (ET kickoff hour) already feeds the NFL
travel feature but had never been a SLICE, so prime-time and Thursday games
could have been systematically mispriced with nothing to show it. Added a
`kickoff_slot` dimension (weekday + hour -> the slate a game sat in). Distinct
from the existing `rest` dimension, which captures Thursday's short week but not
the slot itself.

| slot | n | base | mean_p | cal_gap | excess/g | sig | recal_p |
|---|---|---|---|---|---|---|---|
| Sun early (1pm) | 1424 | 0.562 | 0.577 | +0.015 | −0.06372 | SIG better | 0.3163 |
| Sun late (4pm) | 710 | 0.576 | 0.569 | −0.007 | −0.05694 | SIG better | 0.6101 |
| Mon night | 172 | 0.558 | 0.559 | +0.001 | −0.07500 | SIG better | 0.6605 |
| Sun night | 165 | 0.558 | 0.554 | −0.003 | −0.03535 | n.s. | 0.8835 |
| Thu night | 124 | 0.597 | 0.548 | −0.049 | −0.09063 | SIG better | 0.3719 |
| Sat / other | 71 | 0.549 | 0.588 | +0.039 | −0.03776 | n.s. | 0.6567 |

**Null, and about as clean as a null gets.** Nothing survives Bonferroni
(α=0.0083), and two aggregate numbers settle it:

- `recal_total` **3.6 nats against a null of 6.0** — the dimension carries LESS
  apparent structure than chance produces.
- ceiling if every slot were perfectly recalibrated: **−0.000903**, i.e.
  negative. The in-sample gain does not even cover the parameter cost.

Every slot is already "SIG better" than its own base rate except the two
smallest. Thursday night has the largest calibration gap (−0.049: the model
under-rates home teams on short weeks) but at n=124 with recal_p 0.372 that is
noise, and it points the opposite way from the usual "TNF is chaos" intuition.

**No adoption.** Broadcast window carries nothing the model is missing.

---

## Guard-branch necessity sweep, 2026-08-01

Not a research round — a code-health sweep, recorded because it produced a rule
worth keeping.

Mutation testing is usually read one way: a surviving mutant means the tests are
too weak. Iteration 82 hit the other reading. A mutation that disables a branch
and CANNOT be killed at any tolerance does not mean the test is weak; it means
**the branch is dead**. The test was fine; the code had a special case that
merely looked necessary.

Swept every early-return branch in the repo's pure predicates (40 candidates,
most of them legitimate loop filters) and checked, for each, whether the code
after it already handles the case:

| guard | branch | verdict |
|---|---|---|
| `db.pythag` | `rs<=0 and ra<=0` | NECESSARY — otherwise ZeroDivisionError |
| `nfl_weekly.csv_rows` | `n == 0` | NECESSARY — general expr returns 0, not −1 |
| `nfl_season_guards.v7_npy_error` | `n_npy == n_games` | NECESSARY — else builds an error string |
| `nhl_season_boundary.season_of_game_id` | `len(s) != 10` | NECESSARY — a 5-digit id would decode to 20262027 |
| `pred_ledger.entry_tier` | `t in TIERS` | NECESSARY — else the stamp is discarded |
| `nfl_weekly.fetch_is_safe` | `n_prev <= 0` | **DEAD — removed** |
| `nhl_xg_update.season_block_is_safe` | `n_cur_old <= 0` | **DEAD — removed (iter 82)** |

Both dead branches are the same mistake, written by me, twice: a "first
time / empty" special case that the tolerance comparison already admits, because
row counts are never negative and `n_new >= 0 - tol` is trivially true. The
half of the NFL guard that IS load-bearing — `n_new < 0`, meaning the table
could not be counted — was kept and re-confirmed killable.

**Rule:** when adding a tolerance-based guard, check whether the empty/first
case is already inside the tolerance before writing a branch for it.

---

## Round 4 — ten unscreened channels, NFL environment + NHL xG decomposition, 2026-08-01

Samuel asked for continued NHL/NFL model work. The standing claim here is that
the remaining gap is missing INFORMATION rather than mis-specification, which is
only credible if the channels a critic would name first have actually been
tried. Six NFL columns sat in `data/nfl_games.csv` unscreened; four NHL channels
follow from `nhl_team_xg.csv` carrying xgf/xga/shots/gf/ga.

The adoption bar was **pre-registered before any result was seen**
(`documents/channel_screen_prereg_2026_08_01.md`, committed while the screens
were still running): nested ≥ +0.0010, raw-to-nested gap under 60%, sign
consistency ≥6/10 NFL seasons (≥5/7 NHL), plus survival of an independent
adversarial rebuild.

**All ten baselines reproduced. All ten channels are NULL as named.**

| channel | league | raw | nested | verdict |
|---|---|---|---|---|
| wind | NFL | −0.00012 | −0.00036 | NULL |
| temperature | NFL | +0.00059 | −0.00031 | NULL |
| roof | NFL | +0.00036 | −0.00229 | NULL |
| surface | NFL | +0.00049 | −0.00005 | NULL |
| divisional | NFL | +0.00117 | +0.00117 | NULL — see below |
| referee | NFL | +0.00069 | +0.00044 | NULL |
| time-varying home ice | NHL | +0.00015 | −0.00012 | NULL |
| shot quality | NHL | +0.00137 | +0.00106 | NULL as named — see below |
| finishing above expected | NHL | +0.00011 | −0.00004 | NULL |
| goaltending above expected | NHL | +0.00064 | −0.00082 | NULL |

### Two that cleared the numeric bar and were killed anyway

**Divisional games.** Nested = raw = +0.00117 with ZERO search optimism and 9 of
10 seasons — it passes every numeric criterion I pre-registered. It is still a
null, for reasons the bar did not contain:

- the positive cell is `div × z(qb_delta)`, whose fitted coefficient is
  **positive** in every fold, i.e. the QB edge is AMPLIFIED in divisional games —
  the opposite of the hypothesis under test, so a post-hoc discovery;
- game-level bootstrap (20k resamples) gives [−0.00081, +0.00306], P(δ≤0)=0.121;
- 2015 — the DEV season adjacent to TEST — is −0.00584 while the other nine are
  positive;
- **the decisive test: re-scoring the pre-DEV 2002-2005 window, never part of the
  search, gives −0.0019, negative in 4 of 4 seasons.** Pooled 2002-2015 the gain
  collapses to +0.000275. The positive DEV block is bracketed by negatives on
  both sides in time;
- ~15% of the effect is a ridge parameterisation artifact (`D*zq` vs `(1−D)*zq`
  span the same space yet differ by 1.8e-4).

A meaningless index-parity coin-flip partition also cleared +4.9e-4 on the same
search, which calibrates how much of this is search pressure.

**Shot quality.** Nested +0.00106 for the quality/volume PAIR, but the
decomposition kills the named channel outright: quality ALONE is worth
**+0.00007**, below the solver-noise floor, and is actively harmful at any
meaningful learning rate — its optimum sits at the smallest rate in the grid,
the signature of a channel with no team-level persistence. Robust across a 3×3
sweep of the HA constant and season regression. The hypothesis that the single
xG rating "collapses two skills" is wrong: one of the two skills is empty.

A methodological note from that screen worth keeping: the first version returned
a FALSE null for quality because `quality_diff` has std 0.007 and lbfgs at C=1e6
silently returned an exactly-zero coefficient. A StandardScaler fixed it. Screens
of small-scale features need that check.

### What my own bar got wrong

The pre-registered criteria would have ADMITTED divisional games. What killed it
was an out-of-search temporal window — scoring seasons that were never part of
the search — and I had not required that. **Added to the standing bar: any
candidate must be re-scored on a temporal window outside the search before
adoption.** The DEV/TEST split alone does not provide this, because DEV is where
the searching happens.

Separately, my workflow gated adversarial verification on the agents'
self-assigned verdict strings rather than on my own pre-registered numbers, so
neither of the two candidates that cleared the bar numerically got a verifier.
It did not matter here only because both agents self-rejected with more rigor
than my gate demanded — which is luck, not design. Gate on the numbers.

### The one live follow-up

Shot VOLUME, not quality, is where the movement was: volume-only over the Elo
core scored 0.67189 vs the shipped xG rating's 0.67293, and a re-tuned single-xG
control confirms this is not the shipped rating merely being mis-tuned (re-tuning
buys +0.00005, nested −0.00001). But volume-only nested is +0.00076 with a CI
spanning zero, halving to +0.00031 under forward-temporal validation — **below
the bar already**. It is being screened on its own terms with two adversarial
lenses before anything further is claimed.

### Round 4b — the shot-differential observable, screened + doubly refuted

The one live lead out of Round 4. Replace the NHL rating's observable with plain
shot differential (Corsi/Fenwick-style) instead of xG margin, on the mechanistic
argument that shot volume is more repeatable season to season and should
therefore give a less noisy rating. Screened definitively, then attacked by two
independent rebuilds with different lenses (leakage/temporal, redundancy/spec).

Baseline reproduced by all three agents to 5dp: DEV LOSO Elo-only 0.67631,
elo+rest+b2b 0.67543, shipped +xg_diff **0.67239** — so the shipped xG rating is
worth **+0.00303** over the core, a clean DEV reproduction of its known ~+0.00282.

| variant | raw | nested | bootstrap CI | verdict |
|---|---|---|---|---|
| REPLACE xG with shot-diff | +0.00077 (k=0.10) | +0.00061 | [−0.00077, +0.00199] | spans zero |
| ADD alongside xG | +0.00092 (k=0.15) | +0.00080 | [−0.00023, +0.00183] | spans zero |

**NULL — and this time the MECHANISM is refuted, not just the number.**

The candidate's whole rationale is that shot volume is *more* repeatable and so
should age *better*. The data says the opposite. Per-season value over the
rest/b2b core across all 8 DEV seasons:

```
+0.00584 +0.00607 +0.00417 +0.00450 +0.00326 +0.00630 +0.00212 +0.00070
trend −0.00060/season, r = −0.74
```

while the shipped **xG rating is flat: −0.00009/season, r = −0.27**. Shot
differential was genuinely the better observable in the early 2010s and decays to
approximately zero exactly at the TEST boundary. Forward-temporal splits show the
same thing and flip sign as the scored window moves later (+0.00036 → −0.00152).

That also answers the follow-up the screen raised — whether the decay is an era
effect shared by both ratings. It is not. It is volume-specific. The substantive
reading is the familiar one from public hockey analytics: the league moved past
raw shot counts, and xG did not lose its edge. **This is evidence the shipped
choice of observable is correct and durable, not merely untested.**

Two further findings from the refutations:

- **Concentration.** Excluding the two carrier seasons (2011-12, 2015-16), nested
  ADD falls to **+0.00002** (CI [−0.00129, +0.00131]) and REPLACE to −0.00041.
  Five of seven seasons contain literally nothing, at the solver-noise floor.
- **Collinearity.** corr(shot_diff, xg_diff) = 0.751; with both present the xG
  standardized coefficient collapses +0.1840 → +0.0737 while shot takes +0.1442 —
  a 60% reallocation, i.e. largely one signal split across two columns. Though
  43.4% of shot-diff variance does survive projecting out elo and xG, and a
  control adding a *second xG rating at a different timescale* buys only +0.00013
  versus +0.00092 for xG+shot, so the content is not purely a "faster rating"
  artifact. Real, distinct, and still not worth shipping.

Leakage was tested directly and found absent: perturbing game *i*'s own box score
moves feature[i] by exactly 0; flipping every outcome moves the shot feature by 0;
updates flush before the season-regression boundary; and a deliberate
read-after-update positive control shifts LL by +0.01081, **13× the candidate's
honest effect**, confirming the harness would have detected leakage had it existed.

**Verdict: keep `xg_diff` exactly as shipped. The volume/differential observable
is closed alongside shot quality.**
