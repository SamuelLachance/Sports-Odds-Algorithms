# Bet-selection logic — the out-of-sample verdict

Asked to make the bet-graded logic "logical, mathematical and profitable".
Two workflows, four agents, walk-forward probabilities for MLB and NHL joined to
27,978 / 15,204 historical price rows. The honest answer:

> **The model holds real information the OPENING line lacks. That information is
> worth less than the vig. There is no selection rule with demonstrable positive
> expected value out of sample, in either league.**

Everything below is out-of-sample: the blend is refit each season on strictly
earlier seasons, so no coefficient ever saw the outcome of the game it scores.
Rules were developed on the DEV era only; the later era is quoted once, labelled,
and nothing was tuned against it.

## 1. The model loses to the closing line, and ties the open

| league | n | model (OOS) | de-vigged close | paired t |
|---|---|---|---|---|
| MLB | 26,493 | 0.67589 | **0.67356** | +4.84 (market better) |
| NHL | 11,646 | 0.66886 | **0.66693** | CI [−0.00357, −0.00029] |

MLB loses to the close in **11 of 12 seasons**. Against the OPEN both leagues are
a statistical tie (MLB t=+1.96; NHL +0.00068, CI spans zero).

**Correction to a documented claim.** `documents/` states NHL "ties the market".
Under an honest walk-forward refit it does not: it ties the **opening** line and
loses to the **close**. The orthogonality survives but is weaker than recorded —
encompassing β_model = +0.320 (t=+3.02) pooled, against β_market ≈ 0.82.

## 2. The information is real — CLV is large, clean and highly significant

De-vigged CLV at the open, MLB DEV era, by EV gate:

| gate | CLV (implied-prob pts) | t | line moves our way |
|---|---|---|---|
| ≥0% | +0.909 | **+32.2** | 62% |
| ≥5% | +1.347 | +29.6 | — |
| ≥10% | +1.874 | +21.7 | — |
| ≥20% | +3.096 | +9.1 | **79%** |

This is clean: selection uses only past-fit coefficients. The model genuinely
knows something the opening line does not.

## 3. …and it is worth less than the price of admission

The decisive arithmetic. The book charges **1.32–1.84 probability points** on the
bet side at the open (opening overround 3.86%, closing 2.69%). The model's
information is worth 0.91–3.10 points of CLV. Net:

| gate | CLV | vig cost | **net edge** |
|---|---|---|---|
| ≥0% | +0.91 | −1.84 | **−0.93** |
| ≥5% | +1.35 | −1.75 | **−0.40** |
| ≥8% | +1.71 | −1.65 | **+0.06** |

The edge is **REAL but SUB-VIG**. It first crosses zero at ≈+8% EV, by a margin
indistinguishable from zero. For this to become profitable the opening overround
would have to fall from ~3.9% to **1.88%**. *The only lever is a cheaper entry
price, not a better gate.*

## 4. The 20% gate is actively harmful — winner's curse, quantified

On the selected set the model claims a win rate it does not deliver, and the
overstatement **grows with the gate** — exactly backwards from what a usable gate
needs:

| gate | model claims | actually delivers | overstatement |
|---|---|---|---|
| ≥0% | 52.69% | 49.68% | +3.01 pts |
| ≥5% | 52.14% | 48.07% | +4.07 pts |
| ≥20% | 49.66% | **38.99%** | **+10.66 pts** |

This is selection-conditional bias, not global miscalibration: a walk-forward
Platt recalibration shrinks the logit by only 0.924 and slightly *worsens*
log-loss. You cannot calibrate your way out of it — the gate itself is the
problem, because it selects precisely the games where the model is most wrong.

Consistent with the disagreement buckets: MLB model log-loss degrades
monotonically against the market as disagreement grows (−0.0008 at 0–2pp,
−0.0131 at 7–12pp).

## 5. Search-corrected: nothing survives multiplicity

MLB: 42 pre-specified rules on DEV. Best t = **+1.28**. A Monte-Carlo null
(outcomes drawn from the de-vigged close, same rule menu, 400 reps) gives
E[max |t|] = 1.45 with a 95th percentile of **2.62**. Nothing is close. The best
DEV rule then reads +0.67% ± 2.37% on the confirmatory era — decaying to zero
exactly as an overfit rule should.

NHL: no threshold reaches significance; the apparent gradient is a **regime
artifact** (one contiguous positive block 2012–15, then a negative one).

## 6. Kelly is ruinous on this model

Full Kelly asks for a mean 12.6% of bankroll per bet and ends the DEV era at
0.0003–0.0669 of starting bankroll with **99–100% maximum drawdown**. Quarter
Kelly's best cell ends at 2.905 — but only by riding the regime artifact, and
still through a 41% drawdown.

## What this changes

- **Do not present the EV badge as a profit claim.** It is a disagreement
  indicator with genuine positive CLV and negative net-of-vig expectation.
- The ≥20% gate should not be defended as "healthy". It is the worst cell in the
  menu: +10.66 points of overstatement.
- The CLV ledger is still the right instrument — but it measures *information*,
  not *profit*, and the promotion gate should say so.

## Two corrections to my own earlier claims in this session

1. I said the 20% gate was **"structurally unreachable"**. It is not: 159 of
   13,930 DEV games (1.14%) and 435 of 26,493 overall (1.64%) clear it
   historically. The live drought (max +8.9% this week, 0 ledger rows in 10 days)
   is a live-pricing/archive-window issue, not arithmetic impossibility.
2. I said the old +17.8% backtest was invalid **because of blend contamination**.
   The contamination is real but numerically tiny — ~2e-05 to 1e-04 nats, since
   the blend is only 4–7 coefficients fit on 24k–50k games. The number is still
   bogus, but the cause is **selection on the gate plus preferred-side
   overstatement**, not the leak. Nobody should rehabilitate it by observing that
   the leak was small.
