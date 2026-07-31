# MLB Statcast-era protocol — pre-registered 2026-07-31

## Why a second protocol

The primary locked split (DEV 2002-2015 / TEST 2016-2024 ex-2020) predates
Statcast (2015+): every Statcast-only signal has been structurally
UNTESTABLE, not tested-and-null. This protocol opens that space honestly.
It is declared and committed BEFORE any screening runs (this file's git
timestamp is the proof).

## The split (LOCKED as of this commit)

- **DEV: 2016-2020 seasons** (Statcast mature from 2016; includes the short
  2020 season — DEV can afford its noise).
- **TEST: 2021-2024 seasons** (~9,700 games), scored ONCE per idea, ledger row
  in `data/mlb_statcast_ledger.csv` whether it ships or dies.
- Base model: shipped `full_p` from `data/model_probs.csv` (walk-forward,
  market-blind). Screen = increment over logit(full_p), 5 season-grouped CV
  folds inside DEV, paired bootstrap.
- **Gate to spend the TEST look: DEV delta >= +0.0005 with CI lower bound > 0.**
- Adoption rule unchanged: point estimate on TEST, SIG required to ship.

## Leak rules for Statcast features

Savant leaderboard metrics are season-aggregates. The pre-game feature for a
game in season Y uses **season Y-1 values only** (prior-season snapshot —
unambiguously pre-game). As-of intra-season versions require play-level
reconstruction and are out of scope until a prior-season screen passes.

## Candidate queue (screen in this order)

1. **Catcher framing (Savant catcher_framing)** — the proxy validated on
   Retrosheet-era data but nulled at game level; Statcast measurement is
   sharper. Prior-season framing runs of tonight's starting catchers.
2. **Outs Above Average (fielding)** — team defense beyond DER: prior-season
   team OAA (and tonight's-lineup OAA sum).
3. **Sprint speed** — lineup speed beyond baserunning counts: prior-season
   sprint of tonight's 9.
4. **Pitcher stuff proxies** (fastball velo / whiff rate, prior season) —
   starter quality beyond xFIP/SIERA.

All market-blind (Savant is tracking data, not odds). Anything else follows
the same declare-then-screen order.
