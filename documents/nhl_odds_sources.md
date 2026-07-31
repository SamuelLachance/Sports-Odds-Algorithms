# NHL historical closing line — sourcing record

Opened and **CLOSED 2026-07-31**. The question "does a free historical NHL
closing moneyline exist that we can actually join?" is now answered **yes**, with
a verified loader. This file records what was tried so it is not re-guessed.

Odds use throughout is **EVALUATION ONLY** — benchmark and entropy-floor proxy.
No odds enter any NHL feature or training signal.

## What was tried, in order

| # | Attempt | Result |
|---|---|---|
| 1 | SBR `.xlsx` asset, MLB pattern with `nhl-odds-YYYY.xlsx` under `sportsbookreviewsonline_com_737/` | **404**. Also 404: `nhl-odds-2014-15.xlsx`, `nhl-odds-201415.xlsx`, `nhl odds 2015.xlsx`, `nhl-odds-2015.xls`, `hockey-odds-2015.xlsx`. Control `mlb-odds-2015.xlsx` returned 200/530 KB, so the folder id and the probe were both fine — **there simply is no NHL workbook**. |
| 2 | SBR hockey landing page `…/scoresoddsarchives/nhl/nhloddsarchives.htm` | **200**. Links to 16 per-season pages `…/scoresoddsarchives/nhl-odds-<slug>`, slugs `2007-08` … `2022-23` plus `2021` for the covid season. |
| 3 | One season page, looking for a download link | **No spreadsheet link at all** (only the site logo and CSS live in `wp-content/uploads/`). The page is ~580 KB because **the entire season is rendered inline as a single HTML `<table>`** — this is why the `.xlsx` hunt could never have succeeded. |
| 4 | Wayback Machine | **Not needed** — the live pages serve the full data. Not queried. |
| 5 | Paid/other archives (Odds Warehouse, BigDataBall, checkbestodds) | Not pursued; superseded by a free source that joins at 99.97%. |

**So the answer to the recurring question is: SBR does publish NHL open/close
moneylines back to 2007-08, but as HTML, not as the `.xlsx` the MLB loader
consumes.** That format difference is the whole reason this looked unavailable.

## The source

`https://www.sportsbookreviewsonline.com/scoresoddsarchives/nhl-odds-<slug>`

Two rows per game (`V` then `H`). Column layout differs by era — pre-2014-15
pages have 14 cells (no puck-line pair), later pages 16 — but the first ten are
stable and are the only ones used:

```
Date  Rot  VH  Team  1st  2nd  3rd  Final  Open  Close  [PuckLine PLprice]  OpenOU …
 0     1    2   3     4    5    6    7      8     9
```

`Date` is `MMDD`; the year is `<slug>`'s first year for months ≥ 8, second year
otherwise (the `2021` slug is entirely calendar 2021). `Final` includes the
OT/SO winner, so it agrees with `home_win` in `data/nhl_games.csv`.

## Loader

`phase0/nhl_odds_load.py` → `data/odds_nhl.csv`, mirroring
`phase0/odds_load.py`. Same schema as `data/odds_mlb.csv`
(`date,away,home,away_open,home_open,away_close,home_close,home_win`), decimal
odds, atomic write. **15,203 games, 2010-11 … 2022-23** (the loader starts at
2010-11 because that is where the game spine starts; SBR has 2007-08 onwards if
the spine is ever extended backwards).

Team mapping is by punctuation-stripped lowercase name, and keeps the
*contemporaneous* franchise codes the spine uses — `ATL` and `WPG` are separate
keys, as are `PHX` and `ARI`; `UTA` is mapped but never appears (SBR's archive
stops at 2022-23, before the move).

Per-season counts scraped, vs games in `data/nhl_games.csv`:

| season | scraped | spine | | season | scraped | spine |
|---|---|---|---|---|---|---|
| 2010-11 | 1319 | 1319 | | 2016-17 | 1317 | 1317 |
| 2011-12 | 1315 | 1316 | | 2017-18 | 1355 | 1355 |
| 2012-13 | 806 | 806 | | 2018-19 | 1358 | 1358 |
| 2013-14 | 1322 | 1323 | | 2019-20 | 1080 | 1212 |
| 2014-15 | 1319 | 1319 | | 2020-21 | 951 | 952 |
| 2015-16 | 1321 | 1321 | | 2021-22 | 1401 | 1401 |
| | | | | 2022-23 | 339 | 1400 |

2019-20 loses the bubble playoffs (131 rows arrive unpaired) and 2022-23 stops
in November — SBR abandoned the archive mid-season. **Neither touches DEV.**

## Validation

Join to `data/nhl_games.csv` on exact `(date, home, away)`, no fuzzing:

| season | joined / spine | rate |
|---|---|---|
| 2010-11 | 1317 / 1319 | 0.9985 |
| 2011-12 | 1315 / 1316 | 0.9992 |
| 2012-13 | 806 / 806 | 1.0000 |
| 2013-14 | 1322 / 1323 | 0.9992 |
| 2014-15 | 1319 / 1319 | 1.0000 |
| 2015-16 | 1321 / 1321 | 1.0000 |
| 2016-17 | 1317 / 1317 | 1.0000 |
| 2017-18 | 1354 / 1355 | 0.9993 |
| 2018-19 | 1358 / 1358 | 1.0000 |
| 2019-20 | 1080 / 1212 | 0.8911 |
| 2020-21 | 951 / 952 | 0.9989 |
| 2021-22 | 1401 / 1401 | 1.0000 |
| 2022-23 | 339 / 1400 | 0.2421 |

Zero duplicate join keys (no doubleheaders in hockey), and — the strongest
check available — **0 result mismatches in 15,200 joined games**: SBR's own final
score agrees with the NHL API's `home_win` on every single one. A mis-mapped
team or an off-by-one date would have produced thousands of disagreements.

The handful of misses are all identifiable and are *correct* refusals to match:

* `2018-01-01 NYR @ BUF` (Winter Classic, neutral site) — SBR lists it as
  `BUF @ NYR`, i.e. with the home/away designation reversed. The strict join
  declines it rather than silently pairing the wrong side. 1 DEV game.
* `2014-04-09 CBJ @ DAL` — the resumed remainder of the March 10 game suspended
  after Rich Peverley's cardiac arrest. Never had its own line. 1 DEV game.
* `2010-10-08/09 SJS–CBJ` (Stockholm Premiere), `2012-05-25 NYR @ NJD`,
  `2021-03-04 TOR @ VAN` — outside DEV.

Market sanity, on the 7,927 joined DEV games:

* mean closing overround **1.0331** (≈3.3% vig) — normal for a two-way NHL market;
* mean devigged home probability **0.5476** vs realized home-win rate **0.5494**
  (the right comparison; the *home-favourite rate* — share of games where the
  home side is priced under even — is 70.8%, a different quantity);
* closing-line log loss **0.670717** on DEV, i.e. inside the 0.66–0.67 band a
  correct join was required to produce;
* decile calibration of the devigged close: 8 of 10 deciles within ±1.1 SE.
  D02 (+2.58 SE) and D10 (−2.36 SE) are the only strays and they pull in
  opposite directions, which is what a slightly under-confident line looks like
  — the fitted recalibration slope is 1.114 ± 0.068 (z = +1.67 vs 1, n.s.).

## Consequence

`phase0/nhl_floor.py` → `data/nhl_floor.json` completes the third row of the
depth-program floor table. See `documents/depth_program.md`.
