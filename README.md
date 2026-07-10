# Sharp Odds — Sports Odds Algorithms

**Live site:** [sharpsheettips.com](https://sharpsheettips.com) · **Mirror:** [GitHub Pages](https://samuellachance.github.io/Sports-Odds-Algorithms/)

Daily algorithmic sports betting platform across **NBA, WNBA, CBB, NFL, CFB, NHL, NCAA D1 hockey, MLB, NCAA D1 baseball, international winter baseball leagues**, and **soccer** (Premier League, La Liga, Bundesliga, Serie A, Ligue 1, MLS, Champions League, and major international tournaments). Leagues are activated automatically once there is enough completed-game data for the full three-layer model.

---

## What the site does

| Feature | Description |
|---------|-------------|
| **Daily slate** | Rebuilt **4×/day** America/Toronto (midnight, 6 AM, noon, 6 PM EDT) and on every push via GitHub Actions |
| **Live data** | ESPN schedules, scores, and consensus moneylines/spreads; multi-book enrichment for NBA/NHL/MLB/WNBA when available |
| **Unified model** | Blends legacy **Algo V2**, **power ratings**, and a sport-specific third layer |
| **Algo picks** | Hubáček-style official picks: decorrelated model must beat the de-vigged market by ≥2 pp with ≥2% honest EV and a per-bet-type confidence bar (stricter per-league overrides where walk-forward backtests exist) |
| **Bet tracking** | Freezes odds at record time, grades against ESPN finals, logs implied-probability CLV vs the ESPN consensus closing snapshot, and sizes stakes with portfolio-aware quarter-Kelly (0.25–3u) |
| **League coverage** | Games and team pages for all supported leagues; official tracked picks cover NBA, WNBA, CBB, NHL, MLB, **NFL**, **CFB**, and soccer leagues whose calibrated model beats the closing line |

### Three-layer prediction stack

Each matchup blends three independent signals (equal weight when all layers are available):

| Sport | Leagues | Model |
|-------|---------|-------|
| NBA | Basketball | **EnsembleML** / NBA v2 margin stack — market-aware; holdout log-loss can beat the consensus market, but that is **not** the same as profitable closing-line ROI |
| WNBA | Basketball | **BasketballMatrix** / WNBA v2 — soft-impute SVD + margin model |
| CBB | Basketball | **CBB Torvik** efficiency + calibration when Torvik ratings resolve; falls back to BasketballMatrix |
| MLB | Baseball | **MLB RunCast** — EWMA run efficiency + Monte Carlo + XGBoost with probable-pitcher edge |
| NCAA D1 baseball, winter leagues, WBC | Baseball | [MLB-Model](https://github.com/greerreNFL/MLB-Model) Elo ratings |
| NHL, NCAA D1 hockey | Hockey | **Algo V1** / NHL v2 — weighted factor / margin model |
| NFL, CFB | Football | [nfelo](https://github.com/greerreNFL/nfelo) Elo ratings + EnsembleML head when trained; official picks use **conservative Hubáček gates** pending larger OOS |
| EPL, La Liga, Bundesliga, Serie A, Ligue 1, MLS, UCL, international | Soccer | **Path A** — Dixon–Coles + XGBoost with market-calibrated display probabilities (selected leagues beat the closing line on calibrated holdout) |

Layers 1 and 2 (Algo V2 and power ratings) still apply to **baseball** and **football** leagues. **Hockey** (NHL, NCAA D1 men's and women's) uses **Algo V1** / NHL v2 — eight weighted season factors summed into a signed total, converted to win probability via the original parabolic curve.

Soccer uses a dedicated **three-way blend**: each layer contributes home/draw/away probabilities independently, then the site surfaces projected scores (`xG`), fair 1X2 prices, and value picks when all three layers agree on the same outcome.

A **context layer** (research-grounded, conservative) can nudge blended probabilities when signals are available: favorite-longshot bias (FLB) at open-like prices, news/injury keyword heuristics, steam/line-movement proxies, and **sparse-sample EV caps** for thin international tournaments. Soccer also applies ESPN form/style/injury/neutral-venue context when present. Formations, weather, and coach-change flags are not applied when ESPN does not expose them reliably.

### Opening vs closing ROI (honest brackets)

Walk-forward bet backtests report **opening-line ROI** and, where available, a **closing-line floor**. Morning recorded bets sit between those brackets — opening ROI is an upper bound if you always get the open; closing ROI is closer to what you get if the market moves against you. Do **not** treat opening-line backtests as guaranteed live edge. See `data/pick_strategy.json` notes per league.

### NBA ML pilot (opt-in, not the public default)

A separate leak-free NBA ML pilot (`web/nba_ml/`) was graded out-of-sample against **real closing lines** (2016–2026). Honest result: roughly **−1.5% ATS / −3% ML ROI** with **negative CLV**. The model is well-calibrated but **does not beat the NBA close**; it stays off unless `NBA_ML_ENABLED=1`. Nothing here claims guaranteed profits.

### Database player-rating layer (all sports)

A **database ratings** layer blends external player/team ratings (not derived from our stats) into every matchup when a source is available. Weight starts at **12%** and scales up when completed games are sparse (e.g. early season, international tournaments), up to **+28%** when teams have fewer than 15 games.

| Sport category | Rating source |
|----------------|---------------|
| Soccer (domestic + international) | [soccer-rating.com](https://www.soccer-rating.com) team ratings |
| NBA | [2kratings.com](https://www.2kratings.com) top-8 roster OVR |
| WNBA | [2kratings.com](https://www.2kratings.com) WNBA team pages |
| NFL | [maddenratings.com](https://www.maddenratings.com) team overall |
| MLB, NCAA D1 baseball, winter leagues, WBC | [theshowratings.com](https://www.theshowratings.com) top-8 roster OVR |
| NHL | [nhlratings.net](https://www.nhlratings.net) team overall |

College basketball/football and NCAA hockey/baseball have no stable public rating database in the same mold; those leagues skip this layer when no source resolves. Ratings are cached locally for 12 hours under `data/db_ratings_cache/`.

---

## Quick start (local)

### Requirements

- Python 3.10+ (tested on Python 3.12)
- Bundled historical CSV data in `nba/`, `nhl/`, and `mlb/`

### Install

```powershell
cd C:\Users\Admin\Projects\Sports-Odds-Algorithms
python -m pip install -r requirements.txt
```

### Run the website locally

```powershell
python run_server.py
```

Or with auto-reload during development:

```powershell
python -m uvicorn web.app:app --reload --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

### Verify core algorithms

```powershell
python -m pytest tests/test_hubacek_picks.py tests/test_tracking.py tests/test_clv_service.py tests/test_context_signals.py tests/test_portfolio_sizing.py tests/test_live_odds_enrichment.py tests/test_pick_strategy.py tests/test_bet_advisor.py tests/test_official_picks_hubacek.py tests/test_cbb_pred_model.py -q --tb=short
python smoke_test.py
```

The demo can also use bundled historical team CSV files (no live scraping required). Default example:

- **NBA:** Portland Trail Blazers @ Golden State Warriors on `4-16-2017` (season `2017`)
- Expected Algo V2 result: ~71% Warriors win probability

### Deploy to GitHub Pages

Static assets are built from `web/static/` into `docs/` by `scripts/build_gh_pages.py`. Pushes to `master` trigger the `.github/workflows/pages.yml` workflow.

---

## Project layout

| Path | Purpose |
|------|---------|
| `algo.py` | Core Algo V1/V2 prediction logic |
| `odds_calculator.py` | Team stat analysis and odds formatting |
| `backtester.py` | Historical backtesting utilities |
| `espn_scraper.py` | Legacy ESPN schedule/box score scraper |
| `sports_bettor.py` | Original interactive CLI entry point |
| `web/` | FastAPI backend + static frontend |
| `web/blend_service.py` | Unified model blending (legacy + power + sport layer + DB ratings + context) |
| `web/context_signals.py` | FLB / news heuristics / steam proxies / sparse EV caps |
| `web/portfolio_sizing.py` | Portfolio-aware quarter-Kelly stake sizing |
| `web/live_odds_enrichment.py` | Multi-book odds enrichment (NBA/NHL/MLB/WNBA) |
| `web/clv_service.py` | Implied-probability CLV helpers |
| `web/db_rating_model.py` | External database player/team ratings layer |
| `web/daily_service.py` | Daily slate and pick generation |
| `web/tracking_service.py` | Bet logging, grading, CLV vs ESPN consensus close |
| `scripts/build_gh_pages.py` | GitHub Pages static build |
| `.github/workflows/test.yml` | CI: core pytest + smoke_test on push/PR to master |

---

## Original CLI usage

Algorithms to predict NBA, NHL, and MLB games are included. To utilize, run:

```powershell
python sports_bettor.py
```

Follow the interactive menus for single-team analysis, matchup odds, backtests, and schedule scraping.

---

## Algorithm overview (Algo V2)

**Variables:**

1. Record points = (wins − losses) − (opponent wins − opponent losses)
2. Home/away split differential
3. Home/away split over last 10 games
4. Last 10 games win ratio
5. Average scoring margin
6. Average scoring margin over last 10 games
7. Win streak
8. Home/away win streak (NHL)

**Versions:**

- **Algo V1** — point ranking system summed into a total, mapped to win probability
- **Algo V2** — each factor converted via backtest-derived curves, averaged into a percentage

The NHL algorithm predicted the [2016 Stanley Cup champion](http://smartsoftware.technology/sports.php?view=nhl&season=2016) and its playoff bracket was in the 99th percentile. [Backtest results](http://smartsoftware.technology/sports.php) for betting strategies using the original algorithms are on the upstream research site.

---

## Notes / limitations

- Historical CSV coverage: NBA/NHL through 2017, MLB through 2016 (used for backtests and CLI demos).
- Live ESPN endpoints may change; the production site uses the `web/espn_client.py` integration.
- Third-layer models depend on external repos and may be unavailable for some matchups; the blend falls back to two layers when needed.
- **CLV** is industry-standard implied-probability move vs the recorded price and the ESPN consensus closing snapshot (not a multi-book true close everywhere). Multi-book enrichment improves live prices for NBA/NHL/MLB/WNBA when ESPN exposes providers.
- Opening-line backtest ROI ≠ live closing-line edge. The NBA ML pilot’s honest close ROI is negative; do not advertise “beats the close” for NBA.
- No guaranteed profits. Markets are efficient; gates and sizing are deliberately conservative.

---

## Original documentation

The sections below are preserved from the upstream project.

### Bare-bones algorithm information

**Variables:** 
1) Record_points = ( wins - losses ) - ( other_wins - other_losses)
2) Home_away = (away_record - home_record) - (other tema's away record - other team's home record)
3) Home_away_10_games = Home_away for last 10 games
4) Last_10_games_points = Record_points for last 10 games
5) Avg_points = (total_points/num_games) - (other_total_points/num_games)
6) Avg_points_10_games = Avg_points for last 10 games
7) Win_streak = num consecutive wins
8) Win_streak_home_away = num consecutive wins home or away

*Example:* 
NBA's algo_V2 includes 1, 2, 3, 4, 5, 6
NHL's algo_V2 includes 1, 2, 3, 4, 5, 6, 8
MLB's algo_V2 includes 1, 2, 3, 4, 5, 6

**Backtests:** 

*CSV_output* = Backtest all games for 2nd half of seasons in specified timespan. The supplied algorithm will output a point system or percentage system accompanying its prediction. The results are returned in a csv file.

*Stats* = Backtest all games for 2nd half of seasons in specified timespan. The parameter algorithm will solely calculate wins vs losses for a 1-10 ranking system. The ranking sytem can be points or percentage based. The results are returned in a txt file. 

*Running sports_bettor.py:* Choose league, Backtest algorithm, Algo_V1, output to csv.
This will run a CSV_output backtest using a hardcoded algo_V1. EX: NBA = [10, 10, 5, 5,  8,  8,   3, 3];

*Running sports_bettor.py:* Choose league, Backtest algorith, Algo_V1, stats.
This will run a stats backtest for passed in algo_v1s that test each variable at a time. 

-----

### Creating an algorithm: 
**1)** Test each variable individually to create algo_V1

Menu choices: 4) Backtest algorithm -> 1) Algo_V1 - Uses a point system -> 2) Backtest Algo_V1 stats -> INPUT) Start Date: (middle of first season), End Date: (cur date if end 2nd half of current season, or end date of last season if in 1st half of current season)

* Default: algo_V1 = [-1, -1, -1, -1, -1, -1, -1, -1]
* Each parameter is respective to the variables.
	test each param like [1, -1, -1, -1, -1, -1, -1, -1]
	test each param like [2, -1, -1, -1, -1, -1, -1, -1]
* The results will be output to a txt file "./analyze/backtests/Algo_V1_-1,-1,0.5,-1,-1,-1,-1,-1_7-1-2003_10-1-2015.txt"
	EX output: 

	[1, -1, -1, -1, -1, -1, -1, -1]

	1: 537 - 536: 49.95%
	2: 615 - 716: 53.79%
	3: 640 - 683: 51.62%
	4: 572 - 696: 54.89%
	5: 553 - 654: 54.18%
	6: 506 - 631: 55.50%
	7: 477 - 590: 55.30%
	8: 369 - 586: 61.36%
	9: 369 - 497: 57.39%
	10: 1597 - 2351: 59.55%

	6235 - 7940

* 1-10 in the output file correspond to 1-10 levels in the program. Ideal to have a bell curve type distribution of total games from 1 (most games) to 10 (least games). 
* Also ideal if the percentage of games won start at 50 in level 1, and go to 100% by level 10. Level 10 should not have more games won than level 9. 
* The number used to create the ideal backtest output will be used in Algo_V1
*EX:* NHL = [3, 3, 3, 3, 0.3, 0.6, -1, 6]
* These will be the denominators for the variables. The maximum 1-10 level reached in the output will be the max_points. If level 10 isn't rached, the max level will be adjusted.

	
-----
	
	
**2)** Create algo_V2
	
* The games won percentage for each level in each output for each variable will create a polynomial equation for each variable. 
* Create a best-fit line for all perc_won numbers in the ideal output file. 
* The best-fit line will calculate the odds to win for that variable. 
* Best-fit line should start above 50%, and end below 100%
* 

	
**...(More information to be appended later)**


* The new algorithm should be hardcoded into algo.py to be utilized for odds calculation. 
