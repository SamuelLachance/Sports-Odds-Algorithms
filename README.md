# Sharp Odds — Sports Odds Algorithms

**Live site:** [sharpsheettips.com](https://sharpsheettips.com) · **Mirror:** [GitHub Pages](https://samuellachance.github.io/Sports-Odds-Algorithms/)

Daily algorithmic sports betting platform across **NBA, WNBA, CBB, NFL, CFB, NHL, NCAA D1 hockey, MLB, NCAA D1 baseball, international winter baseball leagues**, and **soccer** (Premier League, La Liga, Bundesliga, Serie A, Ligue 1, MLS, Champions League, and major international tournaments). Leagues are activated automatically once there is enough completed-game data for the full three-layer model.

---

## What the site does

| Feature | Description |
|---------|-------------|
| **Daily slate** | Rebuilt every day at **3:00 AM America/Toronto** (and on every push) via GitHub Actions |
| **Live data** | ESPN schedules, scores, and consensus moneylines/spreads |
| **Unified model** | Blends legacy **Algo V2**, **power ratings**, and a sport-specific third layer |
| **Algo picks** | Ranks moneyline, 1X2 (soccer), and spread opportunities where the model disagrees with the market |
| **Bet tracking** | Logs picks at +40 edge or higher and grades them when games finish |
| **League coverage** | Games, team pages, and picks for all seven supported leagues |

### Three-layer prediction stack

Each matchup blends three independent signals (equal weight when all layers are available):

| Sport | Leagues | Model |
|-------|---------|-------|
| NBA, WNBA, CBB | Basketball | **BasketballMatrix** — soft-impute SVD on offensive-rating × pace matrices |
| MLB, NCAA D1 baseball, winter leagues, WBC | Baseball | [MLB-Model](https://github.com/greerreNFL/MLB-Model) Elo ratings |
| NHL, NCAA D1 hockey | Hockey | [hockey-predictions](https://github.com/greerreNFL/hockey-predictions) Poisson xG model |
| NFL, CFB | Football | [nfelo](https://github.com/greerreNFL/nfelo) Elo ratings |
| EPL, La Liga, Bundesliga, Serie A, Ligue 1, MLS, UCL, international | Soccer | Elo + Pi-ratings + Dixon–Coles Poisson (1X2 and score projections) |

Layers 1 and 2 (Algo V2 and power ratings) still apply to **baseball** and **football** leagues. **Basketball** (NBA, WNBA, CBB) uses a dedicated **BasketballMatrix** model only: completed-game scores build offensive-rating and pace matrices, missing matchups are imputed via Mazumder et al. soft-impute (nuclear-norm SVD), and projected scores drive win probability and spread picks.

Soccer uses a dedicated **three-way blend**: each layer contributes home/draw/away probabilities independently, then the site surfaces projected scores (`xG`), fair 1X2 prices, and value picks when all three layers agree on the same outcome.

A fourth **context adjustment** (ESPN public data only) nudges the blended 1X2 probabilities when signals are available: recent form, season style proxies (possession, shots, direct play, pressing), listed injuries, and neutral-venue flags. Formations, weather, and coach-change flags are not applied when ESPN does not expose them reliably.

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
cd C:\Users\ulach5c\Projects\Sports-Odds-Algorithms
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
| `web/blend_service.py` | Unified model blending (legacy + power + sport layer + DB ratings) |
| `web/db_rating_model.py` | External database player/team ratings layer |
| `web/daily_service.py` | Daily slate and pick generation |
| `web/tracking_service.py` | Bet logging and grading |
| `scripts/build_gh_pages.py` | GitHub Pages static build |

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
