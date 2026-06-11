# Sports Odds Algorithms

System that calculates and uses algorithms to predict the outcome of NBA, NHL, and MLB games. Each league has its own unique algorithm to predict winners, with NBA having the most accurate algorithm in the original research.

[The NHL algorithm predicted the 2016 Stanley Cup Champion team, and its NHL playoff bracket was in the 99th percentile](http://smartsoftware.technology/sports.php?view=nhl&season=2016)

[Backtest results of betting strategies utilizing the algorithms' predictions](http://smartsoftware.technology/sports.php)

---

## Quick start (web demo)

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

### Verify core algorithms

```powershell
python smoke_test.py
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

The demo uses bundled historical team CSV files (no live scraping required). Try the default example:

- **NBA:** Portland Trail Blazers @ Golden State Warriors on `4-16-2017` (season `2017`)
- Expected Algo V2 result: ~71% Warriors win probability

---

## Project layout

| Path | Purpose |
|------|---------|
| `algo.py` | Core Algo V1/V2 prediction logic |
| `odds_calculator.py` | Team stat analysis and odds formatting |
| `backtester.py` | Historical backtesting utilities |
| `espn_scraper.py` | Legacy ESPN schedule/box score scraper |
| `sports_bettor.py` | Original interactive CLI entry point |
| `web/` | FastAPI backend + modern static frontend |

---

## Original CLI usage

Algorithms to predict NBA, NHL, and MLB games are included. To utilize, run:

```powershell
python sports_bettor.py
```

Follow the interactive menus for single-team analysis, matchup odds, backtests, and schedule scraping.

---

## Algorithm overview

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

See the original README sections below for backtesting and algorithm creation workflows.

---

## Push to your GitHub

1. Create a new empty repository on GitHub (do not initialize with a README if you want a clean push).
2. From the project folder:

```powershell
cd C:\Users\ulach5c\Projects\Sports-Odds-Algorithms
git remote remove origin
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git add .
git commit -m "Add modern web UI and Python 3 compatibility fixes"
git push -u origin main
```

If your default branch is `master`, replace `main` accordingly. The repo already includes a large historical dataset — first push may take several minutes.

---

## Notes / limitations

- Historical data coverage: NBA/NHL through 2017, MLB through 2016.
- Live ESPN scraping may need updates if ESPN HTML/API endpoints changed since the original project.
- The web demo intentionally uses bundled CSV data for reliable, reproducible predictions.

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
