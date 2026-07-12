# Sports Odds Algorithms — Agent

Algorithme de prédiction sportive (NBA, NHL, MLB et autres ligues) avec démo web FastAPI et tableau public GitHub Pages.

## Stack

- Python 3.10+ (`requirements.txt`)
- Core : `algo.py`, `odds_calculator.py`, `backtester.py`
- Web : `web/` (FastAPI + frontend statique)
- Sport models : `web/{nba,wnba,nhl,mlb,soccer,nfl,cfb,cbb}_v2` (GradientBoost v2; CBB primary = GB v2 with Torvik fallback; NFL/CFB v2 live for predictions, official picks disabled until backtest clears)
- Données historiques : CSV dans `nba/`, `nhl/`, `mlb/`
- Modules récents : `web/context_signals.py` (FLB / news / sparse EV caps), `web/portfolio_sizing.py` (stake sizing corrélé), `web/live_odds_enrichment.py` (multi-book NBA/NHL/MLB/WNBA), `web/basketball_v2_market.py` (helpers partagés NBA/WNBA live market-aware)

## Commandes utiles

```powershell
python -m pip install -r requirements.txt
python -m pytest tests/ -q --tb=short
python -m pytest tests/ -q -m "not slow"
python smoke_test.py
python scripts/dev_check.py
python scripts/dev_check.py --quick-only
python scripts/dev_check.py --with-v2
python scripts/dev_check.py --compile
python run_server.py
python -m uvicorn web.app:app --reload --host 127.0.0.1 --port 8000
python scripts/check_v2_data.py
```

`scripts/dev_check.py` is the local DX gate: pytest with `-m "not slow"` plus `smoke_test.py` by default (`--quick-only` skips smoke; `--full` runs the entire suite; `--with-v2` also runs `check_v2_data.py`; `--compile` mirrors CI `compileall`). `--full` and `--quick-only` are mutually exclusive.

## Training NFL / CFB / CBB v2

Build the leak-free training table, then train (walk-forward OOS + artifacts under `data/models/{league}_v2/`). Scripts accept `--help` and exit with a clear error if inputs are missing.

```powershell
# NFL (needs data/supplemental/closing-odds/nflverse_games.csv)
python scripts/build_nfl_training_table.py
python scripts/train_nfl_model.py

# CFB (needs data/supplemental/closing-odds/cfb.csv)
python scripts/build_cfb_training_table.py
python scripts/train_cfb_model.py

# CBB (ESPN caches under data/cbb_history/; optional cbb.csv odds join)
python scripts/build_cbb_training_table.py
python scripts/train_cbb_model.py

# Quick OOS metadata summary for all v2 leagues
python scripts/check_v2_data.py
```

Official NFL/CFB/CBB picks stay gated until spread backtests clear; training still refreshes live prediction artifacts.

## Déploiement

- Branche : `master`
- Remote : `https://github.com/SamuelLachance/Sports-Odds-Algorithms`
- Site : `https://samuellachance.github.io/Sports-Odds-Algorithms/`
- Rebuild : **4×/jour** America/Toronto (minuit, 6h, midi, 18h EDT via GitHub Actions `pages.yml`)
- CI tests : `.github/workflows/test.yml` (compileall + suite pytest + `smoke_test.py` on push/PR to `master`; pytest cache restored between runs)

## Conventions

- Préserver la logique Algo V1/V2 dans `algo.py` ; tester avec `smoke_test.py` / pytest après changements core.
- Le board public consomme ESPN + résultats saison courante ; ne pas casser le pipeline Pages.
- NBA/WNBA v2 live : passer `home_moneyline` / `away_moneyline` / `home_spread` active les heads market-aware (`model_variant=market_aware`) via `web/basketball_v2_market.py` ; sans cotes → head pure.
- Ne pas inventer de claims de performance (pas de profits garantis ; le pilot NBA ML a un ROI close négatif).
- Commits clairs en anglais ; pousser sur `master` après changements validés.

## CORS (`CORS_ALLOW_ORIGINS`)

FastAPI CORS is configured in `web/app.py` via `cors_allow_origins()`.

- **Default** (env unset): public Pages origin `https://samuellachance.github.io` plus local API/dev hosts (`127.0.0.1` / `localhost` on ports `8000` and `5173`).
- **Comma-separated allowlist**: `CORS_ALLOW_ORIGINS=https://example.com,http://127.0.0.1:3000`
- **Open sandbox** (local only): `CORS_ALLOW_ORIGINS=*`

Do not set `*` on a publicly reachable API. Restart uvicorn after changing the env var.

## Daily build env knobs

Used by `web/daily_service.py` / `web/live_odds_enrichment.py` / Pages `build_gh_pages.py`:

| Variable | Default | Effect |
|---|---|---|
| `FAST_DAILY_BUILD` | unset / off | When `1`/`true`, skips multi-book fetches (and other slow paths) so CI/Pages stays under timeout. |
| `LIVE_MULTI_BOOK` | on interactively; off under `FAST_DAILY_BUILD` | Set `1` to force multi-book on; `0`/`false` to force off. Slate `summary.line_shopping` reports `on` / `skipped_fast_build` / `off`. |
| `LIVE_MULTI_BOOK_BUDGET_S` | `120` | Wall-time budget (seconds) for cumulative book fetches per build. |
| `NEWS_SIGNALS` | league defaults | Set `0`/`false` to disable headline keyword nudges in the context layer. |

Soccer paper tracking (`web/soccer_paper_tracking.py`) is an **internal** research log graded during Pages deploy — not Hubáček official tracking and not shown as site performance.

## Pytest markers

`pytest.ini` defines a `slow` marker for longer integration / live-artifact checks. Default CI runs the full suite; to skip slow tests locally:

```powershell
python -m pytest tests/ -q -m "not slow"
```
