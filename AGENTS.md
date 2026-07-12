# Sports Odds Algorithms — Agent

Algorithme de prédiction sportive (NBA, NHL, MLB et autres ligues) avec démo web FastAPI et tableau public GitHub Pages.

## Stack

- Python 3.10+ (`requirements.txt`)
- Core : `algo.py`, `odds_calculator.py`, `backtester.py`
- Web : `web/` (FastAPI + frontend statique)
- Sport models : `web/{nba,wnba,nhl,mlb,soccer,nfl,cfb,cbb}_v2` (GradientBoost v2; CBB primary = GB v2 with Torvik fallback; NFL/CFB v2 live for predictions, official picks disabled until backtest clears)
- Données historiques : CSV dans `nba/`, `nhl/`, `mlb/`
- Modules récents : `web/context_signals.py` (FLB / news / sparse EV caps), `web/portfolio_sizing.py` (stake sizing corrélé), `web/live_odds_enrichment.py` (multi-book NBA/NHL/MLB/WNBA)

## Commandes utiles

```powershell
python -m pip install -r requirements.txt
python -m pytest tests/ -q --tb=short
python -m pytest tests/ -q -m "not slow"
python smoke_test.py
python run_server.py
python -m uvicorn web.app:app --reload --host 127.0.0.1 --port 8000
python scripts/check_v2_data.py
```

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
- CI tests : `.github/workflows/test.yml` (suite pytest complète + `smoke_test.py` on push/PR to `master`)

## Conventions

- Préserver la logique Algo V1/V2 dans `algo.py` ; tester avec `smoke_test.py` / pytest après changements core.
- Le board public consomme ESPN + résultats saison courante ; ne pas casser le pipeline Pages.
- Ne pas inventer de claims de performance (pas de profits garantis ; le pilot NBA ML a un ROI close négatif).
- Commits clairs en anglais ; pousser sur `master` après changements validés.

## CORS (`CORS_ALLOW_ORIGINS`)

FastAPI CORS is configured in `web/app.py` via `cors_allow_origins()`.

- **Default** (env unset): public Pages origin `https://samuellachance.github.io` plus local API/dev hosts (`127.0.0.1` / `localhost` on ports `8000` and `5173`).
- **Comma-separated allowlist**: `CORS_ALLOW_ORIGINS=https://example.com,http://127.0.0.1:3000`
- **Open sandbox** (local only): `CORS_ALLOW_ORIGINS=*`

Do not set `*` on a publicly reachable API. Restart uvicorn after changing the env var.

## Pytest markers

`pytest.ini` defines a `slow` marker for longer integration / live-artifact checks. Default CI runs the full suite; to skip slow tests locally:

```powershell
python -m pytest tests/ -q -m "not slow"
```
