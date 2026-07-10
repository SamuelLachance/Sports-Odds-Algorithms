# Sports Odds Algorithms — Agent

Algorithme de prédiction sportive (NBA, NHL, MLB et autres ligues) avec démo web FastAPI et tableau public GitHub Pages.

## Stack

- Python 3.10+ (`requirements.txt`)
- Core : `algo.py`, `odds_calculator.py`, `backtester.py`
- Web : `web/` (FastAPI + frontend statique)
- Données historiques : CSV dans `nba/`, `nhl/`, `mlb/`
- Modules récents : `web/context_signals.py` (FLB / news / sparse EV caps), `web/portfolio_sizing.py` (stake sizing corrélé), `web/live_odds_enrichment.py` (multi-book NBA/NHL/MLB/WNBA)

## Commandes utiles

```powershell
python -m pip install -r requirements.txt
python -m pytest tests/ -q --tb=short
python smoke_test.py
python run_server.py
python -m uvicorn web.app:app --reload --host 127.0.0.1 --port 8000
```

## Déploiement

- Branche : `master`
- Remote : `https://github.com/SamuelLachance/Sports-Odds-Algorithms`
- Site : `https://samuellachance.github.io/Sports-Odds-Algorithms/`
- Rebuild : **4×/jour** America/Toronto (minuit, 6h, midi, 18h EDT via GitHub Actions `pages.yml`)
- CI tests : `.github/workflows/test.yml` (pytest core + `smoke_test.py` on push/PR to `master`)

## Conventions

- Préserver la logique Algo V1/V2 dans `algo.py` ; tester avec `smoke_test.py` / pytest après changements core.
- Le board public consomme ESPN + résultats saison courante ; ne pas casser le pipeline Pages.
- Ne pas inventer de claims de performance (pas de profits garantis ; le pilot NBA ML a un ROI close négatif).
- Commits clairs en anglais ; pousser sur `master` après changements validés.
