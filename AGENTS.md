# Sports Odds Algorithms — Agent

Algorithme de prédiction sportive (NBA, NHL, MLB et autres ligues) avec démo web FastAPI et tableau public GitHub Pages.

## Stack

- Python 3.10+ (`requirements.txt`)
- Core : `algo.py`, `odds_calculator.py`, `backtester.py`
- Web : `web/` (FastAPI + frontend statique)
- Données historiques : CSV dans `nba/`, `nhl/`, `mlb/`

## Commandes utiles

```powershell
python -m pip install -r requirements.txt
python smoke_test.py
python run_server.py
python -m uvicorn web.app:app --reload --host 127.0.0.1 --port 8000
```

## Déploiement

- Branche : `master`
- Remote : `https://github.com/SamuelLachance/Sports-Odds-Algorithms`
- Site : `https://samuellachance.github.io/Sports-Odds-Algorithms/`
- Rebuild quotidien : 3h00 America/Toronto (GitHub Actions `pages.yml`)

## Conventions

- Préserver la logique Algo V1/V2 dans `algo.py` ; tester avec `smoke_test.py` après changements core.
- Le board public consomme ESPN + résultats saison courante ; ne pas casser le pipeline Pages.
- Commits clairs en anglais ; pousser sur `master` après changements validés.
