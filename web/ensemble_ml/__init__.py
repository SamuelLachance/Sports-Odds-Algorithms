"""Per-sport mega-models: XGBoost stacks of all prediction layers + market lines."""

from web.ensemble_ml.apply import apply_ensemble_ml
from web.ensemble_ml.config import ensemble_model_available

__all__ = ["apply_ensemble_ml", "ensemble_model_available"]
