"""HybridConfig and focused search space (WNBA-winner recipe family)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class HybridConfig:
    name: str = "default"
    backend: str = "catboost"  # catboost | lightgbm | both
    depth: int = 6
    learning_rate: float = 0.03
    iterations: int = 500
    l2: float = 4.0
    min_data: int = 40
    subsample: float = 0.9
    use_curves: bool = True
    use_categoricals: bool = True
    use_market: bool = True
    market_blend_w: float = 0.0
    feature_mode: str = "full"  # full | core | curves_plus_core
    early_stopping: int = 40
    seed: int = 42
    target_mode: str = "prob"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def focused_search_space() -> list[HybridConfig]:
    """~10 high-upside configs derived from the WNBA overnight winner family."""
    return [
        HybridConfig(
            name="cb_full_w60",
            backend="catboost",
            depth=7,
            learning_rate=0.04,
            iterations=450,
            feature_mode="full",
            use_curves=True,
            market_blend_w=0.60,
            early_stopping=35,
            l2=5.0,
        ),
        HybridConfig(
            name="cb_full_w50",
            backend="catboost",
            depth=7,
            learning_rate=0.04,
            iterations=450,
            feature_mode="full",
            use_curves=True,
            market_blend_w=0.50,
            early_stopping=35,
        ),
        HybridConfig(
            name="cb_full_w40",
            backend="catboost",
            depth=6,
            learning_rate=0.03,
            iterations=500,
            feature_mode="full",
            use_curves=True,
            market_blend_w=0.40,
            early_stopping=40,
        ),
        HybridConfig(
            name="cb_core_w60",
            backend="catboost",
            depth=7,
            learning_rate=0.04,
            iterations=450,
            feature_mode="curves_plus_core",
            use_curves=True,
            market_blend_w=0.60,
            early_stopping=35,
        ),
        HybridConfig(
            name="cb_core_w35",
            backend="catboost",
            depth=6,
            learning_rate=0.03,
            iterations=500,
            feature_mode="curves_plus_core",
            use_curves=True,
            market_blend_w=0.35,
            early_stopping=40,
        ),
        HybridConfig(
            name="cb_core_w0",
            backend="catboost",
            depth=6,
            learning_rate=0.04,
            iterations=500,
            feature_mode="curves_plus_core",
            use_curves=True,
            market_blend_w=0.0,
            early_stopping=40,
        ),
        HybridConfig(
            name="cb_deep_full_w55",
            backend="catboost",
            depth=8,
            learning_rate=0.025,
            iterations=600,
            feature_mode="full",
            use_curves=True,
            market_blend_w=0.55,
            early_stopping=40,
            l2=6.0,
        ),
        HybridConfig(
            name="lgb_core_w50",
            backend="lightgbm",
            depth=7,
            learning_rate=0.03,
            iterations=600,
            feature_mode="curves_plus_core",
            use_curves=True,
            market_blend_w=0.50,
            early_stopping=40,
        ),
        HybridConfig(
            name="both_full_w50",
            backend="both",
            depth=6,
            learning_rate=0.03,
            iterations=400,
            feature_mode="full",
            use_curves=True,
            market_blend_w=0.50,
            early_stopping=35,
        ),
        HybridConfig(
            name="cb_full_w70",
            backend="catboost",
            depth=7,
            learning_rate=0.04,
            iterations=450,
            feature_mode="full",
            use_curves=True,
            market_blend_w=0.70,
            early_stopping=35,
        ),
        # Adaptive logit-space market stack: per-fold weight fit on prior OOS
        # chunks only; beats fixed prob-blending when the raw model is strong.
        HybridConfig(
            name="cb_full_lstack",
            backend="catboost",
            depth=7,
            learning_rate=0.04,
            iterations=450,
            feature_mode="full",
            use_curves=True,
            market_blend_w=0.0,
            target_mode="logit_stack",
            early_stopping=35,
            l2=5.0,
        ),
        HybridConfig(
            name="cb_core_lstack",
            backend="catboost",
            depth=7,
            learning_rate=0.04,
            iterations=450,
            feature_mode="curves_plus_core",
            use_curves=True,
            market_blend_w=0.0,
            target_mode="logit_stack",
            early_stopping=35,
        ),
        # Market-as-init_score residual learners: trees start from logit(close)
        # and learn the correction, instead of prob-averaging toward the close.
        HybridConfig(
            name="cb_full_offset",
            backend="catboost",
            depth=7,
            learning_rate=0.04,
            iterations=450,
            feature_mode="full",
            use_curves=True,
            market_blend_w=0.0,
            target_mode="offset",
            early_stopping=35,
            l2=5.0,
        ),
        HybridConfig(
            name="cb_core_offset",
            backend="catboost",
            depth=6,
            learning_rate=0.03,
            iterations=600,
            feature_mode="curves_plus_core",
            use_curves=True,
            market_blend_w=0.0,
            target_mode="offset",
            early_stopping=40,
        ),
    ]
