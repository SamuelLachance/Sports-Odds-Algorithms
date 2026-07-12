"""CFB/NFL train market helpers must treat ESPN EVEN (0) as +100."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _load(name: str, relative: str):
    path = PROJECT_ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "relative",
    [
        "scripts/train_cfb_model.py",
        "scripts/train_nfl_model.py",
    ],
)
def test_train_devig_treats_espn_even_zero_as_plus_100(relative: str) -> None:
    mod = _load(f"train_even_{Path(relative).stem}", relative)
    # EVEN home vs -110 away must de-vig, not return None via abs(0) < 100.
    even = mod._devig_home_prob(0.0, -110.0)
    plus = mod._devig_home_prob(100.0, -110.0)
    assert even is not None
    assert plus is not None
    assert even == pytest.approx(plus)
    assert mod._devig_home_prob(-50.0, -110.0) is None


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
