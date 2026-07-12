"""Shared pytest fixtures and quiet defaults."""

from __future__ import annotations

import warnings

# Extra belt-and-suspenders beyond pytest.ini filterwarnings.
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"sklearn\..*")
warnings.filterwarnings("ignore", category=FutureWarning, module=r"sklearn\..*")
warnings.filterwarnings("ignore", category=UserWarning, module=r"sklearn\..*")
