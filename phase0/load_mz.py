"""Safely load mitchellz4/MLB-Model's pickles.

Two problems with the file:

1. It was written under pandas <2.0 and references `pandas.core.indexes.numeric`,
   which no longer exists. A shim maps the removed Index classes onto `pd.Index`.
2. Unpickling executes arbitrary code by design. This is a 2-star repository from a
   stranger, so a plain `pickle.load` is not acceptable. The restricted unpickler
   below allows only numpy/pandas reconstruction callables and raises on anything
   else — notably `os.system`, `subprocess`, `builtins.eval`.
"""

from __future__ import annotations

import io
import pickle
import sys
import types

import numpy as np
import pandas as pd

# --- shim the removed pandas module ---------------------------------------
if "pandas.core.indexes.numeric" not in sys.modules:
    _m = types.ModuleType("pandas.core.indexes.numeric")
    for _n in ("Int64Index", "Float64Index", "UInt64Index", "NumericIndex"):
        setattr(_m, _n, pd.Index)
    sys.modules["pandas.core.indexes.numeric"] = _m

ALLOWED_PREFIXES = ("pandas.", "numpy.", "numpy._core", "collections", "copyreg")
ALLOWED_EXACT = {
    ("builtins", "object"), ("builtins", "list"), ("builtins", "dict"),
    ("builtins", "set"), ("builtins", "tuple"), ("builtins", "frozenset"),
    ("builtins", "int"), ("builtins", "float"), ("builtins", "str"),
    ("builtins", "bool"), ("builtins", "bytes"), ("builtins", "complex"),
    ("numpy", "dtype"), ("numpy", "ndarray"),
}
DENIED_HINTS = ("os", "subprocess", "sys", "shutil", "socket", "builtins.eval",
                "builtins.exec", "posix", "nt", "commands", "popen")


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        full = f"{module}.{name}"
        if (module, name) in ALLOWED_EXACT:
            return super().find_class(module, name)
        if module.split(".")[0] in ("os", "subprocess", "shutil", "socket", "sys"):
            raise pickle.UnpicklingError(f"BLOCKED dangerous global: {full}")
        if full in ("builtins.eval", "builtins.exec", "builtins.compile",
                    "builtins.__import__", "builtins.open", "builtins.getattr"):
            raise pickle.UnpicklingError(f"BLOCKED dangerous global: {full}")
        if module.startswith(ALLOWED_PREFIXES) or module == "builtins":
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"BLOCKED unexpected global: {full}")


def safe_load(path: str):
    with open(path, "rb") as fh:
        return RestrictedUnpickler(io.BytesIO(fh.read())).load()
