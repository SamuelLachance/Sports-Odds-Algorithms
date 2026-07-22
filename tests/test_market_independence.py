"""Market-independence, enforced as a build gate rather than an intention.

The real risk is not that someone puts odds in the feature vector — that rule is
too obvious to break. It is that market data creeps in through an import or a
stray reference. These tests walk the model package's source and fail the build
on any market term or any import into the rendering/market layer.

They also assert the reverse guarantees the model claims: it produces a
game-level probability and it depends only on team + pitcher inputs.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1] / "mlbwp"

# Market terms that must never appear in EXECUTABLE model code — identifiers,
# string literals (URLs, feature names), or import names. Matched at word
# boundaries so a pitcher named "Gaviglio" does not trip "vig". Comments and
# docstrings are excluded on purpose: the package documents that it uses no odds,
# and saying so must not fail the guard.
BLOCKLIST = (
    "odds", "moneyline", "vig", "juice", "sportsbook", "vegas", "pinnacle",
    "draftkings", "fanduel", "betmgm", "bovada", "oddsapi", "devig", "kelly",
    "spread", "handle", "sharp", "steam",
)
_RE = re.compile(r"(?<![a-z0-9])(" + "|".join(BLOCKLIST) + r")(?![a-z0-9])", re.I)

# The model package may not import these — they are the rendering / (future)
# market-comparison layers.
FORBIDDEN_IMPORT_ROOTS = ("mlbwp_site", "market")

MODEL_FILES = sorted(PKG.rglob("*.py"))


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _code_strings(tree: ast.AST) -> list[str]:
    """Identifiers, import names and non-docstring string literals — i.e. the
    executable surface, excluding comments (not in the AST) and docstrings."""
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(
                    getattr(body[0], "value", None), ast.Constant) and isinstance(
                    body[0].value.value, str):
                docstrings.add(id(body[0].value))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
            out.append(node.value)
        elif isinstance(node, ast.Name):
            out.append(node.id)
        elif isinstance(node, ast.Attribute):
            out.append(node.attr)
        elif isinstance(node, ast.arg):
            out.append(node.arg)
        elif isinstance(node, ast.Import):
            out.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.append(node.module)
    return out


@pytest.mark.parametrize("path", MODEL_FILES, ids=lambda p: p.name)
def test_no_market_terms_in_model_code(path: Path):
    tree = ast.parse(_source(path), filename=str(path))
    hits = sorted({m.group(0).lower() for s in _code_strings(tree) for m in _RE.finditer(s)})
    assert not hits, f"{path.name} uses market term(s) in code: {hits}"


@pytest.mark.parametrize("path", MODEL_FILES, ids=lambda p: p.name)
def test_model_does_not_import_render_or_market_layer(path: Path):
    tree = ast.parse(_source(path), filename=str(path))
    bad = []
    for node in ast.walk(tree):
        mods = []
        if isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods = [node.module]
        for m in mods:
            root = m.split(".")[0]
            if root in FORBIDDEN_IMPORT_ROOTS:
                bad.append(m)
    assert not bad, f"{path.name} imports forbidden module(s): {bad}"


def test_artifacts_carry_no_market_features():
    """No market feature may be baked into the frozen model.

    Checks the STRUCTURAL field names — the top-level keys and the model
    parameters — which is where a market input would have to appear. The
    team/pitcher/name-index entries are identifiers (a reliever named Joe Kelly,
    a pitcher named Gaviglio) and are not feature names, so they are not scanned.
    """
    import json

    ratings = json.loads((PKG / "artifacts" / "ratings.json").read_text(encoding="utf-8"))
    structural = list(ratings.keys()) + list(ratings["params"].keys())
    hits = sorted({m.group(0).lower() for s in structural for m in _RE.finditer(s)})
    assert not hits, f"ratings.json has market-named structural field(s): {hits}"
    # The model's only inputs are team ratings and pitcher FIP adjustments.
    assert set(ratings["params"]) == {
        "k_team", "hfa", "team_regress", "beta", "decay", "prior_ip", "season_decay"}


def test_model_produces_game_level_probability():
    """Sanity: the model's public contract is a home win probability in (0,1)."""
    from mlbwp.serve import Predictor
    p = Predictor()
    # two known teams and pitchers from the reconstructed engine
    any_team = next(iter(p.fe.R))
    other = next(t for t in p.fe.R if t != any_team)
    out = p.predict(any_team, other, "", "")
    assert "home_win_prob" in out
    assert 0.0 < out["home_win_prob"] < 1.0
    # decomposition present and market-free
    assert set(out["contributions_pp"]) == {
        "team", "home_field", "home_pitcher", "away_pitcher", "bullpen", "lineup", "power"}
