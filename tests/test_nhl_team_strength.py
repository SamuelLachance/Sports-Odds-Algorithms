"""Team strength: the one NHL team number on the pages (package nhl-ratings).

The front end (mlbwp_site/js/nhl.js, nhlStr) turns each team's walked Elo and xG
states - the payload's t.elo / t.xg, exactly what every served forecast uses -
into the model's own neutral-ice view of the team:

    z   = c_elo * K * (Elo - 1500) + c_xg * xG,   K = ln(10) / 400
    win = 100 / (1 + e^-z)     (chance to beat an average team on neutral ice)

Pinned here:
  S1  the constants in nhl.js ARE the served blend's coefficients
      (data/nhl_model.json blend.coefs; that file is sha-pinned by the 2026-27
      forward-test pre-registration, so they cannot drift under the page);
  S2  the decomposition is the model itself: every unplayed regular-season
      forecast in the served payload is
          logit(hp) = (z_home - z_away) + H + c_rest*(hrest - arest)
                      + c_b2b_home*hb2b + c_b2b_away*ab2b,
      H = intercept + c_elo*K*ha + c_xg*XG_HA, to 0.002 logits (payload rounding);
  S3  the Node implementation equals the Python formula for all 32 teams to
      1e-9, and the strength rank is a permutation of 1..32 ordered by z
      descending, ties broken by team code;
  S4  coherence: strength orders the teams like the season simulation built from
      the same ratings (Spearman with projected points >= 0.95).
Only served, UNPLAYED forecasts are decomposed; no outcome and no TEST-era
metric is computed. The Node part is skipped when Node is absent.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "phase0"))
DATA = ROOT / "site" / "data"
PAYLOAD = DATA / "nhl.json"
MODEL = ROOT / "data" / "nhl_model.json"
NHL_JS = ROOT / "mlbwp_site" / "js" / "nhl.js"
K = math.log(10.0) / 400.0


def _model() -> dict:
    if not MODEL.is_file():
        pytest.skip("data/nhl_model.json absent")
    return json.loads(MODEL.read_text(encoding="utf-8"))


def _payload() -> dict:
    if not PAYLOAD.is_file():
        pytest.skip("site/data/nhl.json not built")
    return json.loads(PAYLOAD.read_text(encoding="utf-8"))


def _xg_ha() -> float:
    src = (ROOT / "phase0" / "nhl_features_eval.py").read_text(encoding="utf-8")
    return float(re.search(r"^XG_HA\s*=\s*([0-9.]+)", src, re.M).group(1))


def js_constants() -> dict:
    src = NHL_JS.read_text(encoding="utf-8")
    m = re.search(r"const NHL_STR=\{elo:(-?[0-9.]+),xg:(-?[0-9.]+),icpt:(-?[0-9.]+)\}", src)
    assert m, "NHL_STR not found in nhl.js"
    return {"elo": float(m.group(1)), "xg": float(m.group(2)), "icpt": float(m.group(3))}


def z_of(t: dict, c_elo: float, c_xg: float) -> float:
    return c_elo * K * (float(t["elo"]) - 1500.0) + c_xg * float(t["xg"])


def strength(teams: dict, c_elo: float, c_xg: float) -> dict:
    """code -> (z, win%, rank by z desc, ties by code)."""
    z = {c: z_of(t, c_elo, c_xg) for c, t in teams.items()}
    order = sorted(z, key=lambda c: (-z[c], c))
    return {c: (z[c], 100.0 / (1.0 + math.exp(-z[c])), order.index(c) + 1) for c in z}


def spearman(a, b) -> float:
    def ranks(x):
        o = sorted(range(len(x)), key=lambda i: x[i])
        r = [0.0] * len(x)
        i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and x[o[j + 1]] == x[o[i]]:
                j += 1
            for k in range(i, j + 1):
                r[o[k]] = (i + j) / 2.0 + 1.0
            i = j + 1
        return r
    ra, rb = ranks(list(a)), ranks(list(b))
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = math.sqrt(sum((x - ma) ** 2 for x in ra))
    vb = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return cov / (va * vb)


# ------------------------------------------------------------------- S1 ----
def test_s1_js_constants_are_the_served_blend_coefficients():
    m = _model()
    js = js_constants()
    c = m["blend"]["coefs"]
    assert js["elo"] == c["elo_logit"] == 0.70016
    assert js["xg"] == c["xg"] == 0.37079
    assert js["icpt"] == m["blend"]["intercept"]      # home ice between equal teams (edge line)


def test_s1_the_constants_file_is_pinned_by_the_forward_test():
    pre = ROOT / "data" / "pv_nhl_forward_prereg_2026_27.json"
    if not pre.is_file():
        pytest.skip("pre-registration absent")
    pins = json.loads(pre.read_text(encoding="utf-8"))["frozen_params_sha256"]
    assert "data/nhl_model.json" in {k.replace("\\", "/") for k in pins}


# ------------------------------------------------------------------- S2 ----
def test_s2_every_unplayed_forecast_is_strength_gap_plus_home_plus_schedule():
    m, n = _model(), _payload()
    c, b = m["blend"]["coefs"], m["blend"]
    H = b["intercept"] + c["elo_logit"] * K * m["elo_cfg"]["ha"] + c["xg"] * _xg_ha()
    assert abs(H - 0.14958) < 5e-5
    T = n["teams"]
    rows = [g for g in n["schedule"]
            if g.get("hs") is None and not g.get("playoff") and g.get("hp") is not None]
    if len(rows) < 50:
        pytest.skip("fewer than 50 unplayed regular-season games in this payload")
    worst = 0.0
    for g in rows:
        hp = float(g["hp"])
        zh, za = z_of(T[g["home"]], c["elo_logit"], c["xg"]), z_of(T[g["away"]], c["elo_logit"], c["xg"])
        pred = ((zh - za) + H + c["rest"] * (g["hrest"] - g["arest"])
                + c["b2b_home"] * g["hb2b"] + c["b2b_away"] * g["ab2b"])
        err = abs(math.log(hp / (1 - hp)) - pred)
        worst = max(worst, err)
        assert err <= 0.002, (g["id"], err)
    assert worst < 0.002


# ------------------------------------------------------------------- S4 ----
def test_s4_strength_orders_teams_like_the_season_simulation():
    m, n = _model(), _payload()
    c = m["blend"]["coefs"]
    P = n.get("proj") or {}
    codes = [t for t in n["teams"] if t in P]
    if len(codes) < 20:
        pytest.skip("no season projection in this payload")
    z = [z_of(n["teams"][t], c["elo_logit"], c["xg"]) for t in codes]
    rho = spearman(z, [P[t]["pts"] for t in codes])
    assert rho >= 0.95, rho


# ------------------------------------------------------------------- S3 ----
_spec = importlib.util.spec_from_file_location("nhl_pages_frontend",
                                               ROOT / "tests" / "test_nhl_pages_frontend.py")
FE = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(FE)

BODY = r"""
;(function(){
  const fs=require("fs"), A=JSON.parse(process.env.NHL_STR_ARGS);
  const J=f=>JSON.parse(fs.readFileSync(f,"utf8"));
  state.board=J(A.data+"/board.json"); state.db=J(A.data+"/db.json");
  try{state.nfl=J(A.data+"/nfl.json");}catch(e){state.nfl=null;}
  state.nhl=J(A.payload); siteLeaguesTidy(); state.league="nhl";
  const I=nhlIdx(), out={};
  for(const [c,t] of Object.entries(state.nhl.teams)){const s=nhlStr(t);
    out[c]={z:s.z,win:s.win,ze:s.ze,zx:s.zx,rk:I.strRk[c],eloRk:I.eloRk[c],xgR:I.xgR[c]};}
  // ties: two teams with identical ratings rank by code
  const tie=JSON.parse(JSON.stringify(state.nhl)); const cs=Object.keys(tie.teams).sort();
  tie.teams[cs[1]].elo=tie.teams[cs[0]].elo; tie.teams[cs[1]].xg=tie.teams[cs[0]].xg;
  state.nhl=tie; const I2=nhlIdx();
  console.log(JSON.stringify({teams:out,tie:[I2.strRk[cs[0]],I2.strRk[cs[1]]]}));
})();
"""


def test_s3_node_strength_equals_python_and_ranks_are_a_permutation(tmp_path):
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    for f in ("board.json", "db.json"):
        if not (DATA / f).is_file():
            pytest.skip(f"site/data/{f} not built")
    from mlbwp_site.build_site import JS
    m, n = _model(), _payload()
    c = m["blend"]["coefs"]
    script = tmp_path / "nhl_strength.js"
    script.write_text(FE.STUB + JS.replace("\nboot();", "\n") + BODY, encoding="utf-8")
    env = dict(os.environ, NHL_STR_ARGS=json.dumps({"data": str(DATA), "payload": str(PAYLOAD)}))
    r = subprocess.run([node, str(script)], capture_output=True, text=True, timeout=240,
                       env=env, encoding="utf-8")
    assert r.returncode == 0, r.stderr[-2000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    js = out["teams"]
    py = strength(n["teams"], c["elo_logit"], c["xg"])
    assert set(js) == set(py) and len(py) == 32
    for code, (z, win, rk) in py.items():
        assert abs(js[code]["z"] - z) <= 1e-9, code
        assert abs(js[code]["win"] - win) <= 1e-9, code
        assert js[code]["rk"] == rk, code
        assert abs(js[code]["ze"] + js[code]["zx"] - z) <= 1e-12
        assert js[code]["eloRk"] == n["teams"][code]["rank"]       # the half's own rank
    assert sorted(v["rk"] for v in js.values()) == list(range(1, 33))
    order = sorted(js, key=lambda k: js[k]["rk"])
    assert all(js[a]["z"] >= js[b]["z"] for a, b in zip(order, order[1:]))
    t0, t1 = out["tie"]
    assert t1 == t0 + 1                     # identical ratings: the lower code ranks first
