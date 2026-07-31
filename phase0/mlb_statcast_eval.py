"""MLB Statcast-era candidates — DEV SCREEN ONLY (2016-2020, the NEW protocol
documents/mlb_statcast_protocol.md). NO TEST-era evaluation lives in this file:
the eval mask hard-asserts season <= 2020; seasons 2021-2024 are the modern
TEST and are spent by the orchestrator only.

Four pre-registered candidates (queue order), all from PRIOR-SEASON Baseball
Savant leaderboard values (leak rule: a game in season Y uses season Y-1
Savant aggregates ONLY — unambiguously pre-game):

  A. framing_sc  prior-season Savant framing runs (rv_tot) of tonight's
                 starting catchers, home - away (start_c from the Retrosheet
                 framing cache data/retro_events/framing.csv).
  B. oaa         prior-season team Outs Above Average diff, plus tonight's
                 starting-9 player-OAA sum diff. OAA starts 2016 -> the
                 feature exists 2017+ (DEV rows are 2017-2020).
  C. sprint      prior-season sprint speed of tonight's starting 9 (mean of
                 crosswalk-matched batters, starting pitchers excluded from
                 the 9), home - away.
  D. stuff       prior-season starter fastball velo diff + whiff% diff
                 (Savant custom leaderboard; starters from gamelog fields
                 101/103).

Endpoints (discovered by probing 2026-07-31; all &csv=true, throttled ~1s):
  framing  leaderboard/catcher-framing?type=catcher&seasonStart=Y&seasonEnd=Y
           &minPitches=1  (the documented &min= is IGNORED by this endpoint;
           minPitches is the real knob — 113 catchers in 2019 vs 64 qualified)
  oaa      leaderboard/outs_above_average?type=Fielder|Fielding_Team
           &startYear=Y&endYear=Y&min=1  (type=player/team silently return
           header-only — the working values are Fielder / Fielding_Team)
  sprint   leaderboard/sprint_speed?min_season=Y&max_season=Y&min=5
  pitcher  leaderboard/custom?year=Y&type=pitcher&min=1&selections=
           player_id,pa,fastball_avg_speed,whiff_percent  (no row cap: 805
           pitchers at min=1 for 2019)

Crosswalk: mlbwp/artifacts/mlbam_to_retro.json (task 27), inverted to
retro -> mlbam here. Team map: static MLBAM team_id -> Retrosheet code.

Screen harness: exactly the shape of phase0/mlb_framing_eval.py --screen —
increment over logit(full_p) from data/model_probs.csv (join gid = home +
date.replace('-','') + dh), 5 season-grouped folds inside DEV (one season per
fold), pooled OOF mean-LL delta + paired bootstrap CI (10k, seed 7).
Gate to recommend the one TEST look: delta >= +0.0005 AND CI lower > 0.

Usage: --pull   (cache leaderboards to data/savant/, atomic, skip-if-cached)
       --build  (join -> data/savant/statcast_features.csv, atomic)
       --screen (DEV-only increment screens; hard cap season 2020)
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os
import sys
import time
import urllib.request

sys.path.insert(0, ".")

SAVDIR = "data/savant"
FEAT = "data/savant/statcast_features.csv"
REPORT = "data/mlb_statcast_dev.json"
FRAMING_CACHE = "data/retro_events/framing.csv"   # per-game start_c per side
CROSSWALK = "mlbwp/artifacts/mlbam_to_retro.json"

DEV_LO, DEV_HI = 2016, 2020   # the screen NEVER touches a season > DEV_HI
N_FOLDS = 5
EPS = 1e-15
GATE_DELTA = 0.0005

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) research-script"}
THROTTLE = 1.1

# seasons of SAVANT VALUES to cache. Feature for game-season Y reads Y-1, so
# priors 2015-2024 serve games 2016-2025. Caching values beyond 2019 does NOT
# evaluate anything on TEST — the screen mask below is what enforces DEV-only.
FRAMING_YEARS = range(2015, 2025)
OAA_YEARS = range(2016, 2025)      # OAA starts 2016
SPRINT_YEARS = range(2015, 2025)
PITCH_YEARS = range(2015, 2025)

ENDPOINTS = {
    "framing": ("https://baseballsavant.mlb.com/leaderboard/catcher-framing"
                "?type=catcher&seasonStart={y}&seasonEnd={y}&team="
                "&sortColumn=rv_tot&sortDirection=desc&csv=true&minPitches=1"),
    "oaa_player": ("https://baseballsavant.mlb.com/leaderboard/outs_above_average"
                   "?type=Fielder&startYear={y}&endYear={y}&split=no&team="
                   "&range=year&min=1&pos=&roles=&viz=hide&csv=true"),
    "oaa_team": ("https://baseballsavant.mlb.com/leaderboard/outs_above_average"
                 "?type=Fielding_Team&startYear={y}&endYear={y}&split=no&team="
                 "&range=year&min=1&pos=&roles=&viz=hide&csv=true"),
    "sprint": ("https://baseballsavant.mlb.com/leaderboard/sprint_speed"
               "?min_season={y}&max_season={y}&position=&team=&min=5&csv=true"),
    "pitcher": ("https://baseballsavant.mlb.com/leaderboard/custom"
                "?year={y}&type=pitcher&filter=&sort=4&sortDir=asc&min=1"
                "&selections=player_id,pa,fastball_avg_speed,whiff_percent"
                "&chart=false&x=fastball_avg_speed&y=fastball_avg_speed"
                "&r=no&chartType=beeswarm&csv=true"),
}

# MLBAM team_id -> Retrosheet team code (2016+ era)
TEAM_MAP = {
    "108": "ANA", "109": "ARI", "110": "BAL", "111": "BOS", "112": "CHN",
    "113": "CIN", "114": "CLE", "115": "COL", "116": "DET", "117": "HOU",
    "118": "KCA", "119": "LAN", "120": "WAS", "121": "NYN", "133": "OAK",
    "134": "PIT", "135": "SDN", "136": "SEA", "137": "SFN", "138": "SLN",
    "139": "TBA", "140": "TEX", "141": "TOR", "142": "MIN", "143": "PHI",
    "144": "ATL", "145": "CHA", "146": "MIA", "147": "NYA", "158": "MIL",
}

F_DATE, F_DH, F_VIS, F_HOME, F_VSP, F_HSP = 0, 1, 3, 6, 101, 103


# ---------------------------------------------------------------- pull ----
def pull():
    os.makedirs(SAVDIR, exist_ok=True)
    plan = [("framing", y) for y in FRAMING_YEARS] + \
           [("oaa_player", y) for y in OAA_YEARS] + \
           [("oaa_team", y) for y in OAA_YEARS] + \
           [("sprint", y) for y in SPRINT_YEARS] + \
           [("pitcher", y) for y in PITCH_YEARS]
    for kind, y in plan:
        path = f"{SAVDIR}/{kind}_{y}.csv"
        if os.path.exists(path) and os.path.getsize(path) > 100:
            print(f"cached  {path}", flush=True)
            continue
        url = ENDPOINTS[kind].format(y=y)
        req = urllib.request.Request(url, headers=UA)
        body = urllib.request.urlopen(req, timeout=90).read().decode(
            "utf-8-sig", errors="replace")
        if body.lstrip()[:9].lower() == "<!doctype":
            print(f"FAIL    {kind} {y}: got HTML, not CSV", flush=True)
            continue
        n = body.count("\n")
        with open(path + ".tmp", "w", encoding="utf-8", newline="") as fh:
            fh.write(body)
        os.replace(path + ".tmp", path)
        print(f"pulled  {path}: {n} rows", flush=True)
        time.sleep(THROTTLE)


# ---------------------------------------------------------------- build ----
def _read_csv(path):
    if not os.path.exists(path):
        return []
    return list(csv.DictReader(open(path, encoding="utf-8-sig")))


def _f(row, key):
    v = (row.get(key) or "").strip()
    try:
        return float(v)
    except ValueError:
        return None


def load_savant():
    """-> per prior-season lookup dicts keyed by MLBAM id (str)."""
    S = {"framing": {}, "framing_rate": {}, "oaa_p": {}, "oaa_t": {},
         "sprint": {}, "velo": {}, "whiff": {}, "sprint_med": {}}
    for y in FRAMING_YEARS:
        # AUDIT FIX: rv_tot is all-zero before 2018 (Savant never backfilled
        # run values) but pct_tot — borderline called-strike rate — is real for
        # every season. Metric = pitch-weighted rate ABOVE season league mean
        # (comparable across the whole era); volume variant scales by pitches.
        rows = [r for r in _read_csv(f"{SAVDIR}/framing_{y}.csv")
                if _f(r, "pct_tot") is not None and (_f(r, "pitches") or 0) > 0]
        tot_p = sum(_f(r, "pitches") for r in rows)
        lg = (sum(_f(r, "pct_tot") * _f(r, "pitches") for r in rows) / tot_p
              if tot_p else 0.0)
        d, dr = {}, {}
        for r in rows:
            above = _f(r, "pct_tot") - lg               # rate above league
            dr[r["id"]] = above * 100.0                 # extra strikes /100 borderline
            d[r["id"]] = above * _f(r, "pitches")       # volume (pseudo-runs scale)
        S["framing"][y], S["framing_rate"][y] = d, dr
    for y in OAA_YEARS:
        S["oaa_p"][y] = {r["player_id"]: _f(r, "outs_above_average")
                         for r in _read_csv(f"{SAVDIR}/oaa_player_{y}.csv")
                         if _f(r, "outs_above_average") is not None}
        S["oaa_t"][y] = {TEAM_MAP.get(r["team_id"]): _f(r, "outs_above_average")
                         for r in _read_csv(f"{SAVDIR}/oaa_team_{y}.csv")
                         if _f(r, "outs_above_average") is not None}
    for y in SPRINT_YEARS:
        d = {r["player_id"]: _f(r, "sprint_speed")
             for r in _read_csv(f"{SAVDIR}/sprint_{y}.csv")
             if _f(r, "sprint_speed") is not None}
        S["sprint"][y] = d
        vals = sorted(d.values())
        S["sprint_med"][y] = vals[len(vals) // 2] if vals else None
    for y in PITCH_YEARS:
        dv, dw = {}, {}
        for r in _read_csv(f"{SAVDIR}/pitcher_{y}.csv"):
            v, w = _f(r, "fastball_avg_speed"), _f(r, "whiff_percent")
            if v is not None:
                dv[r["player_id"]] = v
            if w is not None:
                dw[r["player_id"]] = w
        S["velo"][y], S["whiff"][y] = dv, dw
    return S


def load_crosswalk():
    m2r = json.load(open(CROSSWALK, encoding="utf-8"))
    r2m, coll = {}, 0
    for mid, rid in m2r.items():
        if rid in r2m:
            coll += 1
        r2m[rid] = mid
    print(f"crosswalk: {len(m2r):,} mlbam->retro; inverted {len(r2m):,} "
          f"retro ids ({coll} collisions)", flush=True)
    return r2m


def load_start_catchers():
    """framing.csv (Retrosheet cache) -> gid -> {side: start_c retro id}."""
    sc = {}
    for r in csv.reader(open(FRAMING_CACHE, encoding="utf-8")):
        if r[0] == "game_id" or int(r[2]) < DEV_LO:
            continue
        d = sc.setdefault(r[0], {})
        if r[4]:
            d[int(r[3])] = r[4]
    return sc


def load_lineups():
    lu = {}
    for r in csv.reader(open("data/retro_events/lineups.csv", encoding="utf-8")):
        if r[0] == "game_id":
            continue
        lu.setdefault(r[0], {0: [], 1: []})[int(r[1])].append(r[3])
    return lu


def build():
    S = load_savant()
    r2m = load_crosswalk()
    sc = load_start_catchers()
    lu = load_lineups()
    print(f"start-catcher games (>= {DEV_LO}): {len(sc):,}; "
          f"lineup games: {len(lu):,}", flush=True)

    def sav(table, y, retro_id):
        mid = r2m.get(retro_id)
        if mid is None:
            return None
        return S[table].get(y, {}).get(mid)

    out = []
    cov = {k: [0, 0] for k in ("cat", "oaa_lu", "spr", "sp")}   # matched, total
    for path in sorted(glob.glob("data/retrosheet/gl*.txt")):
        yr = int(os.path.basename(path)[2:6])
        if yr < DEV_LO or yr > 2025:
            continue
        prior = yr - 1
        for r in csv.reader(open(path, encoding="latin-1")):
            if len(r) < 105:
                continue
            gid = r[F_HOME] + r[F_DATE] + r[F_DH]
            row = {"game_id": gid, "date": r[F_DATE], "season": yr}

            # A. framing (starting catchers)
            cats = sc.get(gid, {})
            hc, ac = cats.get(1), cats.get(0)
            if hc and ac:
                fh, fa = sav("framing", prior, hc), sav("framing", prior, ac)
                gh, ga = sav("framing_rate", prior, hc), sav("framing_rate", prior, ac)
                cov["cat"][0] += (fh is not None) + (fa is not None)
                cov["cat"][1] += 2
                if S["framing"].get(prior):
                    row["fr_tot"] = (fh or 0.0) - (fa or 0.0)
                    row["fr_rate"] = (gh or 0.0) - (ga or 0.0)
                    row["fr_nmatch"] = (fh is not None) + (fa is not None)

            # teams + starting pitchers
            home, vis = r[F_HOME], r[F_VIS]
            hsp, vsp = r[F_HSP], r[F_VSP]

            # B. OAA
            ot = S["oaa_t"].get(prior, {})
            if home in ot and vis in ot:
                row["oaa_team"] = ot[home] - ot[vis]
            ln = lu.get(gid)
            if ln and len(ln[0]) >= 9 and len(ln[1]) >= 9 and S["oaa_p"].get(prior):
                sides = {}
                for side in (0, 1):
                    tot = 0.0
                    for pid in ln[side][:9]:
                        v = sav("oaa_p", prior, pid)
                        cov["oaa_lu"][0] += v is not None
                        cov["oaa_lu"][1] += 1
                        tot += v or 0.0
                    sides[side] = tot
                row["oaa_lineup"] = sides[1] - sides[0]

            # C. sprint (starting 9, pitchers excluded)
            if ln and len(ln[0]) >= 9 and len(ln[1]) >= 9 and S["sprint"].get(prior):
                med = S["sprint_med"][prior]
                ok = True
                means, imps = {}, {}
                for side in (0, 1):
                    vals, n = [], 0
                    for pid in ln[side][:9]:
                        if pid in (hsp, vsp):
                            continue
                        n += 1
                        v = sav("sprint", prior, pid)
                        cov["spr"][0] += v is not None
                        cov["spr"][1] += 1
                        vals.append(v)
                    got = [v for v in vals if v is not None]
                    if len(got) < 6:
                        ok = False
                        break
                    means[side] = sum(got) / len(got)
                    imps[side] = sum(v if v is not None else med
                                     for v in vals) / n
                if ok:
                    row["sprint_mean"] = means[1] - means[0]
                    row["sprint_imp"] = imps[1] - imps[0]

            # D. stuff (starting pitchers)
            if hsp and vsp and S["velo"].get(prior):
                vh, va = sav("velo", prior, hsp), sav("velo", prior, vsp)
                wh, wa = sav("whiff", prior, hsp), sav("whiff", prior, vsp)
                cov["sp"][0] += (vh is not None) + (va is not None)
                cov["sp"][1] += 2
                if vh is not None and va is not None:
                    row["velo"] = vh - va
                if wh is not None and wa is not None:
                    row["whiff"] = wh - wa
            out.append(row)

    cols = ["game_id", "date", "season", "fr_tot", "fr_rate", "fr_nmatch",
            "oaa_team", "oaa_lineup", "sprint_mean", "sprint_imp",
            "velo", "whiff"]
    with open(FEAT + ".tmp", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(out)
    os.replace(FEAT + ".tmp", FEAT)
    print(f"wrote {FEAT}: {len(out):,} games", flush=True)
    for k, lab in (("cat", "starting catchers w/ prior framing"),
                   ("oaa_lu", "lineup slots w/ prior player OAA"),
                   ("spr", "lineup slots w/ prior sprint"),
                   ("sp", "starting pitchers w/ prior velo")):
        m, t = cov[k]
        print(f"  match {lab}: {m:,}/{t:,} ({m/max(t,1):.1%})", flush=True)


# ---------------------------------------------------------------- screen ----
CANDIDATES = {
    "A_framing_sc": ["fr_tot", "fr_rate"],
    "B_oaa": ["oaa_team", "oaa_lineup"],
    "C_sprint": ["sprint_mean", "sprint_imp"],
    "D_stuff": ["velo", "whiff", "stuff2"],
}


def screen():
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    feats = {}
    for r in csv.DictReader(open(FEAT, encoding="utf-8")):
        feats[r["game_id"]] = r

    probs = []
    n_dev_probs = 0
    for r in csv.DictReader(open("data/model_probs.csv")):
        season = int(r["season"])
        if season < DEV_LO or season > DEV_HI:      # HARD CAP: no TEST-era rows
            continue
        if r["full_p"] == "":
            continue
        n_dev_probs += 1
        gid = r["home"] + r["date"].replace("-", "") + r["dh"]
        f = feats.get(gid)
        if f is None:
            continue
        probs.append((season, float(r["full_p"]), int(r["y"]), f))
    assert all(DEV_LO <= s <= DEV_HI for s, *_ in probs)
    print(f"DEV games with full_p: {n_dev_probs:,}; with a feature row: "
          f"{len(probs):,}", flush=True)

    def llv(yy, pp):
        pp = np.clip(pp, EPS, 1 - EPS)
        return -(yy * np.log(pp) + (1 - yy) * np.log(1 - pp))

    res = {"dev_years": f"{DEV_LO}-{DEV_HI}", "folds": N_FOLDS,
           "n_dev_probs": n_dev_probs, "gate": f">=+{GATE_DELTA} & CI lo>0",
           "candidates": {}}
    for cand, variants in CANDIDATES.items():
        need = [v for v in variants if v != "stuff2"]
        rows = [(s, p, y, f) for s, p, y, f in probs
                if all(f.get(v) not in (None, "") for v in need)]
        if not rows:
            print(f"\n== {cand}: NO joined rows — skipped")
            res["candidates"][cand] = {"n": 0, "verdict": "NO DATA"}
            continue
        seas = np.array([r[0] for r in rows])
        assert seas.max() <= DEV_HI                 # belt & braces: DEV only
        p = np.clip(np.array([r[1] for r in rows]), EPS, 1 - EPS)
        base_z = np.log(p / (1 - p))
        y = np.array([r[2] for r in rows], float)
        cols = {v: np.array([float(r[3][v]) for r in rows]) for v in need}
        yrs = sorted(set(seas.tolist()))
        print(f"\n== {cand}: n={len(rows):,} ({len(rows)/n_dev_probs:.1%} of "
              f"DEV) seasons {yrs[0]}-{yrs[-1]}")
        for v in need:
            print(f"   {v}: std {cols[v].std():.4f}  nonzero "
                  f"{np.mean(np.abs(cols[v]) > 1e-9):.1%}")
        groups = np.array_split(np.array(yrs), min(N_FOLDS, len(yrs)))
        cres = {"n": int(len(rows)), "coverage": round(len(rows) / n_dev_probs, 4),
                "seasons": f"{yrs[0]}-{yrs[-1]}", "variants": {}}
        for name in variants:
            fcols = [cols["velo"], cols["whiff"]] if name == "stuff2" \
                else [cols[name]]
            d_base = np.empty(len(y)); d_with = np.empty(len(y))
            coefs = []
            for grp in groups:
                te = np.isin(seas, grp); tr = ~te
                fzs = []
                for fc in fcols:
                    mu, sd = fc[tr].mean(), fc[tr].std()
                    fzs.append((fc - mu) / (sd if sd > 0 else 1.0))
                Xb = base_z.reshape(-1, 1)
                X = np.column_stack([base_z] + fzs)
                mb = LogisticRegression(C=1e6, max_iter=2000).fit(Xb[tr], y[tr])
                ms = LogisticRegression(C=1e6, max_iter=2000).fit(X[tr], y[tr])
                d_base[te] = llv(y[te], mb.predict_proba(Xb[te])[:, 1])
                d_with[te] = llv(y[te], ms.predict_proba(X[te])[:, 1])
                coefs.append([round(float(c), 4) for c in ms.coef_[0][1:]])
            d = d_base - d_with                     # positive = feature helps
            rng = np.random.default_rng(7)
            bs = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(axis=1)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            sig = "SIG" if lo > 0 else ("SIG WORSE" if hi < 0 else "n.s.")
            gate = "PASS-GATE" if (d.mean() >= GATE_DELTA and lo > 0) else "no"
            print(f"  {name:<12} OOF base {d_base.mean():.5f} -> "
                  f"{d_with.mean():.5f}  delta {d.mean():+.5f} "
                  f"CI [{lo:+.5f},{hi:+.5f}] {sig}  gate:{gate}  "
                  f"coef/fold {coefs}")
            cres["variants"][name] = {
                "base_ll": round(float(d_base.mean()), 5),
                "with_ll": round(float(d_with.mean()), 5),
                "delta": round(float(d.mean()), 6),
                "ci": [round(float(lo), 6), round(float(hi), 6)],
                "sig": sig, "gate": gate, "fold_coefs": coefs}
        res["candidates"][cand] = cres
    with open(REPORT + ".tmp", "w") as fh:
        json.dump(res, fh, indent=1)
    os.replace(REPORT + ".tmp", REPORT)
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--screen"
    if mode == "--pull":
        pull()
    elif mode == "--build":
        build()
    else:
        screen()
