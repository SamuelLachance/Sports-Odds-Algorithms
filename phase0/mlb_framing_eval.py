"""MLB catcher-framing proxy from Retrosheet pitch sequences — DEV SCREEN ONLY
(2002-2015). NO TEST-era evaluation lives in this file (hard season cap).

Statcast framing starts 2015, useless for DEV. But Retrosheet event files carry
pitch-by-pitch sequences (C = called strike, B = ball) and starting fielders
(position 2 = catcher), so a framing proxy is derivable for the whole DEV era:

  For every taken pitch (B or C; intentional balls I, pitchouts P, auto balls V
  excluded) we know the catcher catching and the pitcher throwing. A catcher's
  framing proxy is his called-strike rate on taken pitches ABOVE the rate his
  pitcher mix would predict:

    exp_called(game, catcher) = sum over pitchers taken_gp * rate_p(as-of)
    residual(game, catcher)   = called_g - exp_called
    framing_c = EWMA(residual) / (EWMA(taken) + K)     (EB shrink toward 0)

  rate_p is the pitcher's own EWMA called-strike rate, shrunk toward the running
  league rate (handles zone-era drift). Feature per game:

    fr_diff = framing(home starting catcher) - framing(away starting catcher)

  read strictly PRE-game (state updates only after the feature is recorded).
  Catcher identity is announced-lineup info (same no-leak argument as the
  absence feature: historical actual starter == announced modulo rare scratches).

Known simplifications (accepted for a screen):
  - pitcher rate is itself catcher-contaminated (two-way EB decomposition would
    fix; kept simple per protocol)
  - no count/handedness/umpire control — count mix and ump draw are noise here
  - mid-PA battery changes attributed to the personnel on the final play record

Pitch-string double-count guard: when a PA is interrupted (SB/WP/sub), Retrosheet
REPEATS the full sequence-so-far on the later record (verified empirically:
5,317 prefix / 0 non-prefix on 2008 sample). We therefore aggregate consecutive
play records with the same (inning, side, batter) and count only the final
(longest) pitch string once.

Base model: shipped full_p from data/model_probs.csv, join by
game_id = home + date.replace('-','') + dh (same recipe as mlb_absence_eval).
Screen harness: 5 season-grouped CV folds within 2002-2015, logistic
recalibration logit(full_p) vs logit(full_p)+feature_z, pooled OOF mean-LL delta
+ paired bootstrap CI. Gate: <= -0.0002 nothing, >= +0.0005 material.

Usage: --build  (stream .EV* files -> data/retro_events/framing.csv, atomic)
       --screen (as-of walk + DEV-only increment screen; hard cap season 2015)
"""
from __future__ import annotations

import csv
import glob
import json
import os
import sys
from collections import defaultdict
from sys import intern

sys.path.insert(0, ".")

FRAMING = "data/retro_events/framing.csv"
NAMES = "data/retro_events/framing_names.json"
REPORT = "data/mlb_framing_dev.json"

DEV_LO, DEV_HI = 2002, 2015   # the screen NEVER touches a season > DEV_HI
N_FOLDS = 5
EPS = 1e-15

# as-of state knobs
DP = 0.995      # pitcher EWMA decay per appearance (game)
DC = 0.998      # catcher EWMA decay per game caught (half-life ~ 3 catcher-seasons)
LD = 0.9995     # league running-rate decay per game (tracks zone-era drift)
KP = 300        # pitcher shrink (taken pitches) toward league rate
MIN_DEN = 0.0   # no hard mask: shrinkage sends unknown catchers to ~0


# ---------------------------------------------------------------- build ----
def parse_file(path: str, out_rows: list, names: dict, season_games: dict):
    """Stream one event file; emit (game_id, date, season, def_side, start_c,
    catcher, pitcher, taken, called) aggregates."""
    game_id = date = None
    season = 0
    cur_p = {0: None, 1: None}   # side -> current pitcher
    cur_c = {0: None, 1: None}   # side -> current catcher
    start_c = {0: None, 1: None}
    agg = {}                     # (def_side, catcher, pitcher) -> [taken, called]
    pending = None               # [key, pitch_str, def_side, catcher, pitcher]

    def flush_pa():
        nonlocal pending
        if pending is None:
            return
        _, seq, ds, ca, pi = pending
        pending = None
        if ca is None or pi is None:
            return
        called = seq.count("C")
        taken = called + seq.count("B")
        if taken:
            k = (ds, ca, pi)
            t = agg.get(k)
            if t is None:
                agg[k] = [taken, called]
            else:
                t[0] += taken
                t[1] += called
    def flush_game():
        flush_pa()
        if game_id is not None and agg:
            for (ds, ca, pi), (tk, cl) in agg.items():
                out_rows.append([game_id, date, season, ds, start_c[ds] or "",
                                 ca, pi, tk, cl])
            season_games[season] = season_games.get(season, 0) + 1
        agg.clear()

    for line in open(path, encoding="latin-1"):
        f = line.rstrip("\n").split(",")
        rec = f[0]
        if rec == "play":
            bat_side = int(f[2])
            key = (f[1], bat_side, f[3])
            seq = f[5]
            if pending is not None and pending[0] == key:
                # interrupted PA: later record repeats the full sequence
                if len(seq) >= len(pending[1]):
                    pending[1] = seq
                ds = 1 - bat_side
                pending[2], pending[3], pending[4] = ds, cur_c[ds], cur_p[ds]
            else:
                flush_pa()
                ds = 1 - bat_side
                pending = [key, seq, ds, cur_c[ds], cur_p[ds]]
        elif rec in ("start", "sub"):
            pid, side, pos = f[1], int(f[3]), f[5]
            if pos == "1":
                cur_p[side] = intern(pid)
            elif pos == "2":
                cur_c[side] = intern(pid)
                names.setdefault(pid, f[2].strip('"'))
                if rec == "start":
                    start_c[side] = intern(pid)
        elif rec == "id":
            flush_game()
            game_id = f[1]
            season = int(f[1][3:7])
            date = None
            cur_p = {0: None, 1: None}
            cur_c = {0: None, 1: None}
            start_c = {0: None, 1: None}
        elif rec == "info" and f[1] == "date":
            date = f[2].replace("/", "")
    flush_game()


def build():
    files = sorted(glob.glob("data/retro_events/*.EV*"))
    print(f"event files: {len(files)}", flush=True)
    rows, names, season_games = [], {}, {}
    for i, path in enumerate(files):
        parse_file(path, rows, names, season_games)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(files)}  rows={len(rows):,}", flush=True)
    with open(FRAMING + ".tmp", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["game_id", "date", "season", "side", "start_c", "catcher",
                    "pitcher", "taken", "called"])
        w.writerows(rows)
    os.replace(FRAMING + ".tmp", FRAMING)
    with open(NAMES + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(names, fh)
    os.replace(NAMES + ".tmp", NAMES)
    tk = sum(r[7] for r in rows)
    cl = sum(r[8] for r in rows)
    print(f"wrote {FRAMING}: {len(rows):,} game-side-battery rows, "
          f"{tk:,} taken pitches, league called-strike rate {cl/tk:.4f}", flush=True)
    for s in sorted(season_games):
        print(f"  {s}: {season_games[s]:,} games with pitch data", flush=True)


# ---------------------------------------------------------------- screen ----
def asof_walk():
    """Chronological walk -> per-game pre-game framing features.
    Returns gid -> dict(variant -> value), plus the end-of-2014 sanity snapshot."""
    games = defaultdict(list)
    meta = {}
    for r in csv.reader(open(FRAMING, encoding="utf-8")):
        if r[0] == "game_id":
            continue
        gid = r[0]
        games[gid].append((int(r[3]), r[5], r[6], int(r[7]), int(r[8])))
        meta[gid] = (r[1], int(r[2]), {**meta.get(gid, (None, None, {}))[2],
                                       int(r[3]): r[4]})
    order = sorted(games, key=lambda g: (meta[g][0], g))

    pnum = defaultdict(float); pden = defaultdict(float)     # pitcher EWMA
    rnum = defaultdict(float); rden = defaultdict(float)     # catcher residual (pitcher-controlled)
    qnum = defaultdict(float)                                # catcher residual (league-only control)
    Lnum, Lden = 0.31, 1.0                                   # league rate prior seed
    feats = {}
    snapshot = None

    def fr(c, k, num):
        return num[c] / (rden[c] + k)

    for gid in order:
        date, season, starters = meta[gid]
        if snapshot is None and season >= 2015:
            lrate = Lnum / Lden
            elig = [(c, fr(c, 1000, rnum), rden[c]) for c in rden if rden[c] >= 3000]
            elig.sort(key=lambda x: -x[1])
            snapshot = {"asof": "pre-2015 (state after 2000-2014)",
                        "league_rate": round(lrate, 4),
                        "top15": [(c, round(v * 100, 2), int(d)) for c, v, d in elig[:15]],
                        "bottom15": [(c, round(v * 100, 2), int(d)) for c, v, d in elig[-15:]]}
        # ---- read feature BEFORE any state update ----
        hc, ac = starters.get(1), starters.get(0)
        if hc and ac:
            feats[gid] = {
                "fr200": fr(hc, 200, rnum) - fr(ac, 200, rnum),
                "fr1000": fr(hc, 1000, rnum) - fr(ac, 1000, rnum),
                "fr3000": fr(hc, 3000, rnum) - fr(ac, 3000, rnum),
                "fr_nopit": fr(hc, 1000, qnum) - fr(ac, 1000, qnum),
                "den_min": min(rden[hc], rden[ac]),
            }
        # ---- update state strictly AFTER reading (leak-safe) ----
        lrate = Lnum / Lden
        cat_upd = defaultdict(lambda: [0.0, 0.0, 0.0])  # c -> [resid, resid_nop, taken]
        pit_upd = defaultdict(lambda: [0.0, 0.0])       # p -> [called, taken]
        g_taken = g_called = 0
        for _, ca, pi, tk, cl in games[gid]:
            prate = (pnum[pi] + KP * lrate) / (pden[pi] + KP)
            u = cat_upd[ca]
            u[0] += cl - tk * prate
            u[1] += cl - tk * lrate
            u[2] += tk
            v = pit_upd[pi]
            v[0] += cl
            v[1] += tk
            g_taken += tk
            g_called += cl
        for ca, (res, resn, tk) in cat_upd.items():
            rnum[ca] = DC * rnum[ca] + res
            qnum[ca] = DC * qnum[ca] + resn
            rden[ca] = DC * rden[ca] + tk
        for pi, (cl, tk) in pit_upd.items():
            pnum[pi] = DP * pnum[pi] + cl
            pden[pi] = DP * pden[pi] + tk
        Lnum = LD * Lnum + g_called
        Lden = LD * Lden + g_taken
    return feats, snapshot


def screen():
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    feats, snapshot = asof_walk()
    print(f"as-of features computed for {len(feats):,} games", flush=True)

    if snapshot:
        names = json.load(open(NAMES, encoding="utf-8"))
        print(f"\nSanity — top framers by proxy, state entering 2015 "
              f"(league rate {snapshot['league_rate']}, min 3000 taken):")
        for c, v, d in snapshot["top15"]:
            print(f"  {names.get(c, c):<22} +{v:.2f} extra called-strike % (den {d:,})")
        print("  --- bottom ---")
        for c, v, d in snapshot["bottom15"][-5:]:
            print(f"  {names.get(c, c):<22} {v:+.2f} (den {d:,})")

    rows = []
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
        rows.append((season, float(r["full_p"]), int(r["y"]), f))
    assert all(DEV_LO <= s <= DEV_HI for s, *_ in rows)
    print(f"\nDEV games with full_p: {n_dev_probs:,}; joined feature: {len(rows):,} "
          f"({len(rows)/max(n_dev_probs,1):.1%} coverage)")

    seas = np.array([r[0] for r in rows])
    p = np.clip(np.array([r[1] for r in rows]), EPS, 1 - EPS)
    base_z = np.log(p / (1 - p))
    y = np.array([r[2] for r in rows], float)
    F = {k: np.array([r[3][k] for r in rows])
         for k in ("fr200", "fr1000", "fr3000", "fr_nopit")}
    dmin = np.array([r[3]["den_min"] for r in rows])
    v = F["fr1000"] * 100
    print(f"feature stats (fr1000 diff, called-strike %): std {v.std():.3f}, "
          f"share |diff|>1%: {np.mean(np.abs(v) > 1.0):.1%}; "
          f"median min-catcher den {np.median(dmin):,.0f} taken pitches")

    dev_seasons = sorted(set(seas.tolist()))
    groups = np.array_split(np.array(dev_seasons), N_FOLDS)

    def llv(yy, pp):
        pp = np.clip(pp, EPS, 1 - EPS)
        return -(yy * np.log(pp) + (1 - yy) * np.log(1 - pp))

    res = {"n": int(len(rows)), "coverage": round(len(rows) / max(n_dev_probs, 1), 4),
           "dev_years": f"{DEV_LO}-{DEV_HI}", "folds": N_FOLDS,
           "sanity": snapshot, "variants": {}}
    for name, feat in F.items():
        d_base = np.empty(len(y)); d_with = np.empty(len(y))
        coefs = []
        for grp in groups:
            te = np.isin(seas, grp); tr = ~te
            mu, sd = feat[tr].mean(), feat[tr].std()
            fz = (feat - mu) / (sd if sd > 0 else 1.0)
            Xb = base_z.reshape(-1, 1)
            X = np.column_stack([base_z, fz])
            mb = LogisticRegression(C=1e6, max_iter=2000).fit(Xb[tr], y[tr])
            ms = LogisticRegression(C=1e6, max_iter=2000).fit(X[tr], y[tr])
            d_base[te] = llv(y[te], mb.predict_proba(Xb[te])[:, 1])
            d_with[te] = llv(y[te], ms.predict_proba(X[te])[:, 1])
            coefs.append(float(ms.coef_[0][1]))
        d = d_base - d_with                     # positive = feature helps
        rng = np.random.default_rng(7)
        bs = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(axis=1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        sig = "SIG" if lo > 0 else ("SIG WORSE" if hi < 0 else "n.s.")
        print(f"  {name:<9} OOF base {d_base.mean():.5f} -> {d_with.mean():.5f}  "
              f"delta {d.mean():+.5f} CI [{lo:+.5f},{hi:+.5f}] {sig}  "
              f"coef/fold {' '.join(f'{c:+.4f}' for c in coefs)}")
        res["variants"][name] = {
            "base_ll": round(float(d_base.mean()), 5),
            "with_ll": round(float(d_with.mean()), 5),
            "delta": round(float(d.mean()), 6),
            "ci": [round(float(lo), 6), round(float(hi), 6)], "sig": sig,
            "fold_coefs": [round(c, 4) for c in coefs]}
    with open(REPORT + ".tmp", "w") as fh:
        json.dump(res, fh, indent=1)
    os.replace(REPORT + ".tmp", REPORT)
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--screen"
    if mode == "--build":
        build()
    else:
        screen()
