"""Breakthrough program (NFL) - DEV screen nfl_movcore: margin INSIDE the Elo core. DEV ONLY.

Candidate: margin-of-victory Elo (log-compressed, autocorrelation-corrected
multiplier, 538-style) REPLACES the shipped win/loss Elo column (col 0) of the
14-feature blend. Zero tuned parameters in the primary arm (A1): the shipped core's
k / hfa / regress are reused and the multiplier is normalised by a constant c
(mean ln(|PD|+1) over non-tie games of 1999-2001, frozen) so the average update
size matches the shipped core.

LEDGER CAVEAT (pre-registered, quoted verbatim from the screen spec): ledger row 10
(SRS: a ridge-Massey margin added beside a frozen DEV-fit blend) was the strongest
DEV signal ever (DEV CV -0.00488) and lost on TEST (-0.00238, CI [-0.00594,
+0.00110]); row 46 shows era-broken features stacking; MOV was deliberately skipped
because of that result. Hence the era guards G1-G3 below.

Arms (all pre-declared; no other variant is run):
  A0  shipped 14 columns (baseline)
  A1  PRIMARY  col 0 := MOV_LGT (raw logit units), untuned
  A2  MOV_LGT added as a z-scored 15th column, col 0 kept
  A3  col 0 := MOV_LGT re-tuned (k, hfa, regress) on standalone MOV-Elo DEV LL,
      exactly the nfl_elo.py search (1200 draws, rng 20260723, scored 2001-2015)
  A4  DIAGNOSTIC (never promotable): non-EPA points, season-to-date mean of
      (margin - k_s * EPA diff), k_s from seasons < s only, z-scored 15th column

Protocol: shipped walk-forward (bt_nfl_ideas_quick3.wf, exec'd from its source,
never edited): for s in 2006..2015 train seasons < s from 1999, ties dropped,
weight 0.5^((s-1-season)/3), LogisticRegression(C=100). All 2670 DEV rows scored
(ties included) with an identical mask in every arm. Paired per-game bootstrap
(10000 reps, fixed seed) + season-block bootstrap.

CONSTITUTION: no season > 2015 is read from any file (asserted); no odds column is
read (column-access whitelist + self-grep); run_elo is called only on the
season <= 2015 list; nfl_elo.main() is never called.
Outputs: data/bt_nfl_nfl_movcore.json, data/bt_nfl_nfl_movcore_feats.npy
"""
from __future__ import annotations

import csv
import json
import math
import sys
import time
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, "phase0")
T0 = time.time()
TEST_ERA = 2016
MAX_SEASON_READ = 2015
DEV_LO, DEV_HI = 2006, 2015
HL, C = 3.0, 100.0
BOOT_REPS = 10000
BOOT_SEED = 20260924
BAR = 0.00150
assert MAX_SEASON_READ < TEST_ERA and DEV_HI < TEST_ERA

# ------------------------------------------------------------------ odds audit ----
# forbidden tokens are assembled so this file's own text never contains them
_FORBID = ["money" + "line", "spread" + "_line", "total" + "_line", "_od" + "ds", "over" + "_od" + "ds",
           "under" + "_od" + "ds", "v" + "ig", "de" + "v" + "ig", "clos" + "ing"]
_SELF = open(__file__, encoding="utf-8").read()
_HITS = [t for t in _FORBID if t in _SELF.lower()]
assert not _HITS, f"odds token in screen source: {_HITS}"
GAMES_COLS_ALLOWED = {"game_id", "season", "game_type", "gameday", "home_team", "away_team",
                      "home_score", "away_score", "location"}
USED_COLS: set[str] = set()


def llv(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def logit_arr(p):
    eps = 1e-12
    return np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1 - eps))


# ------------------------------------------------------------------ spine ---------
D = json.load(open("data/bt_nfl_ideas_dev.json", encoding="utf-8"))
G = D["games"]
X14 = np.load("data/bt_nfl_ideas_dev_X.npy")
N = len(G)
assert N == 4515 and X14.shape == (4515, 14)
SEAS = np.array([g["season"] for g in G])
assert SEAS.min() == 1999 and SEAS.max() <= MAX_SEASON_READ
assert np.all(np.diff(SEAS) >= 0)
Y = np.array([g["y"] for g in G], float)
GID = [g["gid"] for g in G]
HOME = [g["home"] for g in G]
AWAY = [g["away"] for g in G]
NEU = [bool(g["neutral"]) for g in G]
SEAS_L = [int(s) for s in SEAS]

# ------------------------------------------------------------------ scores --------
FRANCHISE = {"STL": "LA", "SD": "LAC", "OAK": "LV"}   # nfl_elo.FRANCHISE (copied, not imported)
SC = {}
ELO_ROWS = []            # load_games() replica restricted to season <= 2015
seasons_read = set()
with open("data/nfl_games.csv", encoding="utf-8") as fh:
    rd = csv.reader(fh)
    hdr = next(rd)
    ix = {c: k for k, c in enumerate(hdr)}

    def col(r, name):
        assert name in GAMES_COLS_ALLOWED, name
        USED_COLS.add(name)
        return r[ix[name]]

    for r in rd:
        s = int(col(r, "season"))
        if s > MAX_SEASON_READ:
            continue                                  # TEST era never parsed further
        seasons_read.add(s)
        if col(r, "home_score") == "" or col(r, "away_score") == "":
            continue
        hs, as_ = int(col(r, "home_score")), int(col(r, "away_score"))
        SC[col(r, "game_id")] = (hs, as_)
        ELO_ROWS.append({"season": s, "date": col(r, "gameday"),
                         "home": FRANCHISE.get(col(r, "home_team"), col(r, "home_team")),
                         "away": FRANCHISE.get(col(r, "away_team"), col(r, "away_team")),
                         "y": 1.0 if hs > as_ else (0.0 if hs < as_ else 0.5),
                         "neutral": col(r, "location") == "Neutral", "type": col(r, "game_type")})
assert max(seasons_read) <= MAX_SEASON_READ
assert USED_COLS <= GAMES_COLS_ALLOWED
ELO_ROWS.sort(key=lambda g: (g["date"],))             # identical to nfl_elo.load_games
HS = np.array([SC[g][0] for g in GID], float)
AS = np.array([SC[g][1] for g in GID], float)
assert np.array_equal(Y, np.where(HS > AS, 1.0, np.where(HS < AS, 0.0, 0.5)))
print(f"[{time.time()-T0:.0f}s] spine {N} rows 1999-{SEAS.max()}, ties {int((Y == .5).sum())}", flush=True)

BASE = json.load(open("data/nfl_elo_base.json", encoding="utf-8"))
K0, HFA0, REG0 = BASE["params"]["k"], BASE["params"]["hfa"], BASE["params"]["regress"]
assert (K0, HFA0, REG0) == (47.43351932834238, 52.14162888646703, 0.3647684062154459)

PD_ALL = np.abs(HS - AS)
C_NORM = float(np.mean(np.log(PD_ALL[(SEAS <= 2001) & (PD_ALL > 0)] + 1.0)))
print(f"c = mean ln(|PD|+1), non-tie 1999-2001 = {C_NORM:.6f}", flush=True)


# ------------------------------------------------------------------ the walk ------
def walk(k, hfa, reg, c, mode, hs=HS, as_=AS, states=False):
    """chronological team-Elo walk over the spine. mode 'wl' = shipped core
    (mult 1); mode 'mov' = margin inside the core. Returns pre-game p for every
    row (recorded BEFORE the game's own update)."""
    R = {}
    out = np.empty(N)
    st = np.empty((N, 4)) if states else None
    prev = None
    ln2c = math.log(2.0) / c
    for i in range(N):
        s = SEAS_L[i]
        if prev is not None and s != prev:
            for t in R:
                R[t] = 1500.0 + (R[t] - 1500.0) * (1.0 - reg)
        prev = s
        h, a = HOME[i], AWAY[i]
        rh = R.setdefault(h, 1500.0)
        ra = R.setdefault(a, 1500.0)
        hh = 0.0 if NEU[i] else hfa
        p = 1.0 / (1.0 + 10 ** (-((rh + hh) - ra) / 400.0))
        out[i] = p                                    # recorded before update
        x, z = hs[i], as_[i]
        y = 1.0 if x > z else (0.0 if x < z else 0.5)
        if mode == "wl":
            mult = 1.0
        else:
            pd = abs(x - z)
            if pd > 0:
                dW = ((rh + hh) - ra) if x > z else (ra - (rh + hh))
                den = 0.001 * dW + 2.2
                assert den > 0
                mult = math.log(pd + 1.0) * 2.2 / den / c
            else:
                mult = ln2c
        d = k * mult * (y - p)
        R[h] += d
        R[a] -= d
        if states:
            st[i] = (rh, ra, R[h], R[a])
    return (out, st) if states else out


P_WL = walk(K0, HFA0, REG0, C_NORM, "wl")
P_MOV, ST_MOV = walk(K0, HFA0, REG0, C_NORM, "mov", states=True)
WL_LGT = logit_arr(P_WL)
MOV_LGT = logit_arr(P_MOV)
SAN = {}

# ---- sanity (c1): code identity with cache column 0
d_id = float(np.max(np.abs(WL_LGT - X14[:, 0])))
print(f"(c1) walk(mult=1) vs cache col 0: max |d logit| = {d_id:.3e}", flush=True)
assert d_id < 1e-9
SAN["c1_identity_max_abs_dlogit"] = d_id

# ---- sanity (c2): run_elo on the season <= 2015 list, standalone DEV LL
from nfl_elo import run_elo  # noqa: E402  (main() is never called)
assert max(g["season"] for g in ELO_ROWS) <= MAX_SEASON_READ
pe_, ye_ = run_elo(ELO_ROWS, k=K0, hfa=HFA0, regress=REG0, score_from=2001)
elo_dev_ll = float(llv(ye_, pe_).mean())
print(f"(c2) run_elo standalone DEV LL 2001-2015 = {elo_dev_ll:.5f} (recorded {BASE['dev_ll']})", flush=True)
assert abs(elo_dev_ll - BASE["dev_ll"]) < 5e-6
m01 = SEAS >= 2001
assert np.max(np.abs(pe_ - P_WL[m01])) < 1e-12
SAN["c2_run_elo_dev_ll"] = round(elo_dev_ll, 6)

# ------------------------------------------------------------------ leak tests ----
LEAK = {}
# L1 first game of each team-season uses the regressed end-of-previous-season rating
last_post, first_seen, n_checked = {}, set(), 0
for i in range(N):
    s = SEAS_L[i]
    for side, t in ((0, HOME[i]), (1, AWAY[i])):
        if (t, s) not in first_seen:
            first_seen.add((t, s))
            exp_r = 1500.0 if t not in last_post else \
                1500.0 + (last_post[t][1] - 1500.0) * (1.0 - REG0) ** (s - last_post[t][0])
            # (all teams play every season here, so the exponent is always 1)
            assert abs(ST_MOV[i, side] - exp_r) < 1e-9, (GID[i], t)
            n_checked += 1
    rh, ra = ST_MOV[i, 0], ST_MOV[i, 1]
    hh = 0.0 if NEU[i] else HFA0
    assert MOV_LGT[i] == logit_arr(np.array([1.0 / (1.0 + 10 ** (-((rh + hh) - ra) / 400.0))]))[0]
    last_post[HOME[i]] = (s, ST_MOV[i, 2]); last_post[AWAY[i]] = (s, ST_MOV[i, 3])
LEAK["first_game_of_season_checks"] = n_checked
print(f"leak L1: {n_checked} team-season first games use the pre-update (regressed) rating", flush=True)

# L2 c from 1999-2001 only
assert SEAS[(SEAS <= 2001)].max() == 2001
LEAK["c_window"] = "1999-2001 non-tie games only"

# ------------------------------------------------------------------ A4 non-EPA -----
PFIX = {"JAC": "JAX", "WSH": "WAS", "STL": "LA", "SD": "LAC", "OAK": "LV"}
GEPA = defaultdict(float)
plays_max_season = 0
with open("data/nfl_plays.csv", encoding="utf-8") as fh:
    rd = csv.reader(fh)
    hdr = next(rd)
    assert hdr == ["game_id", "game_date", "posteam", "defteam", "epa"], hdr
    for gid_, _, pos_, _, epa_ in rd:
        s_ = int(gid_[:4])
        if s_ > MAX_SEASON_READ:
            continue
        plays_max_season = max(plays_max_season, s_)
        GEPA[(gid_, PFIX.get(pos_, pos_))] += float(epa_)
assert plays_max_season <= MAX_SEASON_READ
EPA_H = np.array([GEPA.get((GID[i], HOME[i]), np.nan) for i in range(N)])
EPA_A = np.array([GEPA.get((GID[i], AWAY[i]), np.nan) for i in range(N)])
EPA_OK = ~np.isnan(EPA_H) & ~np.isnan(EPA_A)
print(f"A4: EPA present for both sides in {int(EPA_OK.sum())}/{N} games", flush=True)


def k_table(hs, as_):
    """k_s = no-intercept OLS slope of margin on EPA-sum diff over seasons < s
    (bt_nfl_anatomy5.k_for, incl. its <200-pair fallback of 1.0)."""
    ks = {}
    for s in range(1999, MAX_SEASON_READ + 1):
        m = EPA_OK & (SEAS < s)
        assert not m.any() or SEAS[m].max() < s
        if m.sum() < 200:
            ks[s] = 1.0
            continue
        xs = (EPA_H - EPA_A)[m]; ys_ = (hs - as_)[m]
        ks[s] = float((xs @ ys_) / (xs @ xs))
    return ks


def nonepa_feature(hs=HS, as_=AS):
    ks = k_table(hs, as_)
    stt = defaultdict(list)                           # team -> [(season, value)] of PRIOR games
    f = np.zeros(N)
    for i in range(N):
        s = SEAS_L[i]
        h, a = HOME[i], AWAY[i]

        def cur(t):
            L = [v for (ss, v) in stt[t] if ss == s]
            return (sum(L) / len(L)) if len(L) >= 3 else 0.0
        f[i] = cur(h) - cur(a)                        # before this game's own values exist
        if EPA_OK[i]:
            ed = EPA_H[i] - EPA_A[i]
            mg = hs[i] - as_[i]
            stt[h].append((s, mg - ks[s] * ed))
            stt[a].append((s, -mg + ks[s] * ed))
    return f, ks


NONEPA, KS = nonepa_feature()

# L3 future-perturbation: shuffle all scores on/after 20 random cut dates
rng_l = np.random.default_rng(20260924)
# spine dates: from the scores file (gameday) keyed by gid
_gd = {}
with open("data/nfl_games.csv", encoding="utf-8") as fh:
    rd = csv.reader(fh)
    hdr = next(rd)
    ix2 = {c: k for k, c in enumerate(hdr)}
    for r in rd:
        if int(r[ix2["season"]]) > MAX_SEASON_READ:
            continue
        _gd[r[ix2["game_id"]]] = r[ix2["gameday"]]
GD = np.array([_gd[g] for g in GID])
assert all(GD[i] <= GD[i + 1] for i in range(N - 1)), "spine not date-ordered"
cand_cut = sorted(set(GD[SEAS >= 2002]))             # after the c window (1999-2001)
cuts = sorted(rng_l.choice(cand_cut, size=20, replace=False).tolist())
pert = []
for cut in cuts:
    after = GD >= cut
    idx = np.where(after)[0]
    perm = rng_l.permutation(idx)
    hs2, as2 = HS.copy(), AS.copy()
    hs2[idx], as2[idx] = HS[perm], AS[perm]
    c2 = float(np.mean(np.log(np.abs(hs2 - as2)[(SEAS <= 2001) & (np.abs(hs2 - as2) > 0)] + 1.0)))
    assert c2 == C_NORM
    mv2 = logit_arr(walk(K0, HFA0, REG0, c2, "mov", hs=hs2, as_=as2))
    ne2, _ = nonepa_feature(hs2, as2)
    before = ~after
    ok_m = bool(np.array_equal(mv2[before], MOV_LGT[before]))
    ok_n = bool(np.array_equal(ne2[before], NONEPA[before]))
    chg = int((mv2[after] != MOV_LGT[after]).sum())
    assert ok_m and ok_n, cut
    pert.append({"cut": cut, "n_before": int(before.sum()), "mov_bit_identical": ok_m,
                 "nonepa_bit_identical": ok_n, "n_after_changed_mov": chg})
LEAK["future_perturbation"] = pert
print(f"leak L3: 20 cuts, all pre-cut MOV_LGT / NONEPA bit-identical", flush=True)
LEAK["max_season_read"] = {"nfl_games": max(seasons_read), "nfl_plays": plays_max_season,
                           "spine": int(SEAS.max())}
LEAK["games_columns_read"] = sorted(USED_COLS)
LEAK["odds_self_grep_hits"] = _HITS

# ------------------------------------------------------------------ A3 tune -------
def standalone_ll(p):
    m = SEAS >= 2001
    return float(llv(Y[m], p[m]).mean())


def search(mode):
    rng = np.random.default_rng(20260723)
    best = None
    for _ in range(1200):
        cand = dict(k=float(np.exp(rng.uniform(np.log(2.0), np.log(80.0)))),
                    hfa=float(rng.uniform(0.0, 120.0)), regress=float(rng.uniform(0.0, 0.8)))
        s_ = standalone_ll(walk(cand["k"], cand["hfa"], cand["regress"], C_NORM, mode))
        if best is None or s_ < best[0]:
            best = (s_, cand)
    return best


t1 = time.time()
wl_best = search("wl")
print(f"[{time.time()-T0:.0f}s] W/L search reproduces shipped core: LL {wl_best[0]:.5f} {wl_best[1]}", flush=True)
assert abs(wl_best[1]["k"] - K0) < 1e-12 and abs(wl_best[1]["hfa"] - HFA0) < 1e-12
mov_best = search("mov")
print(f"[{time.time()-T0:.0f}s] MOV search: LL {mov_best[0]:.5f} {mov_best[1]}", flush=True)
MOV_T_LGT = logit_arr(walk(mov_best[1]["k"], mov_best[1]["hfa"], mov_best[1]["regress"], C_NORM, "mov"))
SAN["wl_search_reproduces_shipped"] = {"ll": round(wl_best[0], 6), "params": wl_best[1]}

STANDALONE = {"wl_elo_shipped": round(standalone_ll(P_WL), 6),
              "mov_elo_untuned": round(standalone_ll(P_MOV), 6),
              "mov_elo_tuned": round(mov_best[0], 6), "mov_tuned_params": mov_best[1],
              "per_season": {}}
for s in range(2001, 2016):
    m = SEAS == s
    STANDALONE["per_season"][str(s)] = {"wl": round(float(llv(Y[m], P_WL[m]).mean()), 5),
                                        "mov": round(float(llv(Y[m], P_MOV[m]).mean()), 5)}
print("standalone DEV LL 2001-2015:", {k: v for k, v in STANDALONE.items() if k != "per_season"}, flush=True)

np.save("data/bt_nfl_nfl_movcore_feats.npy", np.column_stack([MOV_LGT, WL_LGT, MOV_T_LGT, NONEPA]))

# ------------------------------------------------------------------ harness -------
# quick3's wf() / rep() exec'd from its source (never edited). The extracted block
# also runs quick3's own baseline assert (0.62292 +- 5e-4) = sanity (a).
_src3 = open("phase0/bt_nfl_ideas_quick3.py", encoding="utf-8").read()
_MARK_A = "RNG3 = np.random.default_rng(20260926)"
_MARK_B = 'print("1. schedule-adjusted process block")'
_blk = _MARK_A + _src3.split(_MARK_A, 1)[1].split(_MARK_B, 1)[0]
assert "def wf(" in _blk and "def rep(" in _blk and "json.dump" not in _blk and "RAW" not in _blk
exec(compile(_blk, "quick3<wf/rep>", "exec"))  # noqa: S102  -> wf, rep, b_ll, b_per, b_vec, OUT
print(f"[{time.time()-T0:.0f}s] (a) quick3 wf baseline DEV LL {b_ll:.6f} (cache {D['dev_ll']:.6f})", flush=True)
SAN["a_baseline_dev_ll"] = round(b_ll, 6)
DEVM = (SEAS >= DEV_LO) & (SEAS <= DEV_HI)
assert DEVM.sum() == 2670 and len(b_vec) == 2670 and SEAS[DEVM].max() < TEST_ERA
SEAS_DEV = SEAS[DEVM]


def wf_full(replace=None, extra=None, s_lo=DEV_LO, s_hi=DEV_HI):
    """mirror of quick3.wf that also returns every fitted coefficient; used for fold
    coefficients and the 2002-2005 replication. Asserted identical to wf on DEV."""
    X = X14.copy()
    if replace:
        for c_, v in replace.items():
            X[:, c_] = v
    per, vec, coefs = {}, [], {}
    for s_ in range(s_lo, s_hi + 1):
        assert s_ < TEST_ERA
        tr = (SEAS < s_) & (Y != 0.5); te = SEAS == s_
        assert SEAS[tr].max() < s_ and SEAS[te].max() < TEST_ERA
        Xs = X
        if extra is not None:
            E = extra.reshape(-1, 1)
            mu, sd = E[tr].mean(0), E[tr].std(0); sd[sd < 1e-12] = 1.0
            Xs = np.column_stack([X, (E - mu) / sd])
        w = 0.5 ** ((s_ - 1 - SEAS[tr]) / HL)
        m = LogisticRegression(C=C, max_iter=5000).fit(Xs[tr], Y[tr], sample_weight=w)
        v = llv(Y[te], m.predict_proba(Xs[te])[:, 1])
        per[s_] = float(v.mean()); vec.append(v)
        coefs[s_] = m.coef_[0].tolist()
    return float(np.concatenate(vec).mean()), per, np.concatenate(vec), coefs


def boot(d, seas, level=95.0, reps=BOOT_REPS, seed=BOOT_SEED):
    """paired per-game bootstrap + season-block bootstrap of mean(d)."""
    rng = np.random.default_rng(seed)
    n = len(d)
    means = np.empty(reps)
    for j in range(0, reps, 1000):
        ii = rng.integers(0, n, size=(min(1000, reps - j), n))
        means[j:j + len(ii)] = d[ii].mean(1)
    q = (100.0 - level) / 2.0
    lo, hi = np.percentile(means, [q, 100.0 - q])
    us = np.unique(seas)
    ssum = np.array([d[seas == s].sum() for s in us]); scnt = np.array([(seas == s).sum() for s in us])
    rngb = np.random.default_rng(seed + 1)
    pick = rngb.integers(0, len(us), size=(reps, len(us)))
    bm = ssum[pick].sum(1) / scnt[pick].sum(1)
    blo, bhi = np.percentile(bm, [q, 100.0 - q])
    return [round(float(lo), 6), round(float(hi), 6)], [round(float(blo), 6), round(float(bhi), 6)]


def slope(coefs_by_fold, j):
    xs = np.array(sorted(coefs_by_fold)); ys_ = np.array([coefs_by_fold[s][j] for s in xs])
    return round(float(np.polyfit(xs, ys_, 1)[0]), 6), {str(s): round(float(v), 4) for s, v in zip(xs, ys_)}


RES = {}
ZERO = np.zeros(N)
ARMS = {                                   # sanity arms run FIRST
    "SAN_b_drop_qb_delta": dict(replace={1: ZERO}),
    "SAN_d_drop_col0": dict(replace={0: ZERO}),
    "A1_replace_col0_untuned": dict(replace={0: MOV_LGT}),
    "A2_add_z15": dict(extra=MOV_LGT),
    "A3_replace_col0_tuned": dict(replace={0: MOV_T_LGT}),
    "A4_nonEPA_points_diag": dict(extra=NONEPA),
}
a0_ll, a0_per, a0_vec, a0_coefs = wf_full()
assert np.max(np.abs(a0_vec - b_vec)) < 1e-12
VEC = {"A0": a0_vec}
for nm, kw in ARMS.items():
    ll_q, per_q, vec_q, cq = wf(**kw)                  # quick3's wf (primary numbers)
    ll_f, per_f, vec_f, coefs_f = wf_full(**kw)        # mirror (coefficients)
    assert np.max(np.abs(vec_q - vec_f)) < 1e-12, nm
    VEC[nm] = vec_q
    d = b_vec - vec_q
    ci, bci = boot(d, SEAS_DEV)
    ci975, bci975 = boot(d, SEAS_DEV, level=97.5)
    per_gain = {str(s): round(b_per[s] - per_q[s], 6) for s in per_q}
    late = SEAS_DEV >= 2013
    ci_late, _ = boot(d[late], SEAS_DEV[late])
    RES[nm] = {"ll": round(ll_q, 6), "gain": round(float(d.mean()), 6), "ci95": ci, "ci95_season_block": bci,
               "ci97_5": ci975, "ci97_5_season_block": bci975,
               "per_season_gain": per_gain, "seasons_pos": sum(1 for v in per_gain.values() if v > 0),
               "gain_2013_15": round(float(d[late].mean()), 6), "ci95_2013_15": ci_late,
               "seasons_pos_2013_15": sum(1 for s in (2013, 2014, 2015) if per_gain[str(s)] > 0)}
    if nm.startswith("A1") or nm.startswith("A3"):
        RES[nm]["col0_coef_slope_per_season"], RES[nm]["col0_coef_by_fold"] = slope(coefs_f, 0)
    if nm.startswith("A2") or nm.startswith("A4"):
        RES[nm]["z15_coef_slope_per_season"], RES[nm]["z15_coef_by_fold"] = slope(coefs_f, 14)
        RES[nm]["col0_coef_slope_per_season"], RES[nm]["col0_coef_by_fold"] = slope(coefs_f, 0)
    print(f"  {nm:<26} LL {ll_q:.6f} gain {d.mean():+.5f} CI95 {ci} block {bci} "
          f"seasons+ {RES[nm]['seasons_pos']}/10 | 2013-15 {d[late].mean():+.5f} {ci_late}", flush=True)
    if nm == "SAN_d_drop_col0":
        cb_, cd_ = -RES["SAN_b_drop_qb_delta"]["gain"], -RES["SAN_d_drop_col0"]["gain"]
        ok_b = 0.0045 <= cb_ <= 0.0075 and RES["SAN_b_drop_qb_delta"]["ci95"][1] < 0
        ok_d = 0.002 <= cd_ <= 0.005
        print(f"  harness gate (b) {ok_b} (d) {ok_d}", flush=True)
        assert ok_b and ok_d, "harness sanity failed: screen void, arms not run"
    if nm.startswith("A"):
        rep(nm, (ll_q, per_q, vec_q, cq))               # quick3's own 4000-rep report
RES["A0_shipped"] = {"ll": round(a0_ll, 6)}
RES["A0_shipped"]["col0_coef_slope_per_season"], RES["A0_shipped"]["col0_coef_by_fold"] = slope(a0_coefs, 0)

# A2 / A3 vs A1 (promotion rule: beat A1 by > 0.0002 AND 97.5% CI vs A0 excludes 0)
for nm in ("A2_add_z15", "A3_replace_col0_tuned"):
    dd = VEC["A1_replace_col0_untuned"] - VEC[nm]
    ci_v, _ = boot(dd, SEAS_DEV)
    RES[nm]["gain_vs_A1"] = round(float(dd.mean()), 6)
    RES[nm]["ci95_vs_A1"] = ci_v
    RES[nm]["promotable"] = bool(dd.mean() > 0.0002 and RES[nm]["ci97_5"][0] > 0)

# ---- sanity (b), (d)
cost_b = -RES["SAN_b_drop_qb_delta"]["gain"]; ci_b = [-RES["SAN_b_drop_qb_delta"]["ci95"][1], -RES["SAN_b_drop_qb_delta"]["ci95"][0]]
cost_d = -RES["SAN_d_drop_col0"]["gain"]; ci_d = [-RES["SAN_d_drop_col0"]["ci95"][1], -RES["SAN_d_drop_col0"]["ci95"][0]]
SAN["b_drop_qb_delta_cost"] = {"cost": round(cost_b, 6), "ci95": ci_b,
                               "pass": bool(0.0045 <= cost_b <= 0.0075 and ci_b[0] > 0),
                               "reference": "bt_nfl_anatomy_harness +0.00603 [+0.00183,+0.01018]; ledger row 3 +0.00689"}
SAN["d_drop_col0_cost"] = {"cost": round(cost_d, 6), "ci95": ci_d, "pass": bool(0.002 <= cost_d <= 0.005),
                           "reference": "bt_nfl_anatomy_harness +0.00361 [+0.00001,+0.00728]"}
print(f"(b) drop qb_delta cost {cost_b:+.5f} {ci_b} pass={SAN['b_drop_qb_delta_cost']['pass']}", flush=True)
print(f"(d) drop col0 cost {cost_d:+.5f} {ci_d} pass={SAN['d_drop_col0_cost']['pass']}", flush=True)
HARNESS_OK = SAN["b_drop_qb_delta_cost"]["pass"] and SAN["d_drop_col0_cost"]["pass"]

# ---- G2 replication 2002-2005 (train seasons < s from 1999)
REPL = {}
r0 = wf_full(s_lo=2002, s_hi=2005)
M25 = (SEAS >= 2002) & (SEAS <= 2005)
for nm in ("A1_replace_col0_untuned", "A2_add_z15", "A3_replace_col0_tuned"):
    r1 = wf_full(s_lo=2002, s_hi=2005, **ARMS[nm])
    dd = r0[2] - r1[2]
    ci_r, bci_r = boot(dd, SEAS[M25])
    REPL[nm] = {"a0_ll": round(r0[0], 6), "ll": round(r1[0], 6), "gain": round(float(dd.mean()), 6),
                "ci95": ci_r, "ci95_season_block": bci_r,
                "per_season_gain": {str(s): round(r0[1][s] - r1[1][s], 6) for s in r1[1]}}
    print(f"  G2 2002-05 {nm:<26} gain {dd.mean():+.5f} CI {ci_r} per {REPL[nm]['per_season_gain']}", flush=True)
REPL["note"] = "A3's params were tuned on standalone LL 2001-2015, so its 2002-05 row is not a clean replication"

# ---- verdict on the PRIMARY arm
A1 = RES["A1_replace_col0_untuned"]
bar_met = bool(A1["gain"] >= BAR and A1["ci95"][0] > 0)
g1 = bool(A1["gain_2013_15"] >= 0.0010 and A1["seasons_pos_2013_15"] >= 2)
g2 = bool(REPL["A1_replace_col0_untuned"]["gain"] >= 0)
pg = np.array(list(A1["per_season_gain"].values()))
tot = pg.sum()
share = float(pg.max() / tot) if tot > 0 else float("nan")
g3 = bool(tot > 0 and share <= 0.5)
if not HARNESS_OK:
    verdict = "HARNESS SANITY FAILED - screen void"
elif bar_met and g1 and g2 and g3:
    verdict = "DEV PASS (era-safe) - eligible for a TEST look by the lead engineer"
elif bar_met:
    verdict = "DEV pass, era-unsafe - not recommended for TEST"
else:
    verdict = "NULL - pre-registered bar not met"
GUARDS = {"bar_met": bar_met, "G1_2013_15": {"pass": g1, "gain": A1["gain_2013_15"],
                                             "seasons_pos": A1["seasons_pos_2013_15"], "ci95": A1["ci95_2013_15"]},
          "G2_2002_05": {"pass": g2, "gain": REPL["A1_replace_col0_untuned"]["gain"]},
          "G3_max_season_share": {"pass": g3, "max_share": None if np.isnan(share) else round(share, 4)}}
print("VERDICT:", verdict, GUARDS, flush=True)

# same guards evaluated (report-only) on the secondary arms; A1 remains the primary
GUARDS_SECONDARY = {}
for nm in ("A2_add_z15", "A3_replace_col0_tuned"):
    r_ = RES[nm]
    pg_ = np.array(list(r_["per_season_gain"].values())); t_ = pg_.sum()
    GUARDS_SECONDARY[nm] = {
        "bar_met": bool(r_["gain"] >= BAR and r_["ci95"][0] > 0),
        "G1": bool(r_["gain_2013_15"] >= 0.0010 and r_["seasons_pos_2013_15"] >= 2),
        "G2": bool(REPL[nm]["gain"] >= 0), "G2_gain": REPL[nm]["gain"], "G2_ci95": REPL[nm]["ci95"],
        "G3": bool(t_ > 0 and pg_.max() / t_ <= 0.5),
        "G3_max_share": round(float(pg_.max() / t_), 4) if t_ > 0 else None}
print("secondary-arm guards:", GUARDS_SECONDARY, flush=True)

# descriptive only (no fit): what the in-core margin adds relative to the W/L core
DIFF = MOV_LGT - WL_LGT
DESC = {"corr_devrows": {
    "mov_lgt__wl_lgt": round(float(np.corrcoef(MOV_LGT[DEVM], WL_LGT[DEVM])[0, 1]), 4),
    "mov_minus_wl__nonEPA": round(float(np.corrcoef(DIFF[DEVM], NONEPA[DEVM])[0, 1]), 4),
    "mov_minus_wl__epa_net_col2": round(float(np.corrcoef(DIFF[DEVM], X14[DEVM, 2])[0, 1]), 4),
    "mov_minus_wl__qb_delta_col1": round(float(np.corrcoef(DIFF[DEVM], X14[DEVM, 1])[0, 1]), 4),
    "nonEPA__epa_net_col2": round(float(np.corrcoef(NONEPA[DEVM], X14[DEVM, 2])[0, 1]), 4)},
    "sd_devrows": {"mov_lgt": round(float(MOV_LGT[DEVM].std()), 4), "wl_lgt": round(float(WL_LGT[DEVM].std()), 4),
                   "mov_minus_wl": round(float(DIFF[DEVM].std()), 4)}}
print("descriptive:", DESC, flush=True)

OUT_ALL = {
    "candidate": "nfl_movcore", "title": "margin-of-victory Elo inside the core (replaces W/L Elo col 0)",
    "protocol": {"dev_scored": "2006-2015 (n=2670, ties included)", "train": "seasons < s from 1999, ties dropped, "
                 "w=0.5^((s-1-season)/3), LogisticRegression(C=100)", "bootstrap": f"{BOOT_REPS} reps seed {BOOT_SEED}",
                 "bar": "gain >= +0.00150 and CI95 lo > 0 (A1 primary)", "max_season_read": MAX_SEASON_READ},
    "construction": {"k": K0, "hfa": HFA0, "regress": REG0, "c": C_NORM,
                     "mult": "ln(PD+1)*2.2/(0.001*dW+2.2)/c ; ties ln(2)/c"},
    "ledger_caveat": ("Ledger row 10 (SRS: a ridge-Massey margin added beside a frozen DEV-fit blend) was the "
                      "strongest DEV signal ever (DEV CV -0.00488). It lost on TEST: -0.00238, CI [-0.00594, "
                      "+0.00110]. Row 46 shows era-broken features stacking. MOV was deliberately skipped "
                      "because of that result."),
    "sanity": SAN, "harness_ok": HARNESS_OK, "leak": LEAK, "standalone_dev_ll_2001_2015": STANDALONE,
    "k_points_per_epa": {str(s): round(v, 4) for s, v in KS.items()},
    "arms": RES, "replication_2002_05": REPL, "guards": GUARDS, "verdict": verdict,
    "guards_secondary_report_only": GUARDS_SECONDARY, "descriptive": DESC,
    "quick3_rep_4000": OUT["results"], "runtime_s": round(time.time() - T0, 1),
}
json.dump(OUT_ALL, open("data/bt_nfl_nfl_movcore.json", "w", encoding="utf-8"), indent=1)
print(f"[{time.time()-T0:.0f}s] wrote data/bt_nfl_nfl_movcore.json", flush=True)
