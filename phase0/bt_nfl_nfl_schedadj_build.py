"""Breakthrough program (NFL) - screen nfl_nfl_schedadj, feature builder. DEV ONLY.

Schedule-adjusted process block, solved EXACTLY:
  * team EPA channels (net / pass / run): at every distinct gameday D, in
    chronological order, solve the joint 64x64 offence/defence system

      o_t (sum_j w_tj n_j + pn) + sum_j w_tj n_j a_u(j) = sum_j w_tj (S_j - n_j lg)
      a_u (sum_j v_uj n_j + pn) + sum_j v_uj n_j o_t(j) = sum_j v_uj (S_j - n_j lg)

    over the team-game observations dated strictly before D, with the exact
    per-team weights of the shipped EWMA (decay^(later own games with play data)
    * (1-season_decay)^(season boundaries)).  Feature for a game on D:
    (o_h - a_h) - (o_a - a_a), raw EPA/play units like the shipped columns.
  * QJ: opponent-adjusted QB increment on the shipped QbElo state, where every
    past dropback is corrected by the AS-OF-D joint pass-defence estimate of the
    opponent faced (retroactive adjustment, consistent with the team solve).
    QJ = incr(home starter) - incr(away starter).

MARKET-BLIND: data/nfl_games.csv is read for game_id / season / gameday ONLY.
No row with season >= 2016 is kept from any file (asserted).  The league anchors
LG_EPA / LGP / LGR are the pre-existing shipped constants (full 1999-2015 means,
shared by every arm; not changed here).

Also builds (for the screen's harness sanity):
  * the zero-cross identity features (must equal cache cols 2/12/13 < 1e-9),
  * the stale-opponent QJ identity (must equal FE['Q'] of bt_nfl_ideas_quick2),
  * the one-pass reference block (quick1 C1c ADJ_* + quick2 FE['Q']) by exec of
    those scripts' builder sections, exactly as bt_nfl_ideas_quick3.py does,
  * a future-perturbation leak test on 20 random cut dates.

Outputs: data/bt_nfl_nfl_schedadj_feats.npy (N x 4: net', pass', run', QJ),
         data/bt_nfl_nfl_schedadj_ref.npz, data/bt_nfl_nfl_schedadj_build.json
"""
from __future__ import annotations

import csv
import json
import sys
import time

import numpy as np

sys.path.insert(0, "phase0")
T0 = time.time()
TEST_ERA = 2016
PBP_FIX = {"JAC": "JAX", "WSH": "WAS", "STL": "LA", "SD": "LAC", "OAK": "LV"}

epaP = json.load(open("data/nfl_epa_rating.json", encoding="utf-8"))["params"]
DEC, PN, SDEC = epaP["decay"], epaP["prior_n"], epaP["season_decay"]
assert abs(DEC - 0.8814) < 1e-4 and abs(PN - 754) < 1 and abs(SDEC - 0.7131) < 1e-4
QSP = json.load(open("data/nfl_qb_replacement.json", encoding="utf-8"))["params"]
QDEC, QPRIOR, QSD = 0.8010595244395629, 650.2103644039794, 0.3915362183690712
assert (QSP["decay"], QSP["prior_db"], QSP["season_decay"]) == (QDEC, QPRIOR, QSD)

D_ = json.load(open("data/bt_nfl_ideas_dev.json", encoding="utf-8"))
G = D_["games"]
N = len(G)
X14 = np.load("data/bt_nfl_ideas_dev_X.npy")
assert X14.shape == (N, 14)
SEAS = np.array([g["season"] for g in G])
assert SEAS.max() < TEST_ERA and SEAS.min() == 1999
GIX = {g["gid"]: i for i, g in enumerate(G)}
TEAMS = sorted({g["home"] for g in G} | {g["away"] for g in G})
T = len(TEAMS)
TIX = {t: k for k, t in enumerate(TEAMS)}
HI = np.array([TIX[g["home"]] for g in G])
AI = np.array([TIX[g["away"]] for g in G])
MAX_SEASON_READ = {}

# ---------------------------------------------------------------- gamedays ---
gday = {}
with open("data/nfl_games.csv", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        if not r["season"] or int(r["season"]) >= TEST_ERA:
            continue
        gday[r["game_id"]] = r["gameday"]          # the only field used from this file
DAYS = np.array([np.datetime64(gday[g["gid"]]).astype(np.int64) for g in G])
assert np.all(np.diff(DAYS) >= 0), "spine not in gameday order"
MAX_SEASON_READ["nfl_games.csv"] = int(max(int(k[:4]) for k in gday))
UDAYS = np.unique(DAYS)
for d in UDAYS:  # one season per day, one game per team per day
    m = DAYS == d
    assert len(set(SEAS[m])) == 1
    tt = np.concatenate([HI[m], AI[m]])
    assert len(tt) == len(set(tt.tolist()))


# ------------------------------------------------------------------- plays ---
def read_plays(path, duel):
    gi, side, epa, isp = [], [], [], []
    ms = 0
    with open(path, encoding="utf-8") as fh:
        rd = csv.reader(fh)
        hdr = next(rd)
        ix = {c: k for k, c in enumerate(hdr)}
        for r in rd:
            gid = r[ix["game_id"]]
            s = int(gid[:4])
            if s >= TEST_ERA:
                continue
            ms = max(ms, s)
            i = GIX[gid]
            t = PBP_FIX.get(r[ix["posteam"]], r[ix["posteam"]])
            if t == G[i]["home"]:
                sd = 0
            else:
                assert t == G[i]["away"], (gid, t)
                sd = 1
            gi.append(i); side.append(sd); epa.append(float(r[ix["epa"]]))
            if duel:
                isp.append(bool(r[ix["passer_player_id"]] or r[ix["receiver_player_id"]]))
    MAX_SEASON_READ[path] = ms
    out = {"gi": np.array(gi), "side": np.array(side), "epa": np.array(epa)}
    if duel:
        out["isp"] = np.array(isp)
    return out


PL = read_plays("data/nfl_plays.csv", False)
DU = read_plays("data/nfl_duel_plays.csv", True)
assert max(MAX_SEASON_READ.values()) <= 2015
print(f"[{time.time()-T0:.0f}s] plays read: net {len(PL['epa'])}, duel {len(DU['epa'])}", flush=True)

# shipped anchors (full 1999-2015 league means; pre-existing shipped constants)
assert SEAS[PL["gi"]].min() >= 1999 and SEAS[DU["gi"]].min() >= 1999
LG_EPA = float(PL["epa"].sum() / len(PL["epa"]))
LGP = float(DU["epa"][DU["isp"]].sum() / DU["isp"].sum())
LGR = float(DU["epa"][~DU["isp"]].sum() / (~DU["isp"]).sum())
print(f"anchors LG_EPA {LG_EPA:.6f} LGP {LGP:.6f} LGR {LGR:.6f}", flush=True)


def aggregate(pl, du):
    """per (game, side) EPA sums and play counts; has_* = the shipped 'tm' truthiness."""
    k = pl["gi"] * 2 + pl["side"]
    Snet = np.bincount(k, pl["epa"], 2 * N).reshape(N, 2)
    Nnet = np.bincount(k, None, 2 * N).reshape(N, 2)
    kd = du["gi"] * 2 + du["side"]
    p = du["isp"]
    Sp = np.bincount(kd[p], du["epa"][p], 2 * N).reshape(N, 2)
    Np_ = np.bincount(kd[p], None, 2 * N).reshape(N, 2)
    Sr = np.bincount(kd[~p], du["epa"][~p], 2 * N).reshape(N, 2)
    Nr = np.bincount(kd[~p], None, 2 * N).reshape(N, 2)
    has_net = np.bincount(pl["gi"], None, N) > 0
    has_duel = np.bincount(du["gi"], None, N) > 0
    return {"net": (Snet, Nnet, has_net), "pass": (Sp, Np_, has_duel), "run": (Sr, Nr, has_duel)}


# ----------------------------------------------------------------- QB lines ---
import nfl_qb_elo as QE  # noqa: E402

_qbw = QE.load_qb_weeks()
qbw = {k: v for k, v in _qbw.items() if k[1] < TEST_ERA}   # drop every >= 2016 line at load
del _qbw
MAX_SEASON_READ["qb_weeks"] = int(max(k[1] for k in qbw))
QL = np.full((N, 2), np.nan)                     # dropbacks of (home, away) starter's weekly line
for i, g in enumerate(G):
    for sd, qid in enumerate((g["home_qb"], g["away_qb"])):
        line = qbw.get((qid, g["season"], g["week"]))
        if line is not None:
            QL[i, sd] = line[1]
QID = [(g["home_qb"], g["away_qb"]) for g in G]
assert max(MAX_SEASON_READ.values()) <= 2015


# ------------------------------------------------------------------ engine ---
class Chan:
    """observation store of one EPA channel; weights are the exact shipped EWMA weights."""

    def __init__(self, lg, prior):
        self.lg, self.prior = lg, prior
        M = 2 * N
        self.t = np.zeros(M, np.int64); self.u = np.zeros(M, np.int64)
        self.S = np.zeros(M); self.n = np.zeros(M); self.day = np.zeros(M, np.int64)
        self.ko = np.zeros(M, np.int64); self.kd = np.zeros(M, np.int64); self.nb = np.zeros(M, np.int64)
        self.co = np.zeros(T, np.int64); self.cd = np.zeros(T, np.int64)
        self.m = 0

    def add(self, t, u, S, n, day, nb):
        j = self.m
        self.t[j], self.u[j], self.S[j], self.n[j], self.day[j], self.nb[j] = t, u, S, n, day, nb
        self.ko[j] = self.co[t]; self.co[t] += 1
        self.kd[j] = self.cd[u]; self.cd[u] += 1
        self.m += 1

    def solve(self, D, nb, cross):
        m = self.m
        if m:
            assert self.day[:m].max() < D, "observation dated on/after the solve date"
        t, u, S, n = self.t[:m], self.u[:m], self.S[:m], self.n[:m]
        sf = (1.0 - SDEC) ** (nb - self.nb[:m])
        w = DEC ** (self.co[t] - 1 - self.ko[:m]) * sf     # offence weights of t
        v = DEC ** (self.cd[u] - 1 - self.kd[:m]) * sf     # defence weights of u
        r = S - n * self.lg
        A = np.zeros((2 * T, 2 * T))
        A[np.arange(T), np.arange(T)] = np.bincount(t, w * n, T) + self.prior
        A[np.arange(T, 2 * T), np.arange(T, 2 * T)] = np.bincount(u, v * n, T) + self.prior
        if cross:
            A[:T, T:] = np.bincount(t * T + u, w * n, T * T).reshape(T, T)   # row o_t, col a_u
            A[T:, :T] = np.bincount(u * T + t, v * n, T * T).reshape(T, T)   # row a_u, col o_t
        b = np.concatenate([np.bincount(t, w * r, T), np.bincount(u, v * r, T)])
        x = np.linalg.solve(A, b)
        return x[:T], x[T:]


class QBStore:
    def __init__(self):
        self.L = {}      # qid -> [opp list, db list, stale list, k list, nb list, day list]
        self.cnt = {}

    def add(self, qid, opp, db, stale, day, nb):
        L = self.L.setdefault(qid, [[], [], [], [], [], []])
        k = self.cnt.get(qid, 0)
        for lst, v in zip(L, (opp, db, stale, k, nb, day)):
            lst.append(v)
        self.cnt[qid] = k + 1

    def incr(self, qid, D, nb, a_pass=None):
        if qid not in self.L:
            return 0.0
        opp, db, stale, k, nbj, day = (np.asarray(x) for x in self.L[qid])
        assert day.max() < D
        w = QDEC ** (self.cnt[qid] - 1 - k) * (1.0 - QSD) ** (nb - nbj)
        adj = a_pass[opp] if a_pass is not None else stale
        return -float(np.sum(w * db * adj)) / (float(np.sum(w * db)) + QPRIOR)


def build(agg, ql, cross=True, qmode="retro"):
    """walk the spine by gameday. qmode 'retro' = as-of-D joint pass defence of past
    opponents (the candidate); 'stale' = opponent's pass-defence deviation frozen at
    the time of each past start (the quick2 identity)."""
    lgs = {"net": LG_EPA, "pass": LGP, "run": LGR}
    pri = {"net": PN, "pass": PN / 2, "run": PN / 2}
    ch = {c: Chan(lgs[c], pri[c]) for c in ("net", "pass", "run")}
    qb = QBStore()
    F = np.zeros((N, 4))
    nb, prev, solved_at = 0, None, {}
    for D in UDAYS:
        idx = np.where(DAYS == D)[0]
        s = SEAS[idx[0]]
        if prev is not None and s != prev:
            nb += 1
        prev = s
        sol = {c: ch[c].solve(D, nb, cross) for c in ch}
        solved_at["pass"] = D
        for k, c in enumerate(("net", "pass", "run")):
            o, a = sol[c]
            F[idx, k] = (o[HI[idx]] - a[HI[idx]]) - (o[AI[idx]] - a[AI[idx]])
        a_pass = sol["pass"][1]
        assert solved_at["pass"] == D           # QJ at D uses a^pass solved at D
        for i in idx:
            inc = [qb.incr(q, D, nb, a_pass if qmode == "retro" else None) for q in QID[i]]
            F[i, 3] = inc[0] - inc[1]
        # ---- post-game updates (enter only systems of later dates)
        for i in idx:
            h, a = HI[i], AI[i]
            for c in ("net", "pass", "run"):
                S, n_, has = agg[c]
                if has[i]:
                    ch[c].add(h, a, S[i, 0], n_[i, 0], D, nb)
                    ch[c].add(a, h, S[i, 1], n_[i, 1], D, nb)
            for sd, (qid, opp) in enumerate(((QID[i][0], a), (QID[i][1], h))):
                if not np.isnan(ql[i, sd]):
                    qb.add(qid, opp, ql[i, sd], a_pass[opp], D, nb)   # a_pass[opp] = as-of-D value
    return F


AGG = aggregate(PL, DU)
t1 = time.time()
FJ = build(AGG, QL, cross=True, qmode="retro")
print(f"[{time.time()-T0:.0f}s] joint build done in {time.time()-t1:.1f}s", flush=True)
FZ = build(AGG, QL, cross=False, qmode="stale")
print(f"[{time.time()-T0:.0f}s] zero-cross identity build done", flush=True)

# ---- identity (c1): zero-cross solver == shipped cols 2/12/13 on every 1999-2015 row
ident = {nm: float(np.abs(FZ[:, k] - X14[:, col]).max())
         for k, (nm, col) in enumerate((("epa_net", 2), ("pass_net", 12), ("run_net", 13)))}
print("identity max|d| vs cache cols:", ident, flush=True)

np.save("data/bt_nfl_nfl_schedadj_feats.npy", FJ)

# ------------------------------------------------ one-pass reference (exec) ---
t1 = time.time()
_src2 = open("phase0/bt_nfl_ideas_quick2.py", encoding="utf-8").read()
_pre2, _rest2 = _src2.split("FE = {}\n", 1)
_qsec = _rest2.split("# ---- Q opponent-adjusted QB increment")[1].split('FE["Q"] = QINC')[0]
_ns2 = {"__name__": "quick2_ref"}
exec(compile(_pre2 + "FE = {}\n# ---- Q" + _qsec + 'FE["Q"] = QINC\n', "quick2<prefix+Q>", "exec"), _ns2)  # noqa: S102
FEQ = _ns2["FE"]["Q"]
_src1 = open("phase0/bt_nfl_ideas_quick.py", encoding="utf-8").read()
_ns1 = {"__name__": "quick1_ref"}
exec(compile(_src1.split("# ============================================= C2 playoff leverage")[0],  # noqa: S102
             "quick1<C1 builders>", "exec"), _ns1)
print(f"[{time.time()-T0:.0f}s] reference builders exec'd in {time.time()-t1:.0f}s", flush=True)
ident["QJ_stale_vs_quick2_FE_Q"] = float(np.abs(FZ[:, 3] - FEQ).max())
print("identity QJ(stale) vs quick2 FE['Q'] max|d|:", ident["QJ_stale_vs_quick2_FE_Q"], flush=True)
np.savez("data/bt_nfl_nfl_schedadj_ref.npz", FZ=FZ, FEQ=FEQ, ADJ_ALL=_ns1["adj_all"],
         ADJ_PASS=_ns1["adj_pass"], ADJ_RUN=_ns1["adj_run"], RAW_ALL=_ns1["raw_all"])

# ------------------------------------------------ future-perturbation test ---
rng = np.random.default_rng(20260924)
cuts = np.sort(rng.choice(UDAYS[10:], size=20, replace=False))
pert = []
for cut in cuts:
    pl2 = dict(PL); du2 = dict(DU)
    fm = DAYS[PL["gi"]] >= cut
    e = PL["epa"].copy(); ii = np.where(fm)[0]; e[ii] = e[rng.permutation(ii)]; pl2["epa"] = e
    fm = DAYS[DU["gi"]] >= cut
    e = DU["epa"].copy(); ii = np.where(fm)[0]; e[ii] = e[rng.permutation(ii)]; du2["epa"] = e
    ql2 = QL.copy(); gm = DAYS >= cut
    ql2[gm] = np.where(np.isnan(ql2[gm]), np.nan, rng.uniform(0, 80, size=ql2[gm].shape))
    Fp = build(aggregate(pl2, du2), ql2, cross=True, qmode="retro")
    before = DAYS < cut
    same = bool(np.array_equal(Fp[before], FJ[before]))
    changed_after = bool(not np.array_equal(Fp[~before], FJ[~before]))
    pert.append({"cut": str(np.datetime64(int(cut), "D")), "n_before": int(before.sum()),
                 "bit_identical_before": same, "changed_on_or_after": changed_after})
    assert same, f"future leak at cut {cut}"
print(f"[{time.time()-T0:.0f}s] future-perturbation: {sum(p['bit_identical_before'] for p in pert)}/20 "
      f"bit-identical; {sum(p['changed_on_or_after'] for p in pert)}/20 changed after cut", flush=True)

json.dump({"n": N, "teams": T, "gamedays": int(len(UDAYS)),
           "anchors_shipped_full_1999_2015": {"LG_EPA": LG_EPA, "LGP": LGP, "LGR": LGR},
           "epa_params": epaP, "qb_params": QSP, "max_season_read": MAX_SEASON_READ,
           "identity_max_abs_diff": ident, "perturbation": pert,
           "feat_cols": ["net_joint", "pass_joint", "run_joint", "QJ_retro"],
           "secs": round(time.time() - T0, 1)},
          open("data/bt_nfl_nfl_schedadj_build.json", "w"), indent=1)
print("wrote data/bt_nfl_nfl_schedadj_feats.npy / _ref.npz / _build.json", flush=True)
