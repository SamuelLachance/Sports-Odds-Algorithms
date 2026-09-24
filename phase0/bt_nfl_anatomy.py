"""Breakthrough program 2026-09-24 -- NFL gap anatomy, step 2 (slice the gap).

DEV ONLY (2006-2015). Reads data/bt_nfl_anatomy_dev.csv (built by
bt_nfl_anatomy_build.py: the shipped 14-feature blend, shipped walk-forward
protocol, DEV seasons only) and the closing moneylines of nfl_games.csv for
seasons <= 2015 ONLY (rows with season >= 2016 are skipped at load time and a
hard assert guards every mask).

MARKET-BLIND: the de-vigged close is an EVALUATION benchmark here. It is used to
measure per-game (model LL - market LL) and the market-minus-model logit
disagreement r; nothing fitted here is a predictor or is served. The one
outcome-only diagnostic (offset logistic y ~ logit(p_model) + v, LOSO cross-fit)
never sees odds.

Output: data/bt_nfl_anatomy.json
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict

import numpy as np

EPS = 1e-12
RNG = np.random.default_rng(20260924)
NB = 10000

# ---------------------------------------------------------------- load (DEV only)
rows = list(csv.DictReader(open("data/bt_nfl_anatomy_dev.csv", encoding="utf-8")))
assert max(int(r["season"]) for r in rows) <= 2015
raw_all = []
for r in csv.DictReader(open("data/nfl_games.csv", encoding="utf-8")):
    if r["home_score"] == "" or int(r["season"]) > 2015:
        continue                                   # TEST seasons never loaded
    raw_all.append(r)
raw = {r["game_id"]: r for r in raw_all}
assert max(int(r["season"]) for r in raw_all) <= 2015


def imp(ml):
    v = float(ml)
    return (-v / (-v + 100.0)) if v < 0 else (100.0 / (v + 100.0))


# ---------------------------------------------------------------- team-game history
DIV = {}
for d, ts in {"AE": "BUF MIA NE NYJ", "AN": "BAL CIN CLE PIT", "AS": "HOU IND JAX TEN",
              "AW": "DEN KC OAK SD", "NE": "DAL NYG PHI WAS", "NN": "CHI DET GB MIN",
              "NS": "ATL CAR NO TB", "NW": "ARI SEA SF STL"}.items():
    for t in ts.split():
        DIV[t] = d

raw_all.sort(key=lambda r: (r["gameday"], r["game_id"]))
qb_starts = defaultdict(int)                 # career starts (since 1999)
team_hist = defaultdict(list)                # team -> list of dicts (past games)
last_coach_season = {}                       # team -> (season, coach) of last game
coach_week1 = {}                             # (team, season) -> coach in first game
prev_season_coach = {}                       # (team, season) -> coach in last game of season-1
standing = defaultdict(lambda: [0.0, 0.0, 0])  # (team, season) -> [wins, losses, played]
pre = {}                                     # gid -> {team: state}
for r in raw_all:
    s, wk, gt = int(r["season"]), int(r["week"]), r["game_type"]
    gid = r["game_id"]
    st = {}
    for side, team, opp, qb, coach in (("h", r["home_team"], r["away_team"], r["home_qb_id"], r["home_coach"]),
                                       ("a", r["away_team"], r["home_team"], r["away_qb_id"], r["away_coach"])):
        H = team_hist[team]
        last16 = H[-16:]
        cnt = defaultdict(int)
        for g in last16:
            cnt[g["qb"]] += 1
        estab = max(cnt, key=cnt.get) if cnt else None
        estab_n = cnt.get(estab, 0) if estab else 0
        prev_qb = H[-1]["qb"] if H else None
        this_season = [g for g in H if g["season"] == s]
        prev_season = [g for g in H if g["season"] == s - 1]
        pd_s = [g["margin"] for g in this_season]
        # previous-season primary QB (most starts)
        pcnt = defaultdict(int)
        for g in prev_season:
            pcnt[g["qb"]] += 1
        prev_primary = max(pcnt, key=pcnt.get) if pcnt else None
        W, L, P = standing[(team, s)]
        rem = max(0, 16 - P) if gt == "REG" else 0
        lc = last_coach_season.get(team)
        if (team, s) not in prev_season_coach:
            prev_season_coach[(team, s)] = lc[1] if (lc is not None and lc[0] == s - 1) else None
        psc = prev_season_coach[(team, s)]
        st[side] = {
            "team": team, "qb": qb, "qb_starts": qb_starts[qb],
            "estab": estab, "estab_n": estab_n, "is_estab": int(qb == estab),
            "qb_changed": int(prev_qb is not None and qb != prev_qb),
            "returning": int(prev_qb is not None and qb != prev_qb and qb == estab),
            "new_to_team_qb": int(cnt.get(qb, 0) == 0),
            "offseason_qb_change": int(prev_primary is not None and qb != prev_primary),
            "prev_margin": H[-1]["margin"] if H and H[-1]["season"] == s else None,
            "pd_pg": (sum(pd_s) / len(pd_s)) if pd_s else None,
            "n_played": len(this_season),
            "W": W, "L": L, "P": P, "max_wins": W + rem,
            "new_coach": int(psc is not None and psc != coach_week1.get((team, s), coach)),
            "interim": int(coach_week1.get((team, s), coach) != coach),
            "coach": coach,
        }
    # division clinch (conservative, ignores tiebreaks) + elimination proxy
    for side in ("h", "a"):
        t = st[side]["team"]
        rivals = [x for x in DIV if DIV[x] == DIV[t] and x != t]
        best_rival_max = max(standing[(x, s)][0] + max(0, 16 - standing[(x, s)][2]) for x in rivals)
        st[side]["clinched_div"] = int(gt == "REG" and st[side]["W"] > best_rival_max)
        st[side]["dead"] = int(gt == "REG" and st[side]["max_wins"] <= 8)
    pre[gid] = st
    # ---- post-game updates
    hs, as_ = int(r["home_score"]), int(r["away_score"])
    for team, qb, coach, marg in ((r["home_team"], r["home_qb_id"], r["home_coach"], hs - as_),
                                  (r["away_team"], r["away_qb_id"], r["away_coach"], as_ - hs)):
        team_hist[team].append({"season": s, "week": wk, "qb": qb, "margin": marg})
        qb_starts[qb] += 1
        coach_week1.setdefault((team, s), coach)
        last_coach_season[team] = (s, coach)
        if gt == "REG":
            sd = standing[(team, s)]
            sd[2] += 1
            if marg > 0:
                sd[0] += 1
            elif marg < 0:
                sd[1] += 1
            else:
                sd[0] += 0.5; sd[1] += 0.5

# ---------------------------------------------------------------- injury reports 2009-2015
pl_pfr = {}
for p in csv.DictReader(open("data/nfl_players.csv", encoding="utf-8")):
    if p["gsis_id"] and p["pfr_id"]:
        pl_pfr[p["gsis_id"]] = p["pfr_id"]
inj = defaultdict(list)                       # (team, season, week) -> [(gsis,pos,status,practice)]
for yr in range(2009, 2016):
    for r in csv.DictReader(open(f"data/inj_{yr}.csv", encoding="utf-8")):
        inj[(r["team"], int(r["season"]), int(r["week"]))].append(
            (r["gsis_id"], r["position"], r["report_status"].strip(), r["practice_status"].strip()))

# trailing snap share (EWMA over team games, 2012+) for injured-player importance
snap_share = {}                                # (gid, team) -> {pfr: share at kickoff}
snap_rows = defaultdict(lambda: defaultdict(dict))
for yr in range(2012, 2016):
    for r in csv.DictReader(open(f"data/snap_{yr}.csv", encoding="utf-8")):
        pct = max(float(r["offense_pct"] or 0), float(r["defense_pct"] or 0))
        snap_rows[r["game_id"]][r["team"]][r["pfr_player_id"]] = pct
share_state = defaultdict(lambda: [0.0, 0.0])   # (team, pfr) -> ewma num/den
team_players = defaultdict(set)
DEC = 0.8
for r in raw_all:
    if int(r["season"]) < 2012:
        continue
    gid = r["game_id"]
    for team in (r["home_team"], r["away_team"]):
        snap_share[(gid, team)] = {pid: (share_state[(team, pid)][0] / share_state[(team, pid)][1])
                                   for pid in team_players[team] if share_state[(team, pid)][1] > 0}
    for team in (r["home_team"], r["away_team"]):
        tbl = snap_rows.get(gid, {}).get(team, {})
        if not tbl:
            continue
        for pid in team_players[team] | set(tbl):
            stt = share_state[(team, pid)]
            stt[0] = DEC * stt[0] + tbl.get(pid, 0.0)
            stt[1] = DEC * stt[1] + 1.0
            team_players[team].add(pid)

OL = {"T", "G", "C", "OL", "OT", "OG"}
SKILL = {"WR", "TE", "RB", "FB"}
DEF = {"DE", "DT", "NT", "DL", "LB", "ILB", "OLB", "MLB", "CB", "S", "SS", "FS", "DB"}


def inj_feats(team, s, wk, gid, starter_qb):
    L = inj.get((team, s, wk))
    if L is None:
        return None
    out = dict(n_out=0, n_doubt=0, n_quest=0, n_od_qb=0, n_od_ol=0, n_od_skill=0, n_od_def=0,
               starter_qb_listed_q=0, w_od=None, w_q=None, n_dnp_active=0)
    sh = snap_share.get((gid, team))
    w_od = w_q = 0.0
    for gsis, pos, stat, prac in L:
        if stat == "Out":
            out["n_out"] += 1
        elif stat == "Doubtful":
            out["n_doubt"] += 1
        elif stat == "Questionable":
            out["n_quest"] += 1
        od = stat in ("Out", "Doubtful")
        if od:
            if pos == "QB":
                out["n_od_qb"] += 1
            elif pos in OL:
                out["n_od_ol"] += 1
            elif pos in SKILL:
                out["n_od_skill"] += 1
            elif pos in DEF:
                out["n_od_def"] += 1
        if gsis == starter_qb and stat in ("Questionable", "Doubtful"):
            out["starter_qb_listed_q"] = 1
        if stat in ("Questionable", "Probable") and prac.startswith("Did Not"):
            out["n_dnp_active"] += 1
        if sh is not None:
            w = sh.get(pl_pfr.get(gsis, ""), 0.0)
            if od:
                w_od += w
            elif stat == "Questionable":
                w_q += w
    if sh is not None:
        out["w_od"], out["w_q"] = w_od, w_q
    return out


# ---------------------------------------------------------------- assemble DEV frame
G = []
for r in rows:
    if r["p_model"] == "":
        continue
    s = int(r["season"])
    assert 2006 <= s <= 2015
    y = float(r["y"])
    if y == 0.5:
        continue
    rr = raw[r["gid"]]
    if not (rr["home_moneyline"] and rr["away_moneyline"]):
        continue
    ph, pa = imp(rr["home_moneyline"]), imp(rr["away_moneyline"])
    pm = ph / (ph + pa)
    p = float(r["p_model"])
    st = pre[r["gid"]]
    wk = int(r["week"])
    ih = inj_feats(rr["home_team"], s, wk, r["gid"], rr["home_qb_id"]) if s >= 2009 else None
    ia = inj_feats(rr["away_team"], s, wk, r["gid"], rr["away_qb_id"]) if s >= 2009 else None
    G.append(dict(gid=r["gid"], s=s, wk=wk, gt=rr["game_type"], y=y, p=p, pm=pm,
                  pe=float(r["p_elo"]), h=st["h"], a=st["a"], ih=ih, ia=ia,
                  hrest=int(r["hrest"]), arest=int(r["arest"]), weekday=rr["weekday"],
                  gametime=rr["gametime"], div=int(rr["div_game"] or 0), roof=rr["roof"],
                  temp=float(rr["temp"]) if rr["temp"] else None,
                  wind=float(rr["wind"]) if rr["wind"] else None,
                  neutral=int(rr["location"] == "Neutral"),
                  X={c: float(r[c]) for c in ("lgt", "qd", "epa", "early", "thfa", "rest", "luck",
                                              "ol", "de", "sk", "rq", "pass", "run")}))
n = len(G)
y = np.array([g["y"] for g in G]); p = np.array([g["p"] for g in G]); pm = np.array([g["pm"] for g in G])
pe = np.array([g["pe"] for g in G])


def llv(yy, pp):
    pp = np.clip(pp, EPS, 1 - EPS)
    return -(yy * np.log(pp) + (1 - yy) * np.log(1 - pp))


def logit(q):
    q = np.clip(q, 1e-9, 1 - 1e-9)
    return np.log(q / (1 - q))


llm, llk, lle = llv(y, p), llv(y, pm), llv(y, pe)
d = llm - llk                                   # >0 = market better
r_dis = logit(pm) - logit(p)                    # market minus model, home-oriented
TOT = d.sum()


def boot_ci(x):
    if len(x) < 5:
        return [None, None]
    idx = RNG.integers(0, len(x), size=(NB, len(x)))
    bs = x[idx].mean(axis=1)
    return [round(float(np.percentile(bs, 2.5)), 5), round(float(np.percentile(bs, 97.5)), 5)]


def summarize(mask, label, orient=None):
    """orient: +1/-1 per game to express r and bias from the slice team's side."""
    m = np.asarray(mask, bool)
    k = int(m.sum())
    if k == 0:
        return {"slice": label, "n": 0}
    o = np.ones(n) if orient is None else np.asarray(orient, float)
    ci = boot_ci(d[m])
    out = {"slice": label, "n": k, "model_ll": round(float(llm[m].mean()), 5),
           "market_ll": round(float(llk[m].mean()), 5), "gap": round(float(d[m].mean()), 5),
           "gap_ci": ci,
           "sig": ("market SIG better" if ci[0] is not None and ci[0] > 0 else
                   "model SIG better" if ci[1] is not None and ci[1] < 0 else "n.s."),
           "share_of_gap": round(float(d[m].sum() / TOT), 4), "n_share": round(k / n, 4),
           "mean_abs_disagree_prob": round(float(np.abs(pm[m] - p[m]).mean()), 4),
           "mean_r_logit": round(float((o[m] * r_dis[m]).mean()), 4),
           "bias_model": round(float((o[m] * (y[m] - p[m])).mean()), 4),
           "bias_market": round(float((o[m] * (y[m] - pm[m])).mean()), 4)}
    return out


RES = {"protocol": {"league": "NFL", "dev_scored": [2006, 2015], "test_touched": False,
                    "market_use": "evaluation / diagnostic only",
                    "model": "shipped 14-feature blend (nfl_season_serve.py feature build), shipped "
                             "walk-forward protocol (train seasons < s, HL 3, C=100); col 7 v7 = 0 on DEV "
                             "(participation 2016+); absence/rq cols live from 2013",
                    "ties_dropped": True, "ml_required": True}}
RES["headline"] = {"n": n, "model_ll": round(float(llm.mean()), 5), "market_ll": round(float(llk.mean()), 5),
                   "elo_only_ll": round(float(lle.mean()), 5),
                   "gap": round(float(d.mean()), 5), "gap_ci": boot_ci(d),
                   "model_minus_elo_only": round(float((lle - llm).mean()), 5),
                   "corr_logit": round(float(np.corrcoef(logit(p), logit(pm))[0, 1]), 4),
                   "mean_abs_disagree": round(float(np.abs(p - pm).mean()), 4),
                   "sd_r": round(float(r_dis.std()), 4)}
print("HEADLINE", RES["headline"])

S = {}
W = np.array([g["wk"] for g in G]); SEA = np.array([g["s"] for g in G])
REG = np.array([g["gt"] == "REG" for g in G])
S["era"] = [summarize(SEA <= 2012, "2006-2012 (no player cols live)"),
            summarize(SEA >= 2013, "2013-2015 (absence+rq live)")]
S["season"] = [summarize(SEA == s, str(s)) for s in range(2006, 2016)]
S["phase"] = [summarize(REG & (W == 1), "week 1"), summarize(REG & (W >= 2) & (W <= 4), "weeks 2-4"),
              summarize(REG & (W >= 5) & (W <= 8), "weeks 5-8"), summarize(REG & (W >= 9) & (W <= 12), "weeks 9-12"),
              summarize(REG & (W >= 13) & (W <= 16), "weeks 13-16"), summarize(REG & (W == 17), "week 17"),
              summarize(~REG, "playoffs")]
fav = np.maximum(pm, 1 - pm)
S["market_fav_size"] = [summarize((fav >= a) & (fav < b), f"market fav p [{a},{b})")
                        for a, b in ((0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01))]
S["market_fav_side"] = [summarize(pm >= 0.5, "home market favourite"), summarize(pm < 0.5, "away market favourite")]
S["disagreement_direction"] = [
    summarize(r_dis <= -0.4, "market much lower on home (r<=-0.4)"),
    summarize((r_dis > -0.4) & (r_dis <= -0.15), "market lower on home (-0.4,-0.15]"),
    summarize(np.abs(r_dis) < 0.15, "|r|<0.15"),
    summarize((r_dis >= 0.15) & (r_dis < 0.4), "market higher on home [0.15,0.4)"),
    summarize(r_dis >= 0.4, "market much higher on home (r>=0.4)")]
# model-favourite relative to market: is the model over-confident?
conf_more = np.abs(logit(p)) > np.abs(logit(pm))
S["model_vs_market_confidence"] = [
    summarize(conf_more & (np.sign(logit(p)) == np.sign(logit(pm))), "same fav, model MORE confident"),
    summarize(~conf_more & (np.sign(logit(p)) == np.sign(logit(pm))), "same fav, model LESS confident"),
    summarize(np.sign(logit(p)) != np.sign(logit(pm)), "different favourite")]

hr = np.array([g["hrest"] for g in G]); ar = np.array([g["arest"] for g in G])
wd = np.array([g["weekday"] for g in G])
S["rest"] = [summarize((hr <= 5) | (ar <= 5), "short week (either rest<=5)"),
             summarize(wd == "Thursday", "Thursday"), summarize(wd == "Monday", "Monday"),
             summarize((hr >= 13) | (ar >= 13), "off a bye (either)"),
             summarize(np.abs(hr - ar) >= 4, "|rest diff|>=4"),
             summarize((hr < 13) & (ar < 13) & (hr > 5) & (ar > 5), "normal rest both")]

# QB situations (team-oriented slices pooled over both sides)
def team_slice(fn, label, cond=None):
    """games where the HOME side satisfies fn -> orient +1, AWAY side -> orient -1.
    Games where both do are dropped (unorientable) and counted separately."""
    hs = np.array([bool(fn(g["h"], g, "h")) for g in G]); as_ = np.array([bool(fn(g["a"], g, "a")) for g in G])
    if cond is not None:
        hs &= cond; as_ &= cond
    one = hs ^ as_
    orient = np.where(hs, 1.0, -1.0)
    res = summarize(one, label + " (one side)", orient)
    res["n_both_sides"] = int((hs & as_).sum())
    return res


S["qb"] = [
    team_slice(lambda t, g, sd: t["is_estab"] == 0 and t["qb_starts"] < 8, "non-established QB with <8 career starts"),
    team_slice(lambda t, g, sd: t["is_estab"] == 0 and t["qb_starts"] >= 8, "non-established QB with >=8 career starts"),
    team_slice(lambda t, g, sd: t["qb_changed"] == 1 and t["returning"] == 0, "QB differs from last game, not returning starter"),
    team_slice(lambda t, g, sd: t["returning"] == 1, "established starter RETURNS after missing last game"),
    team_slice(lambda t, g, sd: t["qb_starts"] == 0, "QB career debut start"),
    team_slice(lambda t, g, sd: t["new_to_team_qb"] == 1 and t["qb_starts"] >= 16, "veteran QB new to team (no start for team in last 16)"),
]
S["qb"].append(summarize(np.array([g["h"]["is_estab"] == 1 and g["a"]["is_estab"] == 1 and g["h"]["qb_changed"] == 0
                                   and g["a"]["qb_changed"] == 0 for g in G]), "both teams: established QB, same as last game"))
S["qb_offseason"] = [
    team_slice(lambda t, g, sd: t["offseason_qb_change"] == 1, "weeks 1-4: starter != last season's primary QB",
               cond=REG & (W <= 4)),
    summarize(REG & (W <= 4) & np.array([g["h"]["offseason_qb_change"] == 0 and g["a"]["offseason_qb_change"] == 0 for g in G]),
              "weeks 1-4: both starters = last season's primary QB")]
S["coach"] = [
    team_slice(lambda t, g, sd: t["new_coach"] == 1, "new head coach this season, weeks 1-4", cond=REG & (W <= 4)),
    team_slice(lambda t, g, sd: t["new_coach"] == 1, "new head coach this season, weeks 5-17", cond=REG & (W >= 5)),
    team_slice(lambda t, g, sd: t["interim"] == 1, "interim/mid-season coach change"),
]
S["standings_late"] = [
    team_slice(lambda t, g, sd: t["dead"] == 1, "weeks 13-17: one team eliminated-proxy (max wins<=8)",
               cond=REG & (W >= 13)),
    summarize(REG & (W >= 13) & np.array([g["h"]["dead"] == 1 and g["a"]["dead"] == 1 for g in G]),
              "weeks 13-17: both eliminated-proxy"),
    team_slice(lambda t, g, sd: t["clinched_div"] == 1, "weeks 13-16: one team clinched division",
               cond=REG & (W >= 13) & (W <= 16)),
    team_slice(lambda t, g, sd: t["clinched_div"] == 1, "week 17: one team clinched division", cond=REG & (W == 17)),
    summarize(REG & (W == 17) & np.array([g["h"]["clinched_div"] == 0 and g["a"]["clinched_div"] == 0
                                          and g["h"]["dead"] == 0 and g["a"]["dead"] == 0 for g in G]),
              "week 17: both alive, neither clinched"),
]
# injuries 2009+
has_inj = np.array([g["ih"] is not None and g["ia"] is not None for g in G])
def injd(key):
    return np.array([(g["ih"][key] - g["ia"][key]) if (g["ih"] is not None and g["ia"] is not None
                                                        and g["ih"][key] is not None and g["ia"][key] is not None)
                     else np.nan for g in G], float)
od_diff = injd("n_out") + injd("n_doubt")
S["injury_2009_2015"] = [
    summarize(has_inj & (np.abs(od_diff) >= 4), "|Out+Doubtful count diff|>=4"),
    summarize(has_inj & (np.abs(od_diff) >= 2) & (np.abs(od_diff) < 4), "|O+D diff| 2-3"),
    summarize(has_inj & (np.abs(od_diff) < 2), "|O+D diff| <2"),
    team_slice(lambda t, g, sd: (g["i" + sd] or {}).get("n_od_qb", 0) >= 1, "a QB listed Out/Doubtful (team side)", cond=has_inj),
    team_slice(lambda t, g, sd: (g["i" + sd] or {}).get("starter_qb_listed_q", 0) == 1, "STARTING QB listed Q/D (plays hurt)", cond=has_inj),
    team_slice(lambda t, g, sd: (g["i" + sd] or {}).get("n_od_ol", 0) >= 2, ">=2 OL Out/Doubtful", cond=has_inj),
    team_slice(lambda t, g, sd: (g["i" + sd] or {}).get("n_od_def", 0) >= 3, ">=3 defenders Out/Doubtful", cond=has_inj),
    team_slice(lambda t, g, sd: (g["i" + sd] or {}).get("n_od_skill", 0) >= 2, ">=2 skill players Out/Doubtful", cond=has_inj),
]
wod = injd("w_od")
okw = ~np.isnan(wod)
S["injury_snapweighted_2012_2015"] = [
    summarize(okw & (np.abs(wod) >= 1.5), "|snap-weighted O+D diff|>=1.5 starters"),
    summarize(okw & (np.abs(wod) >= 0.75) & (np.abs(wod) < 1.5), "|sw O+D diff| 0.75-1.5"),
    summarize(okw & (np.abs(wod) < 0.75), "|sw O+D diff| <0.75"),
]
temp = np.array([g["temp"] if g["temp"] is not None else np.nan for g in G])
wind = np.array([g["wind"] if g["wind"] is not None else np.nan for g in G])
roof = np.array([g["roof"] for g in G])
S["weather_venue"] = [summarize(wind >= 15, "wind>=15mph"), summarize(temp <= 32, "temp<=32F"),
                      summarize(np.isin(roof, ["dome", "closed"]), "dome/closed"),
                      summarize(np.array([g["neutral"] for g in G]) == 1, "neutral site")]
kick = np.array([float(g["gametime"][:2]) + float(g["gametime"][3:5]) / 60 if g["gametime"] else 13.0 for g in G])
S["slot"] = [summarize(kick >= 19.0, "primetime (kick>=19:00 ET)"), summarize((kick >= 15.5) & (kick < 19), "late afternoon"),
             summarize(kick < 15.5, "early"), summarize(np.array([g["div"] for g in G]) == 1, "divisional"),
             summarize(np.array([g["div"] for g in G]) == 0, "non-divisional")]
# margin information the W/L Elo does not carry
pdh = np.array([g["h"]["pd_pg"] if g["h"]["pd_pg"] is not None else np.nan for g in G])
pda = np.array([g["a"]["pd_pg"] if g["a"]["pd_pg"] is not None else np.nan for g in G])
pdd = pdh - pda
okpd = ~np.isnan(pdd) & (np.array([min(g["h"]["n_played"], g["a"]["n_played"]) for g in G]) >= 3)
# residualize PD diff on the model logit to isolate "margin says more than our model"
if okpd.sum() > 50:
    A = np.column_stack([np.ones(okpd.sum()), logit(p[okpd])])
    beta = np.linalg.lstsq(A, pdd[okpd], rcond=None)[0]
    pd_res = np.full(n, np.nan); pd_res[okpd] = pdd[okpd] - A @ beta
    q1, q2 = np.nanpercentile(pd_res, [20, 80])
    S["margin_info"] = [summarize(okpd & (pd_res <= q1), "PD/g diff far BELOW what model implies (bottom 20%)"),
                        summarize(okpd & (pd_res > q1) & (pd_res < q2), "PD/g diff consistent with model (mid 60%)"),
                        summarize(okpd & (pd_res >= q2), "PD/g diff far ABOVE what model implies (top 20%)")]
pm_h = np.array([g["h"]["prev_margin"] if g["h"]["prev_margin"] is not None else np.nan for g in G])
pm_a = np.array([g["a"]["prev_margin"] if g["a"]["prev_margin"] is not None else np.nan for g in G])
S["last_game"] = [
    team_slice(lambda t, g, sd: t["prev_margin"] is not None and t["prev_margin"] >= 21, "team won last game by 21+"),
    team_slice(lambda t, g, sd: t["prev_margin"] is not None and t["prev_margin"] <= -21, "team lost last game by 21+"),
]
RES["slices"] = S

# ---------------------------------------------------------------- continuous: what does r load on?
def ols_t(x, yv):
    ok = ~np.isnan(x) & ~np.isnan(yv)
    xx, yy_ = x[ok], yv[ok]
    if ok.sum() < 30 or xx.std() == 0:
        return None
    A = np.column_stack([np.ones(len(xx)), xx])
    b, *_ = np.linalg.lstsq(A, yy_, rcond=None)
    e = yy_ - A @ b
    # HC1 robust se
    XtXi = np.linalg.inv(A.T @ A)
    meat = (A * e[:, None] ** 2).T @ A
    V = XtXi @ meat @ XtXi * len(xx) / (len(xx) - 2)
    return float(b[1]), float(b[1] / math.sqrt(V[1, 1])), int(ok.sum()), float(np.corrcoef(xx, yy_)[0, 1])


def offset_logit(x, off, yy, cols=1):
    """y ~ sigmoid(off + a + b*x) Newton; returns (a,b), se_b."""
    A = np.column_stack([np.ones(len(x)), x])
    w = np.zeros(2)
    for _ in range(50):
        eta = off + A @ w
        mu = 1 / (1 + np.exp(-eta))
        g_ = A.T @ (yy - mu)
        Hm = (A * (mu * (1 - mu))[:, None]).T @ A + 1e-9 * np.eye(2)
        step = np.linalg.solve(Hm, g_)
        w += step
        if np.abs(step).max() < 1e-10:
            break
    se = math.sqrt(np.linalg.inv(Hm)[1, 1])
    return w, se


def loso_gain(x):
    """outcome-only: cross-fitted (leave-one-season-out) LL gain of adding v to logit(p_model)."""
    ok = ~np.isnan(x)
    gains = np.zeros(n); gains[:] = np.nan
    for s in range(2006, 2016):
        te = ok & (SEA == s); tr = ok & (SEA != s)
        if te.sum() == 0 or tr.sum() < 100:
            continue
        w, _ = offset_logit(x[tr], logit(p[tr]), y[tr])
        pp = 1 / (1 + np.exp(-(logit(p[te]) + w[0] + w[1] * x[te])))
        gains[te] = llm[te] - llv(y[te], pp)
    m = ~np.isnan(gains)
    return float(gains[m].mean()), boot_ci(gains[m]), int(m.sum())


def team_diff(key, fn=lambda v: v):
    return np.array([(fn(g["h"][key]) - fn(g["a"][key])) if (g["h"][key] is not None and g["a"][key] is not None)
                     else np.nan for g in G], float)


CAND = {
    "model:lgt (elo logit)": np.array([g["X"]["lgt"] for g in G]),
    "model:qd": np.array([g["X"]["qd"] for g in G]),
    "model:epa": np.array([g["X"]["epa"] for g in G]),
    "model:thfa": np.array([g["X"]["thfa"] for g in G]),
    "model:rest": np.array([g["X"]["rest"] for g in G]),
    "model:luck": np.array([g["X"]["luck"] for g in G]),
    "model:pass": np.array([g["X"]["pass"] for g in G]),
    "model:run": np.array([g["X"]["run"] for g in G]),
    "model:rq (2013+)": np.where(SEA >= 2013, [g["X"]["rq"] for g in G], np.nan),
    "model:absence ol+de+sk (2013+)": np.where(SEA >= 2013, [g["X"]["ol"] + g["X"]["de"] + g["X"]["sk"] for g in G], np.nan),
    "qb: non-established starter (diff)": team_diff("is_estab", lambda v: 1 - v),
    "qb: inexperienced non-estab (<8 starts) (diff)": np.array(
        [int(g["h"]["is_estab"] == 0 and g["h"]["qb_starts"] < 8) - int(g["a"]["is_estab"] == 0 and g["a"]["qb_starts"] < 8) for g in G], float),
    "qb: log(1+career starts) (diff)": team_diff("qb_starts", lambda v: math.log1p(v)),
    "qb: returning starter (diff)": team_diff("returning"),
    "qb: offseason change wk1-4 (diff)": np.where(REG & (W <= 4), team_diff("offseason_qb_change"), np.nan),
    "coach: new HC (diff)": team_diff("new_coach"),
    "coach: interim (diff)": team_diff("interim"),
    "standings: eliminated-proxy wk13+ (diff)": np.where(REG & (W >= 13), team_diff("dead"), np.nan),
    "standings: clinched div wk13+ (diff)": np.where(REG & (W >= 13), team_diff("clinched_div"), np.nan),
    "standings: win pct to date wk5+ (diff)": np.where(REG & (W >= 5), np.array(
        [(g["h"]["W"] / max(g["h"]["P"], 1)) - (g["a"]["W"] / max(g["a"]["P"], 1)) for g in G]), np.nan),
    "margin: PD/g season-to-date (diff, >=3 games)": np.where(okpd, pdd, np.nan),
    "margin: last-game margin (diff)": pm_h - pm_a,
    "inj: Out+Doubtful count (diff, 2009+)": od_diff,
    "inj: Questionable count (diff, 2009+)": injd("n_quest"),
    "inj: QB O/D (diff, 2009+)": injd("n_od_qb"),
    "inj: OL O/D (diff, 2009+)": injd("n_od_ol"),
    "inj: DEF O/D (diff, 2009+)": injd("n_od_def"),
    "inj: skill O/D (diff, 2009+)": injd("n_od_skill"),
    "inj: starting QB listed Q/D (diff, 2009+)": injd("starter_qb_listed_q"),
    "inj: DNP-in-practice actives (diff, 2009+)": injd("n_dnp_active"),
    "inj: snap-weighted O+D (diff, 2012+)": wod,
    "inj: snap-weighted Questionable (diff, 2012+)": injd("w_q"),
}
# residual of logit(close) after projecting on our live features (pooled DEV OLS): outside-span info
Xspan = np.column_stack([np.ones(n)] + [np.array([g["X"][c] for g in G]) for c in
                                         ("lgt", "qd", "epa", "early", "thfa", "rest", "luck", "ol", "de", "sk", "rq", "pass", "run")]
                        + [logit(p)])
bproj = np.linalg.lstsq(Xspan, logit(pm), rcond=None)[0]
r_perp = logit(pm) - Xspan @ bproj
R2span = 1 - r_perp.var() / logit(pm).var()
RES["span"] = {"R2_close_on_features_plus_our_logit": round(float(R2span), 4),
               "sd_r_perp": round(float(r_perp.std()), 4), "sd_r": round(float(r_dis.std()), 4)}
assoc = []
for name, x in CAND.items():
    a1 = ols_t(x, r_dis); a2 = ols_t(x, r_perp)
    ok = ~np.isnan(x)
    if a1 is None:
        continue
    w, se = offset_logit(x[ok], logit(p[ok]), y[ok])
    lg = loso_gain(x) if not name.startswith("model:") else (None, [None, None], None)
    assoc.append({"var": name, "n": a1[2], "sd": round(float(np.nanstd(x)), 4),
                  "r_on_v_slope": round(a1[0], 4), "r_on_v_t": round(a1[1], 2), "corr_r": round(a1[3], 4),
                  "rperp_on_v_t": round(a2[1], 2), "corr_rperp": round(a2[3], 4),
                  "outcome_offset_coef": round(float(w[1]), 4), "outcome_offset_t": round(float(w[1] / se), 2),
                  "outcome_loso_gain": None if lg[0] is None else round(lg[0], 5),
                  "outcome_loso_gain_ci": lg[1]})
assoc.sort(key=lambda a: -abs(a["rperp_on_v_t"]))
RES["association"] = assoc

# joint: how much of the outside-span residual do all non-model candidates explain (2012-2015 where all live)
names = [k for k in CAND if not k.startswith("model:")]
okj = (SEA >= 2012)
M = np.column_stack([np.nan_to_num(CAND[k]) for k in names])
for lo, lab in ((2006, "2006-2015 (missing -> 0)"), (2012, "2012-2015 (all live)")):
    mm = SEA >= lo
    A = np.column_stack([np.ones(mm.sum()), M[mm]])
    b = np.linalg.lstsq(A, r_perp[mm], rcond=None)[0]
    e = r_perp[mm] - A @ b
    RES.setdefault("joint_rperp_R2", {})[lab] = {
        "n": int(mm.sum()), "R2_in_sample": round(float(1 - e.var() / r_perp[mm].var()), 4),
        "k": len(names)}

# where does the model disagree most: top-5% |r| profile
top = np.abs(r_dis) >= np.percentile(np.abs(r_dis), 95)
RES["top5_disagreement"] = summarize(top, "top 5% |r|")
RES["top5_profile"] = {
    "share_any_nonestab_qb": round(float(np.mean([g["h"]["is_estab"] == 0 or g["a"]["is_estab"] == 0
                                                  for g, t in zip(G, top) if t])), 3),
    "share_any_nonestab_qb_all": round(float(np.mean([g["h"]["is_estab"] == 0 or g["a"]["is_estab"] == 0 for g in G])), 3),
    "share_week_1_4": round(float(np.mean((W <= 4)[top])), 3), "share_week_1_4_all": round(float(np.mean(W <= 4)), 3),
    "share_week_17": round(float(np.mean((W == 17)[top])), 3), "share_week_17_all": round(float(np.mean(W == 17)), 3),
    "share_2013plus": round(float(np.mean((SEA >= 2013)[top])), 3), "share_2013plus_all": round(float(np.mean(SEA >= 2013)), 3),
    "mean_abs_lgt_top": round(float(np.abs(Xspan[top, 1]).mean()), 3), "mean_abs_lgt_all": round(float(np.abs(Xspan[:, 1]).mean()), 3),
}
top_games = []
for i in np.argsort(-np.abs(r_dis))[:25]:
    g = G[i]
    top_games.append({"gid": g["gid"], "p_model": round(float(p[i]), 3), "p_close": round(float(pm[i]), 3), "y": g["y"],
                      "h_qb_estab": g["h"]["is_estab"], "a_qb_estab": g["a"]["is_estab"],
                      "h_qb_starts": g["h"]["qb_starts"], "a_qb_starts": g["a"]["qb_starts"],
                      "h_od": None if g["ih"] is None else g["ih"]["n_out"] + g["ih"]["n_doubt"],
                      "a_od": None if g["ia"] is None else g["ia"]["n_out"] + g["ia"]["n_doubt"],
                      "h_dead": g["h"]["dead"], "a_dead": g["a"]["dead"],
                      "h_clinch": g["h"]["clinched_div"], "a_clinch": g["a"]["clinched_div"]})
RES["top25_games"] = top_games

json.dump(RES, open("data/bt_nfl_anatomy.json", "w"), indent=1)
print("wrote data/bt_nfl_anatomy.json")
