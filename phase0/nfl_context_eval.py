"""NFL game-context channels: weather / travel / officials — DEV-ONLY screens.

Three channels that the shipped 14-feature model does not contain (verified: the
serve column order at nfl_season_serve.py:338-351 has no weather, no travel
distance, no officials).

ONE-LOOK PROTOCOL: this script NEVER touches a season >= 2016.
  * the weather pull is issued only for 1999-2015 stadium-seasons, so TEST-era
    weather is not even downloaded;
  * every candidate column is force-zeroed for seasons >= 2016 and asserted;
  * scoring is cv_ll() (phase0/nfl_coord_tune.py:131), whose index set is the
    `dev` mask = seasons in [2001, 2015]. An explicit assert guards that.

Weather method note (deliberate): archive weather is ACTUALS, which is a known
leak trap for deployment (a sister project manufactured a fake edge that way).
It is used here ONLY as a KNOWN-OPTIMISTIC UPPER BOUND: if actual kickoff
weather adds nothing on DEV, no forecast product could, and the channel is dead.
If it clears the gate the result is reported as "upper bound passes — requires
re-build on historical FORECAST data before any TEST look" and nothing further
is done.

Usage:
    python -X utf8 phase0/nfl_context_eval.py --pull     # network, resumable
    python -X utf8 phase0/nfl_context_eval.py --build    # cheap, no prelude
    python -X utf8 phase0/nfl_context_eval.py --screen   # heavy prelude + CV
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import threading
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np

DEV_LO, DEV_HI = 1999, 2015          # rows we are allowed to build at all
SCORE_LO = 2001                      # DEV_SCORE_FROM — the rows cv_ll actually scores
TEST_FLOOR = 2016                    # nothing at or above this may ever be built/read

GAMES = "data/nfl_games.csv"
WX_CACHE = "data/nfl_weather_dev.csv"
OFFICIALS = "data/nfl_officials.csv"
FEATS = "data/nfl_ctx_features.json"
OUT = "data/nfl_context_eval.json"
WORKERS = 6                          # Open-Meteo archive: generous, but be polite

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

# ---------------------------------------------------------------- stadiums --
# stadium_id -> (lat, lon).  Every stadium_id that appears in a 1999-2015
# completed game.  Roof comes per-game from nfl_games.csv (handles retractables).
STAD = {
    "NYC00": (40.8128, -74.0742),   # Giants Stadium
    "NYC01": (40.8135, -74.0745),   # MetLife
    "GNB00": (44.5013, -88.0622),   # Lambeau
    "CAR00": (35.2258, -80.8528),   # Bank of America
    "BAL00": (39.2780, -76.6227),   # M&T Bank
    "SDG00": (32.7831, -117.1196),  # Qualcomm
    "ATL00": (33.7576, -84.4008),   # Georgia Dome
    "STL00": (38.6329, -90.1885),   # Edward Jones Dome
    "TAM00": (27.9759, -82.5033),   # Raymond James
    "NAS00": (36.1665, -86.7713),   # Nissan
    "OAK00": (37.7516, -122.2005),  # Oakland Coliseum
    "WAS00": (38.9076, -76.8645),   # FedEx
    "MIA00": (25.9580, -80.2389),   # Hard Rock
    "KAN00": (39.0489, -94.4839),   # Arrowhead
    "CLE00": (41.5061, -81.6995),   # Cleveland Browns
    "JAX00": (30.3239, -81.6373),   # EverBank
    "CHI98": (41.8623, -87.6167),   # Soldier Field
    "CHI99": (40.0996, -88.2360),   # Memorial Stadium, Champaign
    "NOR00": (29.9511, -90.0812),   # Superdome
    "CIN00": (39.0955, -84.5161),   # Paul Brown
    "CIN99": (39.0975, -84.5083),   # Cinergy Field
    "PIT00": (40.4468, -80.0158),   # Heinz Field
    "PIT99": (40.4468, -80.0122),   # Three Rivers
    "BUF00": (42.7738, -78.7870),   # Ralph Wilson
    "BUF01": (43.6414, -79.3894),   # Rogers Centre (dome)
    "DEN00": (39.7439, -105.0201),  # Invesco/Empower
    "DEN99": (39.7357, -105.0197),  # Mile High
    "BOS00": (42.0909, -71.2643),   # Gillette
    "BOS99": (42.0925, -71.2644),   # Foxboro
    "SFO00": (37.7136, -122.3861),  # Candlestick
    "SFO01": (37.4033, -121.9694),  # Levi's
    "SEA00": (47.5952, -122.3316),  # Lumen
    "SEA98": (47.5952, -122.3320),  # Kingdome
    "SEA99": (47.6503, -122.3016),  # Husky Stadium
    "MIN00": (44.9739, -93.2581),   # Metrodome
    "MIN98": (44.9765, -93.2246),   # TCF Bank
    "HOU00": (29.6847, -95.4107),   # NRG (retractable)
    "DET00": (42.3400, -83.0456),   # Ford Field
    "DET99": (42.6459, -83.2551),   # Pontiac Silverdome
    "PHI00": (39.9008, -75.1675),   # Lincoln Financial
    "PHI99": (39.9067, -75.1717),   # Veterans
    "PHO00": (33.5276, -112.2626),  # State Farm (retractable)
    "PHO99": (33.4264, -111.9325),  # Sun Devil
    "DAL99": (32.8394, -96.9147),   # Texas Stadium
    "DAL00": (32.7473, -97.0945),   # AT&T (retractable)
    "IND99": (39.7684, -86.1636),   # RCA Dome
    "IND00": (39.7601, -86.1639),   # Lucas Oil (retractable)
    "LON00": (51.5560, -0.2795),    # Wembley
    "SAN00": (29.4169, -98.4791),   # Alamodome
    "BRG00": (30.4118, -91.1836),   # Tiger Stadium (LSU)
    "MEX00": (19.3029, -99.1505),   # Azteca
}
INDOOR_ROOF = {"dome", "closed"}     # per-game; "open" retractables count as outdoor


def haversine(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = (math.sin((la2 - la1) / 2) ** 2
         + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2)
    return 6371.0 * 2 * math.asin(math.sqrt(h))


def load_rows():
    """Completed 1999-2015 games, in the same gameday sort the harness uses."""
    raw = [r for r in csv.DictReader(open(GAMES, encoding="utf-8"))
           if r["home_score"] != ""]
    raw.sort(key=lambda r: (r["gameday"],))
    out = [r for r in raw if DEV_LO <= int(r["season"]) <= DEV_HI]
    assert all(int(r["season"]) < TEST_FLOOR for r in out), "TEST-era row leaked in"
    miss = sorted({r["stadium_id"] for r in out} - set(STAD))
    assert not miss, f"missing stadium coords: {miss}"
    return out


def kick_utc(r):
    """Kickoff instant. nflverse gametime is Eastern (verified: MNF at Pacific
    home stadiums reads 20:30). 1999 has no gametime -> 13:00 ET, matching the
    harness fallback (nfl_big_test.py:56-59). 1999 is outside the scored mask."""
    gt = (r["gametime"] or "").strip()
    try:
        hh, mm = int(gt.split(":")[0]), int(gt.split(":")[1])
    except (ValueError, IndexError):
        hh, mm = 13, 0
    d = datetime.strptime(r["gameday"], "%Y-%m-%d")
    return datetime(d.year, d.month, d.day, hh, mm, tzinfo=ET).astimezone(UTC)


# ================================================================== --pull ==
def pull_weather():
    have = {}
    if os.path.exists(WX_CACHE):
        for r in csv.DictReader(open(WX_CACHE, encoding="utf-8")):
            have[r["game_id"]] = r
    rows = load_rows()
    todo = defaultdict(list)
    for r in rows:
        if r["game_id"] not in have:
            todo[(r["stadium_id"], r["season"])].append(r)
    print(f"weather cache {len(have)} rows; {len(todo)} stadium-seasons to pull "
          f"({sum(len(v) for v in todo.values())} games)", flush=True)

    fh = open(WX_CACHE, "a", newline="", encoding="utf-8")
    w = csv.writer(fh)
    if not have:
        w.writerow(["game_id", "stadium_id", "kick_utc", "temp_c", "wind_kmh", "precip_mm"])
    lock = threading.Lock()
    state = {"done": 0}

    def one(item):
        (sid, season), gs = item
        assert int(season) < TEST_FLOOR, "TEST-era pull attempted"
        lat, lon = STAD[sid]
        ks = {g["game_id"]: kick_utc(g) for g in gs}
        d0 = min(ks.values()).strftime("%Y-%m-%d")
        d1 = max(ks.values()).strftime("%Y-%m-%d")
        url = ("https://archive-api.open-meteo.com/v1/archive"
               f"?latitude={lat}&longitude={lon}&start_date={d0}&end_date={d1}"
               "&hourly=temperature_2m,wind_speed_10m,precipitation&timezone=GMT")
        js = None
        for attempt in range(6):
            try:
                rq = urllib.request.Request(url, headers={"User-Agent": "nfl-ctx-eval"})
                js = json.load(urllib.request.urlopen(rq, timeout=180))
                break
            except Exception as e:               # noqa: BLE001 - transient net
                print(f"  retry {sid} {season} ({attempt}): {e!r}", flush=True)
                time.sleep(2 + 4 * attempt)
        if js is None:
            print(f"  GIVE UP {sid} {season}", flush=True)
            return
        H = js["hourly"]
        idx = {t: i for i, t in enumerate(H["time"])}
        out = []
        for g in gs:
            key = ks[g["game_id"]].strftime("%Y-%m-%dT%H:00")
            i = idx.get(key)
            if i is None:
                continue
            out.append([g["game_id"], sid, key, H["temperature_2m"][i],
                        H["wind_speed_10m"][i], H["precipitation"][i]])
        with lock:
            w.writerows(out)
            fh.flush()
            state["done"] += 1
            if state["done"] % 10 == 0:
                print(f"  [{state['done']}/{len(todo)}]", flush=True)

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        list(ex.map(one, sorted(todo.items())))
    fh.close()
    print("weather pull done", flush=True)


def pull_officials():
    url = ("https://github.com/nflverse/nflverse-data/releases/download/"
           "officials/officials.csv")
    rq = urllib.request.Request(url, headers={"User-Agent": "nfl-ctx-eval"})
    b = urllib.request.urlopen(rq, timeout=180).read()
    tmp = OFFICIALS + ".tmp"
    open(tmp, "wb").write(b)
    os.replace(tmp, OFFICIALS)
    seas = Counter(r["season"] for r in csv.DictReader(open(OFFICIALS, encoding="utf-8")))
    print(f"officials.csv {sum(seas.values())} rows, seasons "
          f"{min(seas)}-{max(seas)}; DEV-era (<=2015) rows: "
          f"{sum(v for k, v in seas.items() if int(k) <= DEV_HI)}")


# ================================================================= --build ==
def build():
    rows = load_rows()
    wx = {}
    if os.path.exists(WX_CACHE):
        for r in csv.DictReader(open(WX_CACHE, encoding="utf-8")):
            wx[r["game_id"]] = (float(r["temp_c"]), float(r["wind_kmh"]), float(r["precip_mm"]))

    # team home venue per (team, season): modal stadium_id of that team's home games
    venue = defaultdict(Counter)
    for r in rows:
        venue[(r["home_team"], r["season"])][r["stadium_id"]] += 1
    home_venue = {}
    for k, c in venue.items():
        home_venue[k] = c.most_common(1)[0][0]
    # carry-forward for teams with no home game recorded in a season (shouldn't happen)
    TZ = {"BUF": 0, "MIA": 0, "NE": 0, "NYJ": 0, "BAL": 0, "CIN": 0, "CLE": 0, "PIT": 0,
          "IND": 0, "JAX": 0, "ATL": 0, "CAR": 0, "TB": 0, "WAS": 0, "NYG": 0, "PHI": 0,
          "DET": 0, "CHI": 1, "GB": 1, "MIN": 1, "DAL": 1, "HOU": 1, "TEN": 1, "KC": 1,
          "NO": 1, "DEN": 2, "ARI": 2, "SEA": 3, "SF": 3, "LAC": 3, "LV": 3, "LA": 3,
          "STL": 1, "SD": 3, "OAK": 3}

    out = {}
    # --- causal (as-of) climate norm per team: expanding mean of that team's own
    #     home-game kickoff temperature. prior = running league mean, n0 = 10.
    tnorm = defaultdict(lambda: [0.0, 0.0])
    lg_t = [0.0, 0.0]
    # --- road-trip bookkeeping
    last_away_date = {}
    consec_road = defaultdict(int)

    for r in rows:
        gid, s = r["game_id"], int(r["season"])
        assert s < TEST_FLOOR
        h, a, sid = r["home_team"], r["away_team"], r["stadium_id"]
        indoor = 1.0 if r["roof"] in INDOOR_ROOF else 0.0
        t_c, wind, prcp = wx.get(gid, (None, None, None))
        has_wx = t_c is not None

        def norm(team):
            st = tnorm[team]
            lgm = lg_t[0] / lg_t[1] if lg_t[1] > 0 else 15.0
            return (st[0] + 10.0 * lgm) / (st[1] + 10.0)

        nh, na = norm(h), norm(a)
        if has_wx and not indoor:
            accl = abs(t_c - na) - abs(t_c - nh)
            cold = max(0.0, na - t_c) - max(0.0, nh - t_c)
        else:
            accl = cold = 0.0

        # travel: distance each side travelled to the venue (0 for a true home game)
        vc = STAD[sid]
        hv = STAD.get(home_venue.get((h, r["season"]), sid), vc)
        av = STAD.get(home_venue.get((a, r["season"])), vc)
        d_home = haversine(hv, vc)
        d_away = haversine(av, vc)
        tz_h, tz_a = TZ.get(h, 0), TZ.get(a, 0)
        try:
            kick = int(r["gametime"].split(":")[0]) + int(r["gametime"].split(":")[1]) / 60.0
        except (ValueError, IndexError):
            kick = 13.0
        # body clock: kickoff hour in each team's home timezone (gametime is ET)
        body_h, body_a = kick - tz_h, kick - tz_a
        gd = datetime.strptime(r["gameday"], "%Y-%m-%d")
        da = last_away_date.get(a)
        days_since_away = (gd - da).days if da else 21
        cr = consec_road[a]

        out[gid] = {
            "season": s, "indoor": indoor,
            "temp_c": t_c if has_wx else None,
            "wind_kmh": wind if has_wx else None,
            "precip_mm": prcp if has_wx else None,
            "gc_temp": (float(r["temp"]) if r["temp"] not in ("", None) else None),
            "gc_wind": (float(r["wind"]) if r["wind"] not in ("", None) else None),
            "accl": accl, "cold": cold,
            "d_away_km": d_away, "d_home_km": d_home,
            "tz_shift": tz_h - tz_a, "body_a": body_a, "body_h": body_h,
            "days_since_away": min(days_since_away, 28), "consec_road": min(cr, 4),
            "referee": (r["referee"] or "").strip(),
            "home": h, "away": a, "neutral": 1.0 if r["location"] != "Home" else 0.0,
        }

        # --- state updates (strictly after the feature is emitted) ---
        if has_wx and not indoor:
            tnorm[h][0] += t_c; tnorm[h][1] += 1.0
            lg_t[0] += t_c; lg_t[1] += 1.0
        last_away_date[a] = gd
        consec_road[a] += 1
        consec_road[h] = 0

    json.dump(out, open(FEATS, "w"), separators=(",", ":"))
    nwx = sum(1 for v in out.values() if v["temp_c"] is not None)
    ind = sum(1 for v in out.values() if v["indoor"])
    print(f"built {len(out)} rows -> {FEATS}")
    print(f"  weather coverage {nwx}/{len(out)} = {nwx/len(out):.3%}   indoor {ind}")
    scored = [v for v in out.values() if SCORE_LO <= v["season"] <= DEV_HI]
    nwx2 = sum(1 for v in scored if v["temp_c"] is not None)
    print(f"  scored rows (2001-2015) {len(scored)}, weather {nwx2} = {nwx2/len(scored):.3%}")
    ref = sum(1 for v in scored if v["referee"])
    print(f"  referee coverage on scored rows {ref}/{len(scored)} = {ref/len(scored):.3%}, "
          f"{len({v['referee'] for v in scored if v['referee']})} distinct")


# ================================================================ --screen ==
def screen():
    sys.path.insert(0, "phase0")
    T0 = time.time()
    coord_src = open("phase0/nfl_coord_tune.py", encoding="utf-8").read()
    g_ = globals()
    exec(coord_src.split("X_CUR0 = X_of(F)")[0], g_)  # noqa: S102
    print(f"[{time.time()-T0:.0f}s] prelude done", flush=True)
    games, seasons, dev, y = g_["games"], g_["seasons"], g_["dev"], g_["y"]
    pe, X_of, F, cv_ll, llv = g_["pe"], g_["X_of"], g_["F"], g_["cv_ll"], g_["llv"]
    PBP_FIX, DEV_YEARS, epaP = g_["PBP_FIX"], g_["DEV_YEARS"], g_["epaP"]
    LogisticRegression = g_["LogisticRegression"]
    KFold = g_["KFold"]

    # ---- HARD ASSERT: nothing at or above 2016 is ever scored ----
    assert seasons[dev].max() <= 2015, f"TEST season in dev mask: {seasons[dev].max()}"
    assert seasons[dev].min() >= SCORE_LO
    print(f"scored rows: {int(dev.sum())}  seasons "
          f"{seasons[dev].min()}-{seasons[dev].max()}", flush=True)

    # ---- rebuild pass/run exactly as nfl_ctx_main_tests.py:27-79 ----
    aggc = defaultdict(lambda: defaultdict(lambda: [0.0, 0, 0.0, 0]))
    with open("data/nfl_duel_plays.csv", encoding="utf-8") as fh:
        rd = csv.reader(fh); hdr = next(rd); ix = {c: i for i, c in enumerate(hdr)}
        for r in rd:
            t_ = PBP_FIX.get(r[ix["posteam"]], r[ix["posteam"]])
            a = aggc[r[ix["game_id"]]][t_]
            if r[ix["passer_player_id"]] or r[ix["receiver_player_id"]]:
                a[0] += float(r[ix["epa"]]); a[1] += 1
            else:
                a[2] += float(r[ix["epa"]]); a[3] += 1
    ps = pn = rs = rn = 0.0
    for gid, tm in aggc.items():
        if int(gid[:4]) in DEV_YEARS:
            for t_, (a_, b_, c_, d_) in tm.items():
                ps += a_; pn += b_; rs += c_; rn += d_
    LGP, LGR = ps / pn, rs / rn
    dec_, pn_, sd_ = epaP["decay"], epaP["prior_n"], epaP["season_decay"]
    offP = defaultdict(lambda: [0.0, 0.0]); offR = defaultdict(lambda: [0.0, 0.0])
    dfaP = defaultdict(lambda: [0.0, 0.0]); dfaR = defaultdict(lambda: [0.0, 0.0])
    def crate(st_, lg_): return (st_[0] + (pn_ / 2) * lg_) / (st_[1] + pn_ / 2)
    f_pass = np.zeros(len(games)); f_run = np.zeros(len(games)); prev = None
    for i, g in enumerate(games):
        if prev is not None and g["season"] != prev:
            for st_ in (list(offP.values()) + list(offR.values())
                        + list(dfaP.values()) + list(dfaR.values())):
                st_[0] *= (1 - sd_); st_[1] *= (1 - sd_)
        prev = g["season"]
        h, a = g["home"], g["away"]
        f_pass[i] = (crate(offP[h], LGP) - crate(dfaP[h], LGP)) - (crate(offP[a], LGP) - crate(dfaP[a], LGP))
        f_run[i] = (crate(offR[h], LGR) - crate(dfaR[h], LGR)) - (crate(offR[a], LGR) - crate(dfaR[a], LGR))
        tm = aggc.get(g["gid"])
        if tm:
            for t_off, opp in ((h, a), (a, h)):
                pS, pN, rS, rN = tm.get(t_off, (0.0, 0, 0.0, 0))
                o = offP[t_off]; o[0] = dec_ * o[0] + pS; o[1] = dec_ * o[1] + pN
                o = offR[t_off]; o[0] = dec_ * o[0] + rS; o[1] = dec_ * o[1] + rN
                d2 = dfaP[opp]; d2[0] = dec_ * d2[0] + pS; d2[1] = dec_ * d2[1] + pN
                d2 = dfaR[opp]; d2[0] = dec_ * d2[0] + rS; d2[1] = dec_ * d2[1] + rN
    X14 = np.column_stack([X_of(F), f_pass, f_run])
    X14[:, 7] = np.load("data/nfl_v7_feature.npy")
    assert X14.shape[1] == 14, X14.shape

    # ---- per-fold cv, mirroring nfl_coord_tune.py:131-140 exactly ----
    def cv_folds(X, C=1e6):
        idx = np.where(dev)[0]
        sc = np.zeros(len(idx)); fold = np.zeros(len(idx), int)
        for k, (tr, va) in enumerate(KFold(5, shuffle=True, random_state=7).split(idx)):
            itr, iva = idx[tr], idx[va]
            fitm = itr[y[itr] != 0.5]
            clf = LogisticRegression(C=C, max_iter=5000).fit(X[fitm], y[fitm])
            sc[va] = llv(y[iva], clf.predict_proba(X[iva])[:, 1])
            fold[va] = k
        per = [float(sc[fold == k].mean()) for k in range(5)]
        return float(sc.mean()), per

    cur_cv, cur_folds = cv_folds(X14)
    ref_cv = cv_ll(X14)
    assert abs(cur_cv - ref_cv) < 1e-12, (cur_cv, ref_cv)
    print(f"[{time.time()-T0:.0f}s] baseline 14-feature DEV CV {cur_cv:.5f}  "
          f"folds {[round(v,5) for v in cur_folds]}", flush=True)

    ctx = json.load(open(FEATS, encoding="utf-8"))
    gids = [g["gid"] for g in games]
    N = len(games)
    seas = np.array([g["season"] for g in games])
    IN_DEVERA = seas < TEST_FLOOR

    def col(fn, default=0.0):
        """Materialise a candidate column; force 0 for every season >= 2016."""
        v = np.full(N, default, float)
        for i, gid in enumerate(gids):
            if not IN_DEVERA[i]:
                continue
            c = ctx.get(gid)
            if c is None:
                continue
            x = fn(c)
            if x is not None and np.isfinite(x):
                v[i] = x
        v[~IN_DEVERA] = 0.0
        assert np.all(v[seas >= TEST_FLOOR] == 0.0)
        return v

    have_wx = np.array([bool(ctx.get(gid, {}).get("temp_c") is not None) for gid in gids])
    outdoor = np.array([1.0 - ctx.get(gid, {}).get("indoor", 1.0) for gid in gids])

    # ---------------- weather columns (ACTUALS = optimistic upper bound) -----
    c_wind = col(lambda c: (c["wind_kmh"] / 10.0) if (c["wind_kmh"] is not None and not c["indoor"]) else 0.0)
    c_temp = col(lambda c: ((c["temp_c"] - 15.0) / 10.0) if (c["temp_c"] is not None and not c["indoor"]) else 0.0)
    c_prcp = col(lambda c: 1.0 if (c["precip_mm"] is not None and not c["indoor"] and c["precip_mm"] > 0.1) else 0.0)
    c_dome = col(lambda c: c["indoor"])
    c_accl = col(lambda c: c["accl"] / 10.0)
    c_cold = col(lambda c: c["cold"] / 10.0)
    # games.csv reported actuals (independent source, outdoor only)
    c_gwind = col(lambda c: (c["gc_wind"] / 10.0) if c["gc_wind"] is not None else 0.0)
    c_gtemp = col(lambda c: ((c["gc_temp"] - 59.0) / 18.0) if c["gc_temp"] is not None else 0.0)

    # ---------------- travel columns ----------------------------------------
    c_dist = col(lambda c: (c["d_away_km"] - c["d_home_km"]) / 1000.0)
    c_ldist = col(lambda c: (math.log1p(c["d_away_km"]) - math.log1p(c["d_home_km"])) / 5.0)
    c_tz = col(lambda c: float(c["tz_shift"]))
    c_body = col(lambda c: (c["body_a"] - c["body_h"]) / 3.0)
    c_bodya = col(lambda c: c["body_a"] / 10.0)
    c_dsa = col(lambda c: c["days_since_away"] / 10.0, default=2.1)
    c_road = col(lambda c: float(c["consec_road"]))

    # ---------------- officials ---------------------------------------------
    # nflverse officials.csv starts at 2015 -> unusable for a DEV walk.  The
    # DEV-testable crew signal is nfl_games.csv `referee` (crew chief), 100%
    # coverage 2001-2015.  Both EWMAs are strictly as-of (crew is announced
    # pre-game, and the state update happens after the feature is emitted).
    refs = [ctx.get(gid, {}).get("referee", "") for gid in gids]
    tot_pts = np.zeros(N)
    for i, g in enumerate(games):
        tot_pts[i] = np.nan
    raw_rows = {r["game_id"]: r for r in csv.DictReader(open(GAMES, encoding="utf-8"))
                if r["home_score"] != "" and int(r["season"]) < TEST_FLOOR}
    for i, gid in enumerate(gids):
        r = raw_rows.get(gid) if seas[i] < TEST_FLOOR else None
        if r:
            tot_pts[i] = float(r["home_score"]) + float(r["away_score"])

    def ref_ewma(decay, prior_n, kind):
        st = defaultdict(lambda: [0.0, 0.0])
        lg = [0.0, 0.0]
        f = np.zeros(N)
        for i in range(N):
            if seas[i] >= TEST_FLOOR:
                break                       # never walk into TEST era
            rf = refs[i]
            if not rf:
                continue
            s = st[rf]
            lgm = lg[0] / lg[1] if lg[1] > 0 else 0.0
            f[i] = (s[0] + prior_n * lgm) / (s[1] + prior_n) - lgm
            if kind == "hfa":
                v = y[i] - pe[i]            # home-win residual vs Elo
            else:
                v = (tot_pts[i] - 43.0) / 10.0 if np.isfinite(tot_pts[i]) else None
            if v is None:
                continue
            s[0] = decay * s[0] + v; s[1] = decay * s[1] + 1.0
            lg[0] = decay * lg[0] + v; lg[1] = decay * lg[1] + 1.0
        f[seas >= TEST_FLOOR] = 0.0
        assert np.all(f[seas >= TEST_FLOOR] == 0.0)
        return f

    c_refhfa = ref_ewma(0.995, 30.0, "hfa")
    c_reftot = ref_ewma(0.995, 30.0, "tot")
    c_refhfa2 = ref_ewma(0.99, 15.0, "hfa")

    # ---------------- interactions -------------------------------------------
    c_wind_x_pass = c_wind * X14[:, 12]
    c_wind_x_elo = c_wind * X14[:, 0]
    c_prcp_x_elo = c_prcp * X14[:, 0]
    c_reftot_x_elo = c_reftot * X14[:, 0]

    CANDS = [
        ("W1  wind (Open-Meteo actual)", [c_wind]),
        ("W2  temp (Open-Meteo actual)", [c_temp]),
        ("W3  precip flag", [c_prcp]),
        ("W4  dome flag", [c_dome]),
        ("W5  wind+temp+precip+dome (block)", [c_wind, c_temp, c_prcp, c_dome]),
        ("W6  wind x pass_net", [c_wind_x_pass]),
        ("W7  wind x elo_logit", [c_wind_x_elo]),
        ("W8  precip x elo_logit", [c_prcp_x_elo]),
        ("W9  acclimation |T-norm| edge", [c_accl]),
        ("W10 cold-only acclim edge", [c_cold]),
        ("W11 FULL weather block (8 cols)", [c_wind, c_temp, c_prcp, c_dome, c_accl,
                                             c_cold, c_wind_x_pass, c_wind_x_elo]),
        ("W12 games.csv wind (2nd source)", [c_gwind]),
        ("W13 games.csv wind+temp", [c_gwind, c_gtemp]),
        ("T1  travel dist diff (km)", [c_dist]),
        ("T2  travel log-dist diff", [c_ldist]),
        ("T3  tz shift (home-away)", [c_tz]),
        ("T4  body-clock diff", [c_body]),
        ("T5  away body-clock hour", [c_bodya]),
        ("T6  days since last away game", [c_dsa]),
        ("T7  consecutive road games", [c_road]),
        ("T8  FULL travel block (dist+tz+body+road)", [c_ldist, c_tz, c_body, c_dsa, c_road]),
        ("O1  ref EWMA home-resid (0.995/30)", [c_refhfa]),
        ("O2  ref EWMA home-resid (0.99/15)", [c_refhfa2]),
        ("O3  ref EWMA total points", [c_reftot]),
        ("O4  ref total x elo_logit", [c_reftot_x_elo]),
        ("O5  FULL officials block", [c_refhfa, c_reftot, c_reftot_x_elo]),
    ]

    res = {}
    print(f"\n{'candidate':44s} {'DEV CV':>9s} {'delta':>9s} {'fold spread':>12s}  gate")
    print("-" * 92)
    for name, cols in CANDS:
        X = np.column_stack([X14] + cols)
        assert np.all(np.isfinite(X))
        s, folds = cv_folds(X)
        delta = s - cur_cv                       # negative = improvement
        fd = [f - c for f, c in zip(folds, cur_folds)]
        gate = "GATE MET" if delta <= -0.0020 else ""
        print(f"{name:44s} {s:9.5f} {delta:+9.5f} "
              f"[{min(fd):+.5f},{max(fd):+.5f}]  {gate}", flush=True)
        res[name] = {"cv": round(s, 5), "delta": round(delta, 5),
                     "folds": [round(v, 5) for v in folds],
                     "fold_delta_min": round(min(fd), 5), "fold_delta_max": round(max(fd), 5),
                     "n_cols": len(cols), "gate_met": bool(delta <= -0.0020)}

    # ---------------- conclusiveness diagnostics ----------------------------
    # (a) in-sample likelihood-ratio test on the DEV rows.  This has NO CV
    #     penalty, so it answers "is there ANY signal in the channel", not
    #     "does it survive out-of-sample".  A null here is the stronger claim.
    from scipy.stats import chi2 as _chi2
    fit = np.where(dev)[0]
    fit = fit[y[fit] != 0.5]

    def insample_ll(X):
        m = LogisticRegression(C=1e6, max_iter=5000).fit(X[fit], y[fit])
        return float(llv(y[fit], m.predict_proba(X[fit])[:, 1]).sum())

    ll0 = insample_ll(X14)
    BLOCKS = {
        "weather (8 cols)": [c_wind, c_temp, c_prcp, c_dome, c_accl, c_cold,
                             c_wind_x_pass, c_wind_x_elo],
        "travel (5 cols)": [c_ldist, c_tz, c_body, c_dsa, c_road],
        "officials (3 cols)": [c_refhfa, c_reftot, c_reftot_x_elo],
    }
    lrt = {}
    print(f"\nin-sample LRT on DEV rows (n={len(fit)}), no CV penalty:")
    for nm, cols in BLOCKS.items():
        ll1 = insample_ll(np.column_stack([X14] + cols))
        stat = 2.0 * (ll0 - ll1)
        p = float(_chi2.sf(max(stat, 0.0), len(cols)))
        lrt[nm] = {"chi2": round(stat, 3), "df": len(cols), "p": round(p, 4)}
        print(f"  {nm:22s} chi2 {stat:7.3f}  df {len(cols)}  p {p:.4f}"
              f"   {'SIGNAL' if p < 0.05 else 'no signal'}")

    # (b) outdoor-only CV: indoor games and any missing-weather rows dilute the
    #     weather columns to zero, which could mask a real outdoor effect.
    out_mask = dev & outdoor.astype(bool) & have_wx

    def cv_sub(X, mask, C=1e6):
        idx = np.where(mask)[0]
        sc = np.zeros(len(idx))
        for tr, va in KFold(5, shuffle=True, random_state=7).split(idx):
            itr, iva = idx[tr], idx[va]
            fm = itr[y[itr] != 0.5]
            m = LogisticRegression(C=C, max_iter=5000).fit(X[fm], y[fm])
            sc[va] = llv(y[iva], m.predict_proba(X[iva])[:, 1])
        return float(sc.mean())

    ob = cv_sub(X14, out_mask)
    sub = {"n": int(out_mask.sum()), "base": round(ob, 5)}
    print(f"\noutdoor-with-weather subset CV (n={int(out_mask.sum())}), base {ob:.5f}:")
    for nm, cols in (("wind only", [c_wind]),
                     ("wind+temp+precip", [c_wind, c_temp, c_prcp]),
                     ("wind x pass_net", [c_wind_x_pass]),
                     ("acclimation edge", [c_accl]),
                     ("full weather (6, no dome)", [c_wind, c_temp, c_prcp, c_accl,
                                                    c_cold, c_wind_x_pass])):
        s = cv_sub(np.column_stack([X14] + cols), out_mask)
        sub[nm] = {"cv": round(s, 5), "delta": round(s - ob, 5)}
        print(f"  {nm:28s} {s:.5f}  ({s-ob:+.5f})"
              f"  {'GATE MET' if s - ob <= -0.0020 else ''}")

    scored_mask = dev
    cov = {
        "scored_rows": int(dev.sum()),
        "scored_seasons": [int(seasons[dev].min()), int(seasons[dev].max())],
        "weather_rows": int((have_wx & scored_mask).sum()),
        "weather_pct": round(float((have_wx & scored_mask).sum() / dev.sum()), 4),
        "outdoor_rows": int((outdoor.astype(bool) & scored_mask).sum()),
        "outdoor_with_weather": int((have_wx & outdoor.astype(bool) & scored_mask).sum()),
        "referee_rows": int(sum(1 for i in np.where(dev)[0] if refs[i])),
        "distinct_refs": len({refs[i] for i in np.where(dev)[0] if refs[i]}),
        "games_csv_wind_rows": int((c_gwind[dev] != 0).sum()),
    }
    print("\ncoverage:", json.dumps(cov, indent=1))
    json.dump({"baseline_cv": round(cur_cv, 5),
               "baseline_folds": [round(v, 5) for v in cur_folds],
               "gate": -0.0020, "coverage": cov, "candidates": res,
               "lrt_insample": lrt, "outdoor_subset": sub},
              open(OUT + ".tmp", "w"), indent=1)
    os.replace(OUT + ".tmp", OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--screen", action="store_true")
    # NB: the prelude exec inside screen() writes into module globals, so keep
    # the parsed flags in locals that it cannot clobber (it defines `a`).
    _args = ap.parse_args()
    _do_pull, _do_build, _do_screen = _args.pull, _args.build, _args.screen
    if _do_pull:
        pull_officials()
        pull_weather()
    if _do_build:
        build()
    if _do_screen:
        screen()
    if not (_do_pull or _do_build or _do_screen):
        ap.print_help()
