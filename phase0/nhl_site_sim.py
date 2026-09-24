"""Monte Carlo season projection for the NHL page (numpy only).

Plays every remaining regular-season game N_SIMS times with the SHIPPED
model's own game probability - the same Elo + rest + back-to-back + xG blend
the schedule rows carry - and reports per team: projected points (mean, SD,
10th/90th percentile), projected W / L / OTL, and the share of simulations in
which the team makes the playoffs, wins its division, and has the league's
best record. Market-blind: no odds anywhere.

  * Played games count as they finished (the standings table).
  * Overtime: a game reaches OT/SO with probability P_OT and the OT/SO winner
    is closer to a coin flip than the regulation winner - home wins OT with
    q = 0.5 + OT_LAMBDA * (p - 0.5). Regulation outcomes are set so the home
    team's TOTAL win probability is exactly the model's p. The loser in OT/SO
    takes a point. Both constants are measured on the DEV seasons only.
  * "Hot" simulation: after each simulated game both teams' Elo moves by the
    model's own update (k, home advantage) so later games in the same
    simulation see it. That is the model's uncertainty about how strong a team
    will be in March; a static-rating simulation is overconfident. The xG team
    rating is held at its current value (no expected goals are simulated).
  * Playoffs: top three per division + two wild cards per conference, by
    points, then regulation wins, then wins, then a random draw.

EARLY-tier by construction: team ratings only, no roster, lineup or goalie.
Deterministic for identical inputs (the seed is derived from them).
"""
from __future__ import annotations

import numpy as np

# DEV regular seasons 2011-12..2017-18 (n = 8,141 games): 23.77% reached OT/SO.
P_OT = 0.2377
# DEV OT/SO games (n = 1,935): home OT/SO win rate ~ 0.5 + 0.454 (p - 0.5) with
# p the model's pre-game home probability (least squares through 0.5).
OT_LAMBDA = 0.454
N_SIMS = 20000


def simulate(remaining: list[dict], elo: dict, xg: dict, meta: dict, base: dict,
             elo_cfg: dict, blend: dict, xg_ha: float, n_sims: int = N_SIMS,
             seed: int = 0, batch: int = 5000) -> tuple[dict, dict]:
    """remaining: unplayed regular-season rows in play order, each with home,
    away and the model inputs hrest, arest, hb2b, ab2b.
    elo / xg: current (walked, boundary-regressed) team states.
    meta: code -> {div, conf}; base: code -> standings row (pts, w, l, otl, rw)."""
    teams = sorted(meta)
    ix = {t: i for i, t in enumerate(teams)}
    T = len(teams)
    games = [g for g in remaining if g["home"] in ix and g["away"] in ix]
    c = blend["coefs"]
    k, ha = float(elo_cfg["k"]), float(elo_cfg["ha"])
    H = np.array([ix[g["home"]] for g in games], dtype=np.int64)
    A = np.array([ix[g["away"]] for g in games], dtype=np.int64)
    # everything in the logit except the Elo term is fixed per game
    const = np.array([
        blend["intercept"]
        + c["rest"] * (g.get("hrest", 3) - g.get("arest", 3))
        + c["b2b_home"] * g.get("hb2b", 0) + c["b2b_away"] * g.get("ab2b", 0)
        + c["xg"] * ((xg.get(g["home"], 0.0) + xg_ha) - xg.get(g["away"], 0.0))
        for g in games], dtype=float)
    R0 = np.array([elo.get(t, 1500.0) for t in teams], dtype=float)
    b = {key: np.array([base.get(t, {}).get(key, 0) for t in teams], dtype=np.int32)
         for key in ("pts", "w", "l", "otl", "rw")}

    divs, confs = {}, {}
    for t in teams:
        divs.setdefault(meta[t]["div"], []).append(ix[t])
        confs.setdefault(meta[t]["conf"], []).append(ix[t])
    divs = {d: np.array(v) for d, v in divs.items()}
    conf_divs = {cf: [d for d in divs if meta[teams[divs[d][0]]]["conf"] == cf] for cf in confs}

    rng = np.random.default_rng(seed)
    tot = {key: np.zeros(T) for key in ("po", "div", "pres", "w", "l", "otl")}
    pts_all = []
    done = 0
    while done < n_sims:
        B = min(batch, n_sims - done)
        R = np.tile(R0, (B, 1))
        pts = np.tile(b["pts"], (B, 1))
        w = np.tile(b["w"], (B, 1))
        lo = np.tile(b["l"], (B, 1))
        otl = np.tile(b["otl"], (B, 1))
        rw = np.tile(b["rw"], (B, 1))
        for j in range(len(games)):
            h, a = H[j], A[j]
            pe = 1.0 / (1.0 + 10.0 ** (-((R[:, h] + ha) - R[:, a]) / 400.0))
            pe = np.clip(pe, 1e-9, 1 - 1e-9)
            p = 1.0 / (1.0 + np.exp(-(const[j] + c["elo_logit"] * np.log(pe / (1 - pe)))))
            q = 0.5 + OT_LAMBDA * (p - 0.5)
            pot = np.minimum(P_OT, np.minimum(p / q, (1 - p) / (1 - q)))
            u = rng.random(B)
            u2 = rng.random(B)
            hreg = u < p - pot * q
            ot = (~hreg) & (u < p - pot * q + pot)
            hwin = hreg | (ot & (u2 < q))
            awin = ~hwin
            areg = awin & ~ot
            pts[:, h] += 2 * hwin + (ot & awin)
            pts[:, a] += 2 * awin + (ot & hwin)
            w[:, h] += hwin
            w[:, a] += awin
            rw[:, h] += hreg
            rw[:, a] += areg
            lo[:, h] += areg
            lo[:, a] += hreg
            otl[:, h] += ot & awin
            otl[:, a] += ot & hwin
            d = k * (hwin - pe)
            R[:, h] += d
            R[:, a] -= d
        key = pts * 1e6 + rw * 1e3 + w + rng.random((B, T)) * 0.5
        made = np.zeros((B, T), dtype=bool)
        for dv, members in divs.items():
            kk = key[:, members]
            order = np.argsort(-kk, axis=1)
            top3 = members[order[:, :3]]
            np.put_along_axis(made, top3, True, axis=1)
            tot["div"][members] += np.bincount(order[:, 0], minlength=len(members))
        for cf, dl in conf_divs.items():
            members = np.concatenate([divs[d] for d in dl])
            kk = np.where(made[:, members], -np.inf, key[:, members])
            order = np.argsort(-kk, axis=1)
            wc = members[order[:, :2]]
            np.put_along_axis(made, wc, True, axis=1)
        tot["po"] += made.sum(axis=0)
        tot["pres"] += np.bincount(np.argmax(key, axis=1), minlength=T)
        tot["w"] += w.sum(axis=0)
        tot["l"] += lo.sum(axis=0)
        tot["otl"] += otl.sum(axis=0)
        pts_all.append(pts)
        done += B
    P = np.concatenate(pts_all, axis=0)
    rem = {t: 0 for t in teams}
    for g in games:
        rem[g["home"]] += 1
        rem[g["away"]] += 1
    out = {}
    for t in teams:
        i = ix[t]
        col = P[:, i]
        out[t] = {
            "pts": round(float(col.mean()), 1), "sd": round(float(col.std()), 1),
            "lo": int(np.percentile(col, 10)), "hi": int(np.percentile(col, 90)),
            "w": round(float(tot["w"][i] / n_sims), 1),
            "l": round(float(tot["l"][i] / n_sims), 1),
            "otl": round(float(tot["otl"][i] / n_sims), 1),
            "po": round(100.0 * float(tot["po"][i]) / n_sims, 2),
            "div": round(100.0 * float(tot["div"][i]) / n_sims, 2),
            "pres": round(100.0 * float(tot["pres"][i]) / n_sims, 2),
            "rem": rem[t],
        }
    info = {"n_sims": n_sims, "games_left": len(games), "p_ot": P_OT,
            "ot_lambda": OT_LAMBDA, "hot_elo_k": k, "seed": seed}
    return out, info
