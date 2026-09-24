"""NHL gap anatomy -- step 4: the backup-goalie games and the era trend.
DEV ONLY, descriptive. Odds are evaluation-only.

  * era split: early DEV (2011-12..2014-15) vs late DEV (2015-16..2017-18) --
    model-open, open->close, model-close. Where does the gap sit in the most
    recent DEV era (the one nearest to how today's market behaves)?
  * backup starts split by WHY: a one-off (the usual #1 started the previous
    game) vs a SPELL (the same non-#1 goalie also started the previous game, i.e.
    the #1 is out for a stretch), and by b2b / no b2b.
  * direction test inside backup games: is the close's gain on line moves
    AGAINST the backup team (goalie information) or elsewhere?
Output: data/bt_nhl_anatomy3.json
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "phase0")
import nhl_depth_eval as Hd  # noqa: E402
from nhl_glicko2_eval import DEV_END, TEST_START, llv  # noqa: E402
from bt_nhl_anatomy import reproduce, join_odds, logit, boot_mean  # noqa: E402

OUT = "data/bt_nhl_anatomy3.json"


def block(m, llm, llo, llc):
    m = np.asarray(m, bool)
    return {"n": int(m.sum()),
            "model_minus_close": round(float((llm - llc)[m].mean()), 5),
            "mc_ci": boot_mean((llm - llc)[m]),
            "model_minus_open": round(float((llm - llo)[m].mean()), 5),
            "mo_ci": boot_mean((llm - llo)[m]),
            "open_to_close": round(float((llo - llc)[m].mean()), 5),
            "oc_ci": boot_mean((llo - llc)[m])}


def show(name, r):
    print(f"  {name:<46} n={r['n']:>5}  model-close {r['model_minus_close']:+.5f} "
          f"[{r['mc_ci'][0]:+.5f},{r['mc_ci'][1]:+.5f}]  model-open {r['model_minus_open']:+.5f} "
          f"[{r['mo_ci'][0]:+.5f},{r['mo_ci'][1]:+.5f}]  open->close {r['open_to_close']:+.5f} "
          f"[{r['oc_ci'][0]:+.5f},{r['oc_ci'][1]:+.5f}]")


def main():
    p, y, seas, G, Xcols = reproduce()
    pc, po, _ = join_odds(G, y)
    keep = np.where(~np.isnan(pc))[0]
    p, y, seas, pc, po = p[keep], y[keep], seas[keep], pc[keep], po[keep]
    G = [G[i] for i in keep]
    Xcols = {k: v[keep] for k, v in Xcols.items()}
    Hd.assert_dev_only(seas)
    llm, llc, llo = llv(y, p), llv(y, pc), llv(y, po)

    F = pd.read_csv("data/bt_nhl_anatomy_feats.csv").set_index("game_id")
    F = F.loc[[g["game_id"] for g in G]]
    assert F.season.max() <= DEV_END < TEST_START
    f = {c: F[c].to_numpy() for c in F.columns if c not in ("date", "home", "away")}
    out = {}

    print("ERA split")
    early = seas <= 20142015
    late = seas >= 20152016
    out["era"] = {"early 2011-12..2014-15": block(early, llm, llo, llc),
                  "late 2015-16..2017-18": block(late, llm, llo, llc)}
    for k, r in out["era"].items():
        show(k, r)

    nm_h, nm_a = f["g_notmodal_home"] == 1, f["g_notmodal_away"] == 1
    spell_h = nm_h & (f["g_changed_home"] == 0)
    spell_a = nm_a & (f["g_changed_away"] == 0)
    oneoff_h = nm_h & (f["g_changed_home"] == 1)
    oneoff_a = nm_a & (f["g_changed_away"] == 1)
    b2bh, b2ba = Xcols["b2b_home"] == 1, Xcols["b2b_away"] == 1
    usual = (f["g_notmodal_home"] == 0) & (f["g_notmodal_away"] == 0)
    B = {
        "both teams start their #1 (modal of last 20)": usual,
        "any non-#1 starter": nm_h | nm_a,
        "non-#1 SPELL (same non-#1 as prev game), any": spell_h | spell_a,
        "non-#1 ONE-OFF (#1 started prev game), any": (oneoff_h | oneoff_a) & ~(spell_h | spell_a),
        "one-off non-#1 on that team's b2b": (oneoff_h & b2bh) | (oneoff_a & b2ba),
        "one-off non-#1 NOT on b2b": ((oneoff_h & ~b2bh) | (oneoff_a & ~b2ba)) & ~(spell_h | spell_a),
        "home non-#1 only": nm_h & ~nm_a,
        "away non-#1 only": nm_a & ~nm_h,
    }
    print("\nGOALIE: why the non-#1 starts")
    out["goalie_why"] = {}
    for k, m in B.items():
        r = block(m, llm, llo, llc)
        out["goalie_why"][k] = r
        show(k, r)

    print("\nGOALIE x ERA")
    out["goalie_x_era"] = {}
    for en, em in (("early", early), ("late", late)):
        for k in ("both teams start their #1 (modal of last 20)", "any non-#1 starter",
                  "non-#1 SPELL (same non-#1 as prev game), any"):
            r = block(B[k] & em, llm, llo, llc)
            out["goalie_x_era"][f"{en}: {k}"] = r
            show(f"{en}: {k}"[:46], r)

    # direction test inside non-#1 games (one side only, for a clean sign)
    lc, lo_ = logit(pc), logit(po)
    one_side = nm_h ^ nm_a
    sgn = np.where(nm_h, -1.0, 1.0)         # + = move toward home is AGAINST the non-#1 team if away
    # move expressed toward the NON-#1 team: positive = line moved toward the backup's side
    mv_toward_backup = np.where(nm_h, lc - lo_, -(lc - lo_))
    print("\nDIRECTION inside one-sided non-#1 games (move = logit shift toward backup's team)")
    out["direction"] = {"mean_move_toward_backup_logit": round(float(mv_toward_backup[one_side].mean()), 4)}
    for nm_, m in (("line moved AGAINST backup team (< -0.02)", one_side & (mv_toward_backup < -0.02)),
                   ("line flat (|move| <= 0.02)", one_side & (np.abs(mv_toward_backup) <= 0.02)),
                   ("line moved TOWARD backup team (> +0.02)", one_side & (mv_toward_backup > 0.02))):
        r = block(m, llm, llo, llc)
        out["direction"][nm_] = r
        show(nm_, r)
    print(f"  mean move toward backup's team: {mv_toward_backup[one_side].mean():+.4f} logit")

    # no-backup games, same direction split relative to HOME for reference
    print("\nREFERENCE: both-#1 games, line move toward home")
    mvh = lc - lo_
    for nm_, m in (("moved toward home (> +0.02)", usual & (mvh > 0.02)),
                   ("flat", usual & (np.abs(mvh) <= 0.02)),
                   ("moved toward away (< -0.02)", usual & (mvh < -0.02))):
        r = block(m, llm, llo, llc)
        out.setdefault("reference_usual", {})[nm_] = r
        show(nm_, r)

    json.dump(out, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
