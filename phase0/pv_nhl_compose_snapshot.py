"""Player-value program, NHL -- COMPOSE end-of-DEV snapshot (DEV ONLY).

Every skater's LAST pre-game composite value of 2017-18 (goals / 60 of his
all-situations TOI, with its component terms) and every goalie's value at his last
2017-18 start, with names. A reading aid for face validity and the site; built
from data/pv_nhl_compose_player_values.parquet and _goalie_values.csv.

    python phase0/pv_nhl_compose_snapshot.py
"""
from __future__ import annotations

import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pv_nhl_compose_core as C  # noqa: E402

OUT = C.D("pv_nhl_compose_snapshot_201718.csv")


def main():
    pv = pd.read_parquet(C.D("pv_nhl_compose_player_values.parquet"))
    assert int(pv.gid.max()) < C.DEV_MAX_GID
    # regular season only (the defense component has no playoff values)
    s = pv[(pv.season == 20172018) & (pv.gtype == 2)].sort_values(["date", "gid"])
    last = s.groupby("pid").tail(1).copy()
    last["n_games_1718"] = last.pid.map(s.groupby("pid").size())
    names = pd.read_csv(C.D("pv_nhl_players.csv"))
    nm = dict(zip(names.pid, names["first"].astype(str) + " " + names["last"].astype(str)))
    last["name"] = last.pid.map(nm)
    cols = ["pid", "name", "grp", "team", "date", "n_games_1718", "m_ev", "m_pp", "m_all",
            "v_all60", "v_ev60", "v_pp60", "v_dd60", "v_game"] + [
        "p_" + c for c in C.SK_COMPS]
    sk = last[cols].rename(columns={"date": "as_of_game_date"})
    gv = pd.read_csv(C.D("pv_nhl_compose_goalie_values.csv"))
    gv = gv[(gv.season == 20172018) & (gv.gid // 10000 % 10 == 2)]
    gl = gv.groupby("g_pid").tail(1).copy()
    gl["n_starts_1718"] = gl.g_pid.map(gv.groupby("g_pid").size())
    g = pd.DataFrame({"pid": gl.g_pid, "name": gl.g_pid.map(nm), "grp": "G",
                      "n_games_1718": gl.n_starts_1718, "v_all60": gl.gv,
                      "sav_mu": gl.sav_mu})
    out = pd.concat([sk, g], ignore_index=True).sort_values(["grp", "v_all60"],
                                                             ascending=[True, False])
    out.to_csv(OUT, index=False, float_format="%.4f")
    print(len(out), "rows ->", OUT)


if __name__ == "__main__":
    main()
