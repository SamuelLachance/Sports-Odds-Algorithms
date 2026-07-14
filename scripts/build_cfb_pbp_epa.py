"""Build leak-free CFB play-level EPA aggregates from cfbfastR PBP.

Downloads season parquet from sportsdataverse releases, aggregates per
game/team EPA + success rate, writes .build-cache/cfb-pbp/game_team_epa.csv
(optional supplemental copy). Includes ``date`` for PIT season-to-date joins.

Usage:
  python scripts/build_cfb_pbp_epa.py --copy-supplemental
  python scripts/build_cfb_pbp_epa.py --start-season 2022 --end-season 2025
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CACHE_DIR = PROJECT_ROOT / ".build-cache" / "cfb-pbp"
SUPP_DIR = PROJECT_ROOT / "data" / "supplemental" / "cfb-pbp"
PBP_URL = (
    "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/"
    "cfbfastR_cfb_pbp/play_by_play_{season}.parquet"
)
USER_AGENT = "Sports-Odds-Algorithms/2.0"


def _download_season(season: int, cache_dir: Path, *, force: bool = False) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"play_by_play_{season}.parquet"
    if path.is_file() and not force and path.stat().st_size > 1_000_000:
        return path
    url = PBP_URL.format(season=season)
    print(f"downloading {url}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=300) as resp:
        path.write_bytes(resp.read())
    return path


def _pick(frame: pd.DataFrame, *names: str) -> str | None:
    lower = {c.lower(): c for c in frame.columns}
    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def _norm_name(name: str) -> str:
    return "".join(ch for ch in str(name or "").lower() if ch.isalnum())


def _espn_name_to_abbr() -> dict[str, str]:
    """Normalized-name -> abbr with tiered mascot-stripped aliases.

    ESPN displays carry mascots ("Ohio State Buckeyes") while cfbfastR uses
    school names ("Ohio State"). Aliases strip 1 then 2 trailing words; an
    earlier tier always wins and ambiguous aliases within a tier are dropped
    (bare "Ohio" must resolve to the Bobcats via 1-word strip, never to the
    Buckeyes via 2-word strip).
    """
    try:
        from web.season_games import _load_espn_team_ids, _normalize_abbr
    except Exception:  # noqa: BLE001
        return {}
    teams: list[tuple[str, str]] = []  # (display, abbr)
    for _eid, abbr, display in _load_espn_team_ids("cfb"):
        if display and abbr:
            # Same normalization as the scoreboard fetch so lake keys join
            # cfb.csv's vocabulary exactly.
            teams.append((str(display), _normalize_abbr("cfb", str(abbr))))
    out: dict[str, str] = {}
    ambiguous: set[str] = set()

    def _claim(key: str, abbr: str, tier_claims: dict[str, str]) -> None:
        if not key or key in out or key in ambiguous:
            return
        prior = tier_claims.get(key)
        if prior is not None and prior != abbr:
            ambiguous.add(key)
            tier_claims.pop(key, None)
            return
        tier_claims[key] = abbr

    # Tier 0: full display + raw abbreviation (never ambiguous in practice).
    tier: dict[str, str] = {}
    for display, abbr in teams:
        _claim(_norm_name(display), abbr, tier)
        _claim(_norm_name(abbr), abbr, tier)
    out.update(tier)
    # Tier 1 then 2: strip trailing mascot words.
    for strip in (1, 2):
        tier = {}
        for display, abbr in teams:
            words = display.split()
            if len(words) > strip:
                _claim(_norm_name(" ".join(words[:-strip])), abbr, tier)
        out.update(tier)
    return out


def _espn_id_to_abbr() -> dict[str, str]:
    """ESPN numeric team id -> normalized scoreboard abbreviation."""
    try:
        from web.season_games import _load_espn_team_ids, _normalize_abbr
    except Exception:  # noqa: BLE001
        return {}
    return {
        str(eid): _normalize_abbr("cfb", str(abbr))
        for eid, abbr, _display in _load_espn_team_ids("cfb")
        if eid and abbr
    }


def _to_abbr(raw: str, mapping: dict[str, str]) -> str:
    name = str(raw or "").lower().strip()
    if not name or name == "nan":
        return ""
    nn = _norm_name(name)
    hit = mapping.get(nn)
    if hit:
        return hit
    # Strict prefix fallback ("Miami" vs "Miami (FL)"): unique abbr among
    # candidates and a small length gap so "Texas A&M-Commerce" can never
    # merge into "Texas A&M".
    candidates = {
        abbr
        for cand, abbr in mapping.items()
        if nn
        and cand
        and (nn.startswith(cand) or cand.startswith(nn))
        and abs(len(cand) - len(nn)) <= 3
    }
    if len(candidates) == 1:
        return next(iter(candidates))
    return name  # last resort: keep full name (won't join, has_epa stays 0)


def _aggregate_season(path: Path) -> pd.DataFrame:
    raw = pd.read_parquet(path)
    game_col = _pick(raw, "game_id", "gameId", "id")
    season_col = _pick(raw, "season", "year")
    date_col = _pick(raw, "start_date", "game_date", "date", "startDate")
    pos_col = _pick(raw, "pos_team", "posteam", "offense", "offense_play", "pos_team_name")
    def_col = _pick(raw, "def_pos_team", "defteam", "defense", "defense_play", "def_pos_team_name")
    epa_col = _pick(raw, "EPA", "epa", "expected_points_added")
    sr_col = _pick(raw, "success", "Success")
    wp_col = _pick(raw, "wp_before", "wp", "win_prob")
    period_col = _pick(raw, "period", "qtr", "quarter")
    diff_col = _pick(raw, "pos_score_diff_start", "score_differential", "pos_score_diff")
    type_col = _pick(raw, "play_type", "playtype")
    if not game_col or not pos_col or not epa_col:
        raise RuntimeError(f"missing required PBP columns in {path.name}: {list(raw.columns)[:40]}")

    name_map = _espn_name_to_abbr()

    def _map_names(series: pd.Series) -> pd.Series:
        return series.map(lambda v: _to_abbr(v, name_map))

    frame = pd.DataFrame(
        {
            "game_id": raw[game_col].astype(str),
            "posteam": _map_names(raw[pos_col]),
            "epa": pd.to_numeric(raw[epa_col], errors="coerce"),
        }
    )
    if def_col:
        frame["defteam"] = _map_names(raw[def_col])
    else:
        frame["defteam"] = ""

    # Prefer exact ESPN team-id keys (the parquet carries home/away ids that
    # match the teams API); name matching stays as fallback only.
    home_name_col = _pick(raw, "home", "home_team")
    away_name_col = _pick(raw, "away", "away_team")
    hid_col = _pick(raw, "home_team_id")
    aid_col = _pick(raw, "away_team_id")
    if home_name_col and away_name_col and hid_col and aid_col:
        id_map = _espn_id_to_abbr()
        if id_map:
            def _ids_to_abbr(ids: pd.Series) -> pd.Series:
                return ids.map(
                    lambda v: id_map.get(str(int(v)), "") if pd.notna(v) else ""
                )

            pos_is_home = raw[pos_col].astype(str) == raw[home_name_col].astype(str)
            pos_by_id = _ids_to_abbr(raw[hid_col]).where(pos_is_home, _ids_to_abbr(raw[aid_col]))
            def_by_id = _ids_to_abbr(raw[aid_col]).where(pos_is_home, _ids_to_abbr(raw[hid_col]))
            pos_known = raw[pos_col].astype(str).isin(
                set(raw[home_name_col].astype(str)) | set(raw[away_name_col].astype(str))
            )
            use_id = pos_known & (pos_by_id.str.len() > 0)
            frame.loc[use_id, "posteam"] = pos_by_id[use_id]
            frame.loc[use_id & (def_by_id.str.len() > 0), "defteam"] = def_by_id[
                use_id & (def_by_id.str.len() > 0)
            ]
    if season_col:
        frame["season"] = pd.to_numeric(raw[season_col], errors="coerce")
    else:
        try:
            frame["season"] = int(path.stem.split("_")[-1])
        except ValueError:
            frame["season"] = 0
    if date_col:
        frame["date"] = pd.to_datetime(raw[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
    else:
        frame["date"] = ""
    if sr_col:
        frame["success"] = pd.to_numeric(raw[sr_col], errors="coerce").fillna(0.0)
    else:
        frame["success"] = 0.0

    # Garbage-time filter: pre-play win prob in [0.04, 0.96] when available,
    # else a margin/period heuristic. Blowout snaps carry no forward signal.
    if wp_col:
        wp = pd.to_numeric(raw[wp_col], errors="coerce")
        competitive = wp.isna() | ((wp >= 0.04) & (wp <= 0.96))
    else:
        competitive = pd.Series(True, index=raw.index)
    if period_col and diff_col:
        period = pd.to_numeric(raw[period_col], errors="coerce").fillna(0)
        diff = pd.to_numeric(raw[diff_col], errors="coerce").fillna(0).abs()
        competitive &= ~(((period >= 4) & (diff > 28)) | ((period >= 3) & (diff > 35)))
    frame["competitive"] = competitive.to_numpy()

    if type_col:
        ptype = raw[type_col].astype(str).str.lower()
        frame["is_pass"] = (ptype.str.contains("pass") | ptype.str.contains("sack")).to_numpy()
        frame["is_rush"] = ptype.str.contains("rush").to_numpy()
    else:
        frame["is_pass"] = False
        frame["is_rush"] = False

    mask = frame["posteam"].notna() & frame["epa"].notna() & (frame["posteam"] != "") & (frame["posteam"] != "nan")
    plays = frame.loc[mask & frame["competitive"]].copy()

    plays["epa_pass"] = plays["epa"].where(plays["is_pass"])
    plays["epa_rush"] = plays["epa"].where(plays["is_rush"])
    off = (
        plays.groupby(["game_id", "season", "date", "posteam"], as_index=False)
        .agg(
            epa_off=("epa", "mean"),
            sr_off=("success", "mean"),
            plays_off=("epa", "size"),
            epa_pass_off=("epa_pass", "mean"),
            epa_rush_off=("epa_rush", "mean"),
        )
        .rename(columns={"posteam": "team"})
    )
    if plays["defteam"].astype(str).str.len().gt(0).any():
        defense = (
            plays.loc[plays["defteam"].astype(str).str.len() > 0]
            .groupby(["game_id", "season", "date", "defteam"], as_index=False)
            .agg(epa_def=("epa", "mean"), sr_def=("success", "mean"), plays_def=("epa", "size"))
            .rename(columns={"defteam": "team"})
        )
        merged = off.merge(defense, on=["game_id", "season", "date", "team"], how="outer")
    else:
        merged = off.copy()
        merged["epa_def"] = np.nan
        merged["sr_def"] = np.nan
        merged["plays_def"] = 0
    for col in ("epa_off", "sr_off", "epa_def", "sr_def", "epa_pass_off", "epa_rush_off"):
        if col in merged.columns:
            merged[col] = merged[col].astype(float)
    merged["plays_off"] = merged.get("plays_off", pd.Series(0, index=merged.index)).fillna(0).astype(int)
    merged["plays_def"] = merged.get("plays_def", pd.Series(0, index=merged.index)).fillna(0).astype(int)
    return merged


def _adjusted_ratings(game_team: pd.DataFrame, *, ridge_lambda: float = 4.0) -> pd.DataFrame:
    """As-of opponent-adjusted EPA ratings via chronological ridge regression.

    For every game date D (per season) fit ``epa_off(t vs o) = off_t - def_o``
    on all games with date <= D of that season, ridge-shrunk toward the
    previous season's final ratings (decayed 0.6, new teams toward 0). Emits
    one row per team per played date with the rating *after* that date — a
    consumer joining rows with date strictly before kickoff sees exactly the
    games played before kickoff (PIT-safe). Season-start carryover rows
    (Aug 10) expose the decayed prior before a team's first game.
    """
    rows = game_team.loc[
        game_team["epa_off"].notna()
        & (game_team["team"].astype(str).str.len() > 0)
        & (game_team["date"].astype(str).str.len() >= 10)
    ].copy()
    # Opponent from the other row of the same game_id.
    by_game: dict[str, list[dict]] = {}
    for rec in rows.to_dict(orient="records"):
        by_game.setdefault(str(rec["game_id"]), []).append(rec)
    obs: list[tuple[int, str, str, str, float]] = []  # season, date, off, def, y
    for recs in by_game.values():
        if len(recs) != 2:
            continue
        a, b = recs
        obs.append((int(a["season"]), str(a["date"]), str(a["team"]), str(b["team"]), float(a["epa_off"])))
        obs.append((int(b["season"]), str(b["date"]), str(b["team"]), str(a["team"]), float(b["epa_off"])))
    if not obs:
        return pd.DataFrame(columns=["season", "date", "team", "adj_epa_off", "adj_epa_def", "n_fit"])
    obs.sort(key=lambda t: (t[0], t[1]))

    out_rows: list[dict] = []
    prior_off: dict[str, float] = {}
    prior_def: dict[str, float] = {}
    obs_frame = pd.DataFrame(obs, columns=["season", "date", "off", "def", "y"])
    for season, season_obs in obs_frame.groupby("season", sort=True):
        # Carryover rows so early-season consumers see the decayed prior.
        for team in set(prior_off) | set(prior_def):
            out_rows.append(
                {
                    "season": int(season),
                    "date": f"{int(season)}-08-10",
                    "team": team,
                    "adj_epa_off": 0.6 * prior_off.get(team, 0.0),
                    "adj_epa_def": 0.6 * prior_def.get(team, 0.0),
                    "n_fit": 0,
                }
            )
        p_off = {t: 0.6 * v for t, v in prior_off.items()}
        p_def = {t: 0.6 * v for t, v in prior_def.items()}
        seen: list[pd.DataFrame] = []
        teams_idx: dict[str, int] = {}

        def _idx(team: str) -> int:
            if team not in teams_idx:
                teams_idx[team] = len(teams_idx)
            return teams_idx[team]

        last_fit_off: dict[str, float] = {}
        last_fit_def: dict[str, float] = {}
        for day, day_obs in season_obs.groupby("date", sort=True):
            seen.append(day_obs)
            fit = pd.concat(seen, ignore_index=True)
            for team in pd.unique(pd.concat([fit["off"], fit["def"]], ignore_index=True)):
                _idx(str(team))
            n_teams = len(teams_idx)
            oi = fit["off"].map(teams_idx).to_numpy()
            di = fit["def"].map(teams_idx).to_numpy()
            prior_vec = np.zeros(2 * n_teams)
            inv = {v: k for k, v in teams_idx.items()}
            for j in range(n_teams):
                prior_vec[j] = p_off.get(inv[j], 0.0)
                prior_vec[n_teams + j] = p_def.get(inv[j], 0.0)
            # Residual vs prior; ridge toward 0 on the residual.
            y = fit["y"].to_numpy(dtype=float)
            y_res = y - (prior_vec[oi] - prior_vec[n_teams + di])
            n_obs = len(y_res)
            x = np.zeros((n_obs, 2 * n_teams))
            x[np.arange(n_obs), oi] = 1.0
            x[np.arange(n_obs), n_teams + di] = -1.0
            xtx = x.T @ x + ridge_lambda * np.eye(2 * n_teams)
            beta = np.linalg.solve(xtx, x.T @ y_res) + prior_vec
            games_per_team = fit.groupby("off").size()
            for team in pd.unique(pd.concat([day_obs["off"], day_obs["def"]], ignore_index=True)):
                team = str(team)
                j = teams_idx[team]
                last_fit_off[team] = float(beta[j])
                last_fit_def[team] = float(beta[n_teams + j])
                out_rows.append(
                    {
                        "season": int(season),
                        "date": str(day),
                        "team": team,
                        "adj_epa_off": float(beta[j]),
                        "adj_epa_def": float(beta[n_teams + j]),
                        "n_fit": int(games_per_team.get(team, 0)),
                    }
                )
        prior_off = last_fit_off
        prior_def = last_fit_def
    return pd.DataFrame(out_rows)


def build(
    *,
    start_season: int,
    end_season: int,
    force: bool = False,
    copy_supplemental: bool = False,
) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    parts: list[pd.DataFrame] = []
    for season in range(start_season, end_season + 1):
        try:
            path = _download_season(season, CACHE_DIR, force=force)
            part = _aggregate_season(path)
            parts.append(part)
            print(f"  season {season}: {len(part)} team-games", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  season {season} failed: {exc}", flush=True)
    if not parts:
        raise SystemExit("no CFB PBP seasons aggregated")
    out = pd.concat(parts, ignore_index=True)
    out_path = CACHE_DIR / "game_team_epa.csv"
    out.to_csv(out_path, index=False)
    print(f"wrote {out_path} rows={len(out)}", flush=True)
    adjusted = _adjusted_ratings(out)
    adj_path = CACHE_DIR / "team_adjusted_epa.csv"
    adjusted.to_csv(adj_path, index=False)
    print(f"wrote {adj_path} rows={len(adjusted)}", flush=True)
    if copy_supplemental:
        SUPP_DIR.mkdir(parents=True, exist_ok=True)
        supp = SUPP_DIR / "game_team_epa.csv"
        out.to_csv(supp, index=False)
        adjusted.to_csv(SUPP_DIR / "team_adjusted_epa.csv", index=False)
        (SUPP_DIR / "README.md").write_text(
            "# CFB PBP EPA\n\nBuilt by `scripts/build_cfb_pbp_epa.py` from cfbfastR "
            "sportsdataverse parquet releases. Joined as season-to-date priors only "
            "(never same-game EPA).\n",
            encoding="utf-8",
        )
        print(f"copied {supp}", flush=True)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-season", type=int, default=2022)
    parser.add_argument("--end-season", type=int, default=2025)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--copy-supplemental", action="store_true")
    args = parser.parse_args()
    build(
        start_season=args.start_season,
        end_season=args.end_season,
        force=args.force,
        copy_supplemental=args.copy_supplemental,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
