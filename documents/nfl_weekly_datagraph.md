# NFL weekly refresh — data-derivation graph

Synthesized 2026-07-30 from three recon passes (serve chain, feature walks, pull layer). All paths relative to `C:\Users\Admin\Projects\Sports-Odds-Algorithms`. Spot-verified against HEAD: `nfl_season_serve.py:76-77` (npy load + length assert), `:96` (repro assert), `:100` (2025 recency anchor), `nfl_site_db.py:28` (standings keep only `season == "2025"` REG rows with scores).

## 0. Exec/import chain (what actually runs)

```
phase0/nfl_season_serve.py:32-33   exec(nfl_coord_tune.py source up to "X_CUR0 = X_of(F)")        [lines 1-140]
phase0/nfl_site_data.py:15-16      exec(same nfl_coord_tune.py prelude)                            [identical chain]
phase0/nfl_coord_tune.py:20-21     exec(nfl_player_rating_system.py up to "# pass 1")              [lines 1-191]
phase0/nfl_coord_tune.py:31        exec(player_rating_system "# pass 1".."# pass 2" segment)       [lines 192-223, Z-tables]
phase0/nfl_player_rating_system.py:26-27  exec(nfl_big_test.py up to "# ---------- variants, DEV CV selection ----------")  [lines 1-211]
phase0/nfl_big_test.py:33-35       imports mlbwp.trueskill (TrueSkill1v1); phase0/nfl_elo.py (DEV_SCORE_FROM, DEV_YEARS, TEST_YEARS, run_elo, FRANCHISE); phase0/nfl_qb_elo.py (QbElo, load_games_qb, load_qb_weeks)
phase0/nfl_season_serve.py:193     from mlbwp.trueskill import TrueSkill1v1 (state walk — dead code since v7 col swap)
phase0/nfl_season_serve.py:455     from nfl_qb_elo import CLIP as QB_CLIP
phase0/nfl_site_data.py:56         from nfl_elo import run_elo (imported, unused)
phase0/nfl_site_db.py              no exec/imports from phase0 — stdlib csv/json only
```

Payload build order (`site/data/nfl.json`, each step read-modify-writes):
`nfl_site_data.py` (creates root: power/boards/mvp/model_card, `status:"preseason"`) → `nfl_site_db.py` (teams/players/divisions) → `nfl_trueskill_players.py:382-423` (swaps player ratings to participation-TS; also emits `nfl_player_ratings_2025.csv`, `nfl_ts_state.json`, `nfl_sal2026.json`) → `nfl_lineups.py` (teams[].lineup, live QB1s; auto-downloads depth charts) → `nfl_season_serve.py` (schedule/proj/model_card, `status:"season"`).

## 1. Data files in the serve chain: producer → source → incremental?

### 1a. Play-by-play reducers (scripted, resumable-by-season)

All share the template: source `https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{y}.parquet`; done-set = distinct `game_id[:4]` prefixes already in output; skip done seasons; `to_csv(mode="a")`. **Season-granular resume only: once any 2026 rows exist, season 2026 is "done" and never re-pulled. Mid-season weekly refresh requires deleting 2026 rows first (or a week-aware mode). All ranges exclude 2026 today.**

| Output | Producer | Range hardcode | Serve-chain consumer |
|---|---|---|---|
| data/nfl_plays.csv | phase0/nfl_pbp_pull.py | `range(1999, 2026)` :22 | nfl_big_test.py:97 (EPA agg), :135 |
| data/nfl_turnovers.csv | phase0/nfl_pbp_pull2.py | `range(1999, 2026)` :22 | nfl_big_test.py:187 (fumble luck) |
| data/nfl_duel_plays.csv | phase0/nfl_pbp_pull3.py | `range(1999, 2026)` :22 | nfl_season_serve.py:38, nfl_site_data.py:20 (pass/run channels) |
| data/nfl_play_ctx2.csv | phase0/nfl_play_ctx2_pull.py | `range(2016, 2026)` :19 | nfl_trueskill_players.py:92 |
| data/nfl_rapm_plays.csv | phase0/nfl_participation_pull.py (also `pbp_participation/pbp_participation_{y}.parquet` :11) | `range(2016, 2026)` :19 | nfl_trueskill_players.py:194 (feature-7 participation spine). **nflverse participation coverage for 2026 unverified** |

(Same-template scripts outside the serve chain, needing the same bumps if their consumers are rerun: nfl_pbp_pull4.py:17 → nfl_plays_wp.csv, nfl_pbp_pull5.py:17 → nfl_fgs.csv, nfl_play_ctx_pull.py:17 → nfl_play_ctx.csv, nfl_micro_pull.py:18 → nfl_play_prot.csv, nfl_st_pull.py, nfl_pen_pull.py.)

### 1b. Ad-hoc nflverse downloads — NO in-repo fetcher (weekly job must add fetch steps)

Runbook `documents/nfl_inseason_runbook.md` has URLs verified 2026-07-30.

| File | Source | Refresh mode | Serve-chain consumer |
|---|---|---|---|
| data/nfl_games.csv | `https://github.com/nflverse/nfldata/raw/master/data/games.csv` (runbook:12; habitatring mirror dead, runbook:17) | evergreen single file, full rewrite | the spine: nfl_elo.py:27, nfl_qb_elo.py:31,37, nfl_big_test.py:52, nfl_season_serve.py:307 (schedule), nfl_site_db.py:27, nfl_trueskill_players.py:112. Currently 7276 completed + 272 blank-score 2026 rows; walks filter `home_score != ""` |
| data/nfl_players.csv | nflverse `players` release | full replace, slowly changing | nfl_player_rating_system.py:47,51; nfl_trueskill_players.py:148,188,297; nfl_site_db.py:72 |
| data/nfl_player_stats.csv | nflverse legacy `player_stats` weekly offense (≤2024, `recent_team`/`sacks` schema) | static | nfl_qb_elo.py:52, nfl_player_rating_system.py:63 |
| data/nfl_player_stats_2025.csv | nflverse `stats_player_week_2025` (NEW schema: `team`/`sacks_suffered`) | static; **2026 needs a new `stats_player_week_2026` file + code branch** | nfl_qb_elo.py:53, nfl_player_rating_system.py:64,80, nfl_site_db.py:86 |
| data/nfl_player_stats_def.csv | nflverse legacy defense weekly | static | nfl_player_rating_system.py:80 |
| data/snap_2012.csv…snap_2025.csv | nflverse `snap_counts_{y}.csv` | one file/season; snap_2026.csv needed | nfl_player_rating_system.py:36-37 (`range(2013, 2026)` — 2012 unread), nfl_site_db.py:102 |
| data/inj_2009…inj_2025.csv | nflverse `injuries/injuries_{y}.csv` (runbook:14) | one file/season; inj_2026.csv appears ~week 1 | nfl_season_serve.py:361, nfl_lineups.py:51 (both try/except — dormant until file exists) |
| data/roster_2026.csv | nflverse weekly-roster (committed 4b9e570) | should be weekly refresh | nfl_season_serve.py:242 (re-homing), nfl_lineups.py:44. (`roster_weekly_2026.csv` is dead — no reader) |
| data/nfl_contracts.parquet | nflverse `contracts` (OTC) | occasional | nfl_trueskill_players.py:139 (cohort salary → Z_SAL). (`nfl_contracts.csv` is a 9-byte "Not Found" stub — dead) |
| data/nfl_draft_picks.csv | nflverse draft_picks | static | nfl_big_test.py:88 (QB pedigree rep_map) |

### 1c. Depth charts (self-refreshing)

| File | Producer | Mechanism |
|---|---|---|
| data/depth_charts_2026.csv | phase0/nfl_lineups.py:23-36 | `https://github.com/nflverse/nflverse-data/releases/download/depth_charts/depth_charts_2026.csv` (:24-25); atomic tmp download, >1MB size gate, graceful degrade to local copy (:29-36); whole-file replace every run; newest snapshot per team (:64-67). Year hardcoded in URL/filename |

### 1d. Derived artifacts (in-repo producers)

| File | Producer | Consumer | Status |
|---|---|---|---|
| data/nfl_ts_state.json | nfl_trueskill_players.py:411 (full deterministic re-walk of 2016-2025 plays each run) | nfl_season_serve.py:266, nfl_lineups.py:107 | end-2025 player TS state |
| data/nfl_sal2026.json | nfl_trueskill_players.py:413 (filter `s_ == 2026` :412) | nfl_season_serve.py:267, nfl_lineups.py:108 | 2026 cohort-salary z |
| data/nfl_player_ratings_2025.csv | nfl_trueskill_players.py:362 | nfl_site_db.py:64 | |
| data/nfl_v7_feature.npy | **ORPHANED** — writer only exists at commit `7dd13d0` (`nfl_ratings_only_model.py` old line 510, cfg7 = V4+V6+{sal_k:1.0, sal_mode:"cohort"}); HEAD writes `nfl_v8_feature.npy` (:953) instead | nfl_season_serve.py:76 with `assert len == len(games)` :77 (7276==7276 today) | **sharpest landmine**: breaks the moment a 2026 final lands in nfl_games.csv |
| data/nfl_qb2026.json | nfl_lineups.py (regenerated each run from the live QB1 resolution — roster/injury filtered + smart-flip; atomic write; prior file persists as fallback if <32 QB1s resolve) | nfl_season_serve.py:167,721 | RESOLVED 2026-07-30: no longer frozen; tracks every depth-chart refresh (07-30 regen confirmed all 32 QB1s of the 07-23 snapshot) |
| data/nfl_player_board_2025.csv | (upstream board build) | nfl_site_data.py:98, nfl_site_db.py:68 | |
| data/nfl_wowy.json | (upstream) | nfl_site_data.py:108 | |
| site/data/nfl.json | build chain in section 0 | site/index.html; tests/test_payload_contracts.py | read at nfl_season_serve.py:603, written :640 |
| data/nfl_season_2026.json | nfl_season_serve.py:644 | sidecar (coefficients/proj) | |

### 1e. Frozen param JSONs (no refresh needed)

nfl_elo_base.json (nfl_big_test.py:42), nfl_qb_elo.json (:43; also nfl_trueskill_players.py:177), nfl_qb_replacement.json (:44), nfl_epa_rating.json (:45 — reused by pass/run channels), nfl_ts_report.json (:46), nfl_qb_pedigree.json (:47), nfl_turnover.json (:48), nfl_snap_absence.json (nfl_player_rating_system.py:30).

## 2. The 14 features: inputs and season-boundary hardcodes

Column order = serve `Xs` order (`nfl_season_serve.py:338-351`); blend = walk-forward logistic, recency half-life 3.

| # | Feature | Built at | Inputs | Boundary logic (file:line) | 2026-append behavior |
|---|---|---|---|---|---|
| 0 | elo_logit | nfl_big_test.py:63-65 via run_elo (nfl_elo.py:43-62) | nfl_games.csv (filter `home_score != ""`, no cap) | in-walk regress at season change nfl_elo.py:49-51; serve manual 2026 boundary nfl_season_serve.py:122-123 (also nfl_site_data.py:72-73) | extends cleanly, but **double-regression** once walked 2026 games cross the in-walk boundary |
| 1 | qb_pedigree | nfl_big_test.py:71-93 (QbElo + rep_map from nfl_draft_picks.csv:88-92) | nfl_games.csv QB ids (nfl_qb_elo.py:28-43); weekly lines nfl_player_stats.csv + hardcoded _2025.csv (nfl_qb_elo.py:52-53); params qb_replacement/qb_pedigree/qb_elo.json | new_season() decay nfl_qb_elo.py:112-117 (trigger nfl_big_test.py:77-78); serve extra nfl_season_serve.py:136 | games extend but 2026 weekly QB lines INVISIBLE (file-list hardcode) — update() silently skips missing lines (nfl_qb_elo.py:105-107) → QB ratings freeze, no error |
| 2 | epa_net | nfl_big_test.py:96-131 | nfl_plays.csv; LG_EPA DEV-only :104-109 | season_decay :117-119; serve extra nfl_season_serve.py:155-156 | 2026 games walk but `agg.get(gid)` → None → silent no-update (:123-124) until pull range bumped + re-pulled |
| 3 | early_kick | nfl_big_test.py:165-174 | nfl_games.csv gametime (:56-59), static TZ table | per-game; `s < 2016` LA quirk :169 | extends cleanly |
| 4 | team_hfa | nfl_big_test.py:176-184 | elo probs + outcomes | **no boundary anywhere** (confirmed nfl_season_serve.py:11,164-169) | extends cleanly |
| 5 | rest_diff | nfl_big_test.py:163 (clip ±7) | nfl_games.csv home_rest/away_rest (:56) | per-game | extends cleanly |
| 6 | fumble_luck | nfl_big_test.py:186-210 | nfl_turnovers.csv; nfl_turnover.json | season_decay :194-197; serve extra nfl_season_serve.py:189-190 | same silent no-update as epa_net |
| 7 | ts_player_v7 | historical col = nfl_v7_feature.npy (nfl_season_serve.py:76-78, length assert :77); serve value = serve_v6() :268-284 (share-weighted `clip(mu-25, ±3) * 1/(1+sig2)` diff) from nfl_ts_state.json + nfl_sal2026.json | state: nfl_trueskill_players.py ← rapm_plays, play_ctx2, games (≥2016 :113), contracts.parquet (cap `min(2027, y0+span)` :155), load_qb_weeks, players.csv | in-walk offseason shrink toward salary target nfl_trueskill_players.py:214-218; serve hand-coded 2026 boundary nfl_season_serve.py:281-282 (mu shrink 2/3 toward `25 + 1*SAL26`, sigma widen +1.5², cap (25/3)²) — duplicated :383-384 in lineup-MC | **HARD BREAK**: any 2026 final grows `games` past 7276 → assert :77 fails. npy has no writer at HEAD. Also active-gates `gid.startswith("2025")` (serve :232-235, trueskill_players :304-306) keep crediting only 2025 actives |
| 8-10 | ol/def/skill_absence | nfl_coord_tune.py:53-115 (rq_and_absence, gate `s >= 2013` :63) | snap_2013..2025.csv (nfl_player_rating_system.py:36-43; same cap nfl_model_snap.py:26); nfl_snap_absence.json | continuous EWMA, no boundary | snap_2026.csv silently ignored until range bumped; missing snap table → features 0, share walk stops, no error. Serve sets cols 8-10 to 0 preseason by design (:347), filled only via injury MC (:352-412, inj_2026.csv Q-tags) |
| 11 | roster_quality | rate6 nfl_coord_tune.py:33-51 (DECAY .95 :30, EB prior 6, floor 5, clamp 1.5); player-week engine nfl_player_rating_system.py:111-221 | OFFW/DEFW: player_stats.csv + hardcoded _2025.csv + _def.csv (:63-64,80); crosswalk players.csv :47-53; snaps | Z-tables frozen to DEV ≤2015 (:202-221, stable); dakota proxy fit 2016-2024, applied ONLY `s == 2025` (:100-109); display gate ACTIVE25 `s == 2025` :391-392 | 2026 weekly stats invisible (file list) → apply_week never fires → silent freeze; even with a 2026 file, dakota stays 0 unless :108 generalized (nflverse dropped dakota post-2024) |
| 12-13 | pass_net / run_net | nfl_season_serve.py:37-75 (duplicated in main-test harnesses) | nfl_duel_plays.csv; epaP reused; LGP/LGR DEV-only :47-52 | season_decay :59-62; serve extra :160-162 (nfl_site_data.py has NO post-walk pass/run boundary — its power table is end-2025 states by design, comments :18,55) | same silent no-update until pull3 bumped + re-pulled |

## 3. End-2025 freeze points in nfl_season_serve.py an in-season mode must change

The freeze is mostly **implicit**: `load_games()` (nfl_elo.py:28-29) drops blank-score rows, so every walk ends at the last played game. The 272 2026 schedule rows are inert until a score lands.

**Invariant for the in-season rewrite: with zero played 2026 games, the new code path must be bit-identical to today's output** — every guard below must reduce to the current behavior when the last walked game is a 2025 game.

1. `:76-77` — v7 npy load + `assert len(v6_hist) == len(games)`. First loud failure on any 2026 final. Fix = regenerate/extend the npy (resurrect the 7dd13d0 writer with 2026-extended inputs, or an append convention reusing nfl_trueskill_players' walk with serve_v6 aggregation) — never bypass the assert silently.
2. `:96` — repro assert `abs(repro - 0.61947) < 0.0005` over TEST 2016-2025. Must keep passing: 2026 rows must not perturb the pre-2026 feature matrix (by design it trips if they do).
3. `:100` — serving-fit recency anchor hardcoded `0.5 ** ((2025 - seasons[tr]) / HL)`. Correct under the no-in-season-refit verdict (commit 7dd13d0: weekly refits tested significantly worse — frozen coefficients, only STATE walks advance). But if left as-is while 2026 rows enter `tr`, they'd get weight >1; the coefficient fit must keep excluding 2026 (train mask), not re-anchor.
4. **Post-walk manual 2026-boundary transforms** — each assumes the walk ended in 2025; once a 2026 game is walked, the in-loop trigger (`g["season"] != prev`) already fired → these double-regress. Guard each with "apply only if last walked game was 2025": Elo `:122-123`; QB Elo `:136`; EPA `:155-156`; pass/run `:160-162`; fumble luck `:189-190`; team-TS units `:208-210` (dead code — computed, never read since v7 swap); player-TS serve_v6 `:281-282` and duplicate `:383-384` (must also NOT re-apply once nfl_ts_state.json is rebuilt with 2026 snaps). Per-team HFA has no boundary anywhere (`:164-169`) — nothing to guard.
5. `:232-235` (also `:271,:289,:369`) — `active25` = players with snap gid startswith "2025" → must become current-season actives; snap loop cap nfl_player_rating_system.py:36 must extend.
6. `:307-308,:327` — schedule = `season=="2026"` rows, `assert len(sched) == 272` (playoff rows in January break this).
7. `:434-442` — serve-feature range sanity vs 2020+ historical distribution; re-examine once features extend.
8. `:459-460` — MC stat dists fit on `season >= 2020` (harmless); `:566` week-1 assert `abs(pmc - ph) < 0.005` holds even for played games (pmc computed pre-conditioning).
9. **ph is re-predicted, not frozen**: `:394` computes `probs = CLF.predict_proba(Xs)[:,1]` for all 272 games from preseason states (Q-lineup MC average :395-412, stored :415). Actual results never feed ph — a rerun with changed inj_2026/roster_2026/qb2026 inputs silently rewrites ph history for played games. `pmc` (:564-565) DOES condition on actual `hs/as` (:537-538; DYN cols 0,1,2,4,12,13 evolve in-sim :492; ts/luck/roster/absence/rest frozen :450-453). An honest refresh needs an explicit pregame-ph freeze: preserve prior payload ph for rows with `hs != None`, or a prediction ledger. Site display (site/index.html:354-355): games within 7 days show ph, farther show pmc; contract test tests/test_payload_contracts.py:33-40 requires `{"w","d","home","away","ph"}`.
10. Downstream: nfl_site_db.py:28 standings keep only `season == "2025"` — weekly 2026 refresh shows stale standings until bumped (:64,:68,:86,:102 all `*_2025` too; :87-88 `season_type != "REG": pass` filter is a no-op bug).

## 4. Proposed ordered weekly chain (pull → walk refresh → serve)

**One-time code touches before week 1** (local, reviewed): all range/file-list bumps in section 5; boundary guards from section 3.4; v7-npy extension path; ph-freeze mechanism; site_db 2026 bumps.

Then weekly, in order:

| Step | Action | CI-safe? |
|---|---|---|
| 1 | Fetch nfl_games.csv (full rewrite, runbook URL) | network-dependent; scriptable, retry-safe (idempotent full replace) — CI with network OK |
| 2 | Fetch inj_2026.csv, snap_2026.csv, stats_player_week_2026, roster_2026.csv, players.csv, contracts.parquet (full replaces) | same — but 2026 stats file needs the new-schema code branch first |
| 3 | Delete current-season rows from nfl_plays/turnovers/duel_plays/play_ctx2/rapm_plays csvs, then run nfl_pbp_pull.py, pull2, pull3, play_ctx2_pull, participation_pull (ranges bumped to 2027) | network + nflverse release lag; participation 2026 availability unverified → must degrade gracefully. Local-only until a week-aware re-pull mode exists (current done-set logic freezes a season after first pull) |
| 4 | Run nfl_trueskill_players.py — full deterministic re-walk; regenerates nfl_ts_state.json / nfl_sal2026.json / nfl_player_ratings_2025.csv (rename semantics for 2026) | deterministic given inputs — CI-safe; but writes into site/data/nfl.json, so must run in payload order |
| 5 | ~~Regenerate nfl_qb2026.json~~ DONE 2026-07-30: nfl_lineups.py now regenerates it each run from its live QB1 derivation (frozen-duplicate path removed; serve untouched) | complete |
| 6 | Payload chain in order: nfl_site_data.py → nfl_site_db.py → nfl_trueskill_players.py (payload section) → nfl_lineups.py (self-fetches depth charts — network) → nfl_season_serve.py | serve is deterministic; nfl_lineups needs network (has graceful local-copy degrade). Serve gates: repro assert :96, sched-count assert :327, range sanity :434-442 all fire loudly — good CI tripwires |
| 7 | Apply ph-freeze: restore prior-payload ph for played rows (or write pregame ledger) before final payload write | CI-safe once implemented |
| 8 | Run tests/test_payload_contracts.py (6 tests) as the build gate | CI-safe |

## 5. OPEN RISKS — hardcoded season literals that silently exclude 2026 (file:line)

**Hard failures (loud, by design):**
- nfl_season_serve.py:76-77 — frozen nfl_v7_feature.npy + length assert; no writer at HEAD (7dd13d0 only). Breaks on first 2026 final.
- nfl_season_serve.py:96 — repro assert 0.61947±0.0005; nfl_season_serve.py:327 — `len(sched) == 272` (playoffs break it).
- nfl_ratings_only_model.py:920 — repro assert 0.62973 (retune harness, if rerun).

**Silent 2026-excluders (the dangerous ones):**
- nfl_player_rating_system.py:36 and nfl_model_snap.py:26 — snap files `range(2013, 2026)`.
- nfl_player_rating_system.py:63-64,80 and nfl_qb_elo.py:52-53 — stats file lists end at `*_2025.csv` (missing weekly lines silently skipped, nfl_qb_elo.py:105-107 → ratings freeze).
- nfl_player_rating_system.py:103,108 — dakota proxy applied only `s == 2025`; :391-392 ACTIVE25 gate `s == 2025`.
- Pull ranges (all `range(..., 2026)`): nfl_pbp_pull.py:22, nfl_pbp_pull2.py:22, nfl_pbp_pull3.py:22, nfl_pbp_pull4.py:17, nfl_pbp_pull5.py:17, nfl_participation_pull.py:19, nfl_micro_pull.py:18, nfl_play_ctx_pull.py:17, nfl_play_ctx2_pull.py:19 — plus season-granular done-sets that freeze an in-progress season after first pull.
- nfl_trueskill_players.py:304-306 — active = `gid.startswith("2025")`; :412 — sal dump `s_ == 2026` (correct for this year, breaks 2027); :155 — contracts cap `min(2027, y0+span)`.
- nfl_ratings_only_model.py:107 — contracts cap `min(2026, ...)` (stale vs trueskill's 2027); :133 inj tuple (2022..2025); :145 `range(2016, 2026)` inj files.
- nfl_season_serve.py:100 — recency anchor 2025 (2026 rows would get weight >1); :122-123,136,155-156,160-162,189-190,208-210,281-282,383-384 — manual 2026 boundaries (double-apply risk); :232-235 (+271,289,369) — `gid.startswith("2025")` actives.
- nfl_site_db.py:28 — standings only `season == "2025"`; :64,:68 `*_2025.csv` ratings/board; :86 stats_2025; :102 snap_2025; :87-88 no-op season_type filter (bug).
- nfl_injury_eval.py:23-25 — `range(2009, 2026)` inj loop (FileNotFoundError-tolerant → inj_2026 silently skipped).
- nfl_lineups.py:23-25 depth_charts_2026 URL/filename year; :44 roster_2026.csv; :51 inj_2026.csv (dormant-until-exists is intended).
- data/nfl_qb2026.json — RESOLVED 2026-07-30: regenerated by nfl_lineups.py every run (was frozen/no-generator).
- Duplicated X14 rebuild blocks needing the same fixes if reused in-season: nfl_adjwin_main_test.py:67, nfl_clone_audit.py:66, nfl_ctx_main_tests.py:79, nfl_ctx_main_tests2.py:66, nfl_margin_head_test.py:75, nfl_oracle_test.py:74, nfl_st_test.py:69, nfl_v7_main_test.py:63.

**Data risks:** nflverse pbp_participation 2026 coverage unverified (feeds feature 7); data/nfl_contracts.csv is a 9-byte "Not Found" stub (dead — parquet is live); roster_weekly_2026.csv unreferenced (dead); stats_player_week_2026 schema branch not yet written.

**Harmless protocol constants (masks simply exclude 2026):** nfl_elo.py:20-22 (DEV_YEARS 1999-2016, DEV_SCORE_FROM 2001, TEST_YEARS 2016-2026); nfl_big_test.py:169 `s < 2016` LA-TZ; nfl_player_rating_system.py:202-221 Z-tables ≤2015; nfl_ratings_only_model.py:886,917,924,218; nfl_season_serve.py:435,460 (≥2020 windows); nfl_trueskill_players.py:113 (≥2016), :123 (sched-len pivot 2021), :179 (SD_QB 2016-2021).
