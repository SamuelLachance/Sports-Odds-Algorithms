"""NFL weekly chain in CI (.github/workflows/nfl-weekly.yml).

Three pieces are pinned here:
  * phase0/nfl_ci_inputs.py - the inputs git does not carry (five play tables,
    frozen nflverse downloads) are listed, validated and rebuilt;
  * phase0/nfl_weekly.py - in CI, absent tables are a failure (never a green
    fetch-only run) and the outcome is written to a status file;
  * phase0/nfl_ci_publish.py - the serve is committed only when validated, and
    no concurrent writer can make origin pair one serve's payload with
    another's ledger (exercised against real temporary git repositories).
"""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "phase0")):
    if p not in sys.path:
        sys.path.insert(0, p)

import nfl_ci_inputs as I  # noqa: E402
import nfl_ci_publish as P  # noqa: E402
import nfl_weekly as W  # noqa: E402

HAVE_GIT = shutil.which("git") is not None


def _tracked() -> set[str]:
    r = subprocess.run(["git", "ls-files"], cwd=str(ROOT), capture_output=True,
                       text=True)
    return set(r.stdout.split())


# ------------------------------------------------------------ input layer
def test_tables_are_exactly_the_weekly_reducers():
    assert {(t.path, t.script) for t in I.TABLES} == set(W.REDUCERS)


def test_every_untracked_input_of_the_traced_run_is_cached():
    """The inputs a traced local run (2026-09-24) opened that git does not
    carry. Drop one from the cache and the CI chain dies on a FileNotFound."""
    traced = {"data/nfl_plays.csv", "data/nfl_turnovers.csv",
              "data/nfl_duel_plays.csv", "data/nfl_play_ctx2.csv",
              "data/nfl_rapm_plays.csv", "data/nfl_players.csv",
              "data/nfl_player_stats.csv", "data/nfl_player_stats_def.csv",
              "data/nfl_player_stats_2025.csv", "data/nfl_draft_picks.csv"}
    traced |= {f"data/snap_{y}.csv" for y in range(2013, 2026)}
    assert traced <= set(I.paths())


@pytest.mark.skipif(not HAVE_GIT, reason="git not available")
def test_the_cache_never_holds_a_tracked_file():
    """A cache restore overwrites the checkout: a tracked file in it (say
    snap_2026.csv) would roll a fresh commit back to a stale copy."""
    assert not set(I.paths()) & _tracked()
    for f in W.FETCHED:
        assert f not in I.paths()


def test_frozen_research_outputs_are_not_rebuilt():
    """nfl_wowy.json comes from a TEST-metric script: CI must never regenerate
    it, so it is checked for, not cached or fetched."""
    assert set(I.FROZEN) == {"data/nfl_player_board_2025.csv", "data/nfl_wowy.json"}
    assert not set(I.FROZEN) & set(I.paths())
    assert all("wowy" not in t.script and "board" not in t.script for t in I.TABLES)


def _table(tmp_path, rows, header=("game_id", "epa"), name="t.csv"):
    p = tmp_path / name
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    return I.Table(name, "phase0/x.py", tuple(header), I.LAST - 1, 2)


def _season_rows(y, n):
    return [[f"{y}_01_A_B", "0.1"]] * n


def test_a_complete_table_passes(tmp_path):
    t = _table(tmp_path, _season_rows(I.LAST - 1, 2) + _season_rows(I.LAST, 3)
               + _season_rows(I.CUR, 1))          # current season is not judged
    assert I.table_errors(t, tmp_path) == []


def test_a_short_or_missing_season_is_invalid(tmp_path):
    """A pull killed mid-append leaves a short season that the pull's
    season-granular done-set would then skip forever."""
    t = _table(tmp_path, _season_rows(I.LAST - 1, 2) + _season_rows(I.LAST, 1))
    assert any("short" in e for e in I.table_errors(t, tmp_path))
    t = _table(tmp_path, _season_rows(I.LAST, 5))
    assert any(str(I.LAST - 1) in e for e in I.table_errors(t, tmp_path))


def test_a_ragged_row_or_wrong_header_is_invalid(tmp_path):
    rows = _season_rows(I.LAST - 1, 2) + _season_rows(I.LAST, 2) + [["2025_x"]]
    t = _table(tmp_path, rows)
    assert any("ragged" in e for e in I.table_errors(t, tmp_path))
    t = _table(tmp_path, _season_rows(I.LAST - 1, 2) + _season_rows(I.LAST, 2))
    t = I.Table(t.path, t.script, ("game_id", "wrong"), t.first, t.min_rows)
    assert any("header" in e for e in I.table_errors(t, tmp_path))
    assert I.table_errors(I.Table("absent.csv", "x", ("a",), 2020, 1), tmp_path) \
        == ["absent.csv: missing"]


def test_static_file_checks(tmp_path):
    p = tmp_path / "s.csv"
    p.write_text("season,a\n2025,1\n2025,2\n", encoding="utf-8")
    ok = I.Static("s.csv", "u", ("a",), 2, ("2025",))
    assert I.static_errors(ok, tmp_path) == []
    assert I.static_errors(I.Static("s.csv", "u", ("b",), 1), tmp_path)
    assert I.static_errors(I.Static("s.csv", "u", ("a",), 3), tmp_path)
    assert I.static_errors(I.Static("s.csv", "u", ("a",), 1, ("2024",)), tmp_path)


def test_bootstrap_rebuilds_only_what_is_broken(tmp_path, monkeypatch):
    t_ok = I.Table("ok.csv", "phase0/ok.py", ("game_id", "epa"), I.LAST, 1)
    t_bad = I.Table("bad.csv", "phase0/bad.py", ("game_id", "epa"), I.LAST, 1)
    s_miss = I.Static("s.csv", "https://x/s.csv", ("a",), 1)
    good = f"game_id,epa\n{I.LAST}_01_A_B,0.1\n"
    (tmp_path / "ok.csv").write_text(good, encoding="utf-8")
    (tmp_path / "bad.csv").write_text("game_id,epa\n1999_01_A_B,0.1\n", encoding="utf-8")
    monkeypatch.setattr(I, "TABLES", (t_ok, t_bad))
    monkeypatch.setattr(I, "STATIC", (s_miss,))
    monkeypatch.setattr(I, "FROZEN", ())
    fetched, ran, existed = [], [], []

    def fake_fetch(path, url, required):
        fetched.append((path, url, required))
        (tmp_path / path).write_text("a\n1\n", encoding="utf-8")
        return "ok"

    def fake_run(script, tmo):
        ran.append(script)
        existed.append((tmp_path / "bad.csv").exists())
        (tmp_path / "bad.csv").write_text(good, encoding="utf-8")
        return True
    monkeypatch.chdir(tmp_path)
    assert I.bootstrap(tmp_path, fetch=fake_fetch, run=fake_run) == 0
    assert fetched == [("s.csv", "https://x/s.csv", True)]
    assert ran == ["phase0/bad.py"]
    assert existed == [False], "an invalid table is deleted before its full re-pull"
    # a second pass is a cache hit
    ran.clear()
    assert I.bootstrap(tmp_path, fetch=fake_fetch, run=fake_run) == 0 and ran == []


def test_bootstrap_fails_when_a_rebuild_does_not_take(tmp_path, monkeypatch):
    monkeypatch.setattr(I, "TABLES", (I.Table("t.csv", "phase0/t.py", ("game_id",), I.LAST, 1),))
    monkeypatch.setattr(I, "STATIC", ())
    monkeypatch.setattr(I, "FROZEN", ("frozen.json",))
    monkeypatch.chdir(tmp_path)
    assert I.bootstrap(tmp_path, fetch=lambda *a: "failed", run=lambda *a: False) == 1
    assert "frozen.json" in I.check(tmp_path, verbose=False)


def test_a_missing_committed_file_does_not_block_caching_good_tables(tmp_path, monkeypatch):
    monkeypatch.setattr(I, "TABLES", ())
    monkeypatch.setattr(I, "STATIC", ())
    monkeypatch.setattr(I, "FROZEN", ("frozen.json",))
    assert I.check(tmp_path, verbose=False)                      # the chain cannot run
    assert I.check(tmp_path, verbose=False, frozen=False) == {}  # the cache may be saved


def test_seed_round_trip_restores_only_the_cached_paths(tmp_path, monkeypatch):
    """The optional parity seed: pack the local inputs, unpack them in CI.
    A seed can never overwrite a tracked file or escape the checkout."""
    import tarfile
    src, dst = tmp_path / "src", tmp_path / "dst"
    (src / "data").mkdir(parents=True)
    t = I.Table("data/t.csv", "phase0/t.py", ("game_id", "epa"), I.LAST, 1)
    s = I.Static("data/s.csv", "u", ("a",), 1)
    (src / "data" / "t.csv").write_text(f"game_id,epa\n{I.LAST}_01_A_B,0.1\n", encoding="utf-8")
    (src / "data" / "s.csv").write_text("a\n1\n", encoding="utf-8")
    monkeypatch.setattr(I, "TABLES", (t,))
    monkeypatch.setattr(I, "STATIC", (s,))
    seed = tmp_path / "seed.tar.gz"
    assert I.pack(str(seed), src) == 0
    with tarfile.open(seed, "r:gz") as tf:
        assert sorted(tf.getnames()) == ["data/s.csv", "data/t.csv"]
    evil = tmp_path / "evil.tar.gz"                      # a hostile or stale seed
    with tarfile.open(evil, "w:gz") as tf:
        for name in ("data/t.csv", "site/data/nfl.json", "../escape.csv"):
            tf.add(str(src / "data" / "t.csv"), arcname=name)
    assert I.unpack(str(evil), dst) == 0
    assert (dst / "data" / "t.csv").exists()
    assert not (dst / "site").exists() and not (tmp_path / "escape.csv").exists()
    assert I.unpack(str(seed), dst) == 0
    assert I.check(dst, verbose=False, frozen=False) == {}
    # an incomplete input set is never packed
    (src / "data" / "s.csv").unlink()
    assert I.pack(str(tmp_path / "bad.tar.gz"), src) == 1
    assert not (tmp_path / "bad.tar.gz").exists()


# ------------------------------------------------------------ nfl_weekly
def test_ci_requires_the_chain(tmp_path, monkeypatch):
    """With the tables missing a local run is a green fetch-only run; in CI
    (NFL_WEEKLY_REQUIRE_CHAIN) it is a failure, so the site cannot silently
    stay on the last serve."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(W, "PROJECT", tmp_path)
    monkeypatch.setattr(W, "fetch", lambda p, u, r: "ok")
    monkeypatch.setattr(W, "REDUCERS", [("absent.csv", "x.py")])
    monkeypatch.delenv(W.REQUIRE_CHAIN_ENV, raising=False)
    assert W.main() == 0
    monkeypatch.setenv(W.REQUIRE_CHAIN_ENV, "1")
    monkeypatch.setenv(W.STATUS_ENV, str(tmp_path / "status.json"))
    assert W.main() == 1
    st = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert st["published"] is False and st["ok"] is False and st["failures"]


def test_status_says_published_only_after_a_validated_swap(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(W, "PROJECT", tmp_path)
    monkeypatch.setattr(W, "fetch", lambda p, u, r: "ok")
    (tmp_path / "table.csv").write_text("game_id\n", encoding="utf-8")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "nfl_games.csv").write_text(
        "game_id,home_score\n2025_01_A_B,20\n", encoding="utf-8")
    monkeypatch.setattr(W, "REDUCERS", [("table.csv", "x.py")])
    monkeypatch.setenv(W.STATUS_ENV, str(tmp_path / "status.json"))
    for built, published, rc in (([], True, 0), (["serve failed"], False, 1)):
        monkeypatch.setattr(W, "build_payload", lambda s, l, b=built: list(b))
        assert W.main() == rc
        st = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
        assert st["published"] is published


def test_publish_list_is_one_serve():
    assert W.PUBLISH[:2] == ["site/data/nfl.json", W.LEDGER_LIVE]
    assert set(W.FETCHED) <= set(W.PUBLISH)
    for f in ("data/nfl_season_2026.json", "data/nfl_v7_feature.npy",
              "data/nfl_qb2026.json"):
        assert f in W.PUBLISH
    assert "data/depth_charts_2026.csv" not in W.PUBLISH     # 53 MB, re-downloaded
    assert not {"site/data/nfl.json", W.LEDGER_LIVE} & set(P.plan({})[1])
    assert P.plan({"published": True}) == ("serve", list(W.PUBLISH))
    assert P.plan({"published": "yes"})[0] == "fetch"


@pytest.mark.skipif(not HAVE_GIT, reason="git not available")
def test_every_published_file_is_tracked():
    """`git add` of an ignored path fails the publish step outright."""
    assert set(W.PUBLISH) <= _tracked()


def test_step_timeouts_are_sized_not_hours():
    """Measured 2026-09-24: no chain step takes a minute locally."""
    for step, tmo in W.PAYLOAD_CHAIN:
        assert 300 <= tmo <= 1200, step
    assert W.PULL_TIMEOUT <= 1200 and W.FINAL_STEP[1] <= 600


def test_workflow_wiring():
    yml = (ROOT / ".github" / "workflows" / "nfl-weekly.yml").read_text(encoding="utf-8")
    assert '"0 9 * * 2"' in yml and '"0 20 * * 5"' in yml      # Tuesday + Friday
    assert "NFL_WEEKLY_REQUIRE_CHAIN" in yml and "NFL_WEEKLY_STATUS" in yml
    assert "phase0/nfl_ci_inputs.py bootstrap" in yml
    assert "phase0/nfl_ci_publish.py" in yml
    assert "actions/cache/restore" in yml and "actions/cache/save" in yml
    assert f"gh release download {I.SEED_TAG} --pattern {I.SEED_ASSET}" in yml
    assert "nfl_ci_inputs.py check --cached" in yml     # the cache-save gate
    for dep in ("numpy==", "pandas==", "pyarrow==", "scikit-learn==", "scipy=="):
        assert dep in yml
    # the publish step owns every git write: no raw pull/push left in the yml
    assert "git push" not in yml and "git pull" not in yml
    assert "timeout-minutes" in yml


# ------------------------------------------------------------ publish (git)
def _git(cwd, *args):
    r = subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t",
                        "-c", "core.autocrlf=false", *args], cwd=str(cwd),
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


def _write(repo, files: dict):
    for f, body in files.items():
        p = Path(repo) / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(body.encode("utf-8"))


BASE = {"site/data/nfl.json": '{"serve": "S0", "finals": 0}\n',
        "data/nfl_ph_ledger.json": '{"1|A|B": {"ph": 0.6}}\n',
        "data/nfl_games.csv": "game_id,home_score\n" + "".join(
            f"2026_{w:02d}_A_B,\n" for w in range(1, 9)),
        "data/nfl_player_ts.csv": "id,mu\n" + "".join(f"p{i},25.0\n" for i in range(8)),
        "data/nfl_season_2026.json": '{"s": 0}\n',
        "data/nfl_v7_feature.npy": "\x93NUMPY\x00\x01base\x00" * 50,   # binary (NULs)
        # tracked, rewritten by nfl_lineups every run, deliberately unpublished
        "data/depth_charts_2026.csv": "team,player\n" + "".join(
            f"T{i},P{i}\n" for i in range(8))}


@pytest.fixture
def repos(tmp_path):
    """A bare origin, the CI checkout and another writer, all at BASE."""
    if not HAVE_GIT:
        pytest.skip("git not available")
    origin = tmp_path / "origin.git"
    _git(tmp_path, "init", "-q", "--bare", "-b", "master", str(origin))
    seed = tmp_path / "seed"
    _git(tmp_path, "clone", "-q", str(origin), str(seed))
    _git(seed, "config", "core.autocrlf", "false")
    _write(seed, BASE)
    _git(seed, "add", "-A")
    _git(seed, "commit", "-q", "-m", "base")
    _git(seed, "push", "-q", "origin", "HEAD:refs/heads/master")
    clones = []
    _write(seed, {"README": "history before the checkout\n"})
    _git(seed, "add", "-A")
    _git(seed, "commit", "-q", "-m", "more history")
    _git(seed, "push", "-q", "origin", "HEAD:refs/heads/master")
    for name in ("ci", "other"):
        c = tmp_path / name
        # the CI clone is shallow, exactly like actions/checkout (fetch-depth 1)
        depth = ["--depth", "1"] if name == "ci" else []
        _git(tmp_path, "clone", "-q", *depth, "-b", "master", origin.as_uri(), str(c))
        _git(c, "config", "core.autocrlf", "false")
        clones.append(c)
    assert _git(clones[0], "rev-parse", "--is-shallow-repository") == "true"
    return origin, clones[0], clones[1], tmp_path


def _status(tmp_path, published: bool) -> str:
    p = tmp_path / "status.json"
    p.write_text(json.dumps({"published": published}), encoding="utf-8")
    return str(p)


def _origin(origin, f):
    return _git(origin, "show", f"master:{f}")


def _other_pushes(other, files: dict):
    _write(other, files)
    _git(other, "add", "-A")
    _git(other, "commit", "-q", "-m", "other writer")
    _git(other, "push", "-q", "origin", "HEAD:refs/heads/master")


def _ci_serve(ci, serve="S1", ledger=True):
    files = {"site/data/nfl.json": f'{{"serve": "{serve}", "finals": 1}}\n',
             "data/nfl_games.csv": BASE["data/nfl_games.csv"].replace(
                 "2026_01_A_B,", "2026_01_A_B,21"),
             "data/nfl_season_2026.json": f'{{"s": "{serve}"}}\n'}
    if ledger:
        files["data/nfl_ph_ledger.json"] = f'{{"1|A|B": {{"ph": 0.6}}, "2|A|B": {{"ph": "{serve}"}}}}\n'
    _write(ci, files)
    return files


def test_a_red_chain_never_commits_the_payload_or_ledger(repos):
    origin, ci, _, tmp = repos
    _ci_serve(ci)            # whatever the chain left in the working tree
    assert P.publish(ci, _status(tmp, False)) == 0
    assert _origin(origin, "site/data/nfl.json") == BASE["site/data/nfl.json"].strip()
    assert _origin(origin, "data/nfl_ph_ledger.json") == BASE["data/nfl_ph_ledger.json"].strip()
    assert "2026_01_A_B,21" in _origin(origin, "data/nfl_games.csv")
    # no status file at all (the chain crashed before writing it) = the same
    assert P.plan(P.load_status(str(tmp / "absent.json")))[0] == "fetch"


def test_a_validated_serve_ships_payload_and_ledger_together(repos):
    origin, ci, _, tmp = repos
    mine = _ci_serve(ci)
    assert P.publish(ci, _status(tmp, True)) == 0
    for f in ("site/data/nfl.json", "data/nfl_ph_ledger.json", "data/nfl_season_2026.json"):
        assert _origin(origin, f) == mine[f].strip()


def test_a_finals_attach_on_origin_does_not_revert_the_serve(repos):
    """refresh.yml attached finals to the OLD payload while the chain ran
    (payload only). The serve supersedes it, with its own ledger."""
    origin, ci, other, tmp = repos
    _other_pushes(other, {"site/data/nfl.json": '{"serve": "S0", "finals": 1}\n'})
    mine = _ci_serve(ci)
    assert P.publish(ci, _status(tmp, True)) == 0
    assert _origin(origin, "site/data/nfl.json") == mine["site/data/nfl.json"].strip()
    assert _origin(origin, "data/nfl_ph_ledger.json") == mine["data/nfl_ph_ledger.json"].strip()


def test_a_newer_serve_on_origin_is_never_overwritten(repos):
    """A local serve pushed payload + ledger while CI ran: its ledger may hold
    receipts for games that kicked off since. CI drops its own serve."""
    origin, ci, other, tmp = repos
    theirs = {"site/data/nfl.json": '{"serve": "LOCAL", "finals": 1}\n',
              "data/nfl_ph_ledger.json": '{"1|A|B": {"ph": 0.6}, "2|A|B": {"ph": "LOCAL"}}\n'}
    _other_pushes(other, theirs)
    _ci_serve(ci)
    assert P.publish(ci, _status(tmp, True)) == 0
    for f, body in theirs.items():
        assert _origin(origin, f) == body.strip()


def test_an_unchanged_ledger_cannot_pair_this_payload_with_anothers_ledger(repos):
    """The exact -X theirs hazard: this serve left the ledger as it was, so a
    rebase would keep ORIGIN's newer ledger under THIS older payload."""
    origin, ci, other, tmp = repos
    theirs = {"site/data/nfl.json": '{"serve": "LOCAL", "finals": 1}\n',
              "data/nfl_ph_ledger.json": '{"1|A|B": {"ph": 0.6}, "2|A|B": {"ph": "LOCAL"}}\n'}
    _other_pushes(other, theirs)
    _ci_serve(ci, ledger=False)
    assert P.publish(ci, _status(tmp, True)) == 0
    assert _origin(origin, "site/data/nfl.json") == theirs["site/data/nfl.json"].strip()
    assert _origin(origin, "data/nfl_ph_ledger.json") == theirs["data/nfl_ph_ledger.json"].strip()


def test_serve_files_are_never_line_merged(repos):
    """A derived CSV touched on both sides (different lines) would merge into a
    hybrid of two serves; the serve's own bytes are what ships."""
    origin, ci, other, tmp = repos
    ts = BASE["data/nfl_player_ts.csv"]
    _other_pushes(other, {"data/nfl_player_ts.csv": ts.replace("p0,25.0", "p0,99.0")})
    mine_ts = ts.replace("p7,25.0", "p7,31.0")
    _ci_serve(ci)
    _write(ci, {"data/nfl_player_ts.csv": mine_ts})
    assert P.publish(ci, _status(tmp, True)) == 0
    assert _origin(origin, "data/nfl_player_ts.csv") == mine_ts.strip()


def test_a_binary_serve_file_changed_on_both_sides_ships_this_serve(repos):
    """nfl_v7_feature.npy is binary: a both-sides change must resolve to this
    serve's bytes, not abort the rebase and lose the serve."""
    origin, ci, other, tmp = repos
    npy = "data/nfl_v7_feature.npy"
    (Path(other) / npy).write_bytes(b"\x93NUMPY\x00\x01theirs\x00" * 50)
    _git(other, "add", "-A")
    _git(other, "commit", "-q", "-m", "other npy")
    _git(other, "push", "-q", "origin", "HEAD:refs/heads/master")
    _ci_serve(ci)
    mine = b"\x93NUMPY\x00\x01mine\x00\x00" * 50
    (Path(ci) / npy).write_bytes(mine)
    assert P.publish(ci, _status(tmp, True)) == 0
    out = subprocess.run(["git", "show", f"master:{npy}"], cwd=str(origin),
                         capture_output=True).stdout
    assert out == mine


def test_fetched_tables_still_merge_with_a_newer_fetch(repos):
    """The spine is not part of one serve: a final another writer fetched on a
    different row survives (-X theirs only decides true conflicts)."""
    origin, ci, other, tmp = repos
    _other_pushes(other, {"data/nfl_games.csv": BASE["data/nfl_games.csv"].replace(
        "2026_08_A_B,", "2026_08_A_B,17")})
    _ci_serve(ci)
    assert P.publish(ci, _status(tmp, True)) == 0
    spine = _origin(origin, "data/nfl_games.csv")
    assert "2026_01_A_B,21" in spine and "2026_08_A_B,17" in spine


def test_nothing_to_publish_is_green(repos):
    _, ci, _, tmp = repos
    assert P.publish(ci, _status(tmp, True)) == 0


def test_the_publisher_refuses_to_run_outside_ci(monkeypatch):
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    assert P.main([]) == 2


def test_a_dirty_unpublished_tracked_file_cannot_cost_a_validated_serve(repos):
    """Reviewed 2026-09-24: nfl_lineups rewrites the tracked, unpublished
    depth chart on every run. With `rebase --autostash`, origin changing it
    too left the path unmerged and the serve's `commit --amend` failed - the
    validated serve was never pushed. It is now set aside before the rebase."""
    origin, ci, other, tmp = repos
    dc = "data/depth_charts_2026.csv"
    theirs = BASE[dc].replace("T0,P0", "T0,THEIRS")
    _other_pushes(other, {dc: theirs})
    mine = _ci_serve(ci)
    _write(ci, {dc: BASE[dc].replace("T0,P0", "T0,MINE")})    # dirty, not published
    assert P.publish(ci, _status(tmp, True)) == 0
    for f in ("site/data/nfl.json", "data/nfl_ph_ledger.json"):
        assert _origin(origin, f) == mine[f].strip()
    assert _origin(origin, dc) == theirs.strip()               # never published
    assert _git(ci, "status", "--porcelain", "--untracked-files=no") == ""


def test_a_dirty_unpublished_tracked_file_does_not_block_a_fetch_commit(repos):
    origin, ci, other, tmp = repos
    dc = "data/depth_charts_2026.csv"
    _other_pushes(other, {dc: BASE[dc].replace("T1,P1", "T1,THEIRS")})
    _ci_serve(ci)
    _write(ci, {dc: BASE[dc].replace("T1,P1", "T1,MINE")})
    assert P.publish(ci, _status(tmp, False)) == 0
    assert "2026_01_A_B,21" in _origin(origin, "data/nfl_games.csv")
    assert "T1,THEIRS" in _origin(origin, dc)
    assert _origin(origin, "site/data/nfl.json") == BASE["site/data/nfl.json"].strip()


def test_the_serve_commit_names_its_input_set(repos):
    origin, ci, _, tmp = repos
    _ci_serve(ci)
    I._set_origin(Path(ci), "nflverse")        # untracked marker, as in CI
    assert P.publish(ci, _status(tmp, True)) == 0
    assert _git(origin, "log", "-1", "--format=%s", "master") == \
        "NFL weekly serve (CI) [inputs: nflverse]"
    assert P.message("fetch", ci) == P.MSG["fetch"]


# ------------------------------------------- current-season re-pull (review)
CUR = W.CUR


def _spine(root, finals_on):
    """nfl_games.csv with one CUR final per date in finals_on + one unplayed."""
    rows = ["game_id,season,gameday,home_score,away_score"]
    rows += [f"{CUR}_{i + 1:02d}_A_B,{CUR},{d},20,17" for i, d in enumerate(finals_on)]
    rows.append(f"{CUR}_18_C_D,{CUR},{CUR}-12-30,,")
    (root / "data").mkdir(exist_ok=True)
    (root / "data" / "nfl_games.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _play_table(root, name, n_cur, n_old=3):
    p = root / name
    rows = ["game_id,epa"] + [f"2025_01_A_B,0.{i}" for i in range(n_old)]
    rows += [f"{CUR}_01_A_B,0.{i}" for i in range(n_cur)]
    p.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return p


@pytest.fixture
def weekly(tmp_path, monkeypatch):
    """A project with one reducer table and a settled season-CUR spine; the
    chain after the re-pull is replaced by a recorder."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(W, "PROJECT", tmp_path)
    monkeypatch.setattr(W, "fetch", lambda p, u, r: "ok")
    monkeypatch.setattr(W, "REDUCERS", [("plays.csv", "phase0/pull.py")])
    monkeypatch.setattr(W, "CUR_OPTIONAL", set())
    monkeypatch.setenv(W.STATUS_ENV, str(tmp_path / "status.json"))
    monkeypatch.delenv(W.INPUTS_ENV, raising=False)
    _spine(tmp_path, ["2000-09-10"])                       # long settled
    built = []
    monkeypatch.setattr(W, "build_payload", lambda s, l: built.append(1) or [])

    def status():
        return json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    return tmp_path, built, status


def _pull_that_appends(root, name, n):
    def run(script, tmo, args=(), env=None):
        with open(root / name, "a", encoding="utf-8") as fh:
            fh.writelines(f"{CUR}_02_A_B,0.{i}\n" for i in range(n))
        return True
    return run


def test_a_silent_repull_failure_is_red_and_restores_the_table(weekly, monkeypatch):
    """Reviewed 2026-09-24: the pull scripts print 'not available yet, skip'
    for a failed season download and exit 0. The run used to publish a serve
    with this season's plays gone (unplayed games moved up to 5.8 pp)."""
    root, built, status = weekly
    table = _play_table(root, "plays.csv", n_cur=40)
    before = table.read_bytes()
    monkeypatch.setattr(W, "run_step", lambda *a, **k: True)   # exit 0, no rows
    assert W.main() == 1
    assert built == [], "the payload chain must not run"
    st = status()
    assert st["published"] is False and "exited 0" in st["failures"][0]
    assert table.read_bytes() == before, "the pre-strip table is restored"
    assert not (root / ("plays.csv" + W.KEEP_SUFFIX)).exists()


def test_a_truncated_repull_is_red(weekly, monkeypatch):
    root, built, _ = weekly
    _play_table(root, "plays.csv", n_cur=40)
    monkeypatch.setattr(W, "run_step",
                        _pull_that_appends(root, "plays.csv", 40 - W.SHRINK_TOL - 1))
    assert W.main() == 1 and built == []


def test_a_crashed_repull_restores_the_table(weekly, monkeypatch):
    root, built, _ = weekly
    table = _play_table(root, "plays.csv", n_cur=40)
    before = table.read_bytes()

    def crash(script, tmo, args=(), env=None):
        with open(table, "a", encoding="utf-8") as fh:
            fh.write(f"{CUR}_02_A")                          # killed mid-append
        return False
    monkeypatch.setattr(W, "run_step", crash)
    assert W.main() == 1 and built == []
    assert table.read_bytes() == before


def test_a_good_repull_proceeds_to_the_chain(weekly, monkeypatch):
    root, built, status = weekly
    table = _play_table(root, "plays.csv", n_cur=40)
    monkeypatch.setattr(W, "run_step", _pull_that_appends(root, "plays.csv", 55))
    assert W.main() == 0 and built == [1]
    assert status()["published"] is True
    assert W.count_current(str(table)) == 55
    assert not (root / ("plays.csv" + W.KEEP_SUFFIX)).exists()
    # an upstream correction inside the tolerance is not a failure
    built.clear()
    monkeypatch.setattr(W, "run_step",
                        _pull_that_appends(root, "plays.csv", 55 - W.SHRINK_TOL))
    assert W.main() == 0 and built == [1]


def test_an_empty_pbp_table_after_settled_finals_is_red(weekly, monkeypatch):
    """No earlier count to compare against (first re-pull of the season, or a
    table cached empty): zero rows once finals are SETTLE_DAYS old is a
    failed download for a pbp-derived table..."""
    root, built, _ = weekly
    _play_table(root, "plays.csv", n_cur=0)
    monkeypatch.setattr(W, "run_step", lambda *a, **k: True)
    assert W.main() == 1 and built == []
    # ... but not for participation, which nflverse publishes after the season
    monkeypatch.setattr(W, "CUR_OPTIONAL", {"plays.csv"})
    assert W.main() == 0 and built == [1]


def test_an_empty_table_right_after_the_first_final_is_not_yet_a_failure(weekly, monkeypatch):
    """The night of the season's first game its pbp may not be published."""
    from datetime import datetime, timezone
    root, built, _ = weekly
    _spine(root, [datetime.now(timezone.utc).date().isoformat()])
    _play_table(root, "plays.csv", n_cur=0)
    monkeypatch.setattr(W, "run_step", lambda *a, **k: True)
    assert W.main() == 0 and built == [1]


def test_a_leftover_pre_strip_copy_is_restored_first(weekly, monkeypatch):
    """A run killed between strip and verification left the table stripped
    and its last verified state in the .prev copy."""
    root, built, _ = weekly
    good = _play_table(root, "plays.csv", n_cur=40).read_bytes()
    (root / ("plays.csv" + W.KEEP_SUFFIX)).write_bytes(good)
    _play_table(root, "plays.csv", n_cur=0)                   # stripped
    monkeypatch.setattr(W, "run_step", lambda *a, **k: True)  # silent failure
    assert W.main() == 1 and built == []
    assert (root / "plays.csv").read_bytes() == good


def test_season_state(tmp_path):
    from datetime import date
    today = date(int(CUR), 9, 24)
    _spine(tmp_path, [])
    spine = str(tmp_path / "data" / "nfl_games.csv")
    assert W.season_state(spine, today) == (False, False)
    _spine(tmp_path, [f"{CUR}-09-23"])
    assert W.season_state(spine, today) == (True, False)
    _spine(tmp_path, [f"{CUR}-09-22"])
    assert W.season_state(spine, today) == (True, True)


def test_repull_verification_covers_every_pbp_reducer():
    assert W.CUR_OPTIONAL <= {p for p, _ in W.REDUCERS}
    assert W.CUR_OPTIONAL == {"data/nfl_rapm_plays.csv"}
    assert 1 <= W.SETTLE_DAYS <= 3


# --------------------------------------------- a failed bootstrap (review)
def test_a_failed_input_bootstrap_still_fetches_but_never_runs_the_model(tmp_path, monkeypatch):
    """Reviewed 2026-09-24: with the frozen research files uncommitted the
    bootstrap step failed and the chain step - fetch layer included - was
    skipped, so CI stopped committing the fetched tables. The chain step now
    runs; NFL_WEEKLY_INPUTS turns the model into a recorded failure."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(W, "PROJECT", tmp_path)
    fetched, ran = [], []
    monkeypatch.setattr(W, "fetch", lambda p, u, r: fetched.append(p) or "ok")
    monkeypatch.setattr(W, "run_step", lambda *a, **k: ran.append(a) or True)
    monkeypatch.setattr(W, "build_payload", lambda s, l: ran.append("build") or [])
    monkeypatch.setenv(W.REQUIRE_CHAIN_ENV, "1")
    monkeypatch.setenv(W.STATUS_ENV, str(tmp_path / "status.json"))
    for outcome in ("failure", "skipped", ""):
        fetched.clear()
        monkeypatch.setenv(W.INPUTS_ENV, outcome)
        assert W.main() == 1
        assert fetched == W.FETCHED and ran == []
        st = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
        assert st["published"] is False and "inputs not ready" in st["failures"][0]


# --------------------------------------------- input provenance + cache key
def _mini_inputs(root, monkeypatch):
    (root / "data").mkdir(exist_ok=True)
    t = I.Table("data/t.csv", "phase0/t.py", ("game_id", "epa"), I.LAST, 1)
    s = I.Static("data/s.csv", "https://x/s.csv", ("a",), 1)
    monkeypatch.setattr(I, "TABLES", (t,))
    monkeypatch.setattr(I, "STATIC", (s,))
    monkeypatch.setattr(I, "FROZEN", ())
    return t, s


def test_the_cache_digest_follows_content_only(tmp_path, monkeypatch):
    _mini_inputs(tmp_path, monkeypatch)
    (tmp_path / "data" / "t.csv").write_text(f"game_id,epa\n{I.LAST}_01_A_B,0.1\n",
                                             encoding="utf-8")
    (tmp_path / "data" / "s.csv").write_text("a\n1\n", encoding="utf-8")
    I._set_origin(tmp_path, "seed")
    d0 = I.digest(tmp_path)
    assert I.digest(tmp_path) == d0 and len(d0) == 32
    I._set_origin(tmp_path, "seed")                  # rewritten, same bytes
    assert I.digest(tmp_path) == d0
    (tmp_path / "data" / "s.csv").write_text("a\n2\n", encoding="utf-8")
    assert I.digest(tmp_path) != d0
    assert I.ORIGIN in I.paths() and I.ORIGIN not in I.input_paths()


def test_input_provenance(tmp_path, monkeypatch, capsys):
    """seed (unpacked release) -> seed+nflverse once bootstrap had to top it
    up; a bootstrap without a seed is nflverse; a cache hit keeps the label.
    Every CI run on a non-seed set warns."""
    import tarfile
    t, s = _mini_inputs(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    good = f"game_id,epa\n{I.LAST}_01_A_B,0.1\n"

    def fetch(path, url, required):
        (tmp_path / path).write_text("a\n1\n", encoding="utf-8")
        return "ok"

    def run(script, tmo):
        (tmp_path / t.path).write_text(good, encoding="utf-8")
        return True
    assert I.origin(tmp_path) == "unknown"
    assert I.bootstrap(tmp_path, fetch=fetch, run=run) == 0
    assert I.origin(tmp_path) == "nflverse"
    assert I.bootstrap(tmp_path, fetch=fetch, run=run) == 0      # cache hit
    assert I.origin(tmp_path) == "nflverse"
    # a seed that carries only the table: bootstrap tops up the static file
    seed = tmp_path / "seed.tar.gz"
    with tarfile.open(seed, "w:gz") as tf:
        tf.add(str(tmp_path / t.path), arcname=t.path)
    (tmp_path / s.path).unlink()
    assert I.unpack(str(seed), tmp_path) == 0 and I.origin(tmp_path) == "seed"
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    capsys.readouterr()
    assert I.bootstrap(tmp_path, fetch=fetch, run=run) == 0
    assert I.origin(tmp_path) == "seed+nflverse"
    assert "::warning title=NFL inputs::input set 'seed+nflverse'" in capsys.readouterr().out
    I._set_origin(tmp_path, "seed")
    assert I.bootstrap(tmp_path, fetch=fetch, run=run) == 0
    assert "::warning" not in capsys.readouterr().out


def test_workflow_wiring_after_review():
    yaml = pytest.importorskip("yaml")
    wf = yaml.safe_load((ROOT / ".github" / "workflows" / "nfl-weekly.yml")
                        .read_text(encoding="utf-8"))
    steps = {s.get("name"): s for s in wf["jobs"]["weekly"]["steps"]}
    boot = steps["Bootstrap missing or invalid inputs"]
    chain = steps["Run weekly NFL chain"]
    assert boot.get("id") == "bootstrap"
    # the chain step (and its fetch layer) runs after a failed bootstrap ...
    assert "!cancelled()" in chain["if"]
    # ... and is told the bootstrap's outcome
    assert chain["env"][W.INPUTS_ENV] == "${{ steps.bootstrap.outcome }}"
    assert chain["env"][W.REQUIRE_CHAIN_ENV] == "1"
    # content-keyed cache, not saved again when unchanged
    assert "nfl_ci_inputs.py digest" in steps["Check inputs before caching"]["run"]
    save = steps["Save play tables + static inputs"]
    assert save["with"]["key"] == "${{ steps.cachecheck.outputs.key }}"
    assert "steps.restore.outputs.cache-matched-key" in save["if"]
    # every transitive dependency pinned too
    pip = steps["Install model deps"]["run"]
    for dep in ("python-dateutil==", "six==", "tzdata==", "joblib==",
                "threadpoolctl==", "narwhals=="):
        assert dep in pip
    src = (ROOT / "phase0" / "nfl_ci_publish.py").read_text(encoding="utf-8")
    assert '"rebase", "--autostash"' not in src
