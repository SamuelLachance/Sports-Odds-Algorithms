"""Commit and push the NFL weekly chain's outputs from CI without ever splitting
a serve (.github/workflows/nfl-weekly.yml, last step, `if: always()`).

What gets committed is decided by the status file phase0/nfl_weekly.py writes
(NFL_WEEKLY_STATUS):

  published  the chain swapped a VALIDATED payload + ledger onto the live
             files -> commit nfl_weekly.PUBLISH (payload, ledger, the small
             derived files that built them, the fetched tables) in ONE commit.
  otherwise  (a step failed, validation refused, the status file is missing)
             -> commit only the fetched tables. The live payload and ledger
             are never staged, so a red job cannot publish a gutted payload.

The race this guards (mirrors the NFL re-attach block in refresh.yml). Both
site/data/nfl.json and data/nfl_ph_ledger.json are single-line JSON, so under
`git pull --rebase -X theirs` any concurrent change to either is a whole-file
conflict that THIS commit wins - and a file this commit did not touch silently
takes origin's version. While the chain runs (~10 min), three writers can move
origin:
  * refresh.yml attaches finals to the published payload (payload only, never
    the ledger). This serve supersedes it: it attached the same finals from
    its own spine fetch. Safe to win.
  * odds.yml writes edges into the payload (payload only). Same; odds.yml
    recomputes edges on this serve on its next 20-minute cycle.
  * a local nfl_weekly run (or another serve) pushes payload + ledger. Its
    ledger may freeze games that kicked off after this run started. Winning
    the rebase would replace those receipts with this run's older numbers,
    and if this run's ledger happened not to change, the rebase would pair
    THIS payload with THAT ledger. So when origin's ledger differs from the
    checkout's, this serve is dropped (not pushed) with a warning: origin
    already carries a newer serve.
After the rebase, every serve file must be byte-identical to what the chain
built and validated, or nothing is pushed and the step fails. Tracked files
the chain rewrote but this commit does not carry (data/depth_charts_2026.csv)
are stashed before the rebase, never autostashed across it (set_aside).

  python phase0/nfl_ci_publish.py --status data/nfl_weekly_status.json

CI-only: it rebases and pushes. Outside GitHub Actions it refuses unless
--allow-local is given (tests). Standard library only. Market-blind.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "phase0"))

import nfl_ci_inputs as I  # noqa: E402  (origin: which input set built a serve)
import nfl_weekly as W  # noqa: E402  (PUBLISH, FETCHED, LEDGER_LIVE)

PAYLOAD = "site/data/nfl.json"
LEDGER = W.LEDGER_LIVE
# the files one serve owns; FETCHED tables may line-merge with a newer fetch
SERVE_FILES = [f for f in W.PUBLISH if f not in W.FETCHED]
BOT = ("glassbox-bot", "glassbox-bot@users.noreply.github.com")
MSG = {"serve": "NFL weekly serve (CI)",
       "fetch": "NFL weekly data refresh (chain did not publish)"}


class GitError(RuntimeError):
    pass


def git(cwd, *args, check: bool = True) -> subprocess.CompletedProcess:
    r = subprocess.run(["git", "-c", f"user.name={BOT[0]}",
                        "-c", f"user.email={BOT[1]}", *args],
                       cwd=str(cwd), capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if check and r.returncode != 0:
        raise GitError(f"git {' '.join(args)} -> {r.returncode}: "
                       f"{(r.stderr or r.stdout).strip()[:400]}")
    return r


def _say(msg: str, level: str | None = None) -> None:
    print(f"[nfl_ci_publish] {msg}", flush=True)
    if level and os.environ.get("GITHUB_ACTIONS"):
        print(f"::{level} title=NFL weekly publish::{msg}", flush=True)


def load_status(path: str | None) -> dict:
    if not path:
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            st = json.load(fh)
        return st if isinstance(st, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def plan(status: dict) -> tuple[str, list[str]]:
    """('serve', PUBLISH) only when the chain says it published a validated
    payload; ('fetch', FETCHED) for anything else, including no status."""
    if status.get("published") is True:
        return "serve", list(W.PUBLISH)
    return "fetch", list(W.FETCHED)


def message(mode: str, cwd=PROJECT) -> str:
    """The commit message; a serve names the input set that built it, so the
    history shows which serves came from the owner's seed and which from a
    fresh nflverse bootstrap (they differ by up to ~3 pp on unplayed games)."""
    if mode != "serve":
        return MSG[mode]
    return f"{MSG[mode]} [inputs: {I.origin(Path(cwd))}]"


def set_aside(cwd) -> None:
    """Stash the tracked changes this commit does not publish before rebasing.

    The chain rewrites tracked files that are deliberately not published (the
    53 MB data/depth_charts_2026.csv on every run; derived files after a red
    chain). They used to ride along via `rebase --autostash`, but when origin
    changed such a file too, re-applying the autostash conflicted: the rebase
    still exited 0, the path was left unmerged, and the serve's `commit
    --amend` then failed - a validated serve lost over a file it never
    published. The runner is discarded after the job, so nothing needs them
    back; `git stash list` keeps them for a local --allow-local run."""
    if git(cwd, "diff", "--quiet", check=False).returncode == 0 and \
            git(cwd, "diff", "--cached", "--quiet", check=False).returncode == 0:
        return
    names = git(cwd, "diff", "HEAD", "--name-only").stdout.split()
    git(cwd, "stash", "push", "-q", "-m",
        "nfl_ci_publish: tracked changes this run did not publish")
    _say(f"set aside {len(names)} unpublished tracked file(s) (git stash): "
         f"{' '.join(names[:6])}{' ...' if len(names) > 6 else ''}")


def ledger_moved(cwd, base: str, ref: str) -> bool:
    """origin's pre-game ledger differs from the one this run started from:
    another serve landed while the chain was running."""
    return git(cwd, "diff", "--quiet", base, ref, "--", LEDGER,
               check=False).returncode != 0


def publish(cwd=PROJECT, status_path: str | None = None, remote: str = "origin",
            branch: str = "master", attempts: int = 3) -> int:
    """0 = pushed, nothing to push, or superseded by a newer serve on origin;
    1 = a rebase or verification failure, or every push attempt lost a race."""
    cwd = Path(cwd)
    mode, files = plan(load_status(status_path))
    serve_files = [f for f in SERVE_FILES if f in files]
    present = [f for f in files if (cwd / f).exists()]
    base = git(cwd, "rev-parse", "HEAD").stdout.strip()
    if present:
        git(cwd, "add", "--", *present)
    if git(cwd, "diff", "--cached", "--quiet", check=False).returncode == 0:
        _say(f"{mode}: no data change")
        return 0
    staged = git(cwd, "diff", "--cached", "--name-only").stdout.split()
    msg = message(mode, cwd)
    git(cwd, "commit", "-q", "-m", msg)
    mine = git(cwd, "rev-parse", "HEAD").stdout.strip()
    _say(f"{mode}: committed {len(staged)} file(s): {' '.join(staged)}")
    set_aside(cwd)
    ref = f"refs/remotes/{remote}/{branch}"
    for attempt in range(1, attempts + 1):
        git(cwd, "fetch", "--quiet", remote, f"+refs/heads/{branch}:{ref}")
        if mode == "serve" and ledger_moved(cwd, base, ref):
            _say("origin's pre-game ledger changed while the chain ran: a newer "
                 "serve is already published. This run's serve is NOT pushed "
                 "(pushing it would overwrite that serve's receipts or pair this "
                 "payload with its ledger).", "warning")
            return 0
        r = git(cwd, "rebase", "-X", "theirs", ref, check=False)
        if r.returncode != 0:
            git(cwd, "rebase", "--abort", check=False)
            _say(f"rebase onto {remote}/{branch} failed, nothing pushed: "
                 f"{(r.stderr or r.stdout).strip()[:300]}", "error")
            return 1
        # a serve is one unit: take this run's bytes for every serve file (a
        # line-level merge of a derived CSV would build a hybrid of two serves)
        keep = [f for f in serve_files
                if git(cwd, "cat-file", "-e", f"{mine}:{f}",
                       check=False).returncode == 0]
        if keep:
            git(cwd, "checkout", mine, "--", *keep)
            if git(cwd, "diff", "--cached", "--quiet", check=False).returncode != 0:
                head = git(cwd, "rev-parse", "HEAD").stdout.strip()
                if head == git(cwd, "rev-parse", ref).stdout.strip():
                    git(cwd, "commit", "-q", "-m", msg)   # rebase emptied ours
                else:
                    git(cwd, "commit", "-q", "--amend", "--no-edit")
            if git(cwd, "diff", "--quiet", mine, "HEAD", "--", *keep,
                   check=False).returncode != 0:
                _say("after the rebase the serve files differ from the "
                     "validated build; nothing pushed", "error")
                return 1
        p = git(cwd, "push", remote, f"HEAD:refs/heads/{branch}", check=False)
        if p.returncode == 0:
            _say(f"{mode}: pushed to {remote}/{branch} (attempt {attempt})")
            return 0
        _say(f"push rejected (attempt {attempt}/{attempts}), refetching: "
             f"{(p.stderr or '').strip()[:200]}")
    _say(f"push failed after {attempts} attempts; this run's {mode} is lost "
         f"(the next scheduled run rebuilds it)", "error")
    return 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--status", default=os.environ.get(W.STATUS_ENV))
    ap.add_argument("--remote", default="origin")
    ap.add_argument("--branch", default=os.environ.get("GITHUB_REF_NAME") or "master")
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--allow-local", action="store_true",
                    help="run outside GitHub Actions (it rebases and pushes)")
    a = ap.parse_args(argv)
    if not os.environ.get("GITHUB_ACTIONS") and not a.allow_local:
        _say("refusing to run outside GitHub Actions (it rebases and pushes); "
             "locally, commit the files nfl_weekly.py prints")
        return 2
    try:
        return publish(PROJECT, a.status, a.remote, a.branch, a.attempts)
    except GitError as ex:
        _say(str(ex), "error")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
