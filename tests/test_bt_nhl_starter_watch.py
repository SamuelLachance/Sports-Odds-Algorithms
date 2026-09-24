"""Start-anchored goalie watcher (phase0/bt_nhl_starter_watch.py), offline.

Pins: no poll at or after the scheduled start; confirmation fires the serving
hook once and is re-checked; a swap after confirmation fires it again; the
archive stores only changed observations.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "phase0"))

import bt_nhl_starter_watch as w  # noqa: E402

START = dt.datetime(2026, 10, 1, 23, 0, tzinfo=dt.timezone.utc)


class Clock:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += dt.timedelta(seconds=s)


def side(starter, confirmed):
    return {"dressed_goalies": [["1", "A"], ["30", "B"]], "starter": starter,
            "starter_id": None, "confirmed": confirmed}


def make_fetch(clock, confirm_at_min, swap_at_min=None):
    calls = []

    def fetch(gid):
        mins = round((START - clock()).total_seconds() / 60, 1)
        calls.append(mins)
        ok = mins <= confirm_at_min
        home = ["30", "B"] if (swap_at_min is not None and mins <= swap_at_min) else ["1", "A"]
        return {"gid": gid, "t_fetch": clock().isoformat(), "min_to_start": mins,
                "state": "PRE", "ro_status": 200, "pre_game": mins > 0,
                "away": side(["35", "C"] if ok else None, ok),
                "home": side(home if ok else None, ok)}
    return fetch, calls


def test_polls_stop_before_start_and_confirm_once(tmp_path, monkeypatch):
    clock = Clock(START - dt.timedelta(minutes=90))
    fetch, calls = make_fetch(clock, confirm_at_min=9)
    hits = []
    monkeypatch.setattr(w, "on_confirmed", lambda rec: hits.append(rec["min_to_start"]))
    arch = tmp_path / "a.jsonl"
    done = w.watch([(1, START)], str(arch), clock=clock, sleeper=clock.sleep, fetch=fetch)
    assert all(m > 0 for m in calls), calls            # never at/after the start
    assert hits == [8.0]                               # first poll inside the bold window
    assert calls[calls.index(8.0):] == [8.0, 4.0, 1.0]  # thinned re-checks to the start
    assert 1 in done
    lines = [json.loads(x) for x in arch.read_text(encoding="utf-8").splitlines()]
    # unconfirmed observations are identical content -> archived once; confirmed once
    assert len(lines) == 2


def test_swap_after_confirmation_fires_hook_again(tmp_path, monkeypatch):
    clock = Clock(START - dt.timedelta(minutes=30))
    fetch, calls = make_fetch(clock, confirm_at_min=13, swap_at_min=5)
    hits = []
    monkeypatch.setattr(w, "on_confirmed", lambda rec: hits.append(
        (rec["min_to_start"], rec["home"]["starter"][0])))
    w.watch([(1, START)], str(tmp_path / "a.jsonl"), clock=clock, sleeper=clock.sleep,
            fetch=fetch)
    assert hits[0] == (13.0, "1")
    assert hits[-1][1] == "30"                          # the swap re-served
    assert all(m > 0 for m in calls)


def test_poll_plan_is_strictly_pre_start():
    now = START - dt.timedelta(minutes=5)
    plan = w.poll_plan(START, now)
    assert plan and all(now < t < START for t in plan)
    assert w.poll_plan(START, START) == []
