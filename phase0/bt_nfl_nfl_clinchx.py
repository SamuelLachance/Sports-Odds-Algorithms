"""Breakthrough program (NFL) - candidate nfl_clinchx, one-command entry point. DEV ONLY.

Runs, in order: the tiebreak unit test, the MC flag build (resumable per season), the
future-perturbation leak test, then the DEV screen (harness sanity + arms A0/AR/A1/A2).
Builder:  phase0/bt_nfl_clinchx_build.py      Screen: phase0/bt_nfl_clinchx_screen.py
Results:  data/bt_nfl_clinchx.json  (identical copy: data/bt_nfl_nfl_clinchx.json)
Run from the repo root:  python phase0/bt_nfl_nfl_clinchx.py
"""
from __future__ import annotations

import os
import runpy
import subprocess
import sys

B = "phase0/bt_nfl_clinchx_build.py"
if not os.path.exists("data/bt_nfl_clinchx_unittest.json"):
    subprocess.run([sys.executable, B, "--unittest"], check=True)
if not os.path.exists("data/bt_nfl_clinchx_flags.npy"):
    subprocess.run([sys.executable, B, "--build"], check=True)
if not os.path.exists("data/bt_nfl_clinchx_perturb.json"):
    subprocess.run([sys.executable, B, "--perturb"], check=True)
runpy.run_path("phase0/bt_nfl_clinchx_screen.py", run_name="__main__")
