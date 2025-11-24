#!/usr/bin/env python3
"""
Unified CLI for running labeler regression, coordination, accuracy, and performance suites.
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_PY = sys.executable

SUITES = {
    "1": (
        "Coordination sanity checks",
        [DEFAULT_PY, "tests/test_coordination_detection.py"],
    ),
    "2": (
        "Accuracy evaluator",
        [DEFAULT_PY, "tests/evaluate_accuracy.py"],
    ),
    "3": (
        "Performance evaluator",
        [DEFAULT_PY, "tests/performance_test.py"],
    ),
}

RUN_ALL_KEY = "a"
QUIT_KEY = "q"


def _build_env():
    """Ensure PYTHONPATH includes repo root so imports succeed."""
    env = os.environ.copy()
    root_str = str(ROOT)
    existing = env.get("PYTHONPATH", "")
    if existing:
        paths = existing.split(os.pathsep)
        if root_str not in paths:
            env["PYTHONPATH"] = os.pathsep.join([root_str, existing])
    else:
        env["PYTHONPATH"] = root_str
    return env


def run(cmd):
    """Execute a command in the repo root and stream output."""
    print(f"\n>>> {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=ROOT, env=_build_env())
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
        return False
    return True


def prompt() -> str:
    """Render the interactive menu and capture the user's selection."""
    print("\nSelect a suite to run:")
    for key, (label, _) in SUITES.items():
        print(f"  {key}. {label}")
    print(f"  {RUN_ALL_KEY}. Run ALL suites sequentially")
    print(f"  {QUIT_KEY}. Quit")
    return input("Choice: ").strip().lower()


def main():
    while True:
        choice = prompt()

        if choice == QUIT_KEY:
            print("Exiting runner.")
            break

        if choice == RUN_ALL_KEY:
            for key in SUITES:
                label, cmd = SUITES[key]
                print(f"\n=== Running: {label} ===")
                if not run(cmd):
                    break
            continue

        if choice in SUITES:
            label, cmd = SUITES[choice]
            print(f"\n=== Running: {label} ===")
            run(cmd)
            continue

        print("Invalid selection. Please try again.")


if __name__ == "__main__":
    main()

