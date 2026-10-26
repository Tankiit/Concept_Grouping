#!/usr/bin/env python3
"""Dispatch preregistered experiments."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=("e0", "e1", "e2", "e5"))
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    module = importlib.import_module(f"experiments.{args.experiment}_" + {
        "e0": "calibration", "e1": "gate", "e2": "comparators", "e5": "write_path"
    }[args.experiment])
    module.run(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
