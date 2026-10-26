#!/usr/bin/env python3
"""Lock complete preregistered thresholds before any experiment is run."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cmcert.prereg import PreregistrationError, write_lock  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--thresholds", default="configs/thresholds.yaml")
    args = parser.parse_args()
    try:
        print(write_lock(ROOT / args.thresholds))
    except PreregistrationError as exc:
        print(f"preregistration refused: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
