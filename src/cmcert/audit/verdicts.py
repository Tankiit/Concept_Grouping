"""Durable verdict records, written before any figure-generation code runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Verdict:
    """A machine-produced per-concept certification decision."""

    concept: str
    read_path: bool
    write_path: bool | None
    statistic: dict[str, float]
    prereg_hash: str


def write_verdicts(verdicts: list[Verdict], path: str | Path) -> Path:
    """Atomically persist decisions before a caller can create a plot."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps([asdict(verdict) for verdict in verdicts], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return output
