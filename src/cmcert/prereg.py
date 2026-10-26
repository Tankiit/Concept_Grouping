"""Hash-locking and verification for preregistered thresholds."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .errors import PreregistrationError


PLACEHOLDER = "???"


def digest(path: str | Path) -> str:
    """Return the SHA-256 digest of the exact config bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lock_path(thresholds_path: str | Path) -> Path:
    path = Path(thresholds_path)
    return path.with_name(f".{path.name}.sha256")


def ensure_complete(thresholds_path: str | Path) -> None:
    path = Path(thresholds_path)
    if PLACEHOLDER in path.read_text(encoding="utf-8"):
        raise PreregistrationError(f"{path} still contains {PLACEHOLDER!r}; preregistration is incomplete")


def write_lock(thresholds_path: str | Path) -> Path:
    """Create or replace the digest lock, but only for a complete config."""
    path = Path(thresholds_path)
    ensure_complete(path)
    destination = lock_path(path)
    destination.write_text(digest(path) + "\n", encoding="ascii")
    return destination


def verify_prereg(thresholds_path: str | Path = "configs/thresholds.yaml") -> None:
    """Refuse runs with placeholders, no lock, or a changed threshold file."""
    path = Path(thresholds_path)
    ensure_complete(path)
    destination = lock_path(path)
    if not destination.is_file():
        raise PreregistrationError(f"no preregistration lock at {destination}; run scripts/prereg_lock.py first")
    recorded = destination.read_text(encoding="ascii").strip()
    actual = digest(path)
    if recorded != actual:
        raise PreregistrationError("threshold hash differs from preregistration lock; relock before running")
