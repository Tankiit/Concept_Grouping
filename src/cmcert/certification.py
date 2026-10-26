"""Structural admission checks shared by all certification experiments."""

from __future__ import annotations

from .datasets.registry import CARDS
from .errors import NativeDoUnavailableError


def require_native_do(dataset: str) -> None:
    """Require an actual generator intervention, never a proxy claim."""
    try:
        card = CARDS[dataset]
    except KeyError as exc:
        raise KeyError(f"unknown dataset {dataset!r}; available: {sorted(CARDS)}") from exc
    if not card.has_native_do:
        raise NativeDoUnavailableError(
            f"{dataset} has has_native_do=False and cannot enter certification code"
        )
