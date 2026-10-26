"""Dataset capabilities and exact full-factorial data-generating processes.

``FactorialDGP`` represents a table containing one rendered image for every
factor combination.  It deliberately exposes factor vectors separately from
images: :meth:`sample_correlated` produces observational factor assignments,
while :meth:`intervene` implements the corresponding exact ``do`` lookup.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from numbers import Integral
from typing import Any, Hashable, Sequence

import numpy as np


@dataclass(frozen=True)
class DatasetCard:
    """Static causal capabilities of a dataset loader."""

    name: str
    factors: tuple[str, ...]
    has_native_do: bool
    do_kind: str  # "render" | "index_lookup" | "image_swap" | "none"

    def __post_init__(self) -> None:
        valid_kinds = {"render", "index_lookup", "image_swap", "none"}
        if self.do_kind not in valid_kinds:
            raise ValueError(f"unknown do_kind {self.do_kind!r}; expected one of {sorted(valid_kinds)}")
        if self.has_native_do != (self.do_kind != "none"):
            raise ValueError("has_native_do must agree with do_kind")


CARDS = {
    "colored_mnist": DatasetCard("colored_mnist", ("parity", "color"), True, "render"),
    "shapes3d": DatasetCard(
        "shapes3d",
        ("floor_hue", "wall_hue", "object_hue", "scale", "shape", "orientation"),
        True,
        "index_lookup",
    ),
    "waterbirds": DatasetCard("waterbirds", ("bird", "background"), False, "none"),
}


class FactorialDGP:
    """Exact DGP backed by a complete full-factorial image table.

    Args:
        images: One image (or arbitrary payload) per row of ``factor_grid``.
        factor_grid: An ``(n_rows, n_factors)`` array-like table.  A structured
            NumPy array is also accepted, in which case its field names may be
            used for ``concept`` and ``correlate``.
        concept: Index or name of the factor to treat as the concept.
        correlate: Index or name of the factor coupled to ``concept`` during
            observational sampling.

    The mixture sampler is defined for factors with equally many levels.  With
    probability ``rho`` it assigns the correlate the same *level index* as the
    concept; otherwise it samples that level independently.  Other factors are
    uniform.  This gives an easily auditable categorical coupling and preserves
    uniform marginals.  ``intervene`` is unit-level: it replaces one coordinate
    and keeps every other coordinate unchanged.
    """

    def __init__(
        self,
        images: Sequence[Any],
        factor_grid: Sequence[Sequence[Hashable]] | np.ndarray,
        concept: int | str,
        correlate: int | str,
    ) -> None:
        grid, names = self._normalise_grid(factor_grid)
        if grid.ndim != 2 or grid.shape[1] == 0:
            raise ValueError("factor_grid must have shape (n_rows, n_factors) with at least one factor")
        if len(images) != len(grid):
            raise ValueError("images and factor_grid must contain the same number of rows")

        self.images = images
        self.factor_grid = grid
        self.factor_names = names
        self.concept = self._factor_index(concept)
        self.correlate = self._factor_index(correlate)
        if self.concept == self.correlate:
            raise ValueError("concept and correlate must refer to different factors")
        self._levels = tuple(tuple(dict.fromkeys(grid[:, j].tolist())) for j in range(grid.shape[1]))
        self._level_to_index = tuple({value: i for i, value in enumerate(levels)} for levels in self._levels)
        self._index = {tuple(row): i for i, row in enumerate(grid.tolist())}
        self.assert_factored()
        if len(self._levels[self.concept]) != len(self._levels[self.correlate]):
            raise ValueError("mixture coupling requires concept and correlate to have the same number of levels")

    @staticmethod
    def _normalise_grid(factor_grid: Sequence[Sequence[Hashable]] | np.ndarray) -> tuple[np.ndarray, tuple[str, ...] | None]:
        raw = np.asarray(factor_grid)
        if raw.dtype.names:
            names = tuple(raw.dtype.names)
            return np.asarray([[row[name] for name in names] for row in raw], dtype=object), names
        if raw.ndim != 2:
            raise ValueError("factor_grid must be a two-dimensional array or a structured array")
        return np.asarray(raw, dtype=object), None

    def _factor_index(self, factor: int | str) -> int:
        if isinstance(factor, Integral):
            index = int(factor)
            if not 0 <= index < self.factor_grid.shape[1]:
                raise IndexError(f"factor index {index} is out of range")
            return index
        if self.factor_names is None:
            raise TypeError("named factors require a structured factor_grid")
        try:
            return self.factor_names.index(factor)
        except ValueError as exc:
            raise KeyError(f"unknown factor {factor!r}; available: {self.factor_names}") from exc

    def sample_correlated(self, n: int, rho: float, rng: np.random.Generator | None = None) -> np.ndarray:
        """Draw ``n`` observational factor vectors using categorical mixture coupling."""
        if not isinstance(n, Integral) or n < 0:
            raise ValueError("n must be a non-negative integer")
        if not 0.0 <= rho <= 1.0:
            raise ValueError("rho must be in [0, 1]")
        rng = np.random.default_rng() if rng is None else rng
        result = np.empty((n, self.factor_grid.shape[1]), dtype=object)
        for j, levels in enumerate(self._levels):
            result[:, j] = rng.choice(levels, size=n)

        concept_indices = rng.integers(len(self._levels[self.concept]), size=n)
        correlate_indices = rng.integers(len(self._levels[self.correlate]), size=n)
        copy_mask = rng.random(n) < rho
        correlate_indices[copy_mask] = concept_indices[copy_mask]
        result[:, self.concept] = np.asarray(self._levels[self.concept], dtype=object)[concept_indices]
        result[:, self.correlate] = np.asarray(self._levels[self.correlate], dtype=object)[correlate_indices]
        return result

    def intervene(self, factor_vecs: Sequence[Sequence[Hashable]] | np.ndarray, j: int | str, value: Hashable) -> Any:
        """Apply a unit-level intervention and return the looked-up image(s).

        A single factor vector returns one image.  A matrix returns a list of
        images in the same order.  Missing factor combinations are impossible
        after :meth:`assert_factored`, but are still reported clearly if input
        values do not belong to this DGP.
        """
        j = self._factor_index(j)
        if value not in self._level_to_index[j]:
            raise ValueError(f"{value!r} is not a level of factor {j}")
        vectors = np.asarray(factor_vecs, dtype=object)
        one_vector = vectors.ndim == 1
        if one_vector:
            vectors = vectors[None, :]
        if vectors.ndim != 2 or vectors.shape[1] != self.factor_grid.shape[1]:
            raise ValueError(f"factor_vecs must have {self.factor_grid.shape[1]} coordinates per vector")
        changed = vectors.copy()
        changed[:, j] = value
        try:
            looked_up = [self.images[self._index[tuple(row)]] for row in changed.tolist()]
        except KeyError as exc:
            raise ValueError(f"factor vector {exc.args[0]!r} is not present in this DGP") from exc
        return looked_up[0] if one_vector else looked_up

    def assert_factored(self) -> None:
        """Raise if the table is not exactly one row per factor combination."""
        expected = int(np.prod([len(levels) for levels in self._levels]))
        unique_rows = len(self._index)
        if unique_rows != len(self.factor_grid):
            raise ValueError("factor_grid contains duplicate factor combinations")
        if len(self.factor_grid) != expected:
            raise ValueError(
                f"factor_grid is incomplete: found {len(self.factor_grid)} rows, expected {expected} "
                "for its observed factor levels"
            )
