"""Formal read- and write-path certificate specification scaffold.

This is not the final estimator or theorem.  Bootstrap intervals are explicit
``TODO(spec)`` placeholders for valid finite-sample one-sided bounds.
"""
from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Mapping, Sequence


class Outcome(str, Enum):
    CERTIFIED = "certified"
    VIOLATION = "demonstrated_violation"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class CertificateSpec:
    """Everything preregistration must lock for one target factor."""
    name: str
    target_factor: str
    expected_affected: frozenset[str]
    expected_unaffected: frozenset[str]
    expected_direction: Mapping[str, int] = field(default_factory=dict)
    s_min: float = 0.5
    delta: float = 0.2
    leakage_estimand: str = "absolute_paired"
    write_path_min_share: float = 0.5
    family_size: int = 1
    alpha: float = 0.05
    n_boot: int = 2000
    seed: int = 0

    def direction(self, detector: str) -> int:
        return int(self.expected_direction.get(detector, 1))

    @property
    def alpha_sensitivity(self) -> float:
        return self.alpha / (self.family_size * 2)

    @property
    def alpha_leakage(self) -> float:
        return self.alpha / (self.family_size * 2)


@dataclass
class ImagePair:
    """Activations for baseline, target-edit, and optional nuisance-edit images."""
    baseline: Mapping[str, float]
    target_edit: Mapping[str, float]
    nuisance_edit: Mapping[str, float] | None = None


def sensitivity_scores(pairs: Sequence[ImagePair], spec: CertificateSpec) -> list[float]:
    """Signed target-edit response across detectors allowed to change."""
    result = []
    for pair in pairs:
        changes = [spec.direction(det) * (pair.target_edit[det] - pair.baseline[det])
                   for det in spec.expected_affected
                   if det in pair.target_edit and det in pair.baseline]
        if not changes:
            raise ValueError(f"no affected detectors present for {spec.name}")
        result.append(statistics.fmean(changes))
    return result


def leakage_scores(pairs: Sequence[ImagePair], spec: CertificateSpec) -> list[float] | None:
    """Nuisance-edit response across detectors expected to remain unchanged."""
    if any(pair.nuisance_edit is None for pair in pairs):
        return None
    result = []
    for pair in pairs:
        assert pair.nuisance_edit is not None
        signed = [pair.nuisance_edit[det] - pair.baseline[det]
                  for det in spec.expected_unaffected
                  if det in pair.nuisance_edit and det in pair.baseline]
        if not signed:
            raise ValueError(f"no unaffected detectors present for {spec.name}")
        if spec.leakage_estimand == "signed_paired":
            result.append(statistics.fmean(signed))
        elif spec.leakage_estimand == "absolute_paired":
            result.append(statistics.fmean([abs(change) for change in signed]))
        else:
            raise ValueError(f"unsupported leakage estimand: {spec.leakage_estimand}")
    return result


# TODO(spec): replace percentile bootstrap with valid finite-sample one-sided CIs.
def _bootstrap(values: Sequence[float], stat: Callable[[Sequence[float]], float], n_boot: int,
               rng: random.Random) -> list[float]:
    if not values:
        raise ValueError("at least one paired unit is required")
    draws = [stat([values[rng.randrange(len(values))] for _ in values]) for _ in range(n_boot)]
    return sorted(draws)


def lower_bound(values: Sequence[float], alpha: float, n_boot: int, rng: random.Random) -> float:
    draws = _bootstrap(values, statistics.fmean, n_boot, rng)
    return draws[max(0, min(len(draws) - 1, math.floor(alpha * len(draws))))]


def upper_bound(values: Sequence[float], alpha: float, n_boot: int, rng: random.Random) -> float:
    draws = _bootstrap(values, statistics.fmean, n_boot, rng)
    return draws[max(0, min(len(draws) - 1, math.ceil((1 - alpha) * len(draws)) - 1))]


@dataclass
class ConceptVerdict:
    name: str
    outcome: Outcome
    reason: str
    sensitivity_est: float | None = None
    sensitivity_lb: float | None = None
    leakage_est: float | None = None
    leakage_ub: float | None = None


def read_path_certificate(pairs: Sequence[ImagePair], spec: CertificateSpec,
                          rng: random.Random | None = None) -> ConceptVerdict:
    """Three-way read-path certificate with Bonferroni and union-bound budget."""
    rng = rng or random.Random(spec.seed)
    sensitivity = sensitivity_scores(pairs, spec)
    leakage = leakage_scores(pairs, spec)
    s_est = statistics.fmean(sensitivity)
    s_lb = lower_bound(sensitivity, spec.alpha_sensitivity, spec.n_boot, rng)
    if leakage is None:
        return ConceptVerdict(spec.name, Outcome.INCONCLUSIVE,
                              "no nuisance intervention: specificity not identified", s_est, s_lb)
    l_est = statistics.fmean(leakage)
    l_ub = upper_bound(leakage, spec.alpha_leakage, spec.n_boot, rng)
    s_ub = upper_bound(sensitivity, spec.alpha_sensitivity, spec.n_boot, rng)
    l_lb = lower_bound(leakage, spec.alpha_leakage, spec.n_boot, rng)
    if s_lb > spec.s_min and l_ub < spec.delta:
        outcome, reason = Outcome.CERTIFIED, "sensitivity LB > s_min and leakage UB < delta"
    elif s_ub < spec.s_min or l_lb > spec.delta:
        outcome, reason = Outcome.VIOLATION, "evidence against sensitivity or for leakage"
    else:
        outcome, reason = Outcome.INCONCLUSIVE, "bounds overlap margins; no decision"
    return ConceptVerdict(spec.name, outcome, reason, s_est, s_lb, l_est, l_ub)


@dataclass
class WritePathResult:
    outcome: Outcome
    share_est: float
    share_lb: float
    reason: str


def write_path_assessment(full_response: Sequence[float], named_reverted: Sequence[float],
                          spec: CertificateSpec, rng: random.Random | None = None) -> WritePathResult:
    """Assess mean [f(z') - f(z_j, z'_-j)] / [f(z') - f(z)].

    TODO(spec): lock the treatment of nonlinear heads, interactions, and
    near-zero full responses before a preregistered run.
    """
    if len(full_response) != len(named_reverted) or not full_response:
        raise ValueError("paired non-empty arrays required")
    shares = [reverted / full for full, reverted in zip(full_response, named_reverted)
              if abs(full) >= 1e-12]
    if not shares:
        return WritePathResult(Outcome.INCONCLUSIVE, float("nan"), float("nan"),
                               "no non-degenerate full response")
    rng = rng or random.Random(spec.seed)
    estimate = statistics.fmean(shares)
    lb = lower_bound(shares, spec.alpha / spec.family_size, spec.n_boot, rng)
    if lb > spec.write_path_min_share:
        return WritePathResult(Outcome.CERTIFIED, estimate, lb,
                               "named representation carries threshold share")
    return WritePathResult(Outcome.INCONCLUSIVE, estimate, lb,
                           "share below threshold or underpowered")
