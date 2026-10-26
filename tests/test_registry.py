import numpy as np
import pytest

from cmcert.datasets import FactorialDGP


def make_dgp():
    grid = np.asarray([[concept, correlate] for concept in range(2) for correlate in range(2)])
    return FactorialDGP(["00", "01", "10", "11"], grid, concept=0, correlate=1)


def test_intervention_is_exact_unit_level_lookup():
    dgp = make_dgp()
    assert dgp.intervene([1, 1], 1, 0) == "10"
    assert dgp.intervene([[0, 0], [1, 0]], 1, 1) == ["01", "11"]


def test_mixture_sampler_has_perfect_coupling_at_one():
    samples = make_dgp().sample_correlated(100, rho=1.0, rng=np.random.default_rng(0))
    assert np.all(samples[:, 0] == samples[:, 1])


def test_rejects_incomplete_factorial_table():
    with pytest.raises(ValueError, match="incomplete"):
        FactorialDGP([0, 1, 2], [[0, 0], [0, 1], [1, 0]], 0, 1)
