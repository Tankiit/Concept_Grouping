"""Colored-MNIST DGP contract.

This file intentionally contains no unregistered coupling or edge-severing
choice.  It becomes executable only after the corresponding preregistration
decision has been fixed.
"""


def sample_correlated(*args, **kwargs):
    raise NotImplementedError(
        "TODO(dgp): preregister the rho coupling and its marginal-preservation properties"
    )


def intervene(*args, **kwargs):
    raise NotImplementedError(
        "TODO(dgp): preregister whether do() is unit-level or population-level and how its edge is severed"
    )


def assert_factored(*args, **kwargs):
    raise NotImplementedError("TODO(assert): specify Colored-MNIST factor coverage and uniqueness checks")
