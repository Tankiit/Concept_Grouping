"""Comparator interface; all comparator choices must be preregistered."""


def evaluate_comparators(*args, **kwargs):
    raise NotImplementedError("TODO(comp:families): preregister comparator models and their tuning budget")
