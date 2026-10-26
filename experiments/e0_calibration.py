from ._common import begin


def choose_mixture(*args, **kwargs):
    raise NotImplementedError("TODO(mix): preregister mixture grid, sampling allocation, and interpolation rule")


def run(config: str) -> None:
    begin(config)
    raise NotImplementedError("TODO(oracle): preregister oracle reader and calibration target")
