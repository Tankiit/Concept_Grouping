"""Project-specific errors with useful, non-silent failure modes."""


class PreregistrationError(RuntimeError):
    """Raised when a run is not covered by the locked preregistration."""


class NativeDoUnavailableError(RuntimeError):
    """Raised when certification is requested for a non-intervenable dataset."""
