from cmcert.prereg import verify_prereg


def begin(config: str) -> None:
    """Mandatory first operation for every experiment."""
    verify_prereg()
