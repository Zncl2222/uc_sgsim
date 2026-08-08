class VariogramDoesNotCompute(Exception):
    """
    Raises when the variogram is not computed yet.
    """

    default_message = 'Please calculate the variogram first !'

    def __init__(self, message: str = default_message):
        self.message = message
        super().__init__(self.message)


class IterationError(Exception):
    """
    Raises when the continous failed reach the limit.
    """

    default_message = (
        'Maximum continuous failed iterations reached. '
        + 'Please change the combination of parameters. Or set '
        + 'higher iteration_limit in sgsim instance.'
    )

    def __init__(self, message: str = default_message):
        self.message = message
        super().__init__(self.message)


class NativeEngineError(RuntimeError):
    """Raised when the native engine reports an invalid or numerical state."""
