"""Exceptions suitable for notebook, pipeline, and CI integrations."""


class AuditFailed(RuntimeError):
    """A report did not meet an explicitly requested quality/coverage gate.

    The report remains available as ``error.report`` for inspection or saving.
    """

    def __init__(self, message, report):
        super().__init__(message)
        self.report = report
