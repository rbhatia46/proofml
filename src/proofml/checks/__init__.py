"""Explicit registry. Plugins are passed by the caller, never auto-imported."""
from typing import Protocol
from ..context import AuditContext
from ..models import CheckResult
from .quality import QualityCheck, TargetCheck
from .splits import SchemaCheck, OverlapCheck, EntityCheck, TemporalCheck
from .leakage import AvailabilityCheck, AssociationCheck
from .drift import DriftCheck
from .forecasting import ForecastIndexCheck, ForecastCadenceCheck, ForecastBoundaryCheck


class Check(Protocol):
    id: str

    def run(self, ctx: AuditContext) -> CheckResult: ...


def default_checks(task: str = "classification") -> tuple[Check, ...]:
    common = (QualityCheck(), TargetCheck(), SchemaCheck(), OverlapCheck(), EntityCheck(),
            TemporalCheck(), AvailabilityCheck(), AssociationCheck(), DriftCheck())
    if task == "forecasting":
        return (*common, ForecastIndexCheck(), ForecastCadenceCheck(), ForecastBoundaryCheck())
    return common
