"""Assemble a report from already validated data; reusable for bounded slices."""
from dataclasses import asdict
from datetime import datetime, timezone

from .._version import __version__
from ..core import run_checks
from ..models import AuditReport
from .checks import default_retrieval_checks
from .data import RetrievalContext


def build_report(data, config, registry):
    builtin_types = {check.id: type(check) for check in default_retrieval_checks()}
    metadata = data.metadata()
    metadata["builtin_check_contracts"] = {check.id: "1" if type(check) is builtin_types.get(check.id) else None for check in registry}
    return AuditReport("1.0", __version__, datetime.now(timezone.utc).isoformat(),
        {"modality": "retrieval", **asdict(config)}, {"evaluation": metadata},
        run_checks(RetrievalContext(data, config), registry, config.disabled_checks))
