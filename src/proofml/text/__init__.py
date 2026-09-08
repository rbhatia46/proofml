"""Local text corpus audits without embeddings, model downloads, or API keys."""
from dataclasses import asdict
from datetime import datetime, timezone

from .._version import __version__
from ..core import run_checks, validate_checks
from ..models import AuditReport
from .checks import default_text_checks
from .config import TextConfig
from .data import TextContext, load_text


def audit_text(train, test=None, *, y=None, y_test=None, config: TextConfig | None = None, checks=None, **options) -> AuditReport:
    """Audit ordered iterables of strings, with optional single-class labels.

    Inputs are consumed once, bounded, and never modified. Labels attach by
    position (no pandas index alignment). Configure normalization explicitly for
    case-sensitive text/code. Custom checks replace the default registry.
    """
    if y_test is not None and (test is None or y is None):
        raise ValueError("y_test requires test documents and training labels y")
    config = TextConfig(**{**(asdict(config) if config is not None else {}), **options})
    registry = validate_checks(default_text_checks() if checks is None else checks, config.disabled_checks)
    train_data = load_text(train, y, config)
    test_data = load_text(test, y_test, config) if test is not None else None
    ctx = TextContext(train_data, test_data, config)
    datasets = {"train": train_data.metadata()}
    if test_data is not None:
        datasets["test"] = test_data.metadata()
    return AuditReport("1.0", __version__, datetime.now(timezone.utc).isoformat(),
        {"modality": "text", **asdict(config)}, datasets, run_checks(ctx, registry, config.disabled_checks))


__all__ = ["audit_text", "TextConfig", "TextContext", "default_text_checks"]
