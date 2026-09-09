"""Audit search/RAG retrieval outputs without coupling to an agent framework."""
from dataclasses import asdict
from ..core import validate_checks
from ..models import AuditReport
from .checks import default_retrieval_checks
from .config import RetrievalConfig
from .data import RetrievalContext, load_retrieval
from .engine import build_report


def audit_retrieval(retrieved, relevant, *, corpus_ids=None, config: RetrievalConfig | None = None, checks=None, **options) -> AuditReport:
    """Evaluate ranked document/chunk IDs against known binary relevance IDs.

    Use matching query-ID mappings to align by key, or nested ordered iterables
    to align by position. No relevance is inferred. Corpus IDs are optional;
    supplying them enables referential-integrity checks. Thresholds are opt-in.
    """
    config = RetrievalConfig(**{**(asdict(config) if config is not None else {}), **options})
    registry = validate_checks(default_retrieval_checks() if checks is None else checks, config.disabled_checks)
    data = load_retrieval(retrieved, relevant, corpus_ids, config)
    return build_report(data, config, registry)


__all__ = ["audit_retrieval", "RetrievalConfig", "RetrievalContext", "default_retrieval_checks"]
