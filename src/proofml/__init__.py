"""Public API. No model, account, or network required."""
from .engine import audit
from .models import AuditReport, CheckResult, Coverage, Finding
from .config import AuditConfig
from ._version import __version__
from .exceptions import AuditFailed
from .text import audit_text, TextConfig
from .retrieval import audit_retrieval, RetrievalConfig
from .policy import AuditPolicy, GateResult
from .suite import AuditSuite, SuitePolicy, SuiteResult
from .retrieval.slices import audit_retrieval_slices

__all__ = ["audit", "audit_text", "audit_retrieval", "AuditConfig", "TextConfig", "RetrievalConfig",
           "AuditReport", "AuditFailed", "CheckResult", "Coverage", "Finding", "AuditPolicy", "GateResult",
           "AuditSuite", "SuitePolicy", "SuiteResult", "audit_retrieval_slices"]
