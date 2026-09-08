"""Expose deterministic evidence to an agent without requiring an LLM here.

Register `audit_training_data` as a tool in your chosen framework. The host
agent controls path access and human interaction; ProofML supplies evidence.
The function also works from ordinary Python, with no credentials.
"""
from proofml import AuditConfig, audit


def audit_training_data(train_path: str, test_path: str, target: str) -> dict:
    """Return structured findings and coverage; do not treat findings as fixes."""
    return audit(train_path, test=test_path, config=AuditConfig(target=target)).to_dict()
