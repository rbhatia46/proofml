"""Explicit text comparison semantics and resource limits."""
from dataclasses import dataclass

from .._validation import disabled_ids, fraction, positive_integer


@dataclass(frozen=True)
class TextConfig:
    normalization: str = "unicode"
    near_duplicate_threshold: float = 0.8
    shingle_size: int = 3
    max_documents: int = 200_000
    max_bytes: int = 100_000_000
    max_tokens_per_document: int = 10_000
    max_candidate_visits: int = 100_000
    max_shingles: int = 500_000
    disabled_checks: tuple[str, ...] = ()

    def __post_init__(self):
        if self.normalization not in {"unicode", "exact"}:
            raise ValueError("normalization must be 'unicode' or 'exact'")
        fraction("near_duplicate_threshold", self.near_duplicate_threshold)
        for name in ("shingle_size", "max_documents", "max_bytes", "max_tokens_per_document", "max_candidate_visits", "max_shingles"):
            positive_integer(name, getattr(self, name))
        object.__setattr__(self, "disabled_checks", disabled_ids(self.disabled_checks))
