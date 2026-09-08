"""Binary relevance evaluation policy, independent of retriever implementation."""
from dataclasses import dataclass

from .._validation import disabled_ids, fraction, positive_integer


@dataclass(frozen=True)
class RetrievalConfig:
    k: int = 10
    min_recall: float | None = None
    min_ndcg: float | None = None
    max_queries: int = 100_000
    max_ids_per_query: int = 10_000
    max_corpus_ids: int = 1_000_000
    max_bytes: int = 100_000_000
    disabled_checks: tuple[str, ...] = ()

    def __post_init__(self):
        for name in ("k", "max_queries", "max_ids_per_query", "max_corpus_ids", "max_bytes"):
            positive_integer(name, getattr(self, name))
        for name in ("min_recall", "min_ndcg"):
            if getattr(self, name) is not None:
                fraction(name, getattr(self, name), allow_zero=True)
        object.__setattr__(self, "disabled_checks", disabled_ids(self.disabled_checks))
