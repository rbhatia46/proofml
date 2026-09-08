"""Read exported retrieval IDs and judgments, never documents or credentials."""
from collections.abc import Mapping, Set
from dataclasses import dataclass
from hashlib import sha256
from itertools import zip_longest
import json

from .._validation import ordered_iterable
from .config import RetrievalConfig


@dataclass(frozen=True)
class RetrievalData:
    retrieved: tuple[tuple[str, ...], ...]
    relevant: tuple[frozenset[str], ...]
    corpus_ids: frozenset[str] | None
    fingerprint: str
    size_bytes: int

    def metadata(self):
        return {"rows": len(self.retrieved), "columns": ["retrieved", "relevant"],
                "sha256": self.fingerprint, "format": "retrieval", "size_bytes": self.size_bytes}


def load_retrieval(retrieved, relevant, corpus_ids, config: RetrievalConfig) -> RetrievalData:
    size = 0
    digest = sha256(b"proofml:retrieval:v1\n")

    def identifiers(values, limit, *, ranked):
        nonlocal size
        iterator = iter(values) if not ranked and isinstance(values, Set) else ordered_iterable(values, "document IDs")
        result = []
        for value in iterator:
            if len(result) >= limit:
                raise ValueError("Document ID input exceeds its configured count limit")
            if not isinstance(value, str) or not value.strip():
                raise TypeError("Document IDs must be nonempty strings; IDs are compared exactly")
            if len(value) > config.max_bytes:
                raise ValueError("Retrieval input exceeds max_bytes")
            size += len(json.dumps(value).encode("utf-8"))
            if size > config.max_bytes:
                raise ValueError("Retrieval input exceeds max_bytes; no sampling was performed")
            result.append(value)
        return tuple(result)

    if isinstance(retrieved, Mapping) or isinstance(relevant, Mapping):
        if not isinstance(retrieved, Mapping) or not isinstance(relevant, Mapping):
            raise TypeError("Use mappings for both retrieved and relevant, or ordered iterables for both")
        if len(retrieved) > config.max_queries or len(relevant) > config.max_queries:
            raise ValueError("Retrieval input exceeds max_queries")
        if retrieved.keys() != relevant.keys():
            raise ValueError("Retrieved and relevant mappings must contain identical query IDs")
        if any(not isinstance(key, str) or not key.strip() for key in retrieved):
            raise TypeError("Query IDs must be nonempty strings")
        triples = ((key, retrieved[key], relevant[key]) for key in retrieved)
    else:
        sentinel = object()
        pairs = zip_longest(ordered_iterable(retrieved, "retrieved"), ordered_iterable(relevant, "relevant"), fillvalue=sentinel)

        def positional():
            for rank, truth in pairs:
                if rank is sentinel or truth is sentinel:
                    raise ValueError("Retrieved and relevant query counts must match exactly")
                yield None, rank, truth

        triples = positional()
    rankings, judgments = [], []
    for query_id, rank, truth in triples:
        if len(rankings) >= config.max_queries:
            raise ValueError("Retrieval input exceeds max_queries; no sampling was performed")
        if query_id is not None and len(query_id) > config.max_bytes:
            raise ValueError("Retrieval input exceeds max_bytes")
        size += len(json.dumps(query_id).encode("utf-8"))
        if size > config.max_bytes:
            raise ValueError("Retrieval input exceeds max_bytes")
        ranking = identifiers(rank, config.max_ids_per_query, ranked=True)
        judgment = frozenset(identifiers(truth, config.max_ids_per_query, ranked=False))
        digest.update(json.dumps([query_id, ranking, sorted(judgment)], separators=(",", ":")).encode("utf-8") + b"\n")
        rankings.append(ranking)
        judgments.append(judgment)
    if not rankings:
        raise ValueError("Retrieval input must contain at least one query")
    corpus = frozenset(identifiers(corpus_ids, config.max_corpus_ids, ranked=False)) if corpus_ids is not None else None
    digest.update(json.dumps(sorted(corpus) if corpus is not None else None, separators=(",", ":")).encode("utf-8"))
    return RetrievalData(tuple(rankings), tuple(judgments), corpus, digest.hexdigest(), size)


@dataclass(frozen=True)
class RetrievalContext:
    data: RetrievalData
    config: RetrievalConfig
