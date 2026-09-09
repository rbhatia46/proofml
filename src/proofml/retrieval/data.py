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
    evaluation_fingerprint: str
    output_fingerprint: str
    corpus_fingerprint: str
    alignment: str
    # IDs are retained only in the private input context to select slices. They
    # are deliberately absent from metadata(), report JSON, and HTML.
    query_ids: tuple[str | int, ...]
    case_sizes: tuple[int, ...]
    corpus_size: int

    def metadata(self):
        return {"rows": len(self.retrieved), "columns": ["retrieved", "relevant"],
                "sha256": self.fingerprint, "format": "retrieval", "size_bytes": self.size_bytes,
                "comparison_contract": "retrieval.binary.v1", "alignment": self.alignment,
                "evaluation_sha256": self.evaluation_fingerprint, "output_sha256": self.output_fingerprint,
                "corpus_sha256": self.corpus_fingerprint}


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

    keyed = isinstance(retrieved, Mapping) or isinstance(relevant, Mapping)
    if keyed:
        if not isinstance(retrieved, Mapping) or not isinstance(relevant, Mapping):
            raise TypeError("Use mappings for both retrieved and relevant, or ordered iterables for both")
        if len(retrieved) > config.max_queries or len(relevant) > config.max_queries:
            raise ValueError("Retrieval input exceeds max_queries")
        if retrieved.keys() != relevant.keys():
            raise ValueError("Retrieved and relevant mappings must contain identical query IDs")
        if any(not isinstance(key, str) or not key.strip() for key in retrieved):
            raise TypeError("Query IDs must be nonempty strings")
        # Canonical query order also stabilizes floating-point accumulation.
        # Otherwise equivalent mappings could trigger a zero-tolerance regression
        # merely because their insertion orders differ.
        triples = ((key, retrieved[key], relevant[key]) for key in sorted(retrieved))
    else:
        sentinel = object()
        pairs = zip_longest(ordered_iterable(retrieved, "retrieved"), ordered_iterable(relevant, "relevant"), fillvalue=sentinel)

        def positional():
            for rank, truth in pairs:
                if rank is sentinel or truth is sentinel:
                    raise ValueError("Retrieved and relevant query counts must match exactly")
                yield None, rank, truth

        triples = positional()
    rankings, judgments, query_ids, case_sizes = [], [], [], []
    evaluation_parts, output_parts = [], []
    for query_id, rank, truth in triples:
        case_start_size = size
        if len(rankings) >= config.max_queries:
            raise ValueError("Retrieval input exceeds max_queries; no sampling was performed")
        if query_id is not None and len(query_id) > config.max_bytes:
            raise ValueError("Retrieval input exceeds max_bytes")
        size += len(json.dumps(query_id).encode("utf-8"))
        if size > config.max_bytes:
            raise ValueError("Retrieval input exceeds max_bytes")
        ranking = identifiers(rank, config.max_ids_per_query, ranked=True)
        judgment = frozenset(identifiers(truth, config.max_ids_per_query, ranked=False))
        # Sort per-case digests later so mapping insertion order is irrelevant.
        # Stable query IDs and judgments identify the benchmark, not its outputs.
        identity = query_id if keyed else len(rankings)
        query_ids.append(identity)
        evaluation_parts.append(sha256(json.dumps([identity, sorted(judgment)], separators=(",", ":")).encode()).digest())
        output_parts.append(sha256(json.dumps([identity, ranking], separators=(",", ":")).encode()).digest())
        digest.update(json.dumps([query_id, ranking, sorted(judgment)], separators=(",", ":")).encode("utf-8") + b"\n")
        rankings.append(ranking)
        judgments.append(judgment)
        case_sizes.append(size - case_start_size)
    if not rankings:
        raise ValueError("Retrieval input must contain at least one query")
    before_corpus_size = size
    corpus = frozenset(identifiers(corpus_ids, config.max_corpus_ids, ranked=False)) if corpus_ids is not None else None
    digest.update(json.dumps(sorted(corpus) if corpus is not None else None, separators=(",", ":")).encode("utf-8"))
    evaluation_hash = sha256(b"proofml:retrieval:evaluation:v1\n" + b"".join(sorted(evaluation_parts))).hexdigest()
    output_hash = sha256(b"proofml:retrieval:outputs:v1\n" + b"".join(sorted(output_parts))).hexdigest()
    corpus_hash = sha256(json.dumps(sorted(corpus) if corpus is not None else None, separators=(",", ":")).encode()).hexdigest()
    return RetrievalData(tuple(rankings), tuple(judgments), corpus, digest.hexdigest(), size,
                         evaluation_hash, output_hash, corpus_hash, "query_id" if keyed else "position",
                         tuple(query_ids), tuple(case_sizes), size - before_corpus_size)


def select_retrieval(data: RetrievalData, indices: tuple[int, ...]) -> RetrievalData:
    """Build an immutable subset without reloading or rehashing the shared corpus.

    Empty subsets are meaningful requested populations: ranking coverage is 0/0
    and its check skips, so a release policy can distinguish absent evidence.
    """
    rankings = tuple(data.retrieved[i] for i in indices)
    judgments = tuple(data.relevant[i] for i in indices)
    query_ids = tuple(data.query_ids[i] for i in indices)
    sizes = tuple(data.case_sizes[i] for i in indices)
    evaluation_parts, output_parts = [], []
    for key, ranking, judgment in zip(query_ids, rankings, judgments):
        evaluation_parts.append(sha256(json.dumps([key, sorted(judgment)], separators=(",", ":")).encode()).digest())
        output_parts.append(sha256(json.dumps([key, ranking], separators=(",", ":")).encode()).digest())
    evaluation_hash = sha256(b"proofml:retrieval:evaluation:v1\n" + b"".join(sorted(evaluation_parts))).hexdigest()
    output_hash = sha256(b"proofml:retrieval:outputs:v1\n" + b"".join(sorted(output_parts))).hexdigest()
    # Subsets have a composite identity, not original input bytes. The separate
    # benchmark/output fingerprints use the same algorithm as an unsliced run.
    fingerprint = sha256(json.dumps(["retrieval_subset.v1", evaluation_hash, output_hash, data.corpus_fingerprint]).encode()).hexdigest()
    return RetrievalData(rankings, judgments, data.corpus_ids, fingerprint, sum(sizes) + data.corpus_size,
                         evaluation_hash, output_hash, data.corpus_fingerprint, data.alignment,
                         query_ids, sizes, data.corpus_size)


@dataclass(frozen=True)
class RetrievalContext:
    data: RetrievalData
    config: RetrievalConfig
