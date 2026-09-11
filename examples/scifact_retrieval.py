"""Local BEIR SciFact retrieval benchmark shared by the notebook and CLI.

No embedding model, GPU, API key, BEIR runtime, or relevance-label training is
required. TF-IDF is an educational baseline, not a leaderboard implementation.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import version
from io import BytesIO, StringIO
import json
from pathlib import Path
import platform
import re
import ssl
from urllib.request import urlopen
from zipfile import ZipFile

from proofml import AuditPolicy, SuitePolicy, __version__, audit_retrieval_slices

URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
ARCHIVE_SHA256 = "536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165"
MEMBER_SHA256 = {
    "scifact/corpus.jsonl": "dec31c8182f3d744c7d2c09423756fd1d17cbef75808db13ba01cc0aab4d1ac6",
    "scifact/queries.jsonl": "8ff84a7c903f722981cd8d595c022660140c51867b27608a6d4910db86080313",
    "scifact/qrels/test.tsv": "0864bb985e0ca2367ba217977e72004d549054b2b06666ed9d4825ac7c21284c",
}
K = 10
SHORT_QUERY_WORDS = 12
GROUPS = ("overall", "short_queries", "long_queries")
MAX_BYTES = 20_000_000


@dataclass(frozen=True)
class Benchmark:
    corpus: dict
    queries: dict
    relevant: dict


def bounded_read(handle, limit=MAX_BYTES):
    payload = handle.read(limit + 1)
    if len(payload) > limit:
        raise ValueError("Benchmark exceeds the example size limit")
    return payload


def jsonl_records(payload):
    records = {}
    for line in payload.decode("utf-8").splitlines():
        item = json.loads(line)
        key = item.get("_id")
        if not isinstance(key, str) or not key or key in records:
            raise ValueError("Invalid or repeated benchmark ID")
        if not isinstance(item.get("text"), str):
            raise ValueError("Benchmark text must be a string")
        records[key] = item
    return records


def parse_benchmark(members):
    """Select ONLY BEIR test qrels; do not mix train queries into evaluation."""
    corpus = jsonl_records(members["scifact/corpus.jsonl"])
    all_queries = jsonl_records(members["scifact/queries.jsonl"])
    reader = csv.DictReader(StringIO(members["scifact/qrels/test.tsv"].decode()), delimiter="\t")
    if reader.fieldnames != ["query-id", "corpus-id", "score"]:
        raise ValueError("Unexpected qrels schema")
    relevant, pairs = {}, set()
    for row in reader:
        qid, did = row["query-id"], row["corpus-id"]
        if row["score"] != "1":
            raise ValueError("This example expects binary positive SciFact test judgments")
        if qid not in all_queries or did not in corpus or (qid, did) in pairs:
            raise ValueError("Unresolved or repeated qrel")
        pairs.add((qid, did))
        relevant.setdefault(qid, set()).add(did)
    if not relevant:
        raise ValueError("No test judgments")
    if any(not isinstance(item.get("title"), str) for item in corpus.values()):
        raise ValueError("Corpus titles must be strings")
    return Benchmark(corpus, {key: all_queries[key]["text"] for key in sorted(relevant)}, relevant)


def load_scifact(*, archive=None, download=False):
    """Explicit bounded HTTPS download or offline archive; pin all input bytes."""
    if (archive is None) == (not download):
        raise ValueError("Choose exactly one of archive or download=True")
    if archive is not None:
        with Path(archive).open("rb") as handle:
            payload = bounded_read(handle)
    else:
        import certifi  # Example-only dependency for portable verified HTTPS.
        context = ssl.create_default_context()
        context.load_verify_locations(cafile=certifi.where())
        with urlopen(URL, timeout=45, context=context) as handle:
            payload = bounded_read(handle)
    if sha256(payload).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("SciFact archive checksum changed; review the source before updating pins")
    members = {}
    with ZipFile(BytesIO(payload)) as package:
        for name, expected in MEMBER_SHA256.items():
            if package.getinfo(name).file_size > MAX_BYTES:
                raise ValueError("Benchmark member too large")
            with package.open(name) as handle:
                data = bounded_read(handle)
            if sha256(data).hexdigest() != expected:
                raise ValueError("SciFact member checksum mismatch")
            members[name] = data
    benchmark = parse_benchmark(members)
    if (len(benchmark.corpus), len(benchmark.queries), sum(map(len, benchmark.relevant.values()))) != (5183, 300, 339):
        raise ValueError("Unexpected SciFact test population")
    return benchmark


def query_groups(queries):
    """Predeclared lexical cohorts, not demographic or clinical subgroups."""
    return {key: "short_queries" if len(re.findall(r"\b\w+\b", text)) <= SHORT_QUERY_WORDS else "long_queries"
            for key, text in queries.items()}


def rank_tfidf(corpus, queries, *, fields=("title", "text"), k=K):
    """Fit vocabulary/IDF on corpus only; score one query at a time in memory.

    L2-normalized TF-IDF dot products are cosine similarities. Equal scores use
    lexicographically sorted document IDs. Zero-score documents are not padded
    into rankings, so an out-of-vocabulary query returns an empty ranking.
    """
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from threadpoolctl import threadpool_limits

    if type(k) is not int or k < 1 or not corpus or not fields or any(f not in {"title", "text"} for f in fields):
        raise ValueError("Supply a corpus, supported fields, and positive k")
    ids = sorted(corpus)
    vectorizer = TfidfVectorizer(lowercase=True, strip_accents="unicode", sublinear_tf=True,
                                 norm="l2", ngram_range=(1, 1), token_pattern=r"(?u)\b\w\w+\b")
    with threadpool_limits(limits=1):
        documents = vectorizer.fit_transform(" ".join(corpus[key][field] for field in fields) for key in ids)
        query_vectors = vectorizer.transform(queries[key] for key in sorted(queries))
        rankings = {}
        for index, key in enumerate(sorted(queries)):
            scores = (query_vectors[index] @ documents.T).toarray().ravel()
            order = np.argsort(-scores, kind="stable")[:k]
            rankings[key] = [ids[i] for i in order if scores[i] > 0]
    return rankings


def release_policy():
    """Illustrative tolerances fixed before measuring; not significance tests."""
    return SuitePolicy(default=AuditPolicy(
        require_checks=("retrieval_inputs", "retrieval_corpus", "retrieval_ranking"),
        min_coverage={"retrieval_ranking": 1}, min_evaluated={"retrieval_ranking": 20},
        max_drop={"retrieval_ranking.recall@10": 0.05, "retrieval_ranking.ndcg@10": 0.05},
    ), require_reports=GROUPS)


def compare_rankings(benchmark, baseline_rankings, candidate_rankings):
    options = {"groups": query_groups(benchmark.queries), "corpus_ids": tuple(benchmark.corpus), "k": K}
    baseline = audit_retrieval_slices(baseline_rankings, benchmark.relevant, **options)
    candidate = audit_retrieval_slices(candidate_rankings, benchmark.relevant, **options)
    return baseline, candidate, release_policy().evaluate(candidate, baseline=baseline)


def metrics_table(baseline, candidate, decision):
    import pandas as pd
    return pd.DataFrame([{
        "cohort": name, "queries": int(candidate.metrics[name]["retrieval_ranking"]["evaluated_queries"]),
        "baseline_recall@10": baseline.metrics[name]["retrieval_ranking"]["recall@10"],
        "candidate_recall@10": candidate.metrics[name]["retrieval_ranking"]["recall@10"],
        "baseline_ndcg@10": baseline.metrics[name]["retrieval_ranking"]["ndcg@10"],
        "candidate_ndcg@10": candidate.metrics[name]["retrieval_ranking"]["ndcg@10"],
        "gate": decision.results[name].status,
    } for name in GROUPS])


def save_run(output, baseline, candidate, decision):
    """Keep input provenance outside report fingerprints, which hash IDs, not text."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    baseline.save(output / "baseline")
    candidate.save(output / "candidate")
    decision.policy.save(output / "policy.json")
    decision.save(output / "decision.json")
    decision.save_junit(output / "decision.xml")
    result = {
        "dataset": "BEIR SciFact test", "source": URL, "archive_sha256": ARCHIVE_SHA256,
        "member_sha256": MEMBER_SHA256, "corpus_documents": 5183, "test_queries": 300,
        "positive_qrels": 339, "k": K, "short_query_max_words": SHORT_QUERY_WORDS,
        "retriever": "scikit-learn unigram TF-IDF; sublinear TF; L2 cosine; Unicode accents stripped; no stop-word list",
        "baseline_fields": ["title", "text"], "candidate_fields": ["title"],
        "tie_break": "lexicographic document ID; zero-score results omitted",
        "training": "vocabulary and IDF fitted on corpus only; no qrel/query fitting or tuning",
        "versions": {"proofml": __version__, **{name: version(name) for name in ("scikit-learn", "numpy", "scipy", "pandas")}},
        "python": platform.python_version(), "gate": decision.status,
        "cohorts": metrics_table(baseline, candidate, decision).to_dict(orient="records"),
        "limitations": ["Binary relevance, unlisted documents treated as nonrelevant; judgments are not exhaustive.",
                        "Cohorts and 0.05 tolerances are illustrative, not calibrated significance tests.",
                        "No query/document text in saved reports; these findings do not verify scientific claims.",
                        "Baseline compatibility hashes IDs and judgments, not document/query contents; pins retain source provenance.",
                        "No claim of latency improvement or leaderboard equivalence; index fields are deliberately different."],
    }
    (output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--download", action="store_true")
    source.add_argument("--archive", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/scifact"))
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("Use a new output directory")
    benchmark = load_scifact(archive=args.archive, download=args.download)
    before = rank_tfidf(benchmark.corpus, benchmark.queries)
    after = rank_tfidf(benchmark.corpus, benchmark.queries, fields=("title",))
    baseline, candidate, decision = compare_rankings(benchmark, before, after)
    print(metrics_table(baseline, candidate, decision).to_string(index=False))
    save_run(args.output, baseline, candidate, decision)
    print("Release decision:", decision.status)


if __name__ == "__main__":
    main()
