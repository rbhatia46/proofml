"""Runnable offline examples: python examples/text_and_retrieval.py.

Synthetic teaching fixtures, not real-world detection benchmarks. No models or
extra dependencies are needed. Reports can be saved with report.save(path).
"""
from proofml import audit_retrieval, audit_text


def main():
    corpus = ["one two three four five six seven eight nine ten", "a different document"]
    holdout = ["one two three four five six seven eight nine ten eleven"]
    text_report = audit_text(corpus, holdout)
    print("Text corpus:", text_report)
    for finding in text_report.findings:
        print(f"  {finding.code}: {finding.title}")

    # Export stable document/chunk IDs from any retriever. Query-ID mappings
    # align by key, so differing dictionary insertion orders cannot mispair rows.
    retrieved = {"query-1": ["doc-a", "doc-b"], "query-2": ["doc-c", "doc-a"]}
    relevant = {"query-2": {"doc-c"}, "query-1": {"doc-a", "doc-b"}}
    retrieval_report = audit_retrieval(retrieved, relevant, k=2, min_recall=0.9,
                                       corpus_ids={"doc-a", "doc-b", "doc-c"})
    print("Retrieval:", retrieval_report)
    print(retrieval_report.metrics["retrieval_ranking"])
    # Require coverage explicitly: missing judgments must not silently make a
    # deployment gate pass just because there were no high-severity findings.
    retrieval_report.raise_for_issues(require_checks=("retrieval_ranking", "retrieval_corpus"))


if __name__ == "__main__":
    main()
