"""Run with: python examples/release_gate.py. No downloads or persistent files.

Small fixtures illustrate gate behavior; the example thresholds and sample
counts are teaching choices, not recommended production requirements.
"""
from pathlib import Path
from tempfile import TemporaryDirectory

from proofml import AuditFailed, AuditPolicy, AuditReport, audit_retrieval


def main():
    judgments = {"query-1": {"doc-a"}, "query-2": {"doc-b"}}
    baseline = audit_retrieval({"query-1": ["doc-a"], "query-2": ["doc-b"]}, judgments, k=1)
    candidate = audit_retrieval({"query-2": ["doc-b"], "query-1": ["wrong-doc"]}, judgments, k=1)
    policy = AuditPolicy(
        require_checks=("retrieval_ranking",),
        min_coverage={"retrieval_ranking": 1.0},
        min_evaluated={"retrieval_ranking": 2},
        max_drop={"retrieval_ranking.recall@1": 0.05},
    )
    with TemporaryDirectory(prefix="proofml-release-example-") as directory:
        root = Path(directory)
        baseline_path, _ = baseline.save(root / "baseline")
        saved_baseline = AuditReport.load(baseline_path)
        decision = policy.evaluate(candidate, baseline=saved_baseline)
        print("Candidate:", decision.status)
        for issue in decision.issues:
            print(issue.message, issue.evidence)
        decision.save(root / "decision.json")
        decision.save_junit(root / "decision.xml")
        try:
            decision.raise_for_issues()
        except AuditFailed:
            print("Regression correctly blocks release.")
        thin = audit_retrieval({"q1": ["a"], "q2": ["b"]}, {"q1": {"a"}, "q2": set()}, k=1)
        coverage_policy = AuditPolicy(min_coverage={"retrieval_ranking": 1})
        print("Perfect score on half the queries:", coverage_policy.evaluate(thin).status)


if __name__ == "__main__":
    main()
