"""Offline demonstration: an improved aggregate can hide a cohort regression.

Run: python examples/slice_regression.py
The fixtures are synthetic. Two observations per cohort is a teaching minimum,
not a production recommendation or evidence of statistical power.
"""
from pathlib import Path
from tempfile import TemporaryDirectory

from proofml import AuditPolicy, SuitePolicy, audit_retrieval_slices


def main():
    relevant = {f"q{i}": {f"doc{i}"} for i in range(8)}
    groups = {f"q{i}": "english" if i < 6 else "spanish" for i in range(8)}
    old = {f"q{i}": [f"doc{i}" if i >= 6 else "miss"] for i in range(8)}
    new = {f"q{i}": [f"doc{i}" if i < 6 else "miss"] for i in range(8)}
    baseline = audit_retrieval_slices(old, relevant, groups=groups, k=1)
    candidate = audit_retrieval_slices(new, relevant, groups=groups, k=1)
    policy = SuitePolicy(
        default=AuditPolicy(require_checks=("retrieval_ranking",), min_coverage={"retrieval_ranking": 1},
                            min_evaluated={"retrieval_ranking": 2}, max_drop={"retrieval_ranking.recall@1": 0.05}),
        require_reports=("overall", "english", "spanish"),
    )
    decision = policy.evaluate(candidate, baseline=baseline)
    for name, result in decision.results.items():
        before = baseline.metrics[name]["retrieval_ranking"]["recall@1"]
        after = candidate.metrics[name]["retrieval_ranking"]["recall@1"]
        print(f"{name}: {before:.0%} -> {after:.0%}; {result.status}")
    print("Release decision:", decision.status)
    assert decision.status == "failed"
    assert decision.results["overall"].passed
    assert decision.results["spanish"].status == "failed"
    # Temporary exports exercise the same artifacts a real CI job would retain.
    with TemporaryDirectory(prefix="proofml-slice-demo-") as directory:
        candidate.save(Path(directory) / "candidate")
        decision.save(Path(directory) / "decision.json")
        decision.save_junit(Path(directory) / "decision.xml")


if __name__ == "__main__":
    main()
