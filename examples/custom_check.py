"""Run from the repository root: python examples/custom_check.py train.csv."""
import sys
from proofml import CheckResult, Finding, audit
from proofml.checks import default_checks


class MinimumRows:
    """Example of a team-specific contract, not a universal statistical rule."""
    id = "example.minimum_rows"

    def run(self, ctx):
        findings = []
        if len(ctx.train.rows) < 100:
            findings.append(Finding(
                "example.small_dataset", "medium", "needs_context", "Small training dataset",
                "This example team's review policy asks for at least 100 observations.",
                "Review evaluation uncertainty and sampling requirements.",
                evidence={"rows": len(ctx.train.rows), "threshold": 100},
            ))
        return CheckResult.complete(self.id, findings)


if __name__ == "__main__":
    report = audit(sys.argv[1], checks=[*default_checks(), MinimumRows()])
    for finding in report.findings:
        print(finding.severity, finding.title)
