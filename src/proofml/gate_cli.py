"""Policy CLI adapter; algorithms stay in the policy and comparison modules."""
from pathlib import Path
import sys

from .models import AuditReport
from .policy import AuditPolicy


def add_gate_parser(commands):
    gate = commands.add_parser("gate", help="Apply a release policy to saved reports")
    gate.add_argument("report", type=Path, help="Candidate report.json")
    gate.add_argument("--policy", type=Path, required=True)
    gate.add_argument("--baseline", type=Path)
    gate.add_argument("--output", type=Path, help="Decision JSON file")
    gate.add_argument("--junit", type=Path, help="JUnit XML file")
    gate.add_argument("--overwrite", action="store_true")


def run_gate(args):
    try:
        sources = {args.report.resolve(), args.policy.resolve()}
        if args.baseline is not None:
            sources.add(args.baseline.resolve())
        outputs = [p for p in (args.output, args.junit) if p is not None]
        if len({p.resolve() for p in outputs}) != len(outputs) or any(p.resolve() in sources for p in outputs):
            raise ValueError("Gate output collides with an input or another output")
        if not args.overwrite and any(p.exists() or p.is_symlink() for p in outputs):
            raise FileExistsError("Gate output exists; use a new path or --overwrite")
        policy = AuditPolicy.from_file(args.policy)
        report = AuditReport.load(args.report)
        baseline = AuditReport.load(args.baseline) if args.baseline is not None else None
        decision = policy.evaluate(report, baseline=baseline)
        if args.output is not None:
            decision.save(args.output, overwrite=args.overwrite)
        if args.junit is not None:
            decision.save_junit(args.junit, overwrite=args.overwrite)
    except (ValueError, TypeError, OSError, UnicodeError) as error:
        print(f"proofml gate: {error}", file=sys.stderr)
        return 2
    print(f"ProofML release gate: {decision.status}")
    for issue in decision.issues:
        print(f"[{issue.code}] {issue.message}")
    return 0 if decision.passed else 1 if decision.status == "failed" else 2
