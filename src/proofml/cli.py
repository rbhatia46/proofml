"""CLI exit contract: 0 completed, 1 finding threshold, 2 input/check/report error."""
from __future__ import annotations
import argparse
import csv
import sys
from dataclasses import replace
from importlib.resources import files
from pathlib import Path
from .checks import default_checks
from .config import AuditConfig
from .engine import audit
from .models import SEVERITY_ORDER
from .reporting import write_reports
from ._version import __version__
from .text import default_text_checks
from .retrieval import default_retrieval_checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="proofml", description="Local ML audits with evidence and no API keys. Text/retrieval audits use the Python API.")
    parser.add_argument("--version", action="version", version=f"proofml {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("checks", help="List built-in check IDs")
    listing.add_argument("--task", choices=("classification", "regression", "forecasting"), default="classification")
    listing.add_argument("--domain", choices=("tabular", "text", "retrieval"), default="tabular")
    run = commands.add_parser("audit", help="Audit a training dataset and optional test split")
    run.add_argument("train", type=Path)
    run.add_argument("--test", type=Path)
    run.add_argument("--config", type=Path, help="Strict JSON config; flags override provided fields")
    run.add_argument("--target")
    run.add_argument("--task", choices=("classification", "regression", "forecasting"))
    run.add_argument("--entity-id")
    run.add_argument("--time-column")
    demo = commands.add_parser("demo", help="Audit a bundled synthetic churn example")
    demo.add_argument("--clean", action="store_true", help="Use the corrected synthetic example for comparison")
    demo.add_argument("--problem", choices=("churn", "forecasting"), default="churn")
    for sub in (run, demo):
        sub.add_argument("--output", type=Path, default=Path("proofml-report"))
        sub.add_argument("--overwrite", action="store_true")
        sub.add_argument("--fail-on", choices=(*SEVERITY_ORDER, "none"), default="high",
                         help="Exit 1 for findings at or above this severity (default: high)")
    args = parser.parse_args(argv)
    if args.command == "checks":
        registry = {"tabular": lambda: default_checks(args.task), "text": default_text_checks,
                    "retrieval": default_retrieval_checks}[args.domain]()
        print("\n".join(check.id for check in registry))
        return 0
    try:
        if args.command == "demo":
            root = files("proofml").joinpath("demo_data")
            prefix = "clean_" if args.clean else ""
            if args.problem == "forecasting":
                prefix = "forecast_" + prefix
            train, test = Path(str(root.joinpath(prefix + "train.csv"))), Path(str(root.joinpath(prefix + "test.csv")))
            config_name = "forecast_config.json" if args.problem == "forecasting" else "config.json"
            config = AuditConfig.from_file(Path(str(root.joinpath(config_name))))
            if args.clean:
                config = replace(config, unavailable_features=())
        else:
            train, test = args.train, args.test
            config = AuditConfig.from_file(args.config) if args.config else AuditConfig()
            overrides = {name: getattr(args, name) for name in ("target", "task", "entity_id", "time_column") if getattr(args, name) is not None}
            config = replace(config, **overrides)
        # Never replace an input with a report, even with explicit output overwrite.
        sources = {train.resolve(), *([test.resolve()] if test is not None else [])}
        if args.command == "audit" and args.config is not None:
            sources.add(args.config.resolve())
        if any((args.output / name).resolve() in sources for name in ("report.json", "report.html")):
            raise ValueError("Report destination collides with an input file")
        report = audit(train, test=test, config=config)
        destinations = write_reports(report, args.output, overwrite=args.overwrite)
    except (ValueError, OSError, csv.Error, UnicodeError) as error:
        print(f"proofml: {error}", file=sys.stderr)
        return 2
    print(f"ProofML: {len(report.findings)} findings | " + " | ".join(f"{count} {status}" for status, count in report.summary["check_counts"].items()))
    for finding in report.findings:
        print(f"[{finding.severity.upper()} / {finding.confidence}] {finding.title}")
    for destination in destinations:
        print(destination.resolve())
    if any(check.status == "error" for check in report.checks):
        return 2
    if args.fail_on != "none" and any(SEVERITY_ORDER[f.severity] >= SEVERITY_ORDER[args.fail_on] for f in report.findings):
        return 1
    return 0
