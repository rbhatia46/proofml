"""Real-data case study: a useful offline score can rely on an unavailable feature.

Run from a checkout after installing `.[pandas]`, scikit-learn, and certifi:
    python examples/bank_marketing.py --download --output artifacts/bank-marketing

Network access is explicit. The example verifies the official archive and CSV,
never extracts arbitrary ZIP paths, and refuses to overwrite an existing run.
Model fitting is for this demonstration only; ProofML itself does not train models.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from importlib.metadata import version
from io import BytesIO
import json
from pathlib import Path
import platform
import ssl
from urllib.error import URLError
from urllib.request import urlopen
from zipfile import ZipFile

from proofml import AuditPolicy, AuditSuite, audit, __version__

SOURCE_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
DATASET_URL = "https://archive.ics.uci.edu/dataset/222/bank+marketing"
ARCHIVE_SHA256 = "e0bf5f5de5b846e2f18e9d90606637267d46dfa260e0f17bb12e605db5efbeb4"
CSV_SHA256 = "74adfc578bf77a7ff4bb1ba4a9f8709d9e3c6907342959c2c8416847e0afb4d8"
CSV_MEMBER = "bank-additional/bank-additional-full.csv"
MAX_ARCHIVE_BYTES = 5_000_000
MAX_CSV_BYTES = 10_000_000
SEED = 42
NUMERIC = ("age", "duration", "campaign", "pdays", "previous", "emp.var.rate",
           "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed")
CATEGORICAL = ("job", "marital", "education", "default", "housing", "loan",
               "contact", "month", "day_of_week", "poutcome")


def bounded_read(stream, limit):
    """Read at most limit+1 bytes so a changed download cannot grow unbounded."""
    payload = stream.read(limit + 1)
    if len(payload) > limit:
        raise ValueError("Dataset input exceeds the case-study size limit")
    return payload


def member_bytes(archive, name, limit):
    """Read one named member in memory, never extract paths onto the filesystem."""
    info = archive.getinfo(name)
    if info.file_size > limit:
        raise ValueError("Archive member exceeds the case-study size limit")
    with archive.open(info) as handle:
        return bounded_read(handle, limit)


def verified_csv(payload):
    if sha256(payload).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("UCI archive checksum changed; review the source, do not bypass verification")
    with ZipFile(BytesIO(payload)) as outer:
        nested = member_bytes(outer, "bank-additional.zip", MAX_ARCHIVE_BYTES)
    with ZipFile(BytesIO(nested)) as inner:
        csv = member_bytes(inner, CSV_MEMBER, MAX_CSV_BYTES)
    if sha256(csv).hexdigest() != CSV_SHA256:
        raise ValueError("UCI CSV checksum changed")
    return csv


def download_context():
    """Keep TLS verification enabled, optionally adding certifi's public roots.

    Some python.org macOS installations lack configured system certificates.
    certifi is an example-only convenience, not a ProofML runtime dependency.
    """
    context = ssl.create_default_context()
    try:
        import certifi
    except ImportError:
        pass
    else:
        context.load_verify_locations(cafile=certifi.where())
    return context


def read_dataset(*, archive=None, download=False):
    """Load the exact 41,188-row dataset; no sample substitution or account needed."""
    import pandas as pd

    if (archive is None) == (not download):
        raise ValueError("Choose exactly one of archive or download=True")
    if archive is not None:
        with Path(archive).open("rb") as handle:
            payload = bounded_read(handle, MAX_ARCHIVE_BYTES)
    else:
        with urlopen(SOURCE_URL, timeout=45, context=download_context()) as response:
            payload = bounded_read(response, MAX_ARCHIVE_BYTES)
    frame = pd.read_csv(BytesIO(verified_csv(payload)), sep=";", keep_default_na=False)
    if len(frame) != 41_188 or set(frame.columns) != set(NUMERIC + CATEGORICAL + ("y",)):
        raise ValueError("Unexpected UCI row count or schema")
    if set(frame["y"]) != {"yes", "no"}:
        raise ValueError("Unexpected outcome labels")
    return frame


def chronological_holdout(frame):
    """Preserve UCI's published ordering, without inventing precise timestamps.

    This is a single row-order holdout, not a validated time/entity boundary:
    exact dates and stable customer identifiers are absent from this CSV.
    """
    if len(frame) < 10:
        raise ValueError("At least ten rows are needed for this demonstration")
    boundary = len(frame) * 4 // 5
    return frame.iloc[:boundary].copy(), frame.iloc[boundary:].copy()


def audit_scenarios(train, test):
    """Keep the business contract constant when removing the offending predictor."""
    contract = {"target": "y", "unavailable_features": ("duration",)}
    reports = {
        "undeclared": audit(train, test, target="y"),
        "duration_included": audit(train, test, **contract),
        "duration_removed": audit(train.drop(columns="duration"), test.drop(columns="duration"), **contract),
    }
    # Narrow on purpose: a pass means only this critical availability contract
    # passed, not that the dataset or a model is ready for deployment.
    policy = AuditPolicy(name="pre_call_availability", severity="critical",
                         require_checks=("feature_availability",))
    decisions = {name: policy.evaluate(report) for name, report in reports.items()}
    expected = {"undeclared": "insufficient", "duration_included": "failed", "duration_removed": "passed"}
    observed = {name: result.status for name, result in decisions.items()}
    if observed != expected:
        raise RuntimeError(f"Case-study expectations changed: {observed}; inspect the audit evidence")
    return AuditSuite(reports), policy, decisions


def evaluate_model(train, test, *, include_duration):
    """Fit the same train-only preprocessing/model recipe on the same row split."""
    from sklearn.compose import ColumnTransformer
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from threadpoolctl import threadpool_limits
    import warnings

    numeric = [name for name in NUMERIC if include_duration or name != "duration"]
    features = numeric + list(CATEGORICAL)
    model = Pipeline([
        ("preprocess", ColumnTransformer([
            ("numeric", StandardScaler(), numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), list(CATEGORICAL)),
        ])),
        ("classifier", LogisticRegression(solver="lbfgs", max_iter=4000, C=1.0, random_state=SEED)),
    ])
    # Treat 'unknown' as a category and pdays=999 as the published sentinel.
    # No tuning, feature selection, calibration, or fitting uses held-out rows.
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(train[features], train["y"].eq("yes").astype(int))
        probabilities = model.predict_proba(test[features])[:, 1]
    truth = test["y"].eq("yes").astype(int)
    return {
        "roc_auc": float(roc_auc_score(truth, probabilities)),
        "average_precision": float(average_precision_score(truth, probabilities)),
        "features": features,
        "solver_iterations": int(model.named_steps["classifier"].n_iter_[0]),
    }


def summarize_findings(report):
    return [{"code": f.code, "severity": f.severity, "columns": list(f.columns), "evidence": f.evidence}
            for f in report.findings]


def run(output, *, archive=None, download=False):
    """Write reproducible aggregate evidence; do not export customer rows or models."""
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("Use a new output directory; existing runs are never overwritten")
    frame = read_dataset(archive=archive, download=download)
    train, test = chronological_holdout(frame)
    suite, policy, decisions = audit_scenarios(train, test)
    print("Audit decisions:", {name: value.status for name, value in decisions.items()}, flush=True)
    # Deliberately fit the rejected variant ONLY to demonstrate why its score
    # is misleading for pre-call use. A production training job would enforce
    # the policy before fitting and would not override the failed gate.
    models = {}
    for name, include in (("duration_included", True), ("duration_removed", False)):
        models[name] = evaluate_model(train, test, include_duration=include)
        print(name, {key: round(models[name][key], 4) for key in ("roc_auc", "average_precision")}, flush=True)
    broader = AuditPolicy(name="broader_review", severity="high",
                          require_checks=("feature_availability", "data_quality", "split_schema", "split_overlap"))
    broad_decision = broader.evaluate(suite.reports["duration_removed"])
    result = {
        "schema_version": "1.0", "dataset": DATASET_URL, "download_url": SOURCE_URL,
        "citation": "Moro, S., Rita, P., & Cortez, P. (2014). Bank Marketing. UCI. DOI:10.24432/C5K306",
        "license": "CC BY 4.0", "archive_sha256": ARCHIVE_SHA256, "csv_sha256": CSV_SHA256,
        "csv_member": CSV_MEMBER,
        "versions": {"proofml": __version__, **{name: version(name) for name in ("pandas", "numpy", "scipy", "scikit-learn", "threadpoolctl")}},
        "python": platform.python_version(),
        "split": {"method": "first 80% / final 20% in original UCI row order; no shuffling",
                  "train_rows": len(train), "test_rows": len(test),
                  "train_positive_rate": float(train["y"].eq("yes").mean()),
                  "test_positive_rate": float(test["y"].eq("yes").mean())},
        "model": {"estimator": "LogisticRegression", "solver": "lbfgs", "C": 1.0, "max_iter": 4000,
                  "random_state": SEED, "numerical_threads": 1, "tuning": "none",
                  "preprocessing": "train-only StandardScaler + OneHotEncoder(handle_unknown='ignore')"},
        "metrics": models,
        "availability_gate": {name: decision.status for name, decision in decisions.items()},
        "broader_gate_after_removal": broad_decision.status,
        "findings": {name: summarize_findings(report) for name, report in suite.reports.items()},
        "limitations": [
            "No synthetic leakage was injected: duration is in the original dataset.",
            "Availability is declared from source documentation, not inferred by ProofML.",
            "Duration removal fixes this one contract, not all point-in-time feature semantics.",
            "Single holdout and estimator; scores are not a significance claim or a production forecast.",
            "Published row order is used; exact timestamps and stable customer IDs are absent.",
            "Other contact features and publication lags of economic indicators need lineage review.",
            "Metrics are computed by scikit-learn, not ProofML; no real business loss was measured.",
        ],
    }
    output.mkdir(parents=True, exist_ok=False)
    suite.save(output / "audits")
    policy.save(output / "availability-policy.json")
    for name, decision in decisions.items():
        decision.save(output / f"{name}-decision.json")
        decision.save_junit(output / f"{name}-decision.xml")
    broad_decision.save(output / "broader-decision.json")
    (output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Broader review after duration removal:", broad_decision.status)
    print("Reports and reproducibility metadata:", output.resolve())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--download", action="store_true", help="Download the checksum-pinned public UCI archive")
    source.add_argument("--archive", type=Path, help="Use an already downloaded official ZIP, offline")
    parser.add_argument("--output", type=Path, default=Path("artifacts/bank-marketing"))
    args = parser.parse_args()
    try:
        run(args.output, archive=args.archive, download=args.download)
    except URLError as error:
        parser.error(f"UCI download failed: {error}. Check connectivity and trusted certificates "
                     "(python -m pip install certifi), or use --archive. Never disable TLS verification.")


if __name__ == "__main__":
    main()
