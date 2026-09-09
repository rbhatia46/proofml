"""Verify the exact wheel in a fresh environment, outside the source checkout.

Usage: python scripts/check_distribution.py dist
No upload occurs. Wheel installation uses --no-index and --no-deps to prove
that the default package requires no dependency downloads at runtime.
"""
import pathlib
import subprocess
import sys
import tempfile
import venv


def verify(directory):
    wheels = list(pathlib.Path(directory).resolve().glob("*.whl"))
    if len(wheels) != 1:
        raise ValueError("Provide a directory containing exactly one release wheel")
    with tempfile.TemporaryDirectory(prefix="proofml-dist-") as temp:
        root = pathlib.Path(temp)
        venv.create(root / "venv", with_pip=True)
        python = root / "venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        def run(*args):
            subprocess.run([str(python), *args], cwd=root, check=True)
        run("-m", "pip", "install", "--no-index", "--no-deps", str(wheels[0]))
        run("-c", "from importlib.metadata import version; import proofml; assert version('proofml') == proofml.__version__")
        for problem in ("churn", "forecasting"):
            run("-m", "proofml", "demo", "--problem", problem, "--clean", "--output", problem)
        (root / "tiny.csv").write_text("x,y\n1,0\n2,1\n3,0\n", encoding="utf-8")
        run("-c", "from proofml import audit; r = audit('tiny.csv', target='y'); r.save('api'); r.raise_for_issues(); print(r)")
        run("-c", "from proofml import audit_text; r = audit_text(['a red bird flies'], ['snow covers every mountain']); r.save('text'); r.raise_for_issues(require_checks=('text_overlap', 'text_near_duplicates')); print(r)")
        run("-c", "from proofml import audit_retrieval; r = audit_retrieval({'q1': ['doc1']}, {'q1': {'doc1'}}, k=1, min_recall=1); r.save('retrieval'); r.raise_for_issues(require_checks=('retrieval_ranking',)); assert r.metrics['retrieval_ranking']['recall@1'] == 1; print(r)")
        run("-c", "from proofml import AuditReport, AuditPolicy; r = AuditReport.load('retrieval/report.json'); p = AuditPolicy(min_coverage={'retrieval_ranking': 1}, min_evaluated={'retrieval_ranking': 1}, max_drop={'retrieval_ranking.recall@1': 0}); p.save('policy.json'); d = p.enforce(r, baseline=r); d.save('decision.json'); d.save_junit('decision.xml'); assert d.passed")
        run("-m", "proofml", "gate", "retrieval/report.json", "--policy", "policy.json", "--baseline", "retrieval/report.json", "--output", "cli-decision.json", "--junit", "cli-decision.xml")
        run("-c", "from proofml import audit_retrieval_slices, AuditPolicy, SuitePolicy, AuditSuite; s = audit_retrieval_slices({'q1': ['a'], 'q2': ['b']}, {'q1': {'a'}, 'q2': {'b'}}, groups={'q1': 'en', 'q2': 'es'}, k=1); s.save('slices'); p = SuitePolicy(default=AuditPolicy(min_coverage={'retrieval_ranking': 1}, max_drop={'retrieval_ranking.recall@1': 0}), require_reports=('overall', 'en', 'es')); p.save('suite-policy.json'); assert p.enforce(s, baseline=AuditSuite.load('slices/suite.json')).passed")
        run("-m", "proofml", "gate", "slices/suite.json", "--suite", "--policy", "suite-policy.json", "--baseline", "slices/suite.json", "--output", "suite-decision.json", "--junit", "suite-decision.xml")
        print("Installed wheel, versions, bundled demos, and all domain APIs verified without runtime dependencies.")


if __name__ == "__main__":
    verify(sys.argv[1])
