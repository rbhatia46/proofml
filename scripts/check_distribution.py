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
        print("Installed wheel, versions, bundled demos, and all domain APIs verified without runtime dependencies.")


if __name__ == "__main__":
    verify(sys.argv[1])
