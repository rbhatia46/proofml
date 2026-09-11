"""Execute tutorial notebooks with fresh kernels and retain real outputs.

Use both --scifact-archive and --bank-archive for offline verification, or opt
into network access with --download. Output must be a new directory. This tool
never overwrites the source notebooks or installs a global Jupyter kernel.
"""
import argparse
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scifact-archive", type=Path)
    parser.add_argument("--bank-archive", type=Path)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if not args.download and (args.scifact_archive is None or args.bank_archive is None):
        parser.error("Provide both offline archives, or explicitly allow --download")
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        parser.error("Use a new output directory")
    for variable, path in (("PROOFML_SCIFACT_ARCHIVE", args.scifact_archive), ("PROOFML_BANK_ARCHIVE", args.bank_archive)):
        if path is not None:
            if not path.is_file():
                parser.error(f"Archive does not exist: {path}")
            os.environ[variable] = str(path.resolve())
        else:
            os.environ.pop(variable, None)
    output.mkdir(parents=True)
    os.environ["IPYTHONDIR"] = str(output / ".ipython")
    os.environ["JUPYTER_RUNTIME_DIR"] = str(output / ".runtime")
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager

    root = Path(__file__).resolve().parents[1]
    for name in ("scifact_retrieval", "bank_marketing"):
        path = root / "examples" / "notebooks" / f"{name}.ipynb"
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        # Execute using this interpreter, regardless of the user's default kernel.
        manager = KernelManager(kernel_name="python3")
        manager.kernel_spec.argv[0] = sys.executable
        client = NotebookClient(notebook, km=manager, timeout=180, allow_errors=False,
                                record_timing=False, resources={"metadata": {"path": str(root)}})
        try:
            client.execute()
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
        nbformat.validate(notebook)
        if any(cell.execution_count is None for cell in notebook.cells if cell.cell_type == "code"):
            raise RuntimeError("Notebook has an unexecuted code cell")
        nbformat.write(notebook, output / path.name)
        print(f"Executed and validated {name}: {sum(c.cell_type == 'code' for c in notebook.cells)} code cells", flush=True)


if __name__ == "__main__":
    main()
