"""Run only Notebook 02 in the project environment, guarding Notebook 01 and raw files."""

import asyncio
import json
import os
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.data.load_data import sha256, discover_datasets


def main():
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("Use the project's .venv Python interpreter.")
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    protected = [ROOT / "notebooks/01_data_understanding.ipynb", *discover_datasets(ROOT / "data/raw").values()]
    before = {str(p.relative_to(ROOT)): sha256(p) for p in protected}
    cache = ROOT / ".cache/jupyter"
    kernel = cache / "kernels/botnet-research"
    kernel.mkdir(parents=True, exist_ok=True)
    (kernel / "kernel.json").write_text(json.dumps({"argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "Botnet research (.venv)", "language": "python"}), encoding="utf-8")
    os.environ["JUPYTER_PATH"] = str(cache) + os.pathsep + os.environ.get("JUPYTER_PATH", "")
    os.environ["MPLCONFIGDIR"] = str(ROOT / ".cache/matplotlib")
    path = ROOT / "notebooks/02_duplicate_target_ambiguity_and_leakage_investigation.ipynb"
    notebook = nbformat.read(path, as_version=4)
    workdir = ROOT if "--root-cwd" in sys.argv else ROOT / "notebooks"
    client = NotebookClient(notebook, timeout=300, startup_timeout=60, kernel_name="botnet-research",
                            resources={"metadata": {"path": str(workdir)}}, allow_errors=False,
                            on_cell_start=lambda cell, cell_index: print(f"Cell {cell_index+1}/{len(notebook.cells)}", flush=True))
    try:
        client.execute()
    finally:
        nbformat.write(notebook, path)
        after = {str(p.relative_to(ROOT)): sha256(p) for p in protected}
        assert before == after, "Notebook 01 or raw files changed"
    nbformat.validate(notebook)
    code = [c for c in notebook.cells if c.cell_type == "code"]
    report = {"execution": "Jupyter/nbclient in project .venv", "working_directory": str(workdir.relative_to(ROOT)),
              "code_cells": len(code), "executed_code_cells": sum(c.execution_count is not None for c in code),
              "errors": sum(o.output_type == "error" for c in code for o in c.outputs),
              "protected_before": before, "protected_after": after, "vscode_ui_run_verified": False}
    (ROOT / "results/tables/02_notebook_execution.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
