"""Execute Notebook 01 with this interpreter and a project-local kernel spec.

Run with .venv's Python from VS Code's terminal. No global kernel registration.
"""

import json
import os
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient


def main() -> None:
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("Run this script with the project's .venv Python.")
    root = Path(__file__).resolve().parents[1]
    jupyter_data = root / ".cache/jupyter"
    kernel_dir = jupyter_data / "kernels/botnet-research"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "Botnet research (local .venv)", "language": "python",
    }), encoding="utf-8")
    os.environ["JUPYTER_PATH"] = str(jupyter_data) + os.pathsep + os.environ.get("JUPYTER_PATH", "")
    os.environ["MPLCONFIGDIR"] = str(root / ".cache/matplotlib")
    notebook_path = root / "notebooks/01_data_understanding.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    execution_dir = root if "--root-cwd" in sys.argv else root / "notebooks"
    print(f"Starting notebook execution from {execution_dir.relative_to(root)}", flush=True)
    client = NotebookClient(notebook, timeout=120, startup_timeout=60,
                            kernel_name="botnet-research",
                            resources={"metadata": {"path": str(execution_dir)}}, allow_errors=False,
                            on_cell_start=lambda cell, cell_index: print(f"Cell {cell_index + 1}/{len(notebook.cells)}", flush=True))
    try:
        client.execute()
    except Exception:
        # Preserve the error output for diagnosis rather than pretend success.
        nbformat.write(notebook, notebook_path)
        raise
    findings = (root / "docs/01_dataset_findings.md").read_text(encoding="utf-8")
    for cell in notebook.cells:
        if cell.metadata.get("findings_snapshot"):
            cell.source = findings + "\n\nSnapshot of the verified run; re-execute the findings cell to refresh."
    nbformat.validate(notebook)
    nbformat.write(notebook, notebook_path)
    report_path = root / "results/tables/01_notebook_execution.json"
    previous = json.loads(report_path.read_text(encoding="utf-8-sig")) if report_path.exists() else {}
    current_directory = str(execution_dir.relative_to(root)) or "."
    verified_directories = set(previous.get("verified_working_directories", []))
    if previous.get("errors") == 0 and previous.get("working_directory"):
        verified_directories.add(previous["working_directory"])
    verified_directories.add(current_directory)
    report = {"execution": "nbclient using project .venv", "working_directory": current_directory,
              "verified_working_directories": sorted(verified_directories),
              "code_cells": sum(c.cell_type == "code" for c in notebook.cells),
              "executed_code_cells": sum(c.cell_type == "code" and c.execution_count is not None for c in notebook.cells),
              "errors": sum(o.output_type == "error" for c in notebook.cells if c.cell_type == "code" for o in c.outputs),
              "vscode_ui_execution_verified": False}
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
