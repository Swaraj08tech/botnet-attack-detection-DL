"""Execute v3 notebooks top to bottom in the project kernel and save their outputs.

Usage:
    .venv/Scripts/python.exe scripts/run_v3_notebooks.py 01 02 ...   (default: all, in order)
"""
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_v3_notebooks import NAMES  # noqa: E402


def run(number):
    path = ROOT / 'notebooks' / NAMES[number]
    nb = nbformat.read(path, 4)
    started = time.time()
    client = NotebookClient(nb, timeout=4 * 3600, startup_timeout=120, kernel_name='botnet-research',
                            resources={'metadata': {'path': str(ROOT / 'notebooks')}}, allow_errors=False,
                            on_cell_start=lambda cell, cell_index: print(f"  {number}: cell {cell_index + 1}/{len(nb.cells)}", flush=True))
    try:
        client.execute()
    finally:
        nbformat.write(nb, path)
    code = [c for c in nb.cells if c.cell_type == 'code']
    errors = sum(o.output_type == 'error' for c in code for o in c.outputs)
    assert errors == 0 and all(c.execution_count for c in code)
    record = dict(notebook=path.relative_to(ROOT).as_posix(), code_cells=len(code), errors=errors,
                  seconds=round(time.time() - started, 1), python=sys.executable)
    out = ROOT / 'results/tables/v3' / f'{number}_execution.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(f'PASS {number}: {len(code)} code cells, 0 errors, {record["seconds"]} s', flush=True)


if __name__ == '__main__':
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    os.environ['JUPYTER_PATH'] = str(ROOT / '.cache/jupyter') + os.pathsep + os.environ.get('JUPYTER_PATH', '')
    os.environ['MPLCONFIGDIR'] = str(ROOT / '.cache/matplotlib')
    for n in (sys.argv[1:] or sorted(NAMES)):
        run(n)
