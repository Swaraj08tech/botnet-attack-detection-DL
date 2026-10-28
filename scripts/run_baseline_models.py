"""Execute Notebook 06 in the project kernel and protect all prior files."""

import asyncio
import json
import os
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.data.load_data import sha256
from src.baseline_audit import protected_files,write_json


def main():
    if sys.prefix==sys.base_prefix: raise RuntimeError('Use project .venv Python')
    if sys.platform=='win32': asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    before={p.relative_to(ROOT).as_posix():sha256(p) for p in protected_files(ROOT)}
    cache=ROOT/'.cache/jupyter'; kernel=cache/'kernels/botnet-research'
    kernel.mkdir(parents=True,exist_ok=True)
    write_json(kernel/'kernel.json',dict(argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
        display_name='Botnet research (.venv)',language='python'))
    os.environ['JUPYTER_PATH']=str(cache)+os.pathsep+os.environ.get('JUPYTER_PATH','')
    os.environ['MPLCONFIGDIR']=str(ROOT/'.cache/matplotlib')
    path=ROOT/'notebooks/06_baseline_models.ipynb'; nb=nbformat.read(path,4)
    client=NotebookClient(nb,timeout=1200,startup_timeout=60,kernel_name='botnet-research',
        resources={'metadata':{'path':str(ROOT/'notebooks')}},allow_errors=False,
        on_cell_start=lambda cell,cell_index:print(f'Cell {cell_index+1}/{len(nb.cells)}',flush=True),
        on_cell_executed=lambda cell,cell_index,execute_reply:print('\n'.join(o.text.strip() for o in cell.outputs if o.output_type=='stream'),flush=True) if cell.cell_type=='code' else None)
    try: client.execute()
    finally:
        nbformat.write(nb,path)
        after={p.relative_to(ROOT).as_posix():sha256(p) for p in protected_files(ROOT)}
        assert before==after,'Earlier file changed'
    nbformat.validate(nb)
    code=[c for c in nb.cells if c.cell_type=='code']
    assert all(c.execution_count is not None for c in code)
    assert not any(o.output_type=='error' for c in code for o in c.outputs)
    manifest_path=ROOT/'data/processed/06/06_artifact_manifest.json'
    manifest=json.loads(manifest_path.read_text())
    for path,expected in manifest['artifact_sha256'].items(): assert sha256(ROOT/path)==expected,path
    write_json(ROOT/'results/tables/06_notebook_execution.json',dict(execution='nbclient/Jupyter in project .venv',
        python=sys.executable,working_directory=str(ROOT/'notebooks'),code_cells=len(code),errors=0,
        notebook_sha256=sha256(ROOT/'notebooks/06_baseline_models.ipynb'),artifact_manifest_sha256=sha256(manifest_path),
        verified_artifacts=len(manifest['artifact_sha256']),protected_before=before,protected_after=after,
        calibration_predicted=False,official_test_parsed=False))
    print(f'PASS: {len(code)} code cells; zero errors; {len(before)} earlier files unchanged; hashes verified.')


if __name__=='__main__': main()
