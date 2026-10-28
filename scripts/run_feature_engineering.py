"""Execute Notebook 04 in the project environment and independently guard prior artifacts."""

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
from src.data.feature_engineering_audit import protected_files


def main():
    if sys.prefix==sys.base_prefix: raise RuntimeError('Use project .venv Python')
    if sys.platform=='win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    before={p.relative_to(ROOT).as_posix():sha256(p) for p in protected_files(ROOT)}
    cache=ROOT/'.cache/jupyter'
    kernel=cache/'kernels/botnet-research'
    kernel.mkdir(parents=True,exist_ok=True)
    (kernel/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
        'display_name':'Botnet research (.venv)','language':'python'}),encoding='utf-8')
    os.environ['JUPYTER_PATH']=str(cache)+os.pathsep+os.environ.get('JUPYTER_PATH','')
    os.environ['MPLCONFIGDIR']=str(ROOT/'.cache/matplotlib')
    path=ROOT/'notebooks/04_feature_engineering.ipynb'
    notebook=nbformat.read(path,as_version=4)
    workdir=ROOT/'notebooks'
    client=NotebookClient(notebook,timeout=600,startup_timeout=60,kernel_name='botnet-research',
        resources={'metadata':{'path':str(workdir)}},allow_errors=False,
        on_cell_start=lambda cell,cell_index:print(f'Cell {cell_index+1}/{len(notebook.cells)}',flush=True))
    try: client.execute()
    finally:
        nbformat.write(notebook,path)
        after={p.relative_to(ROOT).as_posix():sha256(p) for p in protected_files(ROOT)}
        assert before==after,'Earlier input/artifact changed'
    nbformat.validate(notebook)
    code=[c for c in notebook.cells if c.cell_type=='code']
    assert all(c.execution_count is not None for c in code)
    assert not any(o.output_type=='error' for c in code for o in c.outputs)
    manifest_path=ROOT/'data/processed/04/04_artifact_manifest.json'
    manifest=json.loads(manifest_path.read_text())
    for relative,expected in manifest['artifact_sha256'].items():
        assert sha256(ROOT/relative)==expected,relative
    report=dict(execution='nbclient/Jupyter with project .venv',python=sys.executable,
        working_directory=str(workdir),code_cells=len(code),executed_cells=len(code),errors=0,
        notebook_sha256=sha256(path),artifact_manifest_sha256=sha256(manifest_path),
        verified_artifacts=len(manifest['artifact_sha256']),protected_before=before,protected_after=after,
        vscode_ui_execution_verified=False)
    (ROOT/'results/tables/04_notebook_execution.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(f"PASS: {len(code)} cells; zero errors; {len(before)} protected files unchanged; artifact hashes verified.")


if __name__=='__main__': main()
