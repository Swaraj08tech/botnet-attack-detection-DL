"""Fresh-kernel execution and byte-preserving verification for Notebook 07."""
import asyncio
import os
import sys
from pathlib import Path
import nbformat
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.neural_reference_finish import preserved_files, verify_hashes, REVIEW, ORIGINAL
from src.research_revision.common import read_json, write_json
from src.data.load_data import sha256

def main():
    if sys.prefix==sys.base_prefix: raise RuntimeError('Use the project .venv')
    if sys.platform=='win32': asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    before=preserved_files(ROOT)
    seal_path=ROOT/f'docs/{ORIGINAL}/07_deliverable_manifest.json'
    if seal_path.exists():
        seal=read_json(seal_path);seal['status']='stale_execution_started';write_json(seal_path,seal)
    source_paths=['src/neural_reference.py','src/neural_reference_finish.py','src/neural_reference_seal.py',
        'scripts/run_neural_reference.py','scripts/verify_neural_reference.py',
        'scripts/finish_neural_reference_presentation.py','scripts/check_neural_runtime.py',
        'tests/test_neural_reference.py','tests/test_neural_reference_seal.py']
    source_hashes={name:sha256(ROOT/name) for name in source_paths}
    kernel=ROOT/'.cache/jupyter/kernels/botnet-research'; kernel.mkdir(parents=True,exist_ok=True)
    write_json(kernel/'kernel.json',dict(argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],display_name='Botnet research (.venv)',language='python'))
    os.environ['JUPYTER_PATH']=str(ROOT/'.cache/jupyter')+os.pathsep+os.environ.get('JUPYTER_PATH','')
    os.environ['MPLCONFIGDIR']=str(ROOT/'.cache/matplotlib')
    path=ROOT/'notebooks/07_neural_reference_model.ipynb'; nb=nbformat.read(path,4)
    # Clear in-memory execution state, never earlier notebooks.
    for c in nb.cells:
        if c.cell_type=='code': c.outputs=[]; c.execution_count=None
    def output(cell,cell_index,execute_reply):
        for item in cell.get('outputs',[]):
            if item.output_type=='stream': print(item.text.strip(),flush=True)
    client=NotebookClient(nb,timeout=7200,startup_timeout=60,kernel_name='botnet-research',resources={'metadata':{'path':str(ROOT/'notebooks')}},allow_errors=False,
        on_cell_start=lambda cell,cell_index:print(f'Notebook 07 cell {cell_index+1}/{len(nb.cells)}',flush=True),on_cell_executed=output)
    try: client.execute()
    finally:
        nbformat.write(nb,path)
        verify_hashes(ROOT,before);verify_hashes(ROOT,source_hashes)
    nbformat.validate(nb)
    code=[c for c in nb.cells if c.cell_type=='code']
    assert all(c.execution_count is not None for c in code)
    assert not any(o.output_type=='error' for c in code for o in c.outputs)
    manifest_path=ROOT/f'data/processed/{REVIEW}/07_artifact_manifest.json'; manifest=read_json(manifest_path)
    for name,h in manifest['artifact_sha256'].items(): assert sha256(ROOT/name)==h,name
    execution=dict(version=REVIEW,mode='Fresh-kernel replay of frozen models; no predictive model fit or retuning',
        notebook_sha256=sha256(path),python=sys.executable,sources_sha256=source_hashes,
        fresh_kernel=True,code_cells=len(code),errors=0,artifacts_verified=len(manifest['artifact_sha256']),protected_files=len(before),
        protected_unchanged=True,calibration_targets_loaded=False,official_test_records_loaded=False,manifest_sha256=sha256(manifest_path),
        independent_verification='Pending scripts/verify_neural_reference.py; not an ongoing validity claim')
    write_json(ROOT/f'results/tables/{ORIGINAL}/07_execution.json',execution)
    write_json(ROOT/f'results/tables/{REVIEW}/07_execution.json',execution)
    print(f'PASS: {len(code)} executed cells; no errors; {len(before)} earlier files unchanged.',flush=True)

if __name__=='__main__': main()
