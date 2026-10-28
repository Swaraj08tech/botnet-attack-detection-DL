"""Execute revised notebooks in the project kernel, with protected-byte checks."""
import asyncio
import os
import sys
from pathlib import Path
import argparse
import nbformat
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.research_revision.common import protected_check,read_json,write_json
from src.data.load_data import sha256

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('number',choices=['02','03','04','05','06']); args=parser.parse_args()
    if sys.prefix==sys.base_prefix: raise RuntimeError('Use project .venv')
    if sys.platform=='win32': asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    before=protected_check(ROOT)
    kernel=ROOT/'.cache/jupyter/kernels/botnet-research'; kernel.mkdir(parents=True,exist_ok=True)
    write_json(kernel/'kernel.json',dict(argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],display_name='Botnet research (.venv)',language='python'))
    os.environ['JUPYTER_PATH']=str(ROOT/'.cache/jupyter')+os.pathsep+os.environ.get('JUPYTER_PATH','')
    os.environ['MPLCONFIGDIR']=str(ROOT/'.cache/matplotlib')
    path=next((ROOT/'notebooks').glob(args.number+'_*.ipynb')); nb=nbformat.read(path,4)
    def output(cell,cell_index,execute_reply):
        if cell.cell_type=='code':
            for item in cell.outputs:
                if item.output_type=='stream': print(item.text.strip(),flush=True)
    client=NotebookClient(nb,timeout=7200,startup_timeout=60,kernel_name='botnet-research',resources={'metadata':{'path':str(ROOT/'notebooks')}},allow_errors=False,
        on_cell_start=lambda cell,cell_index:print(f'Notebook {args.number}: cell {cell_index+1}/{len(nb.cells)}',flush=True),on_cell_executed=output)
    try: client.execute()
    finally:
        nbformat.write(nb,path); assert protected_check(ROOT)==before
    nbformat.validate(nb); code=[c for c in nb.cells if c.cell_type=='code']
    assert all(c.execution_count is not None for c in code) and not any(o.output_type=='error' for c in code for o in c.outputs)
    manifest_path=ROOT/f'data/processed/revision_v2/{args.number}_artifact_manifest.json'; manifest=read_json(manifest_path)
    for name,expected in manifest['artifact_sha256'].items(): assert sha256(ROOT/name)==expected,name
    write_json(ROOT/f'results/tables/revision_v2/{args.number}_execution.json',dict(notebook=path.relative_to(ROOT).as_posix(),notebook_sha256=sha256(path),
        python=sys.executable,code_cells=len(code),errors=0,manifest_sha256=sha256(manifest_path),verified_artifacts=len(manifest['artifact_sha256']),protected_files=len(before),protected_unchanged=True))
    print(f'PASS Notebook {args.number}: {len(code)} executed cells, zero errors, protected files unchanged.',flush=True)

if __name__=='__main__': main()
