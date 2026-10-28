"""An explicit, read-only live check of a verification claim's exact bytes."""
from pathlib import Path
from src.data.load_data import sha256
from src.research_revision.common import read_json

REFRESH='Run the Notebook 07 tests, scripts/run_neural_reference.py, then scripts/verify_neural_reference.py. Do not edit execution hashes by hand.'

def require_executed_bytes(path, execution):
    actual=sha256(path)
    if actual!=execution['notebook_sha256']:
        raise RuntimeError(f'STALE execution record: saved notebook SHA-256 is {actual}, but executed bytes were {execution["notebook_sha256"]}. '+REFRESH)
    if not execution.get('fresh_kernel') or execution.get('errors')!=0:
        raise RuntimeError('A successful fresh-kernel execution is required. '+REFRESH)
    return actual

def require_live_seal(root, seal_path):
    root=Path(root);seal=read_json(seal_path)
    if seal.get('status')!='verified_for_recorded_bytes':
        raise RuntimeError('No current completed independent seal. '+REFRESH)
    changed=[name for name,h in seal['file_sha256'].items() if not (root/name).is_file() or sha256(root/name)!=h]
    if changed: raise RuntimeError('STALE seal: these recorded files changed: '+', '.join(changed)+'. '+REFRESH)
    return seal
