"""A normal editor save must not inherit an obsolete verification claim."""
import json
import pytest
from src.data.load_data import sha256
from src.neural_reference_seal import require_executed_bytes, require_live_seal

def test_metadata_save_invalidates_executed_bytes_without_changing_record(tmp_path):
    p=tmp_path/'n.ipynb';p.write_text('{"metadata":{}}',encoding='utf-8')
    execution=dict(notebook_sha256=sha256(p),fresh_kernel=True,errors=0);old=execution.copy()
    assert require_executed_bytes(p,execution)==execution['notebook_sha256']
    p.write_text('{"metadata":{"kernelspec":"python3"}}',encoding='utf-8')
    with pytest.raises(RuntimeError,match='STALE execution record'):require_executed_bytes(p,execution)
    assert execution==old

def test_changed_artifact_invalidates_live_seal_without_resealing(tmp_path):
    p=tmp_path/'result.csv';p.write_text('F1\n0.52\n',encoding='utf-8')
    seal=tmp_path/'seal.json';seal.write_text(json.dumps(dict(status='verified_for_recorded_bytes',file_sha256={'result.csv':sha256(p)})),encoding='utf-8')
    original=seal.read_bytes();require_live_seal(tmp_path,seal)
    p.write_text('F1\n0.63\n',encoding='utf-8')
    with pytest.raises(RuntimeError,match='STALE seal'):require_live_seal(tmp_path,seal)
    assert seal.read_bytes()==original

def test_pending_execution_cannot_claim_completed_independent_verification(tmp_path):
    p=tmp_path/'seal.json';p.write_text(json.dumps(dict(status='stale_execution_started',file_sha256={})),encoding='utf-8')
    with pytest.raises(RuntimeError,match='No current completed'):require_live_seal(tmp_path,p)
