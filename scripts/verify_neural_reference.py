"""Verify the final executed bytes last; --check-seal is a read-only validity check."""
from pathlib import Path
import argparse
import contextlib
import datetime
import io
import sys
import xml.etree.ElementTree as ET
import nbformat
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src import neural_reference_finish as review
from src.neural_reference_seal import require_executed_bytes, require_live_seal
from src.research_revision.common import read_json, write_json
from src.data.load_data import sha256

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check-seal',action='store_true');args=parser.parse_args()
    seal_path=ROOT/'docs/neural_reference_v1/07_deliverable_manifest.json'
    if args.check_seal:
        seal=require_live_seal(ROOT,seal_path)
        print(f'PASS live seal: {len(seal["file_sha256"])} files match the exact bytes independently verified at {seal["verified_utc"]}.')
        return
    execution_path=ROOT/'results/tables/neural_reference_v1/07_execution.json';execution=read_json(execution_path)
    notebook=ROOT/'notebooks/07_neural_reference_model.ipynb'
    final_sha=require_executed_bytes(notebook,execution)
    assert execution['version']==review.REVIEW
    for name,h in execution['sources_sha256'].items():assert sha256(ROOT/name)==h,name
    manifest_path=ROOT/f'data/processed/{review.REVIEW}/07_artifact_manifest.json';manifest=read_json(manifest_path)
    assert sha256(manifest_path)==execution['manifest_sha256']
    for section in ('dependencies_sha256','artifact_sha256'):review.verify_hashes(ROOT,manifest[section])
    nb=nbformat.read(notebook,4);nbformat.validate(nb);code=[c for c in nb.cells if c.cell_type=='code']
    assert all(c.execution_count is not None and c.outputs for c in code)
    assert [c.execution_count for c in code]==list(range(1,len(code)+1))
    assert not any(o.output_type=='error' for c in code for o in c.outputs)
    with contextlib.redirect_stdout(io.StringIO()):
        s=review.load(ROOT,record=False);review.audit_search(s);review.verify_models(s);review.verify_controls(s)
    review.verify_hashes(ROOT,manifest['dependencies_sha256']);require_executed_bytes(notebook,execution)
    test_path=ROOT/f'results/tables/{review.REVIEW}/07_tests.xml'
    suites=list(ET.parse(test_path).getroot().iter('testsuite'))
    assert suites and all(int(x.attrib.get('errors',0))+int(x.attrib.get('failures',0))==0 for x in suites)
    test_count=sum(int(x.attrib['tests']) for x in suites)
    test_provenance=read_json(ROOT/f'results/tables/{review.REVIEW}/07_test_provenance.json')
    assert test_provenance['junit_sha256']==sha256(test_path);review.verify_hashes(ROOT,test_provenance['tested_sources_sha256'])
    runtime=read_json(ROOT/f'docs/{review.REVIEW}/07_runtime_check.json');assert runtime['security_policy_changed'] is False
    verified=datetime.datetime.now(datetime.timezone.utc).isoformat();files={}
    for base in ('data/processed','results/tables','docs'):
        for p in (ROOT/base/review.REVIEW).rglob('*'):
            if p.is_file() and p.name!='07_deliverable_manifest.json':files[p.relative_to(ROOT).as_posix()]=sha256(p)
    for name in manifest['artifact_sha256']:files[name]=sha256(ROOT/name)
    files.update(manifest['dependencies_sha256']);files.update(execution['sources_sha256'])
    files[notebook.relative_to(ROOT).as_posix()]=final_sha;files[execution_path.relative_to(ROOT).as_posix()]=sha256(execution_path)
    files['scripts/run_neural_reference_tests.py']=sha256(ROOT/'scripts/run_neural_reference_tests.py')
    result=dict(version=review.REVIEW,status='verified_for_recorded_bytes',verified_utc=verified,
        validity='Historical claim for these SHA-256 values only. Any later save requires a live check and, if stale, tests + fresh-kernel replay + independent verification.',
        verified_notebook_sha256=final_sha,fresh_kernel=True,executed_code_cells=len(code),notebook_errors=0,
        independently_reloaded_MLP_models=6,independently_reloaded_LightGBM_controls=2,fold_local_preprocessing_checks=3,
        tests_passed=test_count,protected_files=len(s['protected']),no_model_retuning_or_new_predictive_fit=True,
        calibration_targets_loaded=False,official_test_records_loaded=False,runtime_smoke_test_status=runtime['status'],
        file_sha256=files,excluded_self=True)
    write_json(seal_path,result);write_json(ROOT/f'docs/{review.REVIEW}/07_deliverable_manifest.json',result)
    require_live_seal(ROOT,seal_path)
    print(f'PASS FINAL: {len(code)} fresh-kernel cells; six MLP models; two exact LightGBM controls; three fold-local preprocessing checks; {test_count} tests; {len(s["protected"])} protected files unchanged.')
    print('Final saved notebook SHA-256:',final_sha)
    print('This seal is valid only while its recorded file hashes match; --check-seal performs a read-only live check.')

if __name__=='__main__':main()
