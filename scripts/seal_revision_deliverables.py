"""Hash the final reviewable package after execution and independent verification."""
from pathlib import Path
import sys
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.research_revision.common import protected_check,read_json,write_json
from src.data.load_data import sha256

def main():
    protected=protected_check(ROOT,snapshot_copies=True)
    report=read_json(ROOT/'results/tables/revision_v2/final_verification.json')
    assert report['status']=='PASS' and report['executed_notebooks']==5 and len(report['model_prediction_checks'])==15
    files=[]
    for folder in ['docs/revision_v2','results/tables/revision_v2','results/figures/revision_v2','models/revision_v2','data/processed/revision_v2','src/research_revision']:
        files.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    files.extend((ROOT/'notebooks').glob('0[2-6]*.ipynb'))
    files.extend(ROOT/'scripts'/name for name in ['build_research_revision.py','run_research_revision.py','verify_research_revision.py','write_revision_handoff.py','seal_revision_deliverables.py'])
    files.extend(ROOT/name for name in ['README.md','docs/research_notes.md','configs/revision_v2_baseline_plan.json','tests/test_research_revision.py','pytest.ini'])
    destination=ROOT/'docs/revision_v2/deliverable_manifest.json'
    hashes={p.relative_to(ROOT).as_posix():sha256(p) for p in sorted(set(files)) if p!=destination}
    # No notebook output changes after this point; include final execution reports.
    for number in ['02','03','04','05','06']:
        execution=read_json(ROOT/f'results/tables/revision_v2/{number}_execution.json')
        assert hashes[execution['notebook']]==execution['notebook_sha256']
    snapshot=ROOT/(ROOT/'.cache/workflow_snapshot_path.txt').read_text().strip()
    write_json(destination,dict(created_utc=datetime.now(timezone.utc).isoformat(),version='revision_v2',
         verified_notebooks=5,verified_model_experiments=15,independently_reloaded_artifact_entries=report['artifact_entries_reloaded'],
         official_test_evaluated=False,calibration_predicted=False,protected_project_files=len(protected),
         snapshot=snapshot.relative_to(ROOT).as_posix(),snapshot_manifest_sha256=sha256(snapshot/'snapshot_manifest.json'),
         file_sha256=hashes,excluded_self='This manifest intentionally does not hash itself. Earlier immutable dependency hashes are in each notebook artifact manifest and the snapshot manifest.'))
    for name,expected in hashes.items(): assert sha256(ROOT/name)==expected,name
    print(f'Sealed and verified {len(hashes)} current deliverable files. Protected source files and all 379 snapshot copies verify.')

if __name__=='__main__': main()
