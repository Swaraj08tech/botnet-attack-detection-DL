"""Record passing relevant tests bound to the tested source hashes."""
from pathlib import Path
import sys
import subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.data.load_data import sha256
from src.research_revision.common import write_json
from src.neural_reference_finish import REVIEW

def main():
    out=ROOT/f'results/tables/{REVIEW}';out.mkdir(parents=True,exist_ok=True)
    names=['src/neural_reference.py','src/neural_reference_finish.py','src/neural_reference_seal.py',
        'tests/test_neural_reference.py','tests/test_neural_reference_seal.py','scripts/run_neural_reference.py','scripts/verify_neural_reference.py']
    before={name:sha256(ROOT/name) for name in names}
    command=[sys.executable,'-m','pytest','-q','tests',f'--junitxml={out/"07_tests.xml"}']
    subprocess.run(command,cwd=ROOT,check=True)
    for name,h in before.items():assert sha256(ROOT/name)==h,name
    write_json(out/'07_test_provenance.json',dict(command=command,tested_sources_sha256=before,junit_sha256=sha256(out/'07_tests.xml')))

if __name__=='__main__':main()
