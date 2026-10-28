"""Write serving/bundle/class_reference.json (how typical each value is for each class) without rebuilding the rest of the bundle.

The model, calibration and preprocessing files are not touched. Uses the fitting (train) role of the official training file only, the
same rows as feature_reference.json; the test file is never read. The full bundle build (build_serving_bundle.py) writes the same file.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from build_serving_bundle import OUT, class_reference, load_raw  # noqa: E402

if __name__ == '__main__':
    predictors = json.loads((OUT / 'preprocessing.json').read_text(encoding='utf-8'))['predictors']
    reference = class_reference(load_raw('train'), predictors, numeric=[])
    (OUT / 'class_reference.json').write_text(json.dumps(reference), encoding='utf-8')
    print('class_reference.json', reference['rows'], 'rows;', len(reference['discrete']), 'discrete and', len(reference['continuous']), 'continuous columns')
