"""Build three explicitly model-curated UI demos, not accuracy benchmarks.

Reuses the existing serving demo pool. Never changes source flows, labels, the
detector, thresholds, or the original unfiltered samples/test_flows.csv.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))

from app.core.analysis import build_analysis, true_labels  # noqa: E402
from app.core.datasets import validate_dataset  # noqa: E402
from app.core.detector import CLASSES, Detector  # noqa: E402

SEED = 20261028
SOURCE = ROOT / 'serving/bundle/demo_flows.csv'
BUNDLE = ROOT / 'serving/bundle'
DEST = ROOT / 'samples'
PLANS = [
    ('Normal', 'LOW', {'Normal': 50}),
    ('Moderate', 'MODERATE', {'Normal': 45, 'Generic': 3, 'Fuzzers': 2}),
    ('Critical', 'CRITICAL', {'Normal': 20, 'Backdoor': 10, 'Shellcode': 10, 'Worms': 10}),
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(detector: Detector, frame: pd.DataFrame) -> tuple[dict, np.ndarray]:
    pred = detector.predict(frame)
    confidence = pred.proba[np.arange(len(frame)), pred.predicted]
    return build_analysis(pred.predicted, confidence, pred.sets, true_labels(frame)), pred.predicted


def main() -> None:
    protected_paths = [SOURCE, DEST / 'test_flows.csv',
                       ROOT / 'backend/app/core/detector.py', ROOT / 'backend/app/core/analysis.py',
                       ROOT / 'backend/app/core/datasets.py',
                       *sorted(p for p in BUNDLE.rglob('*') if p.is_file())]
    before = {p.relative_to(ROOT).as_posix(): sha(p) for p in protected_paths}
    detector = Detector(BUNDLE)
    source = pd.read_csv(SOURCE)
    original = pd.read_csv(DEST / 'test_flows.csv')
    original_summary, _ = summary(detector, original)
    source_summary, predictions = summary(detector, source)
    predicted_names = np.asarray(CLASSES)[predictions]
    correct = predicted_names == source.ground_truth.to_numpy()
    rng = np.random.default_rng(SEED)
    pools = {cls: rng.permutation(np.flatnonzero(correct & (source.ground_truth.to_numpy() == cls))).tolist()
             for cls in CLASSES}
    used: set[int] = set()
    results = []
    contents: dict[Path, bytes] = {}

    for name, expected_risk, composition in PLANS:
        chosen = []
        for cls, count in composition.items():
            available = [i for i in pools[cls] if i not in used]
            if len(available) < count:
                raise RuntimeError(f'{cls}: need {count} unused correct examples, found {len(available)}. '
                                   'Do not silently change labels or relax this plan.')
            take = available[:count]
            chosen.extend(take)
            used.update(take)
        chosen = rng.permutation(chosen).tolist()
        frame = source.iloc[chosen][detector.predictors + ['ground_truth']].reset_index(drop=True)
        # Uploads contain measurements only; reference labels stay in the sidecar.
        payload = frame[detector.predictors].to_csv(index=False, lineterminator='\n').encode('utf-8')
        filename = f'{name.lower()}.csv'
        validation, reloaded = validate_dataset(payload, filename, detector, 10000, {})
        assert validation['valid'], validation['errors']
        assert not validation['has_labels'] and reloaded is not None
        assert list(reloaded.columns) == detector.predictors
        assert len(frame) == 50 and len(frame.columns) == 43
        assert list(frame.columns) == detector.predictors + ['ground_truth']
        assert np.isfinite(frame[detector.numeric].to_numpy(dtype=float)).all()
        # Attach reference labels in memory for verification only, not in the CSV.
        labelled_check = reloaded.assign(ground_truth=frame.ground_truth.to_numpy())
        a, p = summary(detector, labelled_check)
        assert np.array_equal(np.asarray(CLASSES)[p], frame.ground_truth.to_numpy())
        assert a['risk']['level'] == expected_risk, (name, a['risk'])
        assert a['evaluation']['exact_class_correct'] == len(frame)
        pd.testing.assert_frame_equal(detector.validate(frame), detector.validate(reloaded), check_exact=False,
                                      rtol=1e-12, atol=1e-12)
        assert np.array_equal(detector.predict(reloaded).predicted, p)
        assert np.allclose(detector.predict(reloaded).proba, detector.predict(labelled_check).proba,
                           rtol=0, atol=0)
        path = DEST / filename
        if path.exists() and path.read_bytes() != payload:
            raise FileExistsError(f'{path} differs; preserve it and explicitly version new demos instead.')
        contents[path] = payload
        results.append({
            'file': path.relative_to(ROOT).as_posix(),
            'sha256': hashlib.sha256(payload).hexdigest(),
            'rows': len(frame),
            'distinct_feature_patterns': int(len(frame[detector.predictors].drop_duplicates())),
            'label_counts': composition,
            'expected_batch_priority': expected_risk,
            'actual_summary': a,
            'source_rows': [{'output_row_1based': n + 1, 'source_row_0based': int(i),
                             'source_id': int(source.iloc[i]['id']) if 'id' in source else None,
                             'ground_truth': str(source.iloc[i]['ground_truth'])}
                            for n, i in enumerate(chosen)],
            'checks': {'app_csv_validation': True, 'round_trip_alignment': True,
                       'labels_excluded_from_inference': True, 'filename_not_used_for_inference': True,
                       'finite_numeric_values': True, 'expected_priority_matches': True,
                       'upload_contains_only_42_predictors': True, 'upload_has_no_reference_labels': True},
        })

    assert len(used) == sum(x['rows'] for x in results), 'Source rows must be disjoint across demos'
    source_eligible = {c: int(((source.ground_truth.to_numpy() == c) & correct).sum()) for c in CLASSES}
    manifest = {
        'schema_version': 2,
        'upload_format': '42 predictor columns only; row-linked labels are in files[].source_rows[].ground_truth.',
        'purpose': 'Model-curated demonstrations for UI/report testing, NOT independent accuracy estimates.',
        'selection': 'Select only source rows whose current model prediction matches ground_truth, '
                     'then sample without replacement using a fixed seed and fixed class counts. '
                     'Model-dependent selection is deliberate and explicitly disclosed.',
        'seed': SEED,
        'source': SOURCE.relative_to(ROOT).as_posix(),
        'source_rows': len(source),
        'source_provenance': 'Existing serving demo_flows.csv, originally sampled from official UNSW-NB15 test records. '
                             'This script reads no raw CSV and performs no new full-test evaluation.',
        'source_unfiltered_summary': source_summary,
        'correctly_predicted_pool_counts': source_eligible,
        'original_unfiltered_sample_summary': original_summary,
        'protected_files_sha256': before,
        'disjoint_source_rows_across_new_files': True,
        'files': results,
    }
    manifest_path = DEST / 'scenario_manifest.json'
    manifest_payload = (json.dumps(manifest, indent=2, ensure_ascii=True) + '\n').encode('utf-8')
    if manifest_path.exists() and manifest_path.read_bytes() != manifest_payload:
        raise FileExistsError(f'{manifest_path} differs; explicitly version the previous provenance first.')
    assert all(sha(ROOT / rel) == value for rel, value in before.items()), 'A protected input changed'
    DEST.mkdir(exist_ok=True)
    for path, payload in contents.items():
        path.write_bytes(payload)
    manifest_path.write_bytes(manifest_payload)

    for result in results:
        path = ROOT / result['file']
        assert sha(path) == result['sha256']
        saved = pd.read_csv(path)
        assert list(saved.columns) == detector.predictors
        labels = [row['ground_truth'] for row in result['source_rows']]
        a, _ = summary(detector, saved.assign(ground_truth=labels))
        assert a == result['actual_summary']
        print(f"PASS {path.name}: {len(saved)} rows, priority {a['risk']['level']}, "
              f"Normal {a['safe_count']}, attack {a['malicious_count']}, "
              f"class agreement {a['evaluation']['exact_class_correct']}/{len(saved)} (curated)")
    assert all(sha(ROOT / rel) == value for rel, value in before.items()), 'A protected input changed'
    e = original_summary['evaluation']
    print(f"Original unfiltered test_flows.csv unchanged: class agreement {e['exact_class_correct']}/{len(original)}; "
          f"Normal falsely flagged {e['normal_flagged']}/{e['normal']}.")
    print('PASS saved/reloaded app validation, labels/filename excluded from inference, '
          'disjoint source rows, original sample and serving bundle unchanged.')


if __name__ == '__main__':
    main()
