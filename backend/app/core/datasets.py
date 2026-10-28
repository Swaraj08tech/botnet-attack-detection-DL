"""Uploaded datasets: validation before analysis, and staging of the file until the user clicks "Analyze".

Validation gives user-facing messages only; technical details go to the server log.
"""
from __future__ import annotations

import io
import json
import logging
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from app.core.analysis import true_labels
from app.core.detector import CATEGORICAL

log = logging.getLogger('botnet-api.datasets')
STAGE_HOURS = 24          # staged uploads older than this are deleted


def _error(title: str, message: str, items: list | None = None) -> dict:
    return {'title': title, 'message': message, 'items': items or []}


def validate_dataset(content: bytes, filename: str, detector, max_rows: int, descriptions: dict) -> tuple[dict, pd.DataFrame | None]:
    """Check that a CSV can be analysed by the trained model. Returns (report, frame or None)."""
    report = {'filename': filename, 'size_bytes': len(content), 'rows': None, 'columns': None, 'valid': False,
              'errors': [], 'warnings': [], 'has_labels': False, 'required_features': len(detector.predictors)}
    if not filename.lower().endswith('.csv'):
        report['errors'].append(_error('Unsupported file type', 'Please upload a .csv file (comma-separated values).'))
        return report, None
    try:
        frame = pd.read_csv(io.BytesIO(content), low_memory=False)
    except Exception:                                                           # noqa: BLE001  any parser failure -> one clear message
        log.exception('could not read uploaded CSV %s', filename)
        report['errors'].append(_error('File could not be read', 'The file is not a readable CSV. Check that it is a comma-separated text '
                                                                  'file with one header row.'))
        return report, None
    frame.columns = [str(c).strip() for c in frame.columns]
    report['rows'], report['columns'] = int(len(frame)), int(frame.shape[1])
    if len(frame) == 0:
        report['errors'].append(_error('Empty dataset', 'The file has a header but no network flows.'))
        return report, None
    if len(frame) > max_rows:
        report['errors'].append(_error('Dataset too large', f'The file has {len(frame):,} flows; the limit is {max_rows:,}. Split it into smaller files.'))
        return report, None

    missing = [c for c in detector.predictors if c not in frame.columns]
    if missing:
        report['errors'].append(_error('Missing required features',
                                       f'The dataset is missing {len(missing)} of the {len(detector.predictors)} features required by the trained '
                                       'detection model (UNSW-NB15 flow features).',
                                       [{'name': c, 'description': descriptions.get(c, '')} for c in missing]))
        return report, None

    bad_numbers, bad_infinite, bad_negative = [], [], []
    for c in detector.numeric:
        values = pd.to_numeric(frame[c], errors='coerce')
        invalid = values.isna() & frame[c].notna()
        empty = frame[c].isna()
        if invalid.any() or empty.any():
            bad_numbers.append({'name': c, 'description': f'{int(invalid.sum() + empty.sum()):,} flows with an empty or non-numeric value '
                                                           f'(first at flow {int((invalid | empty).to_numpy().argmax()) + 1})'})
            continue
        array = values.to_numpy(dtype=float)
        if not np.isfinite(array).all():
            bad_infinite.append({'name': c, 'description': f'{int((~np.isfinite(array)).sum()):,} infinite values'})
        elif (array < 0).any():
            bad_negative.append({'name': c, 'description': f'{int((array < 0).sum()):,} negative values'})
    if bad_numbers:
        report['errors'].append(_error('Invalid values', 'These features must contain a number in every flow.', bad_numbers))
    if bad_infinite:
        report['errors'].append(_error('Infinite values', 'These features contain infinite values, which the model cannot use.', bad_infinite))
    if bad_negative:
        report['errors'].append(_error('Negative values', 'These features are counts, sizes or times and cannot be negative.', bad_negative))

    for c in CATEGORICAL:
        known = set(detector.vocab[c])
        values = frame[c].astype(str)
        unseen = values[~values.isin(known)]
        if len(unseen):
            examples = ', '.join(sorted(unseen.unique())[:5])
            report['warnings'].append(f'{len(unseen):,} flows have a "{c}" value the model never saw in training ({examples}); '
                                      'it is treated as unknown.')
    extra = [c for c in frame.columns if c not in detector.predictors and c not in ('id', 'label', 'attack_cat', 'ground_truth')]
    if extra:
        report['warnings'].append(f'{len(extra)} extra column(s) will be ignored: {", ".join(extra[:6])}{" ..." if len(extra) > 6 else ""}.')
    if true_labels(frame) is not None:
        report['has_labels'] = True
        report['warnings'].append('True labels found (ground_truth / attack_cat): the results will also compare the predictions with them. '
                                  'The labels are never used to make the predictions.')
    report['valid'] = not report['errors']
    return report, (frame if report['valid'] else None)


class UploadStore:
    """Validated uploads waiting for analysis, kept as files under <data_dir>/uploads."""

    def __init__(self, directory: Path):
        self.dir = directory
        self.dir.mkdir(parents=True, exist_ok=True)

    def save(self, content: bytes, report: dict) -> str:
        self.cleanup()
        upload_id = uuid.uuid4().hex
        (self.dir / f'{upload_id}.csv').write_bytes(content)
        (self.dir / f'{upload_id}.json').write_text(json.dumps(report), encoding='utf-8')
        return upload_id

    def load(self, upload_id: str) -> tuple[bytes, dict] | None:
        if not upload_id.isalnum():
            return None
        data, meta = self.dir / f'{upload_id}.csv', self.dir / f'{upload_id}.json'
        if not data.exists() or not meta.exists():
            return None
        return data.read_bytes(), json.loads(meta.read_text(encoding='utf-8'))

    def delete(self, upload_id: str):
        for suffix in ('.csv', '.json'):
            (self.dir / f'{upload_id}{suffix}').unlink(missing_ok=True)

    def cleanup(self):
        limit = time.time() - STAGE_HOURS * 3600
        for path in self.dir.glob('*'):
            if path.stat().st_mtime < limit:
                path.unlink(missing_ok=True)
