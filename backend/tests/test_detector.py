import numpy as np
import pandas as pd
import pytest

from app.core.detector import CLASSES, Detector, InputError
from tests.conftest import synthetic_flows


@pytest.fixture(scope='module')
def detector(bundle_dir):
    return Detector(bundle_dir)


def test_transform_layout(detector):
    x = detector.transform(synthetic_flows(5))
    assert x.shape == (5, len(detector.columns))
    onehot = x[:, len(detector.numeric):-2]
    assert np.allclose(onehot.sum(axis=1), 3)                    # exactly one category per text column


def test_unknown_category_becomes_zeros(detector):
    frame = synthetic_flows(1)
    frame['proto'] = 'never-seen'
    x = detector.transform(frame)
    proto_cols = [i for i, c in enumerate(detector.columns) if c.startswith('proto=')]
    assert x[0, proto_cols].sum() == 0


def test_engineered_features(detector):
    frame = synthetic_flows(1)
    frame[['sbytes', 'dbytes']] = [900, 100]
    x = detector.transform(frame)
    assert x[0, -2] == 1000 and x[0, -1] == pytest.approx(0.8)


def test_predictions_are_valid(detector):
    pred = detector.predict(synthetic_flows(50))
    assert np.allclose(pred.proba.sum(axis=1), 1)
    assert pred.sets.any(axis=1).all()                          # never an empty prediction set
    assert set(pred.predicted) <= set(range(len(CLASSES)))
    assert all(pred.decision(i) in {'confident', 'attack_type_uncertain', 'review'} for i in range(50))


@pytest.mark.parametrize('change, message', [(lambda f: f.drop(columns=['sttl']), 'missing columns'),
                                             (lambda f: f.assign(dur=-1.0), 'non-negative'),
                                             (lambda f: f.assign(dur='abc'), 'numeric')])
def test_input_validation(detector, change, message):
    with pytest.raises(InputError, match=message):
        detector.predict(change(synthetic_flows(3)))


def test_explanations(detector):
    frame = synthetic_flows(2)
    pred = detector.predict(frame)
    out = detector.explain(frame, pred.predicted, top=5)
    assert len(out) == 2 and len(out[0]) == 5
    assert all(item['feature'] in detector.predictors + detector.engineered for item in out[0])
    full = detector.explain(frame, pred.predicted, top=100)[0]
    raw_margin = detector.booster.predict(detector.transform(frame), raw_score=True)[0, pred.predicted[0]]
    expected = raw_margin - detector.booster.predict(detector.transform(frame), pred_contrib=True).reshape(2, 10, -1)[0, pred.predicted[0], -1]
    assert sum(i['contribution'] for i in full) == pytest.approx(expected, abs=1e-6)   # TreeSHAP adds up to the score


def test_extra_columns_are_ignored(detector):
    frame = synthetic_flows(3).assign(id=1, label=0, attack_cat='Normal')
    assert detector.predict(frame).proba.shape == (3, 10)
    assert isinstance(pd.DataFrame(), pd.DataFrame)
