import numpy as np

from app.core.analysis import NEEDS_REVIEW_BELOW, SEVERITY, build_analysis, confidence_level, risk_level
from app.core.detector import CLASSES

N = CLASSES.index('Normal')
DOS, BACKDOOR, RECON = CLASSES.index('DoS'), CLASSES.index('Backdoor'), CLASSES.index('Reconnaissance')


def one_hot_sets(predicted, extra=None):
    sets = np.zeros((len(predicted), len(CLASSES)), dtype=bool)
    sets[np.arange(len(predicted)), predicted] = True
    for row, cls in (extra or []):
        sets[row, cls] = True
    return sets


def test_confidence_levels_boundaries():
    assert confidence_level(0.90) == 'high' and confidence_level(0.8999) == 'medium'
    assert confidence_level(0.70) == 'medium' and confidence_level(0.6999) == 'low' and NEEDS_REVIEW_BELOW == 0.70


def test_severity_covers_every_attack_class_and_normal_is_not_an_attack():
    assert set(SEVERITY) == set(CLASSES) and SEVERITY['Normal'] == 'None'
    assert {SEVERITY[c] for c in CLASSES if c != 'Normal'} <= {'Critical', 'High', 'Medium', 'Low'}


def test_risk_rules():
    assert risk_level(0.0, 0.0)['level'] == 'LOW'
    assert risk_level(0.03, 0.0)['level'] == 'MODERATE'
    assert risk_level(0.01, 0.001)['level'] == 'MODERATE'          # any serious flow
    assert risk_level(0.20, 0.0)['level'] == 'HIGH'
    assert risk_level(0.10, 0.05)['level'] == 'HIGH'
    assert risk_level(0.50, 0.0)['level'] == 'CRITICAL' and risk_level(0.3, 0.25)['level'] == 'CRITICAL'


def test_build_analysis_counts_by_hand():
    predicted = np.array([N, N, N, N, DOS, DOS, BACKDOOR, RECON])
    confidence = np.array([0.95, 0.92, 0.75, 0.60, 0.91, 0.65, 0.99, 0.80])
    sets = one_hot_sets(predicted, extra=[(5, BACKDOOR)])          # flow 6: DoS with Backdoor also on the shortlist
    a = build_analysis(predicted, confidence, sets)
    assert (a['total_flows'], a['safe_count'], a['malicious_count']) == (8, 4, 4)
    assert a['confidence_distribution'] == {'high': 4, 'medium': 2, 'low': 2} and a['needs_review_count'] == 2
    assert a['attack_type_uncertain_count'] == 1
    dos = next(d for d in a['attack_distribution'] if d['attack_type'] == 'DoS')
    assert dos['count'] == 2 and dos['percentage_of_malicious'] == 0.5 and abs(dos['average_confidence'] - 0.78) < 1e-9
    assert a['severity_distribution'] == {'Critical': 1, 'High': 2, 'Medium': 0, 'Low': 1}
    assert a['summary']['most_detected_attack'] == {'attack_type': 'DoS', 'count': 2}
    assert a['summary']['highest_confidence_attack']['attack_type'] == 'Backdoor'
    assert a['risk']['level'] == 'CRITICAL'                        # 3 of 8 flows are serious (>= 25%)
    assert abs(a['average_confidence'] - confidence.mean()) < 1e-12


def test_evaluation_only_with_labels():
    predicted = np.array([N, DOS])
    sets = one_hot_sets(predicted)
    assert 'evaluation' not in build_analysis(predicted, np.array([0.9, 0.9]), sets)
    ev = build_analysis(predicted, np.array([0.9, 0.9]), sets, truth=np.array(['Normal', 'Backdoor']))['evaluation']
    assert ev['attacks'] == 1 and ev['attacks_caught'] == 1 and ev['exact_class_correct'] == 1
