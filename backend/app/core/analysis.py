"""One analysis of an uploaded dataset: every number the Results page, the charts and the PDF report show comes from here.

All application rules live in this module (and nowhere else):
  * CONFIDENCE_LEVELS  - how a flow's confidence (calibrated probability of its predicted class) is named
  * NEEDS_REVIEW_BELOW - flows below this confidence are flagged for review
  * SEVERITY           - fixed severity of each attack class (an application rule, not a model output)
  * risk_level()       - overall network status of a dataset

"Confidence" is the model's own calibrated probability for the class it predicts (LightGBM + temperature scaling, Notebook 21).
It is NOT the model's test accuracy. "Attack type uncertain" uses the conformal prediction set (90% shortlist, Notebook 17).
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from app.core.detector import CLASSES

NORMAL = 'Normal'
ATTACK_CLASSES = [c for c in CLASSES if c != NORMAL]

# ------------------------------------------------------------------------------------------------ rules (single place)
CONFIDENCE_LEVELS = [('high', 0.90), ('medium', 0.70), ('low', 0.0)]      # level applies when confidence >= its lower bound
NEEDS_REVIEW_BELOW = 0.70

# Severity by attack class, from the UNSW-NB15 class descriptions (Moustafa & Slay, 2015) and the impact of the mapped ATT&CK techniques:
#   Critical: the attacker gains code execution or persistent access, or the attack spreads by itself
#   High:     exploitation of a vulnerability, or loss of availability
#   Medium:   probing for flaws or attacks on cryptography, without direct compromise
#   Low:      information gathering (scans, probes)
SEVERITY = {'Backdoor': 'Critical', 'Shellcode': 'Critical', 'Worms': 'Critical',
            'Exploits': 'High', 'DoS': 'High',
            'Fuzzers': 'Medium', 'Generic': 'Medium',
            'Reconnaissance': 'Low', 'Analysis': 'Low',
            NORMAL: 'None'}
SEVERITY_ORDER = ['Critical', 'High', 'Medium', 'Low']
SEVERITY_REASON = {
    'Critical': 'code execution, persistent remote access or self-spreading code',
    'High': 'exploitation of a vulnerability or loss of availability',
    'Medium': 'probing for flaws or attacks on cryptography, without direct compromise',
    'Low': 'information gathering such as scans and probes',
}

# Overall network status, checked from the top; `serious` = flows of Critical or High severity.
RISK_RULES = [
    ('CRITICAL', 'serious flows >= 25% of all flows, or malicious flows >= 50%', lambda m, s: s >= 0.25 or m >= 0.50),
    ('HIGH', 'serious flows >= 5%, or malicious flows >= 20%', lambda m, s: s >= 0.05 or m >= 0.20),
    ('MODERATE', 'any serious flow, or malicious flows >= 2%', lambda m, s: s > 0 or m >= 0.02),
    ('LOW', 'none of the above', lambda m, s: True),
]


def confidence_level(p: float) -> str:
    return next(name for name, lower in CONFIDENCE_LEVELS if p >= lower)


def rules() -> dict:
    """The rules, as shown to users (tooltips, Model page, PDF)."""
    return {'confidence_levels': {'high': '>= 90%', 'medium': '70% to 89.9%', 'low': '< 70%'},
            'needs_review': f'confidence below {NEEDS_REVIEW_BELOW:.0%}',
            'attack_type_uncertain': 'predicted as an attack, but the 90% shortlist (conformal prediction set) contains more than one attack type',
            'severity': {c: SEVERITY[c] for c in ATTACK_CLASSES}, 'severity_reason': SEVERITY_REASON,
            'risk': [{'level': level, 'rule': text} for level, text, _ in RISK_RULES],
            'confidence_meaning': 'the calibrated probability the model gives to the class it predicts; not the model accuracy'}


def risk_level(malicious_share: float, serious_share: float) -> dict:
    level, rule = next((lv, text) for lv, text, test in RISK_RULES if test(malicious_share, serious_share))
    explanation = {
        'CRITICAL': 'A large share of the analysed flows was classified as malicious or as serious attacks. Investigate the high-confidence '
                    'Critical and High severity threats first.',
        'HIGH': 'A notable share of the flows was classified as malicious. Review the serious threats and the affected hosts.',
        'MODERATE': 'Some malicious flows were found. Review them, starting with the highest severity and confidence.',
        'LOW': 'Few or no malicious flows were found. Keep monitoring; review any low-confidence predictions.',
    }[level]
    return {'level': level, 'rule': rule, 'explanation': explanation}


# ------------------------------------------------------------------------------------------------ per-flow fields
def share(n: float, d: float) -> float:
    return float(n / d) if d else 0.0


def flow_fields(predicted: np.ndarray, confidence: np.ndarray, sets: np.ndarray) -> dict:
    """Vectorised per-flow labels used by the database rows and the summary."""
    classes = np.asarray(CLASSES)[predicted]
    malicious = classes != NORMAL
    attack_members = sets[:, [CLASSES.index(c) for c in ATTACK_CLASSES]].sum(axis=1)
    return {'classes': classes, 'malicious': malicious,
            'severity': np.array([SEVERITY[c] for c in classes]),
            'level': np.array([confidence_level(float(p)) for p in confidence]),
            'needs_review': confidence < NEEDS_REVIEW_BELOW,
            'type_uncertain': malicious & (attack_members > 1)}


# ------------------------------------------------------------------------------------------------ the analysis object
def status_confidence(proba: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Confidence about attack-or-normal: P(any attack) = 1 - P(Normal) for a predicted attack, P(Normal) for a predicted Normal flow.
    The same calibrated probabilities as the class confidence, summed over the nine attack classes (no new model or threshold)."""
    normal = proba[:, CLASSES.index(NORMAL)]
    return np.where(predicted != CLASSES.index(NORMAL), 1.0 - normal, normal)


def status_summary(status: np.ndarray) -> dict:
    levels = Counter(confidence_level(float(p)) for p in status)
    return {'status_confidence_distribution': {lv: int(levels.get(lv, 0)) for lv, _ in CONFIDENCE_LEVELS},
            'average_status_confidence': float(status.mean()) if len(status) else 0.0,
            'status_needs_review_count': int((status < NEEDS_REVIEW_BELOW).sum())}


def build_analysis(predicted: np.ndarray, confidence: np.ndarray, sets: np.ndarray, truth: np.ndarray | None = None,
                   status: np.ndarray | None = None) -> dict:
    f = flow_fields(predicted, confidence, sets)
    total = len(predicted)
    malicious = int(f['malicious'].sum())
    safe = total - malicious

    distribution = []
    for c in ATTACK_CLASSES:
        rows = f['classes'] == c
        n = int(rows.sum())
        if n:
            distribution.append({'attack_type': c, 'count': n, 'percentage_of_malicious': share(n, malicious), 'percentage_of_all': share(n, total),
                                 'average_confidence': float(confidence[rows].mean()), 'severity': SEVERITY[c],
                                 'needs_review': int(f['needs_review'][rows].sum())})
    distribution.sort(key=lambda d: (-d['count'], d['attack_type']))

    levels = Counter(f['level'].tolist())
    severity_counts = Counter(f['severity'][f['malicious']].tolist())
    serious = severity_counts.get('Critical', 0) + severity_counts.get('High', 0)
    most = distribution[0] if distribution else None
    top_confidence = max(distribution, key=lambda d: (d['average_confidence'], d['count'])) if distribution else None

    analysis = {
        'total_flows': total, 'safe_count': safe, 'malicious_count': malicious,
        'safe_percentage': share(safe, total), 'malicious_percentage': share(malicious, total),
        'average_confidence': float(confidence.mean()) if total else 0.0,
        'confidence_distribution': {lv: int(levels.get(lv, 0)) for lv, _ in CONFIDENCE_LEVELS},
        'needs_review_count': int(f['needs_review'].sum()),
        'attack_type_uncertain_count': int(f['type_uncertain'].sum()),
        'attack_distribution': distribution,
        'severity_distribution': {s: int(severity_counts.get(s, 0)) for s in SEVERITY_ORDER},
        'risk': risk_level(share(malicious, total), share(serious, total)),
        'summary': {'most_detected_attack': most and {'attack_type': most['attack_type'], 'count': most['count']},
                    'attack_types_detected': len(distribution),
                    'highest_confidence_attack': top_confidence and {'attack_type': top_confidence['attack_type'],
                                                                     'average_confidence': top_confidence['average_confidence']},
                    'flows_requiring_review': int(f['needs_review'].sum())},
        'rules': rules(),
        **(status_summary(status) if status is not None else {}),
    }
    if truth is not None:
        analysis['evaluation'] = evaluate(truth, f['classes'], sets)
    return analysis


def evaluate(truth: np.ndarray, predicted: np.ndarray, sets: np.ndarray) -> dict | None:
    """Only when the uploaded file itself carries true labels (ground_truth / attack_cat): how the predictions compare to them."""
    known = np.isin(truth, CLASSES)
    if not known.any():
        return None
    truth, predicted, sets = truth[known], predicted[known], sets[known]
    is_attack, flagged = truth != NORMAL, predicted != NORMAL
    in_set = np.array([sets[i, CLASSES.index(truth[i])] for i in range(len(truth))], dtype=bool)
    per_class = []
    for c in CLASSES:
        rows = truth == c
        if not rows.any():
            continue
        wrong = Counter(predicted[rows & (predicted != c)].tolist()).most_common(1)
        per_class.append({'class': c, 'flows': int(rows.sum()), 'correct': int((predicted[rows] == c).sum()), 'in_set': int(in_set[rows].sum()),
                          'flagged_as_attack': int(flagged[rows].sum()), 'confused_with': wrong[0][0] if wrong else None,
                          'confused_count': wrong[0][1] if wrong else 0})
    return {'labelled_flows': int(known.sum()), 'attacks': int(is_attack.sum()), 'attacks_caught': int((is_attack & flagged).sum()),
            'normal': int((~is_attack).sum()), 'normal_flagged': int((~is_attack & flagged).sum()),
            'attack_vs_normal_correct': int((is_attack == flagged).sum()), 'exact_class_correct': int((truth == predicted).sum()),
            'truth_in_set': int(in_set.sum()), 'per_class': per_class}


def true_labels(frame) -> np.ndarray | None:
    """True classes when the file has them: a `ground_truth` column (sample files) or UNSW-NB15's own `attack_cat` column."""
    for column in ('ground_truth', 'attack_cat'):
        if column in frame.columns:
            labels = frame[column].fillna('').astype(str).str.strip().to_numpy()
            if np.isin(labels, CLASSES).any():
                return labels
    return None


# ------------------------------------------------------------------------------------------------ grouped benchmark view
CONFUSABLE = ('DoS', 'Analysis', 'Backdoor')
GROUP_NAME = 'DoS / Analysis / Backdoor'


def grouped_evaluation(evaluation: dict | None) -> dict | None:
    """The official-test results with DoS, Analysis and Backdoor counted as ONE group, computed from the stored confusion matrix.

    In UNSW-NB15 many flows of these three classes have identical measurements but different labels, so no model can separate them
    (best possible F1 on the test file: Analysis 0.16, Backdoor 0.19). This view adds no predictions and uses no labels beyond the
    confusion matrix measured once; the 10-class results stay the official ones.
    """
    if not evaluation or 'confusion_matrix' not in evaluation:
        return None
    labels = evaluation['confusion_matrix']['labels']
    cm = np.asarray(evaluation['confusion_matrix']['matrix'])
    groups = [c for c in labels if c not in CONFUSABLE] + [GROUP_NAME]
    index = {c: (groups.index(GROUP_NAME) if c in CONFUSABLE else groups.index(c)) for c in labels}
    merged = np.zeros((len(groups), len(groups)), dtype=int)
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            merged[index[a], index[b]] += cm[i, j]
    per_class = []
    for k, c in enumerate(groups):
        tp, col, row = merged[k, k], merged[:, k].sum(), merged[k].sum()
        precision = tp / col if col else 0.0
        recall = tp / row if row else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append({'class': c, 'precision': float(precision), 'recall': float(recall), 'f1': float(f1), 'support': int(row)})
    within = cm[np.ix_([labels.index(c) for c in CONFUSABLE], [labels.index(c) for c in CONFUSABLE])]
    return {'groups': groups, 'per_class': per_class, 'accuracy': float(np.trace(merged) / merged.sum()),
            'macro_f1': float(np.mean([p['f1'] for p in per_class])), 'group': GROUP_NAME,
            'within_group_confusions': int(within.sum() - np.trace(within)),
            'note': 'DoS, Analysis and Backdoor counted as one group (computed from the official-test confusion matrix; no new evaluation). '
                    'The 10-class results remain the official scores.'}
