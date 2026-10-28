"""Build the serving bundle for detector v2 (the web application's model) and verify it.

The bundle holds no pickles: JSON preprocessing, LightGBM's native text model, calibration, conformal thresholds,
the ATT&CK mapping, the knowledge-base passages, a small demo sample and a manifest with SHA-256 hashes.

Checks (the script stops if one fails):
  1. the lean serving detector (backend/app/core/detector.py) reproduces the research detector v2 exactly
     on the model_selection rows and the official test file (same probabilities, same predicted class);
  2. the calibration and conformal settings come only from the calibration rows (as in Notebooks 17 and 21).

Usage:  .venv/Scripts/python.exe scripts/build_serving_bundle.py   ->  serving/bundle/
"""
import datetime
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'backend'))
from src.final_detector import FinalDetector, load_raw, apply_temperature           # noqa: E402  research code
from src.notebook_helpers import CLASSES, MODELS, TABLES, read_json                    # noqa: E402
from app.core.detector import Detector                                               # noqa: E402  serving code

OUT = ROOT / 'serving/bundle'
ALPHA = 0.10


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean_text(text):
    text = str(text).replace('\uf0b7', '-').replace('\ufffd', '-').replace('\u2022', '-')
    return re.sub(r'\s+', ' ', text).strip()


def attack_mitigations():
    """ATT&CK mitigations linked to each technique, with the technique-specific advice of the 'mitigates' relationship."""
    data = json.loads((ROOT / 'data/knowledge_base/mitre/enterprise-attack.json').read_text(encoding='utf-8'))
    objects = {o['id']: o for o in data['objects']}

    def external_id(o):
        ref = next((r for r in o.get('external_references', []) if r.get('source_name') == 'mitre-attack'), None)
        return ref['external_id'] if ref else None

    rows = []
    for o in data['objects']:
        if o['type'] != 'relationship' or o.get('relationship_type') != 'mitigates' or o.get('revoked'):
            continue
        source, target = objects.get(o['source_ref']), objects.get(o['target_ref'])
        if not source or not target or source['type'] != 'course-of-action' or source.get('x_mitre_deprecated') or source.get('revoked'):
            continue
        advice = clean_text(re.sub(r'\(Citation:[^)]*\)', '', o.get('description', '')))
        if advice:
            rows.append({'technique': external_id(target), 'mitigation': external_id(source), 'name': source['name'], 'text': advice})
    return pd.DataFrame(rows).dropna().drop_duplicates()


def conformal_thresholds(p_cal, y_cal, alpha=ALPHA):
    """Class-wise split-conformal thresholds: the ceil((n+1)(1-alpha))-th smallest score of each class."""
    thresholds = []
    for k in range(len(CLASSES)):
        scores = np.sort(1 - p_cal[y_cal == k, k])
        rank = int(np.ceil((len(scores) + 1) * (1 - alpha)))
        q = scores[rank - 1] if rank <= len(scores) else 1.0          # too few rows: include the class always
        thresholds.append(float(1 - q))
    return thresholds


def test_evaluation(test, proba, saved):
    """Evaluation of the deployed (serving) model on the labelled official test file: multiclass + attack-vs-normal metrics,
    confusion matrix and per-class scores. Cross-checked against the saved Notebook 21 numbers."""
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
    y = pd.Categorical(test.attack_cat, categories=CLASSES).codes
    pred = proba.argmax(1)
    out = {'split': 'official UNSW-NB15 testing file (82,332 flows, never used for training or tuning)', 'rows': int(len(y)),
           'accuracy': float(accuracy_score(y, pred))}
    for avg in ('macro', 'weighted'):
        p, r, f, _ = precision_recall_fscore_support(y, pred, average=avg, zero_division=0)
        out[avg] = {'precision': float(p), 'recall': float(r), 'f1': float(f)}
    attack, flagged = y != CLASSES.index('Normal'), pred != CLASSES.index('Normal')
    p, r, f, _ = precision_recall_fscore_support(attack, flagged, average='binary', zero_division=0)
    out['binary'] = {'accuracy': float((attack == flagged).mean()), 'precision': float(p), 'recall': float(r), 'f1': float(f),
                     'false_positive_rate': float((flagged & ~attack).sum() / (~attack).sum())}
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=range(len(CLASSES)), zero_division=0)
    out['per_class'] = [{'class': c, 'precision': float(p[k]), 'recall': float(r[k]), 'f1': float(f[k]), 'support': int(s[k])}
                        for k, c in enumerate(CLASSES)]
    out['confusion_matrix'] = {'labels': CLASSES, 'matrix': confusion_matrix(y, pred, labels=range(len(CLASSES))).tolist()}
    top2 = np.argsort(-proba, axis=1)[:, :2]
    out['top2_accuracy'] = float((top2 == y[:, None]).any(axis=1).mean())
    # Ceiling: identical flows with different labels allow at most the majority label of each group (a memoriser that sees the test labels).
    predictors = Detector(OUT).predictors
    key = test[predictors].astype(str).agg('|'.join, axis=1)
    memoriser = pd.Categorical(key.map(test.groupby(key).attack_cat.agg(lambda s: s.value_counts().index[0])), categories=CLASSES).codes
    m_attack = memoriser != CLASSES.index('Normal')
    _, _, m_f1, _ = precision_recall_fscore_support(attack, m_attack, average='binary', zero_division=0)
    out['ceiling'] = {'accuracy': float(accuracy_score(y, memoriser)),
                      'macro_f1': float(precision_recall_fscore_support(y, memoriser, average='macro', zero_division=0)[2]),
                      'attack_f1': float(m_f1),
                      'meaning': 'best possible score on this file: identical flows with different labels can only receive one answer'}
    ceiling_f1 = precision_recall_fscore_support(y, memoriser, labels=range(len(CLASSES)), zero_division=0)[2]
    for k, row in enumerate(out['per_class']):
        row['ceiling_f1'] = float(ceiling_f1[k])
    assert abs(out['macro']['f1'] - saved['macro F1']) < 1e-9 and abs(out['accuracy'] - saved['accuracy']) < 1e-9, (out, saved)
    assert abs(out['binary']['recall'] - saved['attack recall']) < 1e-9, out['binary']
    return out


def model_comparison(final, deployed=None, version='v2'):
    """All models evaluated on the official test file (Notebooks 19, 21 and 22); only the metrics that were saved there."""
    table = pd.read_csv(TABLES / '19_test_results_all_models.csv')
    table = table[~table.model.str.startswith('FINAL DETECTOR')]
    rows = [{'model': r.model, 'runs': int(r.runs), 'macro_f1': float(r['test macro F1']), 'accuracy': float(r['test accuracy']),
             'attack_recall': float(r['attack recall']), 'false_positive_rate': float(r['false-positive rate']), 'deployed': False}
            for _, r in table.iterrows()]
    v2 = final['v2']
    rows.insert(0, {'model': 'LightGBM v2 (shift-aware)' + ('' if deployed else ', deployed'), 'runs': 1, 'macro_f1': v2['macro F1'], 'accuracy': v2['accuracy'],
                    'attack_recall': v2['attack recall'], 'false_positive_rate': v2['false-positive rate'], 'deployed': deployed is None})
    if version == 'v4':
        v3 = read_json(MODELS / '22_final_results_v3.json')['v3']
        rows.insert(0, {'model': 'LightGBM v3 (v2 + class offsets)', 'runs': 1, 'macro_f1': v3['macro F1'], 'accuracy': v3['accuracy'],
                        'attack_recall': v3['attack recall'], 'false_positive_rate': v3['false-alarm rate'], 'deployed': False})
    if deployed:
        label = {'v3': 'LightGBM v3 (v2 + class offsets, deployed)', 'v4': 'Detector v4 (Notebook 23, deployed)'}[version]
        rows.insert(0, {'model': label, 'runs': 1, 'macro_f1': deployed['macro F1'], 'accuracy': deployed['accuracy'],
                        'attack_recall': deployed['attack recall'], 'false_positive_rate': deployed['false-positive rate'], 'deployed': True})
    return {'split': 'official UNSW-NB15 testing file', 'source': 'results/tables/v3/19_test_results_all_models.csv and Notebooks 21-22',
            'rows': rows}


def knowledge_sources():
    """Source register of the knowledge base: publisher, title, version and dates read from the downloaded documents themselves
    (PDF metadata, ATT&CK matrix object) and the download manifest. Nothing is filled in from memory; unknown fields stay None."""
    from pypdf import PdfReader
    kb = ROOT / 'data' / 'knowledge_base'
    manifest = json.loads((kb / 'MANIFEST.json').read_text(encoding='utf-8'))

    def pdf_info(rel):
        meta = PdfReader(str(kb / rel)).metadata or {}
        created = str(meta.get('/CreationDate', ''))[2:10]
        return {'title': str(meta.get('/Title', '')).strip() or None,
                'pdf_created': f'{created[:4]}-{created[4:6]}-{created[6:8]}' if len(created) == 8 else None}

    attack = json.loads((kb / 'mitre/enterprise-attack.json').read_text(encoding='utf-8'))
    matrix = next((o for o in attack['objects'] if o['type'] == 'x-mitre-matrix'), {})
    newest = max(o.get('modified', '') for o in attack['objects'] if o['type'] == 'attack-pattern')
    files = manifest['files']
    attack_doc = {'publisher': 'The MITRE Corporation', 'title': 'MITRE ATT&CK, Enterprise matrix (STIX 2 bundle)',
                  'version': f"matrix object modified {matrix.get('modified', '')[:10]}; newest technique modified {newest[:10]}",
                  'download_url': files['mitre/enterprise-attack.json']['url'], 'document_sha256': files['mitre/enterprise-attack.json']['sha256'],
                  'retrieved': manifest['downloaded'], 'pdf_created': None, 'link_kind': 'attack'}
    docs = {'MITRE ATT&CK': attack_doc, 'MITRE ATT&CK mitigation': attack_doc | {'link_kind': 'mitigation'}}
    for source, rel, publisher, version in [('NIST SP 800-61r3', 'nist/NIST.SP.800-61r3.pdf', 'NIST', 'SP 800-61 Revision 3'),
                                            ('NIST SP 800-94', 'nist/NIST.SP.800-94.pdf', 'NIST', 'SP 800-94'),
                                            ('CISA IR Playbooks', 'cisa/CISA_Incident_and_Vulnerability_Response_Playbooks.pdf', 'CISA', None)]:
        docs[source] = {'publisher': publisher, **pdf_info(rel), 'version': version, 'download_url': files[rel]['url'],
                        'document_sha256': files[rel]['sha256'], 'retrieved': manifest['downloaded'], 'link_kind': 'pdf'}
    return {'downloaded': manifest['downloaded'], 'documents': docs,
            'note': 'pdf_created comes from each PDF file\'s metadata; publication dates are not recorded separately.'}


def feature_reference(frame, predictors, numeric):
    """Percentiles of every numeric feature (and the two engineered byte features) in the FITTING rows only, so the report can say
    "higher than x% of training flows" instead of calling a value high. Categorical columns: share of each value."""
    data = frame[predictors].copy()
    total = data.sbytes.astype(float) + data.dbytes.astype(float)
    data['fe_total_bytes'] = total
    data['fe_byte_balance'] = np.where(total > 0, (data.sbytes.astype(float) - data.dbytes.astype(float)) / total.replace(0, np.nan), 0.0)
    qs = np.linspace(0, 100, 101)
    out = {'source': 'fitting (train) role of the official training file', 'rows': int(len(data)),
           'distinct_patterns': int(len(frame[predictors].drop_duplicates())), 'percentiles': {}, 'categories': {}}
    for c in list(numeric) + ['fe_total_bytes', 'fe_byte_balance']:
        out['percentiles'][c] = [float(v) for v in np.percentile(data[c].astype(float), qs)]
    for c in ('proto', 'service', 'state'):
        out['categories'][c] = {str(k): float(v) for k, v in data[c].astype(str).value_counts(normalize=True).items()}
    return out


def class_reference(frame, predictors, numeric):
    """How typical a value is for each class, from the FITTING rows only (never the test file), so the report can say
    "TTL 254 occurs in 85% of training Exploits flows and 3% of training Normal flows" instead of calling a value typical or unusual.

    Discrete columns (at most 400 distinct values): share of each value within each class (values below 0.1% omitted).
    Continuous columns: 25th, 50th and 75th percentile within each class (the middle half of that class)."""
    data = frame[predictors].copy()
    total = data.sbytes.astype(float) + data.dbytes.astype(float)
    data['fe_total_bytes'] = total
    data['fe_byte_balance'] = np.where(total > 0, (data.sbytes.astype(float) - data.dbytes.astype(float)) / total.replace(0, np.nan), 0.0)
    labels = frame['attack_cat'].astype(str).str.strip()
    out = {'source': 'fitting (train) role of the official training file', 'rows': int(len(data)),
           'class_rows': {c: int((labels == c).sum()) for c in CLASSES}, 'discrete': {}, 'continuous': {}}
    for c in list(predictors) + ['fe_total_bytes', 'fe_byte_balance']:
        col = data[c]
        if c in ('proto', 'service', 'state') or col.nunique() <= 400:
            col = col.astype(str) if c in ('proto', 'service', 'state') else col.astype(float).map(lambda v: f'{v:g}')
            out['discrete'][c] = {}
            for k in CLASSES:
                shares = col[labels == k].value_counts(normalize=True)
                out['discrete'][c][k] = {str(v): round(float(s), 4) for v, s in shares.items() if s >= 0.001}
        else:
            out['continuous'][c] = {k: [float(x) for x in np.percentile(col[labels == k].astype(float), [25, 50, 75])] for k in CLASSES}
    return out


def feature_descriptions(predictors):
    """Descriptions of the 42 input columns from the official NUSW-NB15_features.csv (names differ slightly in the CSV files)."""
    path = RAW / 'unsw_nb15_full' / 'NUSW-NB15_features.csv'
    official = {}
    if path.exists():
        table = pd.read_csv(path, encoding='latin1')
        # The official CSV lost the "fl" ligature ("?ow packet size"); repair that one known artefact.
        official = {str(n).replace(' ', '').strip().lower(): str(d).strip().replace('?ow', 'flow') for n, d in zip(table.Name, table.Description)}
    alias = {'sinpkt': 'sintpkt', 'dinpkt': 'dintpkt', 'smean': 'smeansz', 'dmean': 'dmeansz', 'response_body_len': 'res_bdy_len'}
    out = {}
    for c in predictors:
        out[c] = official.get(alias.get(c, c), 'Present in the official training/testing files; not described in the feature list.')
    return out


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    version = next((v for v in ('v4', 'v3') if (MODELS / f'final_detector_{v}.json').exists()
                    and read_json(MODELS / f'final_detector_{v}.json').get('deployed')), 'v2')
    research = FinalDetector.load(version)
    spec = read_json(MODELS / f'final_detector_{version}.json')
    if version == 'v2':
        research.temperature = read_json(MODELS / '21_calibration_v2.json')['temperature']
    assert not spec.get('dropped_features') and not spec.get('prior_ratio'), 'serving code assumes no dropped columns / prior correction'
    print('serving detector:', version, '-', spec['name'])
    pre = research.pre

    # 1. preprocessing as JSON (exactly the fitted tree pipeline of Notebook 03 + Notebook 04 features)
    ct = pre.tree_pipeline
    numeric = list(ct.transformers_[0][2])
    encoder = ct.named_transformers_['cat']
    vocab = {c: [str(v) for v in cats] for c, cats in zip(ct.transformers_[1][2], encoder.categories_)}
    columns = numeric + [f'{c}={v}' for c in ['proto', 'service', 'state'] for v in vocab[c]] + ['fe_total_bytes', 'fe_byte_balance']
    assert len(columns) == len(pre.columns['tree']), (len(columns), len(pre.columns['tree']))
    (OUT / 'preprocessing.json').write_text(json.dumps({'predictors': pre.predictors, 'numeric': numeric, 'vocab': vocab,
                                                        'engineered': ['fe_total_bytes', 'fe_byte_balance'], 'columns': columns}, indent=1))

    # 2. LightGBM in its native text format
    research.lightgbm.booster_.save_model(str(OUT / 'model.txt'))
    if research.partner is not None:
        research.partner.booster_.save_model(str(OUT / 'model_partner.txt'))
    if research.xgboost is not None:
        research.xgboost.get_booster().save_model(str(OUT / 'model_xgb.json'))
    if research.binary is not None:
        research.binary.booster_.save_model(str(OUT / 'model_binary.txt'))

    # 3. calibration (temperature from Notebook 21) and class-wise conformal thresholds from the calibration rows only
    temperature = research.temperature
    cal = load_raw('calibration')
    y_cal = pd.Categorical(cal.attack_cat, categories=CLASSES).codes
    p_cal = research.predict_proba(cal)                                 # temperature (+ class offsets for v3)
    (OUT / 'calibration.json').write_text(json.dumps({'temperature': temperature, 'alpha': ALPHA,
                                                      'class_thresholds': conformal_thresholds(p_cal, y_cal),
                                                      'class_bias': None if research.class_bias is None else research.class_bias.tolist(),
                                                      'blend_weight': research.blend_weight,
                                                      'fitted_on': 'calibration role of the training file (17,534 rows)'}, indent=1))

    # 4. verify: serving detector == research detector v2
    serving = Detector(OUT)
    report, test_proba = {}, None
    for name, frame in [('model_selection', load_raw('model_selection')), ('test', load_raw('test'))]:
        a = research.predict_proba(frame)
        b = serving.predict(frame).proba
        if name == 'test':
            test_proba = b
        report[name] = {'rows': len(frame), 'max_abs_difference': float(np.abs(a - b).max()),
                        'same_predicted_class': bool((a.argmax(1) == b.argmax(1)).all())}
        assert report[name]['max_abs_difference'] < 1e-9 and report[name]['same_predicted_class'], report
    print('verification:', report)

    # 5. reports: ATT&CK mapping, technique names, knowledge-base passages
    shutil.copy(TABLES / '18_class_to_attack_mapping.csv', OUT / 'attack_mapping.csv')
    from src.rag import attack_techniques
    attack_techniques()[['technique', 'name', 'tactics', 'url']].to_csv(OUT / 'attack_techniques.csv', index=False)
    chunks = pd.read_csv(ROOT / 'data/knowledge_base/processed_chunks/chunks.csv')
    chunks['text'] = chunks.text.map(clean_text)                       # PDF bullet glyphs -> plain text
    chunks.to_csv(OUT / 'knowledge_chunks.csv', index=False)
    attack_mitigations().to_csv(OUT / 'attack_mitigations.csv', index=False)

    # 6. demo sample: 3,000 test flows (stratified, fixed seed); ground truth kept in a separate column for the demo only
    test = load_raw('test')
    sample = pd.concat([g.sample(min(len(g), 300), random_state=0) for _, g in test.groupby('attack_cat')]).sample(frac=1, random_state=0)
    sample = sample.rename(columns={'attack_cat': 'ground_truth'}).drop(columns=['label'])
    assert 'ground_truth' in sample.columns and sample.ground_truth.nunique() == len(CLASSES)
    sample.to_csv(OUT / 'demo_flows.csv', index=False)

    # 7. model card numbers (official test file, Notebook 21)
    final = read_json(MODELS / '21_final_results_v2.json')
    test_raw = load_raw('test')
    y_test = pd.Categorical(test_raw.attack_cat, categories=CLASSES).codes.astype(int)
    from src.flow_networks import score
    deployed_metrics = {k.replace('(of 13)', '(of 44)'): v for k, v in score(y_test, test_proba).items()}
    saved = {'v2': lambda: final['v2'], 'v3': lambda: read_json(MODELS / '22_final_results_v3.json')['v3'],
             'v4': lambda: read_json(MODELS / '23_final_results_v4.json')['v4']}[version]()
    evaluation = test_evaluation(test_raw, test_proba, saved)
    ms = load_raw('model_selection')
    y_ms = pd.Categorical(ms.attack_cat, categories=CLASSES).codes.astype(int)
    p_ms = serving.predict(ms).proba
    from sklearn.metrics import accuracy_score, f1_score
    in_distribution = {'split': 'model_selection rows of the official training file (17,534 flows, same distribution as training)',
                       'accuracy': float(accuracy_score(y_ms, p_ms.argmax(1))), 'macro_f1': float(f1_score(y_ms, p_ms.argmax(1), average='macro')),
                       'attack_f1': float(f1_score(y_ms != CLASSES.index('Normal'), p_ms.argmax(1) != CLASSES.index('Normal')))}
    card = {'name': spec['name'], 'version': version, 'task': '10-class flow classification (UNSW-NB15 features)',
            'test_metrics': deployed_metrics, 'test_per_class_f1': {c['class']: round(c['f1'], 4) for c in evaluation['per_class']},
            'v1_test_metrics': final['v1'], **({'v2_test_metrics': final['v2']} if version != 'v2' else {}),
            **({'v3_test_metrics': read_json(MODELS / '22_final_results_v3.json')['v3']} if version == 'v4' else {}),
            'evaluation': evaluation, 'in_distribution': in_distribution,
            'comparison': model_comparison(final, deployed_metrics if version != 'v2' else None, version),
            'dataset': {'name': 'UNSW-NB15 (official training and testing CSV files)', 'training_rows': 175_341, 'test_rows': 82_332,
                        'fitting_rows': 140_273, 'model_selection_rows': 17_534, 'calibration_rows': 17_534},
            'input_features': len(pre.predictors), 'model_columns': len(columns), 'classes': CLASSES,
            'limitations': [
                'Trained and tested on UNSW-NB15 testbed traffic; scores on other networks are unknown.',
                f"About {deployed_metrics['false-positive rate'] * 100:.0f}% of Normal test flows were flagged as attacks because of a shift between the official files; re-calibrate on labelled recent traffic before deployment.",
                'Analysis, Backdoor and DoS share identical measurements with other classes; their type is often uncertain.',
                'Input must be completed flows in the UNSW-NB15 42-column format; extracting them from packets needs a separate flow extractor.']}
    (OUT / 'model_card.json').write_text(json.dumps(card, indent=1, default=float))
    (OUT / 'feature_descriptions.json').write_text(json.dumps(feature_descriptions(pre.predictors), indent=1))
    (OUT / 'knowledge_sources.json').write_text(json.dumps(knowledge_sources(), indent=1))
    (OUT / 'feature_reference.json').write_text(json.dumps(feature_reference(load_raw('train'), pre.predictors, numeric)))
    (OUT / 'class_reference.json').write_text(json.dumps(class_reference(load_raw('train'), pre.predictors, numeric)))

    # 8. manifest with hashes
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    manifest = {'created': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'), 'source_commit': commit,
                'detector': spec['name'], 'verification': report,
                'files': {p.name: {'bytes': p.stat().st_size, 'sha256': sha256(p)} for p in sorted(OUT.iterdir())}}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    print(f'bundle written to {OUT.relative_to(ROOT)}: {len(manifest["files"])} files, '
          f'{sum(f["bytes"] for f in manifest["files"].values()) / 1e6:.1f} MB')


if __name__ == '__main__':
    main()
