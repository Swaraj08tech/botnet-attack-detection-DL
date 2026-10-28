"""Test fixtures: a small synthetic model bundle with the real column layout, so tests never need the real 16 MB model."""
import json
import os
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
os.environ.setdefault('JWT_SECRET', 'test-secret-not-for-production-0123456789')

from app.core.detector import CLASSES  # noqa: E402

PREDICTORS = ['dur', 'proto', 'service', 'state', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl', 'sload', 'dload', 'sloss',
              'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat', 'smean', 'dmean',
              'trans_depth', 'response_body_len', 'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm',
              'ct_dst_src_ltm', 'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports']
NUMERIC = [c for c in PREDICTORS if c not in ('proto', 'service', 'state')]
VOCAB = {'proto': ['tcp', 'udp'], 'service': ['-', 'dns', 'http'], 'state': ['CON', 'FIN', 'INT']}


def synthetic_flows(n, seed=0):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({c: rng.integers(0, 500, n).astype(float) for c in NUMERIC})
    frame['sbytes'] += 28
    for c, values in VOCAB.items():
        frame[c] = rng.choice(values, n)
    return frame[PREDICTORS]


@pytest.fixture(scope='session')
def bundle_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp('bundle')
    columns = NUMERIC + [f'{c}={v}' for c in ['proto', 'service', 'state'] for v in VOCAB[c]] + ['fe_total_bytes', 'fe_byte_balance']
    (out / 'preprocessing.json').write_text(json.dumps({'predictors': PREDICTORS, 'numeric': NUMERIC, 'vocab': VOCAB,
                                                        'engineered': ['fe_total_bytes', 'fe_byte_balance'], 'columns': columns}))
    sys.path.insert(0, str(BACKEND))
    from app.core.detector import Detector
    frame = synthetic_flows(3000)
    labels = (frame.sttl.to_numpy() // 50).astype(int) % len(CLASSES)        # a learnable synthetic rule
    probe = Detector.__new__(Detector)
    probe.predictors, probe.numeric, probe.vocab = PREDICTORS, NUMERIC, VOCAB
    x = Detector.transform(probe, frame)
    booster = lgb.train({'objective': 'multiclass', 'num_class': 10, 'verbosity': -1, 'num_leaves': 7, 'seed': 0},
                        lgb.Dataset(x, labels), num_boost_round=20)
    booster.save_model(str(out / 'model.txt'))
    (out / 'calibration.json').write_text(json.dumps({'temperature': 1.1, 'alpha': 0.1, 'class_thresholds': [0.2] * 10}))
    pd.DataFrame({'class': CLASSES, 'techniques': ['T1046', 'T1205', 'T1498', 'T1190', 'T1595', 'T1600', '', 'T1595', 'T1203', 'T1210'],
                  'confidence': 'test', 'rationale': 'test', 'names': ''}).to_csv(out / 'attack_mapping.csv', index=False)
    pd.DataFrame({'technique': ['T1046', 'T1498', 'T1190', 'T1595'], 'name': ['Network Service Discovery', 'Network Denial of Service',
                  'Exploit Public-Facing Application', 'Active Scanning'], 'tactics': 'x', 'url': ''}).to_csv(out / 'attack_techniques.csv', index=False)
    pd.DataFrame({'chunk_id': ['K1', 'K2', 'K3', 'K4'], 'source': ['MITRE ATT&CK', 'MITRE ATT&CK', 'CISA IR Playbooks', 'NIST SP 800-61r3'],
                  'location': ['T1498', 'T1046', 'page 1', 'page 2'],
                  'text': ['T1498 Network Denial of Service flood traffic exhausts bandwidth', 'T1046 Network Service Discovery scanning ports',
                           'Isolate affected systems and networks during incident response containment', 'Detection and analysis of network incidents']
                  }).to_csv(out / 'knowledge_chunks.csv', index=False)
    demo = synthetic_flows(200, seed=1)
    demo['ground_truth'] = np.asarray(CLASSES)[(demo.sttl.to_numpy() // 50).astype(int) % 10]
    demo.to_csv(out / 'demo_flows.csv', index=False)
    (out / 'model_card.json').write_text(json.dumps({'name': 'synthetic test model', 'version': 'test', 'test_metrics': {}, 'limitations': []}))
    (out / 'manifest.json').write_text(json.dumps({'detector': 'synthetic test model', 'created': 'test', 'source_commit': 'test', 'verification': {}}))
    return out


@pytest.fixture()
def client(bundle_dir, tmp_path):
    from fastapi.testclient import TestClient

    from app.config import Settings
    from app.main import create_app
    cfg = Settings(bundle_dir=bundle_dir, database_url=f"sqlite:///{(tmp_path / 'test.db').as_posix()}", admin_email='admin@test.local',
                   admin_password='admin-password-123', ollama_url='http://127.0.0.1:9', jwt_secret='test-secret-not-for-production-0123456789',
                   data_dir=tmp_path / 'data')
    with TestClient(create_app(cfg)) as c:
        yield c


@pytest.fixture()
def admin_headers(client):
    r = client.post('/api/auth/login', data={'username': 'admin@test.local', 'password': 'admin-password-123'})
    assert r.status_code == 200, r.text
    return {'Authorization': f"Bearer {r.json()['access_token']}"}
