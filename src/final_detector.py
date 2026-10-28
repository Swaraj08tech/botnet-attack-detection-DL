"""The final detector chosen in Notebook 15, usable on any raw UNSW-NB15 flow table.

Notebooks 16-19 (and later the web application) use this one entry point:

    detector = FinalDetector.load()
    proba = detector.predict_proba(raw_frame)          # raw CSV rows -> probabilities for the 10 classes

Inside, the frozen preprocessing of Notebooks 03-05 is applied exactly as it was fitted on the train rows:
    03 tree / scaled pipelines -> 04 engineered features (+ their scaler) -> 05 total_and_byte44 column selection.
Notebook 17 adds calibration (temperature) and conformal thresholds on top (see apply_temperature / prediction_sets).
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import torch
from scipy import sparse

from src.notebook_helpers import CATEGORICAL, CLASSES, MODELS, TARGETS, read_json
from src import flow_networks as fn

SOURCE_FEATURES = None   # the 42 original predictors, filled on first use


def add_flow_features(frame):
    """Notebook 04 formulas (no learned parts)."""
    total_bytes = frame.sbytes + frame.dbytes
    total_packets = frame.spkts + frame.dpkts
    out = pd.DataFrame(index=frame.index)
    out['fe_total_bytes'] = total_bytes.astype(float)
    out['fe_byte_balance'] = np.divide((frame.sbytes - frame.dbytes).astype(float), total_bytes.astype(float),
                                       out=np.zeros(len(frame)), where=total_bytes.to_numpy() > 0)
    out['fe_packet_balance'] = np.divide((frame.spkts - frame.dpkts).astype(float), total_packets.astype(float),
                                         out=np.zeros(len(frame)), where=total_packets.to_numpy() > 0)
    return out


class Preprocessor:
    """Raw rows -> tree and scaled matrices of the total_and_byte44 set, exactly as saved in Notebook 05."""

    def __init__(self, feature_set='total_and_byte44'):
        self.tree_pipeline = joblib.load(MODELS / '03_tree_pipeline.joblib')
        self.scaled_pipeline = joblib.load(MODELS / '03_scaled_pipeline.joblib')
        self.engineered_scaler = joblib.load(MODELS / '04_engineered_scaler.joblib')
        names04 = read_json(MODELS / '04_feature_names.json')
        sets = read_json(MODELS / '05_feature_sets.json')
        self.predictors = [c for c in sets['sets']['original42']]
        self.columns = {v: sets['columns'][f'{feature_set}/{v}'] for v in ('tree', 'scaled')}
        self.keep = {v: [names04[v].index(n) for n in self.columns[v]] for v in ('tree', 'scaled')}
        self.feature_set = feature_set

    def transform(self, frame):
        engineered = add_flow_features(frame)
        new = ['fe_total_bytes', 'fe_byte_balance', 'fe_packet_balance']
        tree = sparse.hstack([self.tree_pipeline.transform(frame[self.predictors]), sparse.csr_matrix(engineered[new].to_numpy())], format='csr')
        scaled = sparse.hstack([self.scaled_pipeline.transform(frame[self.predictors]),
                                sparse.csr_matrix(self.engineered_scaler.transform(engineered[new]))], format='csr')
        return {'tree': tree[:, self.keep['tree']].astype(np.float64).tocsr(),
                'scaled': scaled[:, self.keep['scaled']].astype(np.float64).tocsr()}


def network_tensors(scaled, columns, branch_order, text=fn.TEXT):
    """Scaled matrix -> (numbers in branch order, text codes) as used by FlowNet (same logic as FlowData)."""
    m = sparse.csc_matrix(scaled)

    def column_of(feature):
        for prefix in ('log__', 'plain__', ''):
            if prefix + feature in columns:
                return columns.index(prefix + feature)
        raise KeyError(feature)

    numbers = torch.from_numpy(m[:, [column_of(f) for f in branch_order]].toarray().astype(np.float32))
    codes = []
    for t in text:
        block = [i for i, c in enumerate(columns) if c.startswith(f'cat__{t}_')]
        values = m[:, block].toarray()
        codes.append(np.where(values.max(axis=1) > 0, values.argmax(axis=1) + 1, 0))
    return numbers, torch.from_numpy(np.stack(codes, axis=1).astype(np.int64))


class FinalDetector:
    def __init__(self, spec):
        self.spec = spec
        self.kind = spec['detector']                      # 'lightgbm', 'proposed' or 'blend'
        self.pre = Preprocessor()
        self.temperature = 1.0
        self.lightgbm = joblib.load(MODELS.parent.parent / spec['lightgbm_model']) if self.kind in ('lightgbm', 'blend') else None
        self.networks = []
        if self.kind in ('proposed', 'blend'):
            data = fn.FlowData('total_and_byte44')
            self.branch_order = data.branch_a + data.branch_b
            config = {k: v for k, v in spec['network_config'].items() if k != 'feature_set'}
            for seed in spec['network_seeds']:
                model, _ = fn.train_network(data, config, seed)       # loads the saved checkpoint
                self.networks.append(model)

        # Detector v2 (Notebook 20) may use only some tree columns and may correct for different class shares (label shift).
        self.keep_columns = None
        if spec.get('dropped_features'):
            dropped = set(spec['dropped_features'])
            self.keep_columns = [i for i, n in enumerate(self.pre.columns['tree']) if source_feature(n) not in dropped]
        self.prior_ratio = np.asarray(spec['prior_ratio']) if spec.get('prior_ratio') else None
        # Detector v3 (Notebook 22): optional blend partner and class-wise score offsets fitted on the calibration rows.
        self.partner = joblib.load(MODELS.parent.parent / spec['blend_partner']) if spec.get('blend_partner') else None
        self.blend_weight = float(spec.get('blend_weight', 1.0))
        self.class_bias = np.asarray(spec['class_bias'], dtype=float) if spec.get('class_bias') is not None else None
        # Detector v4 (Notebook 23): optional XGBoost ensemble member and optional attack-vs-normal model (two-stage).
        self.xgboost = joblib.load(MODELS.parent.parent / spec['xgboost_model']) if spec.get('xgboost_model') else None
        self.binary = joblib.load(MODELS.parent.parent / spec['binary_model']) if spec.get('binary_model') else None
        if spec.get('temperature'):
            self.temperature = float(spec['temperature'])

    @classmethod
    def load(cls, version='v1'):
        """version 'v1': the detector chosen in Notebook 15 (calibrated in 17); 'v2': the shift-aware detector of Notebook 20."""
        if version == 'v2':
            return cls(read_json(MODELS / 'final_detector_v2.json'))
        if version == 'v3':
            return cls(read_json(MODELS / 'final_detector_v3.json'))
        if version == 'v4':
            return cls(read_json(MODELS / 'final_detector_v4.json'))
        detector = cls(read_json(MODELS / 'final_detector.json'))
        calibration = MODELS / '17_calibration.json'
        if calibration.exists():
            detector.calibration = read_json(calibration)
            detector.temperature = detector.calibration['temperature']
        return detector

    @property
    def name(self):
        return self.spec['name']

    def raw_proba(self, inputs):
        """Uncalibrated probabilities from prepared inputs (dict with 'tree' and 'scaled')."""
        parts = []
        if self.lightgbm is not None:
            tree = inputs['tree'] if self.keep_columns is None else inputs['tree'][:, self.keep_columns]
            p = self.lightgbm.predict_proba(tree)
            if self.prior_ratio is not None:
                p = p * self.prior_ratio[None, :]
                p = p / p.sum(axis=1, keepdims=True)
            if self.partner is not None:
                partner = self.partner.predict_proba(inputs['tree'])
                p = self.blend_weight * p + (1 - self.blend_weight) * partner
            if self.xgboost is not None:                        # equal-weight ensemble; XGBoost sees dense inputs (zeros are values, not missing)
                p = (p + self.xgboost.predict_proba(dense(inputs['tree']))) / 2
            if self.binary is not None:                         # two-stage: P(normal) from the attack-vs-normal model, type shares from p
                p = two_stage(p, self.binary.predict_proba(inputs['tree'])[:, 1])
            parts.append(p)
        if self.networks:
            numbers, codes = network_tensors(inputs['scaled'], self.pre.columns['scaled'], self.branch_order)
            parts.append(np.mean([fn.predict_proba(m, numbers, codes) for m in self.networks], axis=0))
        return np.mean(parts, axis=0)

    def predict_proba(self, frame, calibrated=True):
        p = self.raw_proba(self.pre.transform(frame))
        if not calibrated:
            return p
        return apply_bias(apply_temperature(p, self.temperature), self.class_bias)


def source_feature(column):
    """Encoded tree column name -> the original feature it comes from ('num__sttl' -> 'sttl', 'cat__proto_tcp' -> 'proto')."""
    if column.startswith('cat__'):
        return column[5:].split('_')[0]
    return column.split('__', 1)[-1]


def apply_temperature(proba, temperature):
    """Temperature scaling on log-probabilities: T > 1 softens over-confident outputs, T < 1 sharpens them."""
    logp = np.log(np.clip(proba, 1e-12, 1.0)) / temperature
    logp -= logp.max(axis=1, keepdims=True)
    p = np.exp(logp)
    return p / p.sum(axis=1, keepdims=True)


def dense(matrix):
    return matrix.toarray() if hasattr(matrix, 'toarray') else np.asarray(matrix)


def two_stage(proba, p_attack):
    """Combine an attack-vs-normal probability with the attack-type shares of a 10-class model:
    P(Normal) = 1 - p_attack;  P(c) = p_attack * p(c) / sum of p over the attack classes."""
    from src.notebook_helpers import CLASSES
    normal = CLASSES.index('Normal')
    out = np.array(proba, dtype=float, copy=True)
    attack = np.ones(out.shape[1], dtype=bool); attack[normal] = False
    shares = out[:, attack] / np.clip(out[:, attack].sum(axis=1, keepdims=True), 1e-12, None)
    out[:, attack] = shares * p_attack[:, None]
    out[:, normal] = 1 - p_attack
    return out


def apply_bias(proba, bias):
    """Class-wise score offsets (a prior correction): p'(c) is proportional to p(c) * exp(bias_c). No offsets = unchanged."""
    if bias is None:
        return proba
    logp = np.log(np.clip(proba, 1e-12, 1.0)) + np.asarray(bias)[None, :]
    logp -= logp.max(axis=1, keepdims=True)
    p = np.exp(logp)
    return p / p.sum(axis=1, keepdims=True)


def prediction_sets(proba, thresholds):
    """Conformal prediction sets: every class whose probability is at least its threshold (1 - q_class)."""
    thresholds = np.asarray(thresholds)
    sets = proba >= thresholds[None, :]
    empty = ~sets.any(axis=1)
    sets[empty, proba[empty].argmax(axis=1)] = True        # never return an empty set: keep the top class
    return sets


def load_raw(role_or_path):
    """Raw rows of a role ('train', 'model_selection', 'calibration') from the training CSV, or the test CSV ('test')."""
    from src.notebook_helpers import DATA, TEST_CSV, TRAIN_CSV
    if role_or_path == 'test':
        return pd.read_csv(TEST_CSV, low_memory=False)
    raw = pd.read_csv(TRAIN_CSV, low_memory=False)
    roles = pd.read_csv(DATA / '02_roles.csv')
    return raw[roles.role.to_numpy() == role_or_path].reset_index(drop=True)
