"""Detector v2 for serving: raw UNSW-NB15 flow rows -> calibrated class probabilities, prediction sets and explanations.

Everything is loaded from the serving bundle (built by scripts/build_serving_bundle.py):
  preprocessing.json  column order + category vocabularies (replaces the scikit-learn pipeline, no pickles)
  model.txt           LightGBM model in its native text format
  calibration.json    temperature and class-wise conformal thresholds
The bundle builder checks that this module reproduces the research detector (Notebooks 20-21) exactly.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

CLASSES = ['Analysis', 'Backdoor', 'DoS', 'Exploits', 'Fuzzers', 'Generic', 'Normal', 'Reconnaissance', 'Shellcode', 'Worms']
NORMAL = CLASSES.index('Normal')
CATEGORICAL = ['proto', 'service', 'state']


class InputError(ValueError):
    """Raised when uploaded flow rows do not match the expected UNSW-NB15 format."""


@dataclass
class Prediction:
    proba: np.ndarray            # (rows, 10) calibrated probabilities
    sets: np.ndarray             # (rows, 10) booleans: conformal prediction sets
    predicted: np.ndarray        # (rows,) class index

    def decision(self, i: int) -> str:
        members = np.flatnonzero(self.sets[i])
        if len(members) == 1:
            return 'confident'
        return 'review' if NORMAL in members else 'attack_type_uncertain'


class Detector:
    def __init__(self, bundle_dir: str | Path):
        self.dir = Path(bundle_dir)
        pre = json.loads((self.dir / 'preprocessing.json').read_text(encoding='utf-8'))
        self.predictors: list[str] = pre['predictors']                 # the 42 raw columns a request must contain
        self.numeric: list[str] = pre['numeric']                       # numeric columns in model order
        self.vocab: dict[str, list[str]] = pre['vocab']                # categories per text column, in model order
        self.engineered: list[str] = pre['engineered']                 # engineered columns appended at the end
        self.columns: list[str] = pre['columns']                       # final model column names (for explanations)
        self.booster = lgb.Booster(model_file=str(self.dir / 'model.txt'))
        partner = self.dir / 'model_partner.txt'                        # optional blend partner (detector v3, Notebook 22)
        self.partner = lgb.Booster(model_file=str(partner)) if partner.exists() else None
        xgb_file, binary = self.dir / 'model_xgb.json', self.dir / 'model_binary.txt'   # optional members of detector v4 (Notebook 23)
        self.xgb = None
        if xgb_file.exists():
            import xgboost
            self.xgb = xgboost.Booster()
            self.xgb.load_model(str(xgb_file))
        self.binary = lgb.Booster(model_file=str(binary)) if binary.exists() else None
        cal = json.loads((self.dir / 'calibration.json').read_text(encoding='utf-8'))
        self.temperature = float(cal['temperature'])
        self.blend_weight = float(cal.get('blend_weight', 1.0))
        self.class_bias = np.asarray(cal['class_bias'], dtype=float) if cal.get('class_bias') is not None else None
        self.thresholds = np.asarray(cal['class_thresholds'], dtype=float)
        self.alpha = float(cal['alpha'])
        manifest = self.dir / 'manifest.json'
        self.manifest = json.loads(manifest.read_text(encoding='utf-8')) if manifest.exists() else {}
        self._source = [self._source_feature(c) for c in self.columns]

    # ------------------------------------------------------------------------------------------------- preprocessing
    def validate(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.predictors if c not in frame.columns]
        if missing:
            raise InputError(f'missing columns: {missing[:8]}{" ..." if len(missing) > 8 else ""}')
        frame = frame[self.predictors].copy()
        for c in self.numeric:
            values = pd.to_numeric(frame[c], errors='coerce')
            if values.isna().any():
                raise InputError(f'column {c} must be numeric (row {int(values.isna().to_numpy().argmax()) + 1})')
            if not np.isfinite(values.to_numpy(dtype=float)).all() or (values < 0).any():
                raise InputError(f'column {c} must contain finite, non-negative values')
            frame[c] = values.astype(float)
        for c in CATEGORICAL:
            frame[c] = frame[c].astype(str)
        return frame

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        """Raw rows -> the model matrix: numbers unchanged, text one-hot (unknown values -> all zeros), engineered features."""
        frame = self.validate(frame)
        parts = [frame[self.numeric].to_numpy(dtype=float)]
        for c in CATEGORICAL:
            lookup = {v: i for i, v in enumerate(self.vocab[c])}
            onehot = np.zeros((len(frame), len(self.vocab[c])))
            idx = frame[c].map(lookup)
            known = idx.notna().to_numpy()
            onehot[np.flatnonzero(known), idx[known].astype(int).to_numpy()] = 1.0
            parts.append(onehot)
        total = frame.sbytes.to_numpy() + frame.dbytes.to_numpy()
        balance = np.divide(frame.sbytes.to_numpy() - frame.dbytes.to_numpy(), total, out=np.zeros(len(frame)), where=total > 0)
        parts.append(np.column_stack([total, balance]))                # fe_total_bytes, fe_byte_balance (Notebook 04)
        return np.hstack(parts)

    # ------------------------------------------------------------------------------------------------- prediction
    def raw_proba(self, x: np.ndarray) -> np.ndarray:
        p = self.booster.predict(x)
        if self.partner is not None:
            p = self.blend_weight * p + (1 - self.blend_weight) * self.partner.predict(x)
        if self.xgb is not None:                                        # equal-weight ensemble with XGBoost (dense input: zeros are values)
            import xgboost
            p = (p + self.xgb.predict(xgboost.DMatrix(x, missing=np.nan))) / 2
        if self.binary is not None:                                     # two-stage: P(normal) from the attack-vs-normal model
            p_attack = self.binary.predict(x)
            attack = np.arange(p.shape[1]) != NORMAL
            shares = p[:, attack] / np.clip(p[:, attack].sum(axis=1, keepdims=True), 1e-12, None)
            p = p.copy()
            p[:, attack] = shares * p_attack[:, None]
            p[:, NORMAL] = 1 - p_attack
        return p

    def calibrate(self, proba: np.ndarray) -> np.ndarray:
        """Temperature scaling, then the class-wise offsets of detector v3 (p'(c) proportional to p(c) * exp(b_c)), if any."""
        logp = np.log(np.clip(proba, 1e-12, 1.0)) / self.temperature
        logp -= logp.max(axis=1, keepdims=True)
        p = np.exp(logp)
        p /= p.sum(axis=1, keepdims=True)
        if self.class_bias is None:
            return p
        logp = np.log(np.clip(p, 1e-12, 1.0)) + self.class_bias[None, :]
        logp -= logp.max(axis=1, keepdims=True)
        p = np.exp(logp)
        return p / p.sum(axis=1, keepdims=True)

    def prediction_sets(self, proba: np.ndarray) -> np.ndarray:
        sets = proba >= self.thresholds[None, :]
        empty = ~sets.any(axis=1)
        sets[empty, proba[empty].argmax(axis=1)] = True
        return sets

    def predict(self, frame: pd.DataFrame) -> Prediction:
        proba = self.calibrate(self.raw_proba(self.transform(frame)))
        return Prediction(proba=proba, sets=self.prediction_sets(proba), predicted=proba.argmax(axis=1))

    # ------------------------------------------------------------------------------------------------- explanation
    @staticmethod
    def _source_feature(column: str) -> str:
        return column.split('=', 1)[0] if '=' in column else column

    def explain(self, frame: pd.DataFrame, class_index: np.ndarray, top: int = 7) -> list[list[dict]]:
        """Exact TreeSHAP contributions (LightGBM pred_contrib) for each row's predicted class, grouped by raw column.

        Values are in the model's score units (log-odds); positive pushes towards the class, negative away from it."""
        x = self.transform(frame)
        contrib = self.booster.predict(x, pred_contrib=True)              # (rows, (F + 1) * classes)
        n_features = x.shape[1]
        contrib = contrib.reshape(len(x), len(CLASSES), n_features + 1)[:, :, :n_features]
        raw = frame[self.predictors].reset_index(drop=True)
        out = []
        for r in range(len(x)):
            per_column = pd.Series(contrib[r, class_index[r]], index=self._source).groupby(level=0).sum()
            order = per_column.abs().sort_values(ascending=False).index[:top]
            out.append([{'feature': f, 'value': self._display_value(raw, r, f), 'contribution': float(per_column[f])} for f in order])
        return out

    def _display_value(self, raw: pd.DataFrame, r: int, feature: str):
        if feature in raw.columns:
            v = raw.iloc[r][feature]
            return v.item() if hasattr(v, 'item') else v
        if feature == 'fe_total_bytes':
            return float(raw.iloc[r]['sbytes']) + float(raw.iloc[r]['dbytes'])
        if feature == 'fe_byte_balance':
            s, d = float(raw.iloc[r]['sbytes']), float(raw.iloc[r]['dbytes'])
            return (s - d) / (s + d) if s + d > 0 else 0.0
        return None

    # ------------------------------------------------------------------------------------------------- full explanation (report v2)
    def explain_full(self, frame: pd.DataFrame, class_index: np.ndarray) -> list[dict]:
        """The complete TreeSHAP decomposition of the predicted class's score for each row, grouped by original column.

        LightGBM score (margin) of class c = base value + sum of all contributions (exact for the base booster).
        The deployed detector turns margins into class scores as  margin / temperature + offset_c  before the softmax, so the same
        contributions, divided by the temperature, decompose the deployed class score exactly; they are NOT shares of the probability.
        """
        if self.partner is not None or self.xgb is not None or self.binary is not None:
            raise NotImplementedError('the full decomposition covers a single LightGBM booster; this bundle adds other members')
        x = self.transform(frame)
        contrib = self.booster.predict(x, pred_contrib=True).reshape(len(x), len(CLASSES), x.shape[1] + 1)
        raw = frame[self.predictors].reset_index(drop=True)
        out = []
        for r in range(len(x)):
            c = int(class_index[r])
            per_column = pd.Series(contrib[r, c, :-1], index=self._source).groupby(level=0).sum()
            order = per_column.abs().sort_values(ascending=False).index
            base, total = float(contrib[r, c, -1]), float(per_column.sum())
            offset = float(self.class_bias[c]) if self.class_bias is not None else 0.0
            out.append({'class': CLASSES[c], 'base_value': base, 'margin': base + total,
                        'temperature': self.temperature, 'class_offset': offset, 'deployed_class_score': (base + total) / self.temperature + offset,
                        'contributions': [{'feature': f, 'value': self._display_value(raw, r, f), 'contribution': float(per_column[f])} for f in order]})
        return out
