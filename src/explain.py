"""Model-agnostic explanations (SHAP and LIME) of the final detector, in terms of the 42 original measured columns.

The detector itself works on preprocessed inputs, but an analyst reads raw columns (sbytes, sttl, service, ...).
So we explain a function f(raw 42 columns) -> class probabilities. The text columns (proto, service, state) are coded
as integers for the explainers and decoded back to text inside f. The engineered features are recomputed from the raw
columns inside the detector, so their effect is credited to the raw columns they come from.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from src.notebook_helpers import CATEGORICAL, CLASSES


class RawSpace:
    """Converts between raw flow rows and the numeric matrix used by the explainers."""

    def __init__(self, train_frame, predictors):
        self.predictors = list(predictors)
        self.cat_index = [self.predictors.index(c) for c in CATEGORICAL]
        self.vocab = {c: sorted(train_frame[c].astype(str).unique()) for c in CATEGORICAL}
        self.reference = {c: (train_frame[c].mode().iloc[0] if c in CATEGORICAL else float(train_frame[c].median()))
                          for c in self.predictors}

    def encode(self, frame):
        x = frame[self.predictors].copy()
        for c in CATEGORICAL:
            lookup = {v: i for i, v in enumerate(self.vocab[c])}
            x[c] = x[c].astype(str).map(lambda v: lookup.get(v, -1))
        return x.to_numpy(dtype=float)

    def decode(self, x):
        frame = pd.DataFrame(np.asarray(x, dtype=float), columns=self.predictors)
        for c in CATEGORICAL:
            codes = np.clip(np.rint(frame[c].to_numpy()).astype(int), -1, len(self.vocab[c]) - 1)
            frame[c] = [self.vocab[c][k] if k >= 0 else '__unseen__' for k in codes]
        for c in self.predictors:
            if c not in CATEGORICAL:
                frame[c] = frame[c].astype(float)
        return frame


def make_predict_fn(detector, space, calibrated=False, batch=50_000):
    """f(encoded matrix) -> probabilities, in batches (explainers call it with many rows)."""
    def f(x):
        x = np.asarray(x, dtype=float)
        out = [detector.predict_proba(space.decode(x[i:i + batch]), calibrated=calibrated) for i in range(0, len(x), batch)]
        return np.vstack(out)
    return f


def shap_values(detector, space, rows_encoded, background_encoded, seed=0):
    """Permutation SHAP values: array (rows, 42 features, 10 classes), in probability units."""
    f = make_predict_fn(detector, space)
    masker = shap.maskers.Independent(background_encoded, max_samples=len(background_encoded))
    explainer = shap.PermutationExplainer(f, masker, seed=seed)
    result = explainer(rows_encoded, max_evals=2 * len(space.predictors) + 1, silent=True)
    return result.values, result.base_values


def top_features(values_row, predicted_class, predictors, k=5):
    """Top-k features pushing the predicted class up or down, ordered by absolute contribution."""
    contrib = values_row[:, predicted_class]
    order = np.argsort(-np.abs(contrib))[:k]
    return [(predictors[i], float(contrib[i])) for i in order]
