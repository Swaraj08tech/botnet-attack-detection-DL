"""Feature-name sets and descriptive selection tools; no predictive estimator."""

import numpy as np
import pandas as pd

ENGINEERED = ['fe_total_bytes', 'fe_byte_balance', 'fe_packet_balance']
CATEGORICAL = ['proto', 'service', 'state']


def comparison_sets(original):
    """Fixed comparisons declared before evidence; original names and order."""
    if len(original) != 42 or len(set(original)) != 42 or set(original) & {'id', 'label', 'attack_cat'}:
        raise ValueError('Expected the original 42-predictor contract')
    additions = {
        'original42': [],
        'all45': ENGINEERED,
        'total_and_byte44': ENGINEERED[:2],
        'total_and_packet44': [ENGINEERED[0], ENGINEERED[2]],
        'total_only43': ENGINEERED[:1],
    }
    reasons = {
        'original42': 'Reference: no engineered additions.',
        'all45': 'Reference: all three engineered additions.',
        'total_and_byte44': 'Test total bytes and byte direction without packet balance.',
        'total_and_packet44': 'Test total bytes and packet direction without byte balance.',
        'total_only43': 'Test volume alone, without either directional balance.',
    }
    universe = original + ENGINEERED
    return {name: {'retained': original + extra, 'omitted': [c for c in universe if c not in original + extra],
                   'reason': reasons[name]} for name, extra in additions.items()}


def encoded_sources(encoded_names, original_names):
    """Map complete one-hot groups to their original source feature."""
    universe = set(original_names) | set(ENGINEERED)
    result = []
    for name in encoded_names:
        prefix, separator, suffix = name.partition('__')
        if not separator:
            raise ValueError(f'Unrecognized encoded name: {name}')
        if prefix == 'cat':
            matches = [c for c in CATEGORICAL if suffix.startswith(c + '_')]
            if len(matches) != 1:
                raise ValueError(f'Unknown categorical group: {name}')
            source = matches[0]
        elif prefix in ('num', 'log', 'plain', 'engineered'):
            source = suffix
        else:
            raise ValueError(f'Unknown transformer: {name}')
        if source not in universe:
            raise ValueError(f'Unexpected source: {source}')
        result.append(source)
    if len(encoded_names) != len(set(encoded_names)):
        raise ValueError('Repeated encoded column names')
    return result


def column_indices(encoded_names, original_names, retained):
    sources = encoded_sources(encoded_names, original_names)
    if len(retained) != len(set(retained)) or not set(retained).issubset(set(sources)):
        raise ValueError('Unknown or repeated retained features')
    # Keep frozen transformer output order. The source-name lists have their own fixed order.
    indices = [i for i, source in enumerate(sources) if source in retained]
    if set(sources[i] for i in indices) != set(retained):
        raise ValueError('Missing retained group')
    return indices


def fit_bin_edges(values, bins=8):
    """Unsupervised quantiles from inner training rows only; duplicate cuts collapse."""
    x = np.asarray(values, dtype=float)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError('Finite nonempty values required')
    return np.unique(np.quantile(x, np.arange(1, bins) / bins))


def apply_bins(values, edges):
    x = np.asarray(values, dtype=float)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite values')
    return np.searchsorted(np.asarray(edges), x, side='right')


def association_score(feature, target, weights=None):
    """Cramer's V, or its descriptive fractional-weight analogue. No p-values.

    Constant observed features or targets return NaN, not an invented zero.
    Weighted rows keep conflicting labels but limit an original group's total mass.
    """
    x = pd.factorize(np.asarray(feature))[0]
    y = pd.factorize(np.asarray(target))[0]
    if (x < 0).any() or (y < 0).any() or len(x) != len(y):
        raise ValueError('Aligned nonmissing feature/target required')
    w = np.ones(len(x)) if weights is None else np.asarray(weights, dtype=float)
    if len(w) != len(x) or not np.isfinite(w).all() or (w <= 0).any():
        raise ValueError('Positive finite aligned weights required')
    k, m = len(np.unique(x)), len(np.unique(y))
    if min(k, m) <= 1:
        return float('nan')
    observed = np.zeros((k, m), dtype=float)
    np.add.at(observed, (x, y), w)
    n = observed.sum()
    expected = observed.sum(axis=1)[:, None] * observed.sum(axis=0)[None, :] / n
    chi = np.sum((observed - expected) ** 2 / expected)
    return float(np.sqrt(max(0, chi / n / min(k - 1, m - 1))))
