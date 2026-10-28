"""Small, label-free completed-flow features. No fitted state or target inputs."""

import numpy as np
import pandas as pd
from scipy import sparse


RETAINED = ['fe_total_bytes', 'fe_byte_balance', 'fe_packet_balance']
SOURCES = ['sbytes', 'dbytes', 'spkts', 'dpkts', 'dur']


def safe_ratio(numerator, denominator):
    """Divide only for positive denominators; zero means a documented sentinel."""
    n, d = np.asarray(numerator, dtype=np.float64), np.asarray(denominator, dtype=np.float64)
    if n.shape != d.shape or not np.isfinite(n).all() or not np.isfinite(d).all() or (d < 0).any():
        raise ValueError('Finite equal-shape inputs and nonnegative denominators required')
    out = np.zeros_like(n)
    np.divide(n, d, out=out, where=d > 0)
    if not np.isfinite(out).all():
        raise ValueError('Ratio overflow; do not silently clip')
    return out


def candidate_features(frame):
    """Uses named source measurements only; preserves the input index and order."""
    x = frame[SOURCES].astype(np.float64)
    if not np.isfinite(x.to_numpy()).all() or (x.to_numpy() < 0).any():
        raise ValueError('Expected finite nonnegative source measurements')
    b, p = x.sbytes + x.dbytes, x.spkts + x.dpkts
    result = pd.DataFrame({
        'fe_total_bytes': b,
        'fe_byte_balance': safe_ratio(x.sbytes - x.dbytes, b),
        'fe_packet_balance': safe_ratio(x.spkts - x.dpkts, p),
        'fe_packets_per_second': safe_ratio(p, x.dur),
        'fe_source_bytes_per_packet': safe_ratio(x.sbytes, x.spkts),
        'fe_destination_bytes_per_packet': safe_ratio(x.dbytes, x.dpkts),
    }, index=frame.index)
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError('Nonfinite engineered features')
    return result


def retained_features(frame):
    return candidate_features(frame)[RETAINED]


def proposal_table():
    common = 'Available after flow completion if the supplied counters are faithfully extracted; early availability unverified'
    rows = [
        ('fe_total_bytes', 'sbytes + dbytes', 'sbytes, dbytes', 'Total bidirectional recorded byte volume; not application payload size', 'No denominator', 'Sum of original bytes; not an exact copy of a single column'),
        ('fe_byte_balance', '(sbytes - dbytes) / (sbytes + dbytes)', 'sbytes, dbytes', 'Signed byte direction balance: +1 source-only, -1 destination-only, 0 equal volume', '0 when total bytes is 0; distinguish absent traffic using original counters', 'Bounded interaction of original counts, not new measured information'),
        ('fe_packet_balance', '(spkts - dpkts) / (spkts + dpkts)', 'spkts, dpkts', 'Signed packet direction balance; source/destination do not necessarily mean client/server', '0 when total packets is 0; distinguish absent traffic using original counters', 'Bounded interaction of original counts; may correlate with byte balance'),
        ('fe_packets_per_second', '(spkts + dpkts) / dur', 'spkts, dpkts, dur', 'Completed-flow average packet count per duration unit; seconds assumption not extractor-verified', '0 when dur is 0; sentinel is not a measured rate', 'Check rate and (spkts + dpkts - 1) / dur; rate extractor formula is unresolved'),
        ('fe_source_bytes_per_packet', 'sbytes / spkts', 'sbytes, spkts', 'Recorded bytes per source-direction packet', '0 when spkts is 0; sentinel is not a measured packet size', 'Check smean; integer rounding can explain small differences'),
        ('fe_destination_bytes_per_packet', 'dbytes / dpkts', 'dbytes, dpkts', 'Recorded bytes per destination-direction packet', '0 when dpkts is 0; sentinel is not a measured packet size', 'Check dmean; integer rounding can explain small differences'),
    ]
    return pd.DataFrame([dict(zip(['feature', 'formula', 'source_columns', 'network_meaning', 'zero_handling', 'redundancy_question'], r),
                              offline_availability=common) for r in rows])


def canonical_csr(matrix):
    x = sparse.csr_matrix(matrix, dtype=np.float64, copy=True)
    x.sum_duplicates(); x.eliminate_zeros(); x.sort_indices()
    if not np.isfinite(x.data).all():
        raise ValueError('Nonfinite representation')
    return x


def sparse_row_keys(matrix):
    """Exact index/value tuples: hash lookup is always resolved by full equality."""
    x = canonical_csr(matrix)
    return [(tuple(x.indices[x.indptr[i]:x.indptr[i+1]]), tuple(x.data[x.indptr[i]:x.indptr[i+1]]))
            for i in range(x.shape[0])]


def overlap_counts(keys_a, keys_b):
    a, b = set(keys_a), set(keys_b)
    shared = a & b
    return {'shared_vectors': len(shared), 'role_a_rows_with_match': sum(k in shared for k in keys_a),
            'role_b_rows_with_match': sum(k in shared for k in keys_b)}
