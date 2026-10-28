"""Formula edge cases and exact comparison, not model-performance assertions."""

import unittest
import numpy as np
import pandas as pd
from scipy import sparse

from src.data.feature_engineering import retained_features, safe_ratio, sparse_row_keys, overlap_counts


class FeatureEngineeringTests(unittest.TestCase):
    def test_empty_direction_and_true_balance_are_finite(self):
        frame=pd.DataFrame({'sbytes':[0,100,50],'dbytes':[0,0,50],
            'spkts':[0,2,1],'dpkts':[0,0,1],'dur':[0,0,1]},index=[8,4,9])
        e=retained_features(frame)
        self.assertEqual(e.index.tolist(),[8,4,9])
        np.testing.assert_array_equal(e.fe_total_bytes,[0,100,100])
        np.testing.assert_array_equal(e.fe_byte_balance,[0,1,0])
        np.testing.assert_array_equal(e.fe_packet_balance,[0,1,0])
        self.assertTrue(np.isfinite(e.to_numpy()).all())

    def test_targets_and_ids_do_not_change_formulas(self):
        frame=pd.DataFrame({'sbytes':[10],'dbytes':[30],'spkts':[2],'dpkts':[6],'dur':[1],
                            'id':[1],'label':[0],'attack_cat':['Normal']})
        first=retained_features(frame)
        frame[['id','label','attack_cat']]=[99,1,'DoS']
        pd.testing.assert_frame_equal(first,retained_features(frame))

    def test_invalid_inputs_fail_instead_of_silent_clipping(self):
        with self.assertRaises(ValueError): safe_ratio([1],[np.nan])
        with self.assertRaises(ValueError): safe_ratio([1],[-1])

    def test_exact_sparse_overlap_does_not_round_near_matches(self):
        a=sparse.csr_matrix([[0,1],[0,1]])
        b=sparse.csr_matrix([[0,1+1e-12],[0,1]])
        result=overlap_counts(sparse_row_keys(a),sparse_row_keys(b))
        self.assertEqual(result,{'shared_vectors':1,'role_a_rows_with_match':2,'role_b_rows_with_match':1})


if __name__=='__main__': unittest.main()
