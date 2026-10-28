"""Selection safeguards: whole categorical groups, fitted cuts, exact matches."""

import unittest
import numpy as np
from scipy import sparse
from scipy.stats.contingency import association

from src.data.feature_selection import comparison_sets, column_indices, fit_bin_edges, apply_bins, association_score
from src.data.feature_selection_audit import sparse_keys
from src.data.feature_engineering import overlap_counts


class FeatureSelectionTests(unittest.TestCase):
    def setUp(self):
        self.original=['proto','service','state','is_ftp_login','ct_ftp_cmd']+[f'x{i}' for i in range(37)]

    def test_fixed_sets_preserve_originals_and_both_ftp_columns(self):
        sets=comparison_sets(self.original)
        self.assertEqual([len(s['retained']) for s in sets.values()],[42,45,44,44,43])
        for spec in sets.values():
            self.assertTrue(set(self.original).issubset(spec['retained']))
            self.assertFalse(set(spec['retained']) & set(spec['omitted']))
        self.assertEqual(sets['total_and_packet44']['omitted'],['fe_byte_balance'])

    def test_categorical_feature_keeps_all_one_hot_columns(self):
        names=['cat__proto_a/n','cat__proto_tcp','cat__service_-','cat__service_http','num__x0','engineered__fe_total_bytes']
        self.assertEqual(column_indices(names,self.original,['proto','fe_total_bytes']),[0,1,5])
        with self.assertRaises(ValueError): column_indices(names,self.original,['proto_tcp'])
        with self.assertRaises(ValueError): column_indices(['target__label'],self.original,['label'])

    def test_holdout_extremes_do_not_refit_quantile_cuts(self):
        edges=fit_bin_edges([0,1,2,3],bins=4)
        before=edges.copy()
        self.assertEqual(apply_bins([-100,100],edges).tolist(),[0,3])
        np.testing.assert_array_equal(edges,before)
        np.testing.assert_array_equal(edges,[.75,1.5,2.25])

    def test_association_agrees_with_independent_scipy_counts(self):
        counts=np.array([[100,150],[203,322],[420,700],[320,210]])
        x=[]; y=[]
        for i in range(4):
            for j in range(2):
                x.extend([i]*counts[i,j]); y.extend([j]*counts[i,j])
        self.assertAlmostEqual(association_score(x,y),association(counts,method='cramer',correction=False),places=12)
        self.assertAlmostEqual(association_score(x,y,np.ones(len(x))*0.5),association_score(x,y),places=12)
        self.assertTrue(np.isnan(association_score([0,0],[0,1])))

    def test_sparse_comparison_checks_values_and_normalizes_zero_storage(self):
        a=sparse.csr_matrix([[0,1],[0,1]])
        b=sparse.csr_matrix([[-0.0,1],[0,1+1e-12]])
        counts=overlap_counts(sparse_keys(a),sparse_keys(b))
        self.assertEqual(counts,{'shared_vectors':1,'role_a_rows_with_match':2,'role_b_rows_with_match':1})


if __name__=='__main__': unittest.main()
