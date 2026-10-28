"""Check research-sensitive overlap counts and hash-prefilter correctness."""

import unittest
from unittest.mock import patch

import pandas as pd

from src.data.validate_data import duplicate_overlap


class DuplicateOverlapTests(unittest.TestCase):
    def setUp(self):
        self.train = pd.DataFrame({"id": [1, 2, 3], "x": [1, 1, 2],
            "service": ["dns", "dns", "http"], "label": [0, 1, 0],
            "attack_cat": ["Normal", "DoS", "Normal"]})
        self.test = pd.DataFrame({"id": [9, 10], "x": [1, 3],
            "service": ["dns", "http"], "label": [1, 0],
            "attack_cat": ["DoS", "Normal"]})

    def check_counts(self):
        before_train, before_test = self.train.copy(), self.test.copy()
        result = duplicate_overlap(self.train, self.test)
        self.assertEqual(result["shared_unique_feature_records"], 1)
        self.assertEqual(result["train_rows_with_test_match"], 2)
        self.assertEqual(result["test_rows_with_train_match"], 1)
        self.assertEqual(result["matching_row_pairs"], 2)
        self.assertEqual(result["shared_feature_records_with_conflicting_targets"], 1)
        self.assertEqual(result["matched_pairs_with_different_targets"], 1)
        pd.testing.assert_frame_equal(before_train, self.train)
        pd.testing.assert_frame_equal(before_test, self.test)

    def test_many_to_one_match_and_conflicting_targets(self):
        self.check_counts()

    def test_numeric_dtype_difference_does_not_hide_equal_records(self):
        self.test["x"] = self.test["x"].astype(float)
        self.check_counts()

    def test_hash_collision_does_not_create_false_matches(self):
        with patch("pandas.util.hash_pandas_object", side_effect=lambda df, **kwargs: pd.Series(0, index=df.index, dtype="uint64")):
            self.check_counts()


if __name__ == "__main__":
    unittest.main()
