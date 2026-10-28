"""Research-critical invariants for exact grouping and validation isolation."""

import unittest
import numpy as np
import pandas as pd

from src.data.duplicate_investigation import build_groups, cross_split_overlap, exact_feature_ids
from src.data.split_data import grouped_validation_assignments, assert_exact_partition_disjointness


class ExactGroupTests(unittest.TestCase):
    def test_targets_do_not_define_groups(self):
        frame = pd.DataFrame({"id": [1, 2, 3], "x": [5, 5, 7], "service": ["dns", "dns", "-"],
                              "label": [0, 1, 1], "attack_cat": ["Normal", "DoS", "DoS"]})
        groups = build_groups(frame)
        self.assertEqual(groups.row_groups.tolist(), [0, 0, 1])
        self.assertTrue(groups.summary.loc[0, "binary_conflict"])
        mutated = frame.copy()
        mutated["id"] += 100
        mutated["label"] = 0
        mutated["attack_cat"] = "Normal"
        np.testing.assert_array_equal(groups.row_groups, build_groups(mutated).row_groups)

    def test_same_multi_target_sets_are_still_conflicting(self):
        a = pd.DataFrame({"id": [1, 2, 3], "x": [1, 1, 2],
                          "label": [0, 1, 0], "attack_cat": ["Normal", "DoS", "Normal"]})
        b = pd.DataFrame({"id": [6, 7, 8, 9], "x": [1.0, 1.0, 2.0, 3.0],
                          "label": [0, 1, 0, 1], "attack_cat": ["Normal", "DoS", "Normal", "DoS"]})
        matches, mapping = cross_split_overlap(build_groups(a), build_groups(b))
        self.assertEqual(len(matches), 2)
        self.assertEqual(matches.category_status.tolist(), ["conflicting", "consistent"])
        self.assertEqual(matches.binary_status.tolist(), ["conflicting", "consistent"])
        self.assertEqual(mapping.tolist(), [0, 1, -1])

    def test_no_approximate_matching_and_no_missing_imputation(self):
        frame = pd.DataFrame({"x": [1.0, 1.0 + 1e-12]})
        ids, _ = exact_feature_ids(frame, ["x"])
        self.assertNotEqual(ids[0], ids[1])
        frame.loc[1, "x"] = np.nan
        with self.assertRaises(ValueError):
            exact_feature_ids(frame, ["x"])
        with self.assertRaises(ValueError):
            exact_feature_ids(pd.DataFrame({"label": [0, 1]}), ["label"])


class ValidationProtocolTests(unittest.TestCase):
    def test_reproducible_complete_assignments_and_conflict_isolation(self):
        records = []
        for group in range(80):
            for repeat in range(1 + group % 3):
                category = "Normal" if group % 2 == 0 else "AttackA"
                if group == 0 and repeat == 0:
                    category = "AttackA"
                records.append({"id": len(records) + 1, "x": group, "service": "tcp",
                                "label": int(category != "Normal"), "attack_cat": category})
        # Add a genuinely contradictory copy to group 0.
        records.append({"id": len(records) + 1, "x": 0, "service": "tcp", "label": 0, "attack_cat": "Normal"})
        frame = pd.DataFrame(records)
        before = frame.copy(deep=True)
        groups = build_groups(frame)
        first = grouped_validation_assignments(frame, groups)
        second = grouped_validation_assignments(frame, groups)
        pd.testing.assert_frame_equal(first, second)
        pd.testing.assert_frame_equal(frame, before)
        self.assertEqual(len(first), len(frame))
        self.assertEqual(first.groupby("feature_group_id").role.nunique().max(), 1)
        self.assertTrue((first.label.to_numpy() == frame.label.to_numpy()).all())
        checks = assert_exact_partition_disjointness(frame, first, groups.columns)
        self.assertTrue(all(value == 0 for value in checks.values()))
        # Corrupt one assignment: the independent exact-value check must reject it.
        positions = np.flatnonzero(groups.row_groups == 0)
        first.loc[positions[0], "role"] = "train"
        first.loc[positions[-1], "role"] = "model_selection"
        with self.assertRaises(AssertionError):
            assert_exact_partition_disjointness(frame, first, groups.columns)


if __name__ == "__main__":
    unittest.main()
