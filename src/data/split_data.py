"""Training-file-only grouped validation assignments. No preprocessing or fitting."""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.data.duplicate_investigation import FeatureGroups, exact_feature_ids


def grouped_validation_assignments(train: pd.DataFrame, groups: FeatureGroups,
                                   seed: int = 42, calibration_seed: int = 43) -> pd.DataFrame:
    """Use prespecified fold 0 of five; divide validation into two group folds.

    Target values balance partitions only, never define group membership.
    Neither the test frame, test labels, nor overlap masks are accepted here.
    """
    if len(train) != len(groups.row_groups):
        raise ValueError("Training frame and group assignments differ in length.")
    y = train.attack_cat.to_numpy()
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    fit, validation = next(outer.split(np.zeros((len(train), 1)), y, groups.row_groups))
    inner = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=calibration_seed)
    selection_local, calibration_local = next(inner.split(
        np.zeros((len(validation), 1)), y[validation], groups.row_groups[validation]))
    roles = np.full(len(train), "train", dtype=object)
    roles[validation[selection_local]] = "model_selection"
    roles[validation[calibration_local]] = "calibration"
    result = pd.DataFrame({"row_position": np.arange(len(train)), "csv_line": np.arange(len(train)) + 2,
                           "source_id": train.id.to_numpy(), "feature_group_id": groups.row_groups,
                           "partition": np.where(roles == "train", "train", "validation"), "role": roles,
                           "label": train.label.to_numpy(), "attack_cat": train.attack_cat.to_numpy()})
    assert len(fit) == result.partition.eq("train").sum()
    assert result.groupby("feature_group_id").partition.nunique().max() == 1
    assert result.groupby("feature_group_id").role.nunique().max() == 1
    return result


def assert_exact_partition_disjointness(train: pd.DataFrame, assignments: pd.DataFrame,
                                        predictors: list[str]) -> dict:
    """Independently reconstruct exact keys from source rows, not stored IDs."""
    if len(assignments) != len(train) or not np.array_equal(assignments.row_position, np.arange(len(train))):
        raise AssertionError("Assignments must cover every source row once in source order.")
    keys = {}
    for role in ("train", "model_selection", "calibration"):
        mask = assignments.role.eq(role).to_numpy()
        _, lookup = exact_feature_ids(train.iloc[np.flatnonzero(mask)], predictors)
        keys[role] = set(lookup)
    report = {}
    for a, b in (("train", "model_selection"), ("train", "calibration"),
                 ("model_selection", "calibration")):
        count = len(keys[a] & keys[b])
        report[f"{a}_vs_{b}_exact_vectors"] = count
        if count:
            raise AssertionError(f"Exact feature overlap between {a} and {b}: {count}")
    report["train_vs_validation_exact_vectors"] = len(keys["train"] & (keys["model_selection"] | keys["calibration"]))
    return report


def partition_class_balance(assignments: pd.DataFrame) -> pd.DataFrame:
    """Report row and distinct-vector counts; groups may contain several classes."""
    rows = []
    masks = {"official_train": np.ones(len(assignments), dtype=bool),
             "train": assignments.partition.eq("train").to_numpy(),
             "validation": assignments.partition.eq("validation").to_numpy()}
    masks.update({role: assignments.role.eq(role).to_numpy() for role in ("model_selection", "calibration")})
    for partition, mask in masks.items():
        data = assignments.loc[mask]
        for target in ("label", "attack_cat"):
            for value in sorted(assignments[target].unique(), key=str):
                selected = data[target].eq(value)
                original = assignments[target].eq(value)
                total = int(original.sum())
                rows.append({"partition": partition, "target": target, "class": str(value),
                             "rows": int(selected.sum()), "independent_feature_groups": int(data.loc[selected, "feature_group_id"].nunique()),
                             "partition_class_pct": selected.mean() * 100 if len(data) else np.nan,
                             "share_of_original_class_pct": selected.sum() / total * 100,
                             "original_class_pct": original.mean() * 100,
                             "distribution_difference_pp": (selected.mean() - original.mean()) * 100 if len(data) else np.nan})
    return pd.DataFrame(rows)
