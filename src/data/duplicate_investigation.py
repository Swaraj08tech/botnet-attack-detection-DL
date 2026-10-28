"""Exact feature-vector diagnostics without changing observations or labels."""

from dataclasses import dataclass
from itertools import combinations
import json

import numpy as np
import pandas as pd

from src.data.validate_data import feature_columns


@dataclass
class FeatureGroups:
    """Group IDs are local first-occurrence indices, bound to the source hash."""

    columns: list[str]
    row_groups: np.ndarray
    key_to_group: dict[tuple, int]
    summary: pd.DataFrame
    category_counts: pd.DataFrame
    binary_counts: pd.DataFrame


def exact_feature_ids(frame: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, dict]:
    """Use exact tuple equality after Python dictionary hashing, never hash alone.

    Missing predictors are rejected rather than implicitly filled or normalized.
    No rounding, tolerance, label or identifier enters the tuple.
    """
    if not columns or frame[columns].isna().any().any():
        raise ValueError("Exact grouping requires nonempty, nonmissing predictors.")
    if set(columns) & {"id", "label", "attack_cat"}:
        raise ValueError("IDs and targets cannot define feature groups.")
    ids = np.empty(len(frame), dtype=np.int64)
    lookup: dict[tuple, int] = {}
    for position, key in enumerate(frame[columns].itertuples(index=False, name=None)):
        group = lookup.get(key)
        if group is None:
            group = len(lookup)
            lookup[key] = group
        ids[position] = group
    return ids, lookup


def build_groups(frame: pd.DataFrame) -> FeatureGroups:
    """Assign groups without targets; summarize targets only after assignment."""
    columns = feature_columns(frame)
    ids, lookup = exact_feature_ids(frame, columns)
    records = frame[["label", "attack_cat"]].reset_index(drop=True).copy()
    records.insert(0, "group_id", ids)
    category_counts = pd.crosstab(records.group_id, records.attack_cat).sort_index()
    binary_counts = pd.crosstab(records.group_id, records.label).sort_index()
    sizes = np.bincount(ids)
    first = np.full(len(lookup), len(frame), dtype=np.int64)
    np.minimum.at(first, ids, np.arange(len(frame)))
    summary = pd.DataFrame({"group_id": np.arange(len(lookup)), "rows": sizes,
                            "first_row_position": first, "first_source_id": frame.iloc[first]["id"].to_numpy()})
    for target, counts in (("binary", binary_counts), ("category", category_counts)):
        present = counts.to_numpy() > 0
        summary[f"{target}_classes"] = present.sum(axis=1)
        summary[f"{target}_conflict"] = present.sum(axis=1) > 1
        labels = counts.columns.to_numpy()
        summary[f"{target}_combination"] = [json.dumps(labels[row].tolist()) for row in present]
    return FeatureGroups(columns, ids, lookup, summary, category_counts, binary_counts)


def repetition_summary(frame: pd.DataFrame, groups: FeatureGroups, split: str) -> tuple[dict, pd.DataFrame]:
    sizes = groups.summary.rows.to_numpy()
    repeated = sizes[groups.row_groups] > 1
    overall = {"dataset": split, "rows": len(frame), "unique_vectors": len(sizes),
               "repeated_groups": int((sizes > 1).sum()), "participating_rows": int(sizes[sizes > 1].sum()),
               "excess_duplicate_rows": int((sizes - 1).sum()), "largest_group": int(sizes.max())}
    rows = []
    for target in ("label", "attack_cat"):
        for value in sorted(frame[target].unique(), key=str):
            mask = frame[target].eq(value).to_numpy()
            unique = np.unique(groups.row_groups[mask]).size
            rows.append({"dataset": split, "target": target, "class": str(value), "rows": int(mask.sum()),
                         "independent_feature_groups": int(unique),
                         "rows_in_repeated_groups": int((mask & repeated).sum()),
                         "repeated_group_row_pct": float((mask & repeated).sum() / mask.sum() * 100),
                         "excess_rows_within_class": int(mask.sum() - unique)})
    return overall, pd.DataFrame(rows)


def ambiguity_tables(frame: pd.DataFrame, groups: FeatureGroups, split: str):
    """Return per-task counts, class combinations, pairs, and bounded examples."""
    totals, combos, pairs, examples = [], [], [], []
    for target, prefix, counts in (("label", "binary", groups.binary_counts),
                                   ("attack_cat", "category", groups.category_counts)):
        selected = groups.summary.loc[groups.summary[f"{prefix}_conflict"]]
        affected = np.isin(groups.row_groups, selected.group_id)
        totals.append({"dataset": split, "target": target, "affected_groups": len(selected),
                       "affected_rows": int(affected.sum()), "affected_row_pct": affected.mean() * 100})
        for combo, table in selected.groupby(f"{prefix}_combination", sort=True):
            combos.append({"dataset": split, "target": target, "class_combination": combo,
                           "groups": len(table), "rows": int(table.rows.sum())})
        # Each unordered pair contributes once per feature group, not per row pair.
        pair_counts = {}
        for gid in selected.group_id:
            present = counts.columns[counts.loc[gid].to_numpy() > 0].tolist()
            for a, b in combinations(present, 2):
                pair_counts[(str(a), str(b))] = pair_counts.get((str(a), str(b)), 0) + 1
        pairs.extend({"dataset": split, "target": target, "class_a": a, "class_b": b,
                      "shared_conflicting_groups": n} for (a, b), n in sorted(pair_counts.items()))
        # Largest three conflict groups, one real source row per target value.
        for gid in selected.sort_values(["rows", "group_id"], ascending=[False, True]).head(3).group_id:
            positions = np.flatnonzero(groups.row_groups == gid)
            candidates = frame.iloc[positions]
            for value in sorted(candidates[target].unique(), key=str):
                position = positions[np.flatnonzero(candidates[target].eq(value).to_numpy())[0]]
                example = frame.iloc[position].to_dict()
                example.update(dataset=split, conflict_target=target, group_id=int(gid),
                               row_position=int(position), csv_line=int(position + 2))
                examples.append(example)
    return pd.DataFrame(totals), pd.DataFrame(combos), pd.DataFrame(pairs), pd.DataFrame(examples)


def cross_split_overlap(train: FeatureGroups, test: FeatureGroups) -> tuple[pd.DataFrame, np.ndarray]:
    """Match exact predictor tuples across independently built groups.

    A consistent group has one identical target in the union of both splits.
    A conflicting group has more than one target in that union, even when the
    same set of multiple classes occurs in both splits.
    """
    if train.columns != test.columns:
        raise ValueError("Predictor order must match for exact comparison.")
    matches = np.full(len(test.summary), -1, dtype=np.int64)
    rows = []
    for key, test_id in test.key_to_group.items():
        train_id = train.key_to_group.get(key)
        if train_id is None:
            continue
        matches[test_id] = train_id
        record = {"train_group_id": train_id, "test_group_id": test_id,
                  "train_rows": int(train.summary.loc[train_id, "rows"]),
                  "test_rows": int(test.summary.loc[test_id, "rows"])}
        for prefix, a, b in (("binary", train.binary_counts, test.binary_counts),
                              ("category", train.category_counts, test.category_counts)):
            va = set(a.columns[a.loc[train_id].to_numpy() > 0].tolist())
            vb = set(b.columns[b.loc[test_id].to_numpy() > 0].tolist())
            record[f"{prefix}_status"] = "consistent" if len(va | vb) == 1 else "conflicting"
            record[f"{prefix}_union"] = json.dumps(sorted(va | vb))
        rows.append(record)
    columns = ["train_group_id", "test_group_id", "train_rows", "test_rows",
               "binary_status", "binary_union", "category_status", "category_union"]
    return pd.DataFrame(rows, columns=columns), matches


def overlap_by_class(frame: pd.DataFrame, groups: FeatureGroups, matches: pd.DataFrame,
                     split: str) -> pd.DataFrame:
    rows = []
    key = f"{split}_group_id"
    for target, prefix in (("label", "binary"), ("attack_cat", "category")):
        for status in ("consistent", "conflicting"):
            selected = matches.loc[matches[f"{prefix}_status"].eq(status), key]
            overlap = np.isin(groups.row_groups, selected)
            for value in sorted(frame[target].unique(), key=str):
                mask = overlap & frame[target].eq(value).to_numpy()
                rows.append({"dataset": split, "target": target, "status": status, "class": str(value),
                             "participating_rows": int(mask.sum()),
                             "participating_vectors": int(np.unique(groups.row_groups[mask]).size)})
    return pd.DataFrame(rows)


def policy_class_counts(frame: pd.DataFrame, groups: FeatureGroups, split: str) -> pd.DataFrame:
    """Quantify policy and rejected alternatives without changing the dataset."""
    keep_all = np.ones(len(frame), dtype=bool)
    binary_clear = ~groups.summary.binary_conflict.to_numpy()[groups.row_groups]
    category_clear = ~groups.summary.category_conflict.to_numpy()[groups.row_groups]
    labeled_unique = ~frame.duplicated(subset=groups.columns + ["label", "attack_cat"]).to_numpy()
    policies = {"adopted_keep_all_labels_and_rows": keep_all,
                "diagnostic_only_remove_binary_conflict_groups": binary_clear,
                "diagnostic_only_remove_category_conflict_groups": category_clear,
                "diagnostic_only_one_per_exact_labeled_record": labeled_unique}
    rows = []
    for policy, keep in policies.items():
        for target in ("label", "attack_cat"):
            for value in sorted(frame[target].unique(), key=str):
                original = frame[target].eq(value).to_numpy()
                rows.append({"dataset": split, "policy": policy, "target": target, "class": str(value),
                             "original_rows": int(original.sum()), "retained_rows": int((original & keep).sum()),
                             "excluded_rows": int((original & ~keep).sum())})
    return pd.DataFrame(rows)
