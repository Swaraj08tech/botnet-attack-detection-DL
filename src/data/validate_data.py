"""Deterministic, read-only checks. Flags are investigation prompts, not cleaning rules."""

from collections import Counter, defaultdict

import numpy as np
import pandas as pd

TARGETS = ("label", "attack_cat")
IDENTIFIERS = ("id",)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Candidate predictors for diagnostics only; no columns are removed."""
    return [c for c in df if c not in (*TARGETS, *IDENTIFIERS)]


def feature_summary(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    rows = []
    for column in df:
        s = df[column]
        role = "Target" if column in TARGETS else "Identifier" if column in IDENTIFIERS else "Boolean" if pd.api.types.is_bool_dtype(s) else "Numerical" if pd.api.types.is_numeric_dtype(s) else "Categorical"
        frequencies = s.value_counts(dropna=False)
        rows.append({"Dataset": dataset, "Column": column, "Data Type": str(s.dtype), "Unique Values": s.nunique(dropna=False), "Missing Count": int(s.isna().sum()), "Missing %": float(s.isna().mean() * 100), "Likely Role": role, "Dominant Value %": float(frequencies.iloc[0] / len(s) * 100) if len(s) else np.nan, "Leakage Concern": "Exclude both targets from predictors" if column in TARGETS else "Index may encode row ordering; investigate" if column in IDENTIFIERS else ""})
    return pd.DataFrame(rows)


def class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in TARGETS:
        if target not in df:
            continue
        counts = df[target].value_counts(dropna=False)
        for value, count in counts.items():
            rows.append({"Target": target, "Class": str(value), "Count": int(count), "Percentage": float(count / len(df) * 100), "Majority-to-Class Ratio": float(counts.max() / count), "Rare (<1%)": bool(count / len(df) < 0.01)})
    return pd.DataFrame(rows)


def numeric_statistics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in df.select_dtypes(include="number"):
        s = df[column]
        finite = s[np.isfinite(s)]
        stats = finite.describe(percentiles=[0.25, 0.5, 0.75]).to_dict()
        stats.update({"Column": column, "Negative Count": int((s < 0).sum()), "Zero %": float((s == 0).mean() * 100), "Infinite Count": int(np.isinf(s).sum()), "Skew": float(finite.skew()), "Large Magnitude (>1e6) Count": int((finite.abs() > 1e6).sum())})
        flags = []
        if stats["Negative Count"]:
            flags.append("negative: validate feature meaning")
        if stats["Zero %"] >= 90:
            flags.append("at least 90% zeros: inspect applicability")
        if abs(stats["Skew"]) > 5:
            flags.append("absolute skew >5")
        if stats["Large Magnitude (>1e6) Count"]:
            flags.append("magnitude >1e6: check units")
        stats["Investigation Flags"] = "; ".join(flags)
        rows.append(stats)
    return pd.DataFrame(rows)


def quality_summary(df: pd.DataFrame, dataset: str) -> dict:
    predictors = feature_columns(df)
    numeric = [c for c in predictors if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    categorical = [c for c in predictors if not pd.api.types.is_numeric_dtype(df[c])]
    return {"Dataset": dataset, "Rows": len(df), "Columns": len(df.columns), "Candidate Features": len(predictors), "Numerical Features": len(numeric), "Categorical Features": len(categorical), "Boolean Features": sum(pd.api.types.is_bool_dtype(df[c]) for c in predictors), "Missing Values": int(df.isna().sum().sum()), "Infinite Values": int(np.isinf(df.select_dtypes(include="number")).sum().sum()), "Duplicate Rows": int(df.duplicated().sum()), "Duplicate Features (excluding id and targets)": int(df.duplicated(subset=predictors).sum()), "Duplicate Labeled Records (excluding id)": int(df.duplicated(subset=[c for c in df if c not in IDENTIFIERS]).sum()), "Number of Classes": int(df["attack_cat"].nunique()) if "attack_cat" in df else None, "Binary Classes": int(df["label"].nunique()) if "label" in df else None}


def compare_schemas(train: pd.DataFrame, test: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    common = [c for c in train if c in test]
    report = {"missing_columns_in_test": sorted(set(train) - set(test)), "extra_columns_in_test": sorted(set(test) - set(train)), "same_column_order": list(train) == list(test), "dtype_mismatches": {c: [str(train[c].dtype), str(test[c].dtype)] for c in common if train[c].dtype != test[c].dtype}, "candidate_feature_counts": [len(feature_columns(train)), len(feature_columns(test))]}
    rows = []
    for c in common:
        if not pd.api.types.is_numeric_dtype(train[c]) or c in TARGETS:
            train_values = set(train[c].dropna().tolist())
            test_values = set(test[c].dropna().tolist())
            rows.append({"Column": c, "Train Only Values": repr(sorted(train_values - test_values, key=str)), "Test Only Values": repr(sorted(test_values - train_values, key=str)), "Train Unique": len(train_values), "Test Unique": len(test_values), "Train Missing %": train[c].isna().mean() * 100, "Test Missing %": test[c].isna().mean() * 100})
    return report, pd.DataFrame(rows)


def distribution_comparison(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in feature_columns(train):
        if c not in test or not pd.api.types.is_numeric_dtype(train[c]) or not pd.api.types.is_numeric_dtype(test[c]):
            continue
        a = train[c][np.isfinite(train[c])]
        b = test[c][np.isfinite(test[c])]
        pooled_sd = np.sqrt((a.var() + b.var()) / 2)
        rows.append({"Column": c, "Train Mean": a.mean(), "Test Mean": b.mean(), "Train Median": a.median(), "Test Median": b.median(), "Absolute Standardized Mean Difference": abs(a.mean() - b.mean()) / pooled_sd if pooled_sd > 0 else np.nan, "Train Zero %": (train[c] == 0).mean() * 100, "Test Zero %": (test[c] == 0).mean() * 100, "Train Missing %": train[c].isna().mean() * 100, "Test Missing %": test[c].isna().mean() * 100})
    return pd.DataFrame(rows).sort_values("Absolute Standardized Mean Difference", ascending=False)


def duplicate_overlap(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Hash-filter candidates, then verify exact tuples (no reliance on hash equality).

    Excludes id and targets only for diagnostics. Neither source dataset is merged,
    resplit, edited, or deduplicated. Counts include all participating source rows.
    """
    columns = feature_columns(train)
    if set(columns) != set(feature_columns(test)):
        return {"status": "not checked: predictor schemas differ"}
    # Hashing may differ for numerically equal values with different dtypes.
    # In that case bypass the prefilter and compare exact tuples for all rows.
    same_dtypes = all(train[c].dtype == test[c].dtype for c in columns)
    ha = pd.util.hash_pandas_object(train[columns], index=False) if same_dtypes else None
    hb = pd.util.hash_pandas_object(test[columns], index=False) if same_dtypes else None
    shared = set(ha) & set(hb) if same_dtypes else None
    targets = [c for c in TARGETS if c in train and c in test]

    def records(df, hashes):
        counts = Counter()
        labels = defaultdict(Counter)
        candidates = df.loc[hashes.isin(shared), columns + targets] if shared is not None else df[columns + targets]
        for row in candidates.itertuples(index=False, name=None):
            values = tuple(None if pd.isna(v) else v for v in row)
            key = values[:len(columns)]
            counts[key] += 1
            labels[key][values[len(columns):]] += 1
        return counts, labels

    ca, la = records(train, ha)
    cb, lb = records(test, hb)
    keys = ca.keys() & cb.keys()
    conflicting_keys = sum(set(la[k]) != set(lb[k]) or len(set(la[k]) | set(lb[k])) > 1 for k in keys)
    mismatched_pairs = sum(na * nb for k in keys for va, na in la[k].items() for vb, nb in lb[k].items() if va != vb)
    return {"status": "verified exact tuples after hash filtering" if same_dtypes else "verified exact tuples without hash filtering: dtype differences", "excluded_columns": list(IDENTIFIERS + TARGETS), "shared_unique_feature_records": len(keys), "train_rows_with_test_match": sum(ca[k] for k in keys), "test_rows_with_train_match": sum(cb[k] for k in keys), "matching_row_pairs": sum(ca[k] * cb[k] for k in keys), "shared_feature_records_with_conflicting_targets": int(conflicting_keys), "matched_pairs_with_different_targets": int(mismatched_pairs)}


def leakage_screen(df: pd.DataFrame) -> pd.DataFrame:
    """Check low-cardinality deterministic label mappings; no predictor fitting."""
    rows = []
    if "label" not in df:
        return pd.DataFrame()
    for column in df:
        if column == "label":
            continue
        n = df[column].nunique(dropna=False)
        if n <= 100:
            groups = df.groupby(column, dropna=False)["label"].agg(["nunique", "size"])
            pure = groups["nunique"] == 1
            rows.append({"Column": column, "Unique Values": n, "Deterministic Label Mapping": bool(pure.all()), "Rows in Pure Groups %": float(groups.loc[pure, "size"].sum() / len(df) * 100), "Interpretation": "Target-derived: exclude from predictors" if column in TARGETS else "Association alone does not prove leakage; investigate source and availability"})
    return pd.DataFrame(rows)


def redundant_columns(df: pd.DataFrame) -> list[list[str]]:
    """List exactly equal columns without changing them."""
    columns = list(df)
    return [[a, b] for i, a in enumerate(columns) for b in columns[i + 1:] if df[a].equals(df[b])]
