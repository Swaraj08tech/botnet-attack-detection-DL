"""Author Notebook 01; run explicitly, never overwrite an existing notebook."""

from pathlib import Path
import textwrap

import nbformat


def build_notebook() -> None:
    root = Path(__file__).resolve().parents[1]
    destination = root / "notebooks/01_data_understanding.ipynb"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination.name}")
    cells = []

    def md(text):
        cells.append(nbformat.v4.new_markdown_cell(textwrap.dedent(text).strip()))

    def code(text):
        cells.append(nbformat.v4.new_code_cell(textwrap.dedent(text).strip()))

    md("""
    # 01 — Data Understanding and Validation

    ## Objective
    Validate the supplied UNSW-NB15 training and testing CSVs separately, before
    preprocessing, feature engineering, or modelling. Inspect schema, data quality,
    target distributions, repeated records, and possible leakage.

    Raw data remains immutable. Diagnostic subsets and aggregations do not change
    source frames or files. The official splits are never combined or resplit.

    Test inspection is limited to the explicitly requested dataset audit. It must
    not inform later hyperparameter, feature-ranking, threshold, or calibration decisions.
    Any evaluation integrity issues require a documented protocol before modelling.
    """)
    md("""
    ## Imports and Project Paths
    In VS Code, select the project `.venv` Python interpreter as the Jupyter kernel,
    then **Run All**. Required notebook packages are listed in `requirements.txt`.
    This notebook never installs dependencies.

    The first cell displays the actual interpreter and fails outside a virtual
    environment. Root discovery works from the repository root or `notebooks/`.
    One bootstrap adds the detected root to Python's module search path so imports
    use the reusable `src/` package. No machine-specific paths are embedded.
    """)
    code("""
    from pathlib import Path
    import sys
    import platform
    import importlib.metadata
    import json

    print("Interpreter:", sys.executable)
    print("Python:", platform.python_version())
    print("Virtual environment:", sys.prefix != sys.base_prefix)
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("Select the project's .venv Python interpreter as the VS Code notebook kernel.")

    current = Path.cwd().resolve()
    candidates = (current, *current.parents)
    PROJECT_ROOT = next((p for p in candidates if (p / "README.md").is_file()
                         and all((p / part).is_dir() for part in ("data/raw", "src", "notebooks"))), None)
    if PROJECT_ROOT is None:
        raise FileNotFoundError("Run the notebook from within the project directory.")
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from IPython.display import display, Markdown
    from src.utils.paths import find_project_root
    from src.data.load_data import discover_datasets, load_datasets, sha256
    from src.data.validate_data import (
        TARGETS, IDENTIFIERS, feature_columns, feature_summary, class_distribution,
        quality_summary, numeric_statistics, compare_schemas, distribution_comparison,
        duplicate_overlap, leakage_screen, redundant_columns,
    )

    assert PROJECT_ROOT == find_project_root()
    RAW_DIR = PROJECT_ROOT / "data/raw"
    TABLE_DIR = PROJECT_ROOT / "results/tables"
    FIGURE_DIR = PROJECT_ROOT / "results/figures"
    for directory in (TABLE_DIR, FIGURE_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.max_columns", 50)
    pd.set_option("display.max_rows", 60)
    plt.rcParams.update({"figure.dpi": 110, "font.size": 10})
    versions = pd.DataFrame([
        {"Package": name, "Version": importlib.metadata.version(name)}
        for name in ("pandas", "numpy", "matplotlib", "ipykernel", "nbformat", "nbclient")
    ])
    display(versions)
    versions.to_csv(TABLE_DIR / "01_environment_versions.csv", index=False)
    """)
    md("""
    ## Dataset Loading
    Discover filenames rather than assume them. Naming identifies the supplied
    split; independent official download provenance has not been established.
    Record hashes before reading and compare with the placement manifest.
    Pandas default type inference is used; category strings are not stripped,
    renamed, encoded, or otherwise cleaned.
    """)
    code("""
    paths = discover_datasets(RAW_DIR)
    hashes_before = {split: sha256(path) for split, path in paths.items()}
    manifest = json.loads((PROJECT_ROOT / "docs/initialization_manifest.json").read_text(encoding="utf-8-sig"))
    expected_hashes = {Path(item["path"]).name: item["sha256"].lower() for item in manifest["datasets"]["files"]}
    for split, path in paths.items():
        print(split, path.relative_to(PROJECT_ROOT), path.stat().st_size, "bytes")
        assert hashes_before[split] == expected_hashes[path.name], "Raw file changed since placement"
    train_df, test_df = load_datasets(paths)
    datasets = {"train": train_df, "test": test_df}
    shapes_before = {name: df.shape for name, df in datasets.items()}
    """)
    md("## Dataset Overview")
    code("""
    for name, df in datasets.items():
        display(Markdown(f"### {name.title()} dataset"))
        print("Rows:", len(df), "Columns:", len(df.columns), "Shape:", df.shape)
        print("Columns:", df.columns.tolist())
        display(df.head())
        display(df.dtypes.rename("dtype").to_frame())
    """)
    md("""
    ## Schema Analysis
    Count predictors separately from the ID and two targets. Role assignments
    are hypotheses for later review, not a feature-selection operation.
    Numeric 0/1 fields are reported as numeric if pandas infers integers;
    boolean-like value sets are listed separately from true boolean dtypes.
    """)
    code("""
    summaries = {name: feature_summary(df, name) for name, df in datasets.items()}
    summary_table = pd.concat(summaries.values(), ignore_index=True)
    summary_table.to_csv(TABLE_DIR / "feature_summary.csv", index=False)
    for name, df in datasets.items():
        print(name, "Targets:", [c for c in TARGETS if c in df])
        print("Identifiers:", [c for c in IDENTIFIERS if c in df])
        print("Candidate predictors:", feature_columns(df))
        print("Boolean dtype columns:", df.select_dtypes(include="bool").columns.tolist())
        print("Numeric boolean-like fields:", [c for c in df.select_dtypes(include="number")
              if set(df[c].dropna().unique()) <= {0, 1}])
        print("Exactly redundant columns:", redundant_columns(df))
        display(summaries[name])
    """)
    md("""
    ## Data Quality
    Missing counts include pandas NaN/NA values. Infinite values are checked
    separately. Near-constant means at least 99% of rows have one value (a
    descriptive threshold, not a removal rule). Literal blank strings and
    placeholder `-` strings are counted because pandas does not equate every
    sentinel with a missing value. Their meaning remains unverified.
    """)
    code("""
    quality = pd.DataFrame([quality_summary(df, name) for name, df in datasets.items()])
    quality.to_csv(TABLE_DIR / "dataset_summary.csv", index=False)
    placeholder_rows = []
    for name, df in datasets.items():
        display(Markdown(f"### {name.title()} quality"))
        display(summaries[name][["Column", "Missing Count", "Missing %"]])
        print("Constant columns:", summaries[name].loc[summaries[name]["Unique Values"] == 1, "Column"].tolist())
        print("Near-constant columns (>=99% dominance):", summaries[name].loc[
            (summaries[name]["Dominant Value %"] >= 99) & (summaries[name]["Unique Values"] > 1), "Column"].tolist())
        for c in df.select_dtypes(include=["object", "string"]):
            placeholder_rows.append({"Dataset": name, "Column": c,
                "Blank String Count": int(df[c].str.fullmatch(r"\\s*", na=False).sum()),
                "Dash Placeholder Count": int(df[c].eq("-").sum()),
                "Edge Whitespace Count": int(df[c].ne(df[c].str.strip()).fillna(False).sum())})
    placeholders = pd.DataFrame(placeholder_rows)
    placeholders.to_csv(TABLE_DIR / "01_categorical_placeholders.csv", index=False)
    display(quality)
    display(placeholders)
    """)
    md("## Target Analysis")
    code("""
    distributions = {name: class_distribution(df) for name, df in datasets.items()}
    for name, df in datasets.items():
        print(name, {c: sorted(df[c].dropna().unique().tolist(), key=str) for c in TARGETS if c in df})
        display(distributions[name])
        distributions[name].to_csv(TABLE_DIR / f"class_distribution_{name}.csv", index=False)
    """)
    code("""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, (name, df) in zip(axes, datasets.items()):
        counts = df["label"].value_counts().reindex([0, 1], fill_value=0)
        percentages = counts / len(df) * 100
        bars = ax.bar(["Normal (0)", "Attack (1)"], percentages, color=["#527b94", "#b05d45"])
        ax.bar_label(bars, labels=[f"{n:,}\\n{p:.1f}%" for n, p in zip(counts, percentages)], padding=4)
        ax.set(title=name.title(), ylim=(0, 82), ylabel="Share of dataset (%)")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Binary target distribution")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "01_binary_class_distribution.png", dpi=180, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    """)
    code("""
    categories = train_df["attack_cat"].value_counts().index.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), sharey=True)
    for ax, (name, df) in zip(axes, datasets.items()):
        counts = df["attack_cat"].value_counts().reindex(categories, fill_value=0)
        bars = ax.barh(categories, counts / len(df) * 100, color="#527b94")
        ax.bar_label(bars, labels=[f"{v:,}" for v in counts], padding=3, fontsize=9)
        ax.set(title=name.title(), xlabel="Share of dataset (%)", xlim=(0, 54))
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].invert_yaxis()
    fig.suptitle("Attack-category distribution (including Normal)")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "01_attack_class_distribution.png", dpi=180, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    """)
    md("""
    ## Train/Test Comparison
    Explicitly inspect missing/extra columns, ordering, feature counts, dtypes,
    category differences and missing-value rates. Category lists are observations,
    not vocabulary fitted to both splits. Future encoders must be fitted on the
    training partition only and support unseen categories.
    """)
    code("""
    schema_report, category_comparison = compare_schemas(train_df, test_df)
    print("MISSING COLUMNS in test:", schema_report["missing_columns_in_test"])
    print("EXTRA COLUMNS in test:", schema_report["extra_columns_in_test"])
    print("DTYPE MISMATCHES:", schema_report["dtype_mismatches"])
    print("Same column ordering:", schema_report["same_column_order"])
    print("Candidate feature counts:", schema_report["candidate_feature_counts"])
    print("CATEGORY DIFFERENCES:")
    display(category_comparison)
    category_comparison.to_csv(TABLE_DIR / "01_category_comparison.csv", index=False)
    (TABLE_DIR / "01_schema_comparison.json").write_text(json.dumps(schema_report, indent=2), encoding="utf-8")
    class_shift = pd.DataFrame({name: df["attack_cat"].value_counts(normalize=True) * 100
                                for name, df in datasets.items()}).fillna(0)
    class_shift["Test minus Train (percentage points)"] = class_shift["test"] - class_shift["train"]
    display(class_shift)
    class_shift.to_csv(TABLE_DIR / "01_target_distribution_shift.csv", index_label="Class")
    """)
    md("""
    ## Statistical Analysis
    Statistics are descriptive. Negative values require feature-specific meaning;
    large magnitudes can be valid given units. Flags use >1e6 absolute magnitude,
    >=90% zeros, or absolute skew >5; these thresholds only focus investigation.
    Infinite values are counted separately and omitted from finite statistics.
    No outliers are removed and no features are transformed.
    """)
    code("""
    statistics = {name: numeric_statistics(df) for name, df in datasets.items()}
    for name, table in statistics.items():
        display(Markdown(f"### {name.title()} numerical statistics"))
        display(table)
        table.to_csv(TABLE_DIR / f"01_numeric_statistics_{name}.csv", index=False)
        display(table.loc[table["Investigation Flags"].ne(""),
            ["Column", "min", "max", "Negative Count", "Zero %", "Skew", "Investigation Flags"]])
    numeric_shift = distribution_comparison(train_df, test_df)
    display(numeric_shift)
    numeric_shift.to_csv(TABLE_DIR / "01_numeric_distribution_comparison.csv", index=False)
    """)
    md("""
    Absolute standardized mean difference divides the mean gap by the square root
    of the average within-split variance. It is a descriptive comparison only,
    not scaling performed on model inputs or a criterion for feature selection.
    Large differences motivate protocol review, not test-driven tuning.
    """)
    md("""
    ## Class Imbalance
    Imbalance ratio is largest class count divided by smallest class count.
    A class with <1% of a split is called rare for this audit. These thresholds
    are descriptive. Small class counts limit the stability of future class-wise
    metrics; macro F1 and per-class recall will need attention.
    No resampling is performed.
    """)
    code("""
    imbalance_rows = []
    for name, table in distributions.items():
        for target in TARGETS:
            subset = table.loc[table["Target"].eq(target)].sort_values("Count", ascending=False)
            if subset.empty:
                continue
            imbalance_rows.append({"Dataset": name, "Target": target,
                "Majority Class": subset.iloc[0]["Class"], "Minority Class": subset.iloc[-1]["Class"],
                "Majority Count": int(subset.iloc[0]["Count"]), "Minority Count": int(subset.iloc[-1]["Count"]),
                "Imbalance Ratio": subset.iloc[0]["Count"] / subset.iloc[-1]["Count"]})
        print(name, "Rare attack categories:")
        display(table.loc[table["Target"].eq("attack_cat") & table["Rare (<1%)"]])
    imbalance = pd.DataFrame(imbalance_rows)
    display(imbalance)
    imbalance.to_csv(TABLE_DIR / "01_imbalance_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 4.6))
    y = np.arange(len(categories))
    for offset, (name, df) in zip((-0.18, 0.18), datasets.items()):
        counts = df["attack_cat"].value_counts().reindex(categories, fill_value=0)
        ax.barh(y + offset, counts, height=0.34, label=name.title())
    ax.set_yticks(y, categories)
    ax.set_xscale("log")
    ax.set(title="Class imbalance across supplied splits", xlabel="Record count (log scale)")
    ax.invert_yaxis()
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "01_class_imbalance.png", dpi=180, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    """)
    md("""
    ## Duplicate Analysis
    Full-row duplicates include IDs. Two additional diagnostics exclude ID alone
    and exclude ID plus both targets. Counts use pandas `duplicated()` default:
    excess rows beyond the first occurrence, not all members of duplicate groups.

    Cross-split overlap excludes ID and targets. Hashes efficiently identify
    candidates, then exact feature tuples confirm matches. Report unique matching
    feature vectors, participating rows in each split, and target conflicts.
    Pair counts can exceed participating rows due to many-to-many matches.
    Repeated feature vectors are not necessarily duplicate network events.
    """)
    code("""
    display(quality[["Dataset", "Duplicate Rows", "Duplicate Labeled Records (excluding id)",
                     "Duplicate Features (excluding id and targets)"]])
    overlap = duplicate_overlap(train_df, test_df)
    print(json.dumps(overlap, indent=2))
    pd.DataFrame([overlap]).to_csv(TABLE_DIR / "01_cross_split_overlap.csv", index=False)
    conflict_rows = []
    for name, df in datasets.items():
        predictors = feature_columns(df)
        grouped = df.groupby(predictors, dropna=False, observed=True)[list(TARGETS)].nunique()
        conflict_rows.append({"Dataset": name, "Feature Groups With Binary Conflict": int((grouped["label"] > 1).sum()),
            "Feature Groups With Attack-category Conflict": int((grouped["attack_cat"] > 1).sum())})
    target_conflicts = pd.DataFrame(conflict_rows)
    display(target_conflicts)
    target_conflicts.to_csv(TABLE_DIR / "01_within_split_target_conflicts.csv", index=False)
    """)
    md("""
    ## Leakage Investigation
    `label` and `attack_cat` must both be kept out of candidate predictors for
    either task. Audit the relation between the targets explicitly. Review ID
    ordering and low-cardinality features that deterministically map to labels.
    Pure groups for a feature alone do not prove leakage: small group size and
    genuine predictive associations can produce the same effect. This screening
    cannot establish feature provenance or availability at prediction time.
    """)
    code("""
    leakage_tables = []
    target_consistency = {}
    id_diagnostics = []
    for name, df in datasets.items():
        display(Markdown(f"### {name.title()} leakage checks"))
        contingency = pd.crosstab(df["attack_cat"], df["label"], dropna=False)
        display(contingency)
        contingency.to_csv(TABLE_DIR / f"01_target_cross_tab_{name}.csv")
        target_consistency[name] = int((df["attack_cat"].eq("Normal") != df["label"].eq(0)).sum())
        print("Normal/category vs binary-label inconsistency:", target_consistency[name])
        screen = leakage_screen(df)
        screen.insert(0, "Dataset", name)
        leakage_tables.append(screen)
        display(screen)
        if "id" in df:
            id_diagnostics.append({"Dataset": name, "ID Unique": bool(df["id"].is_unique),
                "ID Monotonic Increasing": bool(df["id"].is_monotonic_increasing),
                "ID Min": int(df["id"].min()), "ID Max": int(df["id"].max()),
                "Adjacent Label Transitions": int(df["label"].ne(df["label"].shift()).sum() - 1),
                "ID-label Pearson Association": float(df["id"].corr(df["label"]))})
            # Diagnostic ordering bins only; not temporal windows or transformed inputs.
            bins = pd.cut(df["id"], bins=4)
            display(pd.crosstab(bins, df["label"], normalize="index") * 100)
    leakage = pd.concat(leakage_tables, ignore_index=True)
    leakage.to_csv(TABLE_DIR / "01_leakage_screen.csv", index=False)
    ids = pd.DataFrame(id_diagnostics)
    display(ids)
    ids.to_csv(TABLE_DIR / "01_identifier_diagnostics.csv", index=False)
    print("No columns have been removed.")
    """)
    md("""
    ## Integrity Verification
    Confirm raw file hashes and in-memory frame contents have not changed.
    Report artifacts are summaries and figures, never modified source CSV copies.
    """)
    code("""
    hashes_after = {split: sha256(path) for split, path in paths.items()}
    assert hashes_before == hashes_after, "Raw source file changed during audit"
    reload_train, reload_test = load_datasets(paths)
    pd.testing.assert_frame_equal(train_df, reload_train)
    pd.testing.assert_frame_equal(test_df, reload_test)
    assert {name: df.shape for name, df in datasets.items()} == shapes_before
    del reload_train, reload_test
    print("PASS: raw SHA-256 hashes and both in-memory dataframes are unchanged.")
    (TABLE_DIR / "01_raw_integrity.json").write_text(json.dumps(
        {"hashes_before": hashes_before, "hashes_after": hashes_after,
         "source_unchanged": True, "dataframes_unchanged": True}, indent=2), encoding="utf-8")
    """)
    md("""
    ## Key Research Findings
    The next cell generates findings from the computed tables after all checks
    succeed. Re-running the notebook refreshes the findings report from current
    outputs. The dated Markdown snapshot at the end is populated after verification.
    """)
    code("""
    def listed(values):
        return ", ".join(str(v) for v in values) or "none"

    def rare_classes(name):
        table = distributions[name]
        rare = table.loc[table["Target"].eq("attack_cat") & table["Rare (<1%)"]]
        return "; ".join(f"{r['Class']}: {r['Count']:,} ({r['Percentage']:.3f}%)" for _, r in rare.iterrows())

    categorical = summaries["train"].loc[summaries["train"]["Likely Role"].eq("Categorical"), "Column"].tolist()
    constant = {name: table.loc[table["Unique Values"].eq(1), "Column"].tolist() for name, table in summaries.items()}
    negative = {name: table.loc[table["Negative Count"].gt(0), "Column"].tolist() for name, table in statistics.items()}
    largest = statistics["train"].sort_values("max", ascending=False).iloc[0]
    flagged = statistics["train"].loc[statistics["train"]["Investigation Flags"].ne(""), "Column"].tolist()
    unseen = category_comparison.loc[category_comparison["Test Only Values"].ne("[]")]
    unseen_text = "; ".join(f"{r['Column']}: {r['Test Only Values']}" for _, r in unseen.iterrows()) or "none"
    q = quality.set_index("Dataset")
    numeric_features = int(q.loc["train", "Numerical Features"])
    train_attack = distributions["train"].query("Target == 'label' and Class == '1'")["Percentage"].iloc[0]
    test_attack = distributions["test"].query("Target == 'label' and Class == '1'")["Percentage"].iloc[0]
    ratios = imbalance.set_index(["Dataset", "Target"])
    observed_columns = set(train_df.columns)
    dns_named = [c for c in train_df if any(token in c.lower() for token in ("dns", "domain", "query", "entropy"))]
    timestamp_named = [c for c in train_df if c.lower() in {"stime", "ltime", "timestamp", "start_time", "end_time"}]
    findings_lines = [
        "# Key Dataset Findings",
        "",
        f"1. **Training size:** {len(train_df):,} rows, {train_df.shape[1]} columns (`{paths['train'].name}`).",
        f"2. **Testing size:** {len(test_df):,} rows, {test_df.shape[1]} columns (`{paths['test'].name}`).",
        f"3. **Feature count:** {len(feature_columns(train_df))} candidate predictors after separately identifying ID and the two targets. No columns were removed.",
        f"4. **Numerical features:** {numeric_features} candidate predictors. Counts of numeric columns including ID and targets differ from predictor counts.",
        f"5. **Categorical features:** {listed(categorical)}; no actual boolean dtypes in either split." if int(q["Boolean Features"].sum()) == 0 else f"5. **Categorical features:** {listed(categorical)}; boolean counts are recorded in dataset_summary.csv.",
        f"6. **Targets:** label (binary) and attack_cat ({train_df['attack_cat'].nunique()} observed categories, including Normal). Normal/label=0 contradictions: {target_consistency}.",
        f"7. **Data quality:** missing cells train/test {q.loc['train', 'Missing Values']}/{q.loc['test', 'Missing Values']}; infinite cells {q.loc['train', 'Infinite Values']}/{q.loc['test', 'Infinite Values']}. Constant columns: {constant}. Dash placeholders are counted separately from NaN.",
        f"8. **Duplicates:** full-row train/test {q.loc['train', 'Duplicate Rows']:,}/{q.loc['test', 'Duplicate Rows']:,}; excluding ID and targets {q.loc['train', 'Duplicate Features (excluding id and targets)']:,}/{q.loc['test', 'Duplicate Features (excluding id and targets)']:,} excess rows. Shared unique feature vectors: {overlap.get('shared_unique_feature_records', 'not checked'):,}; participating train/test rows: {overlap.get('train_rows_with_test_match', 0):,}/{overlap.get('test_rows_with_train_match', 0):,}. Shared vectors with conflicting target combinations: {overlap.get('shared_feature_records_with_conflicting_targets', 0):,}. Matching identical features do not establish identical source events.",
        f"9. **Imbalance:** attack share is {train_attack:.2f}% in train vs {test_attack:.2f}% in test. Multiclass majority/minority ratio is {ratios.loc[('train', 'attack_cat'), 'Imbalance Ratio']:.2f}:1 in train and {ratios.loc[('test', 'attack_cat'), 'Imbalance Ratio']:.2f}:1 in test.",
        f"10. **Rare attack categories (<1%):** train: {rare_classes('train')}. Test: {rare_classes('test')}. These limited counts make later class-wise estimates sensitive to a few errors.",
        f"11. **Numerical flags:** negative-value columns {negative}; largest observed train maximum is {largest['Column']}={largest['max']:.6g}. Investigation flags affect {listed(flagged)}. These flags do not establish invalid values.",
        f"12. **Train/test differences:** missing columns {schema_report['missing_columns_in_test']}; extra columns {schema_report['extra_columns_in_test']}; same order {schema_report['same_column_order']}; dtype mismatches {schema_report['dtype_mismatches']}. Test-only category values: {unseen_text}. Largest numeric mean differences: {listed(numeric_shift.head(5)['Column'])}.",
        "13. **Leakage risks:** both targets must be excluded from predictors. attack_cat carries binary-label information. Unique sequential ID may reflect target ordering. Repeated vectors across splits can inflate apparent generalization, and conflicting targets warrant provenance and ambiguity investigation. Low-cardinality purity is a screening diagnostic, not proof of leakage.",
        "14. **Future preprocessing requirements:** establish target policy and a validation protocol; fit categorical encoders and numeric transforms only on the training partition; support unknown categorical values; investigate placeholders, skew and repeated/conflicting records before deciding how to handle them. No preprocessing choice has been fitted here.",
        f"15. **Open questions:** independently verify source provenance and feature meanings. DNS/domain/query/entropy-named columns: {listed(dns_named)}; explicit start/end timestamp-named columns: {listed(timestamp_named)}. Their absence by name does not prove all DNS-related information is absent, but DNS modality and temporal ordering remain unestablished. Attack categories contain a literal Botnet class: {'Botnet' in set(train_df['attack_cat'])}. The broad IDS dataset does not by itself establish botnet-specific ground truth.",
        "",
        "## Conclusion",
        "The source files pass the recorded integrity, missingness, and schema checks. Repeated feature vectors, target conflicts, ID ordering and split differences require investigation before preprocessing or modelling. No records were changed or removed.",
        "",
        "**Recommended Notebook 02:** Duplicate, Target Ambiguity, and Leakage Investigation. Review within-split target conflicts, cross-split overlap, ID ordering, and feature provenance alongside the reference paper. Document an evaluation protocol while retaining the official test file intact. Do not create Notebook 02 until review.",
    ]
    findings_text = "\\n".join(findings_lines)
    (PROJECT_ROOT / "docs/01_dataset_findings.md").write_text(findings_text + "\\n", encoding="utf-8")
    display(Markdown(findings_text))
    """)
    md("""
    # Key Dataset Findings
    Pending the first verified execution. The findings-generating cell above is
    authoritative on a rerun. This snapshot will be filled from observed results.
    """)
    cells[-1].metadata["findings_snapshot"] = True
    md("""
    ## Conclusion
    Review the generated findings and saved tables before deciding the next
    research stage. No Notebook 02, model, preprocessing, or feature engineering
    is created by this notebook. Outputs contain no transformed dataset copies.
    """)
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata = {"kernelspec": {"display_name": "Python (project .venv)", "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "version": "3.11"}}
    nbformat.validate(notebook)
    nbformat.write(notebook, destination)
    print(f"Created {destination.relative_to(root)} with {len(cells)} cells")


if __name__ == "__main__":
    build_notebook()
