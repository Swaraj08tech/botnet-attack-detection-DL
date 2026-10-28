"""Build the staged investigation notebook without altering Notebook 01."""

from pathlib import Path
import textwrap
import nbformat


def main():
    root = Path(__file__).resolve().parents[1]
    path = root / "notebooks/02_duplicate_target_ambiguity_and_leakage_investigation.ipynb"
    if path.exists():
        raise FileExistsError("Notebook 02 already exists; edit it explicitly.")
    cells = []

    def md(source):
        cells.append(nbformat.v4.new_markdown_cell(textwrap.dedent(source).strip()))

    def code(source):
        cells.append(nbformat.v4.new_code_cell(textwrap.dedent(source).strip()))

    md("""
    # 02 — Duplicate, Target Ambiguity, and Leakage Investigation

    Establish an auditable validation and evaluation protocol before preprocessing.
    Observations come from the source files and Notebook 01's saved outputs.
    Hypotheses remain distinct from decisions. No model is fitted and no source
    record is deleted, relabeled, scaled, encoded, or resampled.

    Run all cells in order using the project's `.venv` kernel in VS Code.
    Dependencies are already declared in `requirements.txt`; no installation or
    network request is made by this notebook. Approximate split fractions do not
    override group integrity. The entire supplied testing file remains reserved.
    """)
    md("""
    ## 1. Setup and integrity

    Reuse the existing project-root discovery, loading, hashing and audit helpers.
    Notebook 01 and its saved artifacts are read-only evidence. The exact-group
    implementation in Notebook 02 independently reconstructs its critical counts.
    Paths work from the repository root and `notebooks/`.
    """)
    code("""
    from pathlib import Path
    import sys, json, platform, importlib.metadata
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, ListedColormap, BoundaryNorm
    from IPython.display import display, Markdown
    import nbformat

    print("Python:", platform.python_version(), "Interpreter:", sys.executable)
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("Select the project .venv interpreter as the notebook kernel.")
    current = Path.cwd().resolve()
    root = next((p for p in (current, *current.parents)
                 if (p / "README.md").is_file() and all((p / d).is_dir() for d in ("data/raw", "src", "notebooks"))), None)
    if root is None:
        raise FileNotFoundError("Run from inside the project.")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.utils.paths import find_project_root
    from src.data.load_data import discover_datasets, load_datasets, sha256
    from src.data.validate_data import feature_columns, quality_summary, numeric_statistics, duplicate_overlap
    from src.data.duplicate_investigation import (
        build_groups, repetition_summary, ambiguity_tables, cross_split_overlap,
        overlap_by_class, policy_class_counts,
    )
    from src.data.split_data import (
        grouped_validation_assignments, assert_exact_partition_disjointness, partition_class_balance,
    )
    ROOT = find_project_root()
    TABLES, FIGURES, SPLITS = [ROOT / p for p in ("results/tables", "results/figures", "results/splits")]
    for directory in (TABLES, FIGURES, SPLITS):
        directory.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "configs/02_validation_protocol.json").read_text())
    pd.set_option("display.max_columns", 24)
    pd.set_option("display.max_rows", 45)
    plt.rcParams.update({"figure.dpi": 110, "font.size": 10})
    versions = {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "matplotlib", "scikit-learn", "ipykernel", "nbformat", "nbclient")}
    display(pd.Series(versions, name="version").to_frame())
    pd.Series(versions, name="version").to_csv(TABLES / "02_environment_versions.csv", index_label="package")
    sections = []

    def save_table(table, name):
        table.to_csv(TABLES / f"02_{name}.csv", index=False)
        return table

    def show_figure(fig, name):
        fig.tight_layout()
        fig.savefig(FIGURES / f"02_{name}.png", dpi=170, bbox_inches="tight")
        plt.show()
        plt.close(fig)

    def interpret(title, observation, hypothesis, decision):
        text = f"### {title} — interpretation\\n\\n**Observation:** {observation}\\n\\n**Hypothesis / limitation:** {hypothesis}\\n\\n**Decision:** {decision}"
        sections.append(text)
        display(Markdown(text))
    """)
    code("""
    paths = discover_datasets(ROOT / "data/raw")
    notebook01_path = ROOT / "notebooks/01_data_understanding.ipynb"
    protected = [notebook01_path, *paths.values(), ROOT / "docs/01_dataset_findings.md",
                 *sorted(TABLES.glob("01_*")), *sorted(FIGURES.glob("01_*")),
                 *[TABLES / n for n in ("dataset_summary.csv", "feature_summary.csv", "class_distribution_train.csv", "class_distribution_test.csv")]]
    protected_before = {str(p.relative_to(ROOT)): sha256(p) for p in protected}
    assert sha256(notebook01_path) == "97bcae6a8e4a2700704d4a3ea1424289f28dc5afd0feebc9cc875b8721c125dc", "Notebook 01 differs from the reviewed source; review before continuing"
    placement = json.loads((ROOT / "docs/initialization_manifest.json").read_text(encoding="utf-8-sig"))
    for entry in placement["datasets"]["files"]:
        assert sha256(ROOT / entry["path"]) == entry["sha256"].lower()
    notebook01 = nbformat.read(notebook01_path, as_version=4)
    assert not any(o.output_type == "error" for c in notebook01.cells if c.cell_type == "code" for o in c.outputs)
    print("Reviewed Notebook 01:", len(notebook01.cells), "cells; saved code executions:",
          sum(c.cell_type == "code" and c.execution_count is not None for c in notebook01.cells))
    print((ROOT / "docs/01_dataset_findings.md").read_text(encoding="utf-8")[:850])
    train_df, test_df = load_datasets(paths)
    datasets = {"train": train_df, "test": test_df}
    groups = {name: build_groups(df) for name, df in datasets.items()}
    prior_quality = pd.read_csv(TABLES / "dataset_summary.csv").set_index("Dataset")
    prior_conflicts = pd.read_csv(TABLES / "01_within_split_target_conflicts.csv").set_index("Dataset")
    checks = []
    for name, df in datasets.items():
        now = quality_summary(df, name)
        for metric in ("Rows", "Columns", "Missing Values", "Infinite Values", "Duplicate Rows",
                       "Duplicate Features (excluding id and targets)", "Duplicate Labeled Records (excluding id)"):
            expected = int(prior_quality.loc[name, metric])
            checks.append({"dataset": name, "metric": metric, "prior": expected, "reproduced": int(now[metric]), "match": expected == now[metric]})
        for prefix, prior_column in (("binary", "Feature Groups With Binary Conflict"), ("category", "Feature Groups With Attack-category Conflict")):
            actual = int(groups[name].summary[f"{prefix}_conflict"].sum())
            expected = int(prior_conflicts.loc[name, prior_column])
            checks.append({"dataset": name, "metric": prior_column, "prior": expected, "reproduced": actual, "match": actual == expected})
    reproduction = save_table(pd.DataFrame(checks), "audit_reproduction")
    assert reproduction["match"].all()
    display(reproduction)
    interpret("1. Integrity", f"All {len(checks)} initial audit checks reproduce Notebook 01; source and prior notebook hashes are recorded.",
              "Matching the local placement manifest proves preservation, not independent official-file authentication.",
              "Use the supplied files separately. Recheck all protected hashes at completion; cross-split counts are independently confirmed in section 5.")
    """)
    md("""
    ## 2. Target definition

    Primary task: classify `attack_cat`, including Normal. Secondary task: classify
    binary `label` (0 Normal, 1 Attack). Neither task is a verified botnet classifier.
    The predictor contract excludes `id`, `label` and `attack_cat`, leaving all 42
    observed predictors. This is target/identifier separation, not feature selection.

    Intended label meanings are supported by Table VII of the dataset creators'
    paper. Local contradictions and class spellings must still be checked. See
    [the source and semantics register](../docs/02_feature_semantics_sources.md).
    """)
    code("""
    predictors = feature_columns(train_df)
    assert len(predictors) == 42 and predictors == feature_columns(test_df)
    assert not set(predictors) & {"id", "label", "attack_cat"}
    target_checks = []
    for name, df in datasets.items():
        target_checks.append({"dataset": name, "binary_classes": str(sorted(df.label.unique().tolist())),
                              "attack_categories": str(sorted(df.attack_cat.unique().tolist())),
                              "normal_binary_inconsistencies": int((df.attack_cat.eq("Normal") != df.label.eq(0)).sum()),
                              "literal_botnet_rows": int(df.attack_cat.str.casefold().eq("botnet").sum())})
    display(save_table(pd.DataFrame(target_checks), "target_definition_checks"))
    (TABLES / "02_predictor_contract.json").write_text(json.dumps({"predictors": predictors,
        "excluded": config["excluded_predictor_columns"], "targets": ["label", "attack_cat"]}, indent=2), encoding="utf-8")
    interpret("2. Targets", "Both files have binary labels and ten attack categories including Normal; no literal Botnet label and no Normal/binary inconsistency are observed.",
              "An attack category cannot be renamed Botnet without event-level provenance. Intended feature definitions do not establish extractor correctness or causal availability.",
              "Use an offline IDS task description. Preserve both labels as outcomes only; exclude ID from prediction and grouping.")
    """)
    md("""
    ## 3. Within-split duplicates

    Exact groups use all 42 parsed predictor values in the same order, with no
    rounding or target information. Python dictionary lookups verify full tuple
    equality after hashing. Group IDs are first-occurrence indices local to a file;
    the source hash and row order bind their meaning. They are audit metadata.

    **Participating rows** includes every member of a group with size >1.
    **Excess rows** is the sum of group size minus one. Class group counts may
    overlap because a single vector can have several class labels. Distinct
    vectors are not necessarily statistically independent network events.
    """)
    code("""
    repetition, class_repetition = [], []
    for name, df in datasets.items():
        overall, classes = repetition_summary(df, groups[name], name)
        repetition.append(overall)
        class_repetition.append(classes)
        group_export = groups[name].summary.join(groups[name].category_counts.add_prefix("attack_rows__"))
        group_export = group_export.join(groups[name].binary_counts.add_prefix("binary_rows__"))
        save_table(group_export, f"{name}_feature_groups")
    repetition = save_table(pd.DataFrame(repetition), "within_split_repetition")
    class_repetition = save_table(pd.concat(class_repetition, ignore_index=True), "class_repetition")
    display(repetition)
    display(class_repetition.loc[class_repetition.target.eq("attack_cat")])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    bins = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500]
    labels = ["1", "2", "3–4", "5–9", "10–19", "20–49", "50–99", "100–199", "200–499"]
    for ax, (name, grouped) in zip(axes, groups.items()):
        counts, _ = np.histogram(grouped.summary.rows, bins=bins)
        bars = ax.bar(labels, counts, color="#527b94")
        ax.set(yscale="log", title=name.title(), xlabel="Rows per exact vector", ylabel="Feature groups (log scale)")
        ax.tick_params(axis="x", rotation=45)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Repeated predictor vectors: group-size distribution")
    show_figure(fig, "group_size_distribution")
    """)
    code("""
    category_order = sorted(train_df.attack_cat.unique())
    fig, ax = plt.subplots(figsize=(10, 5))
    y = np.arange(len(category_order))
    for offset, (name, color) in zip((-0.18, 0.18), (("train", "#527b94"), ("test", "#b66d4b"))):
        values = class_repetition.loc[class_repetition.dataset.eq(name) & class_repetition.target.eq("attack_cat")].set_index("class").reindex(category_order)
        ax.barh(y + offset, values.repeated_group_row_pct, height=0.34, label=name.title(), color=color)
    ax.set_yticks(y, category_order)
    ax.set(xlim=(0, 100), xlabel="Rows belonging to repeated feature groups (%)", title="Class-wise repetition")
    ax.invert_yaxis(); ax.legend()
    show_figure(fig, "class_wise_repetition")
    r = repetition.set_index("dataset")
    interpret("3. Repetition", f"Train: {r.loc['train','unique_vectors']:,} unique vectors; {r.loc['train','participating_rows']:,} participating rows and {r.loc['train','excess_duplicate_rows']:,} excess rows. Test: {r.loc['test','unique_vectors']:,} unique vectors; {r.loc['test','participating_rows']:,} participating rows and {r.loc['test','excess_duplicate_rows']:,} excess rows.",
              "Repeated aggregate measurements can arise from separate events, feature quantization or extraction behavior; identity of source events is unproven.",
              "Retain rows. Use complete feature groups as the unit of partitioning and report both row support and distinct-vector support.")
    """)
    md("""
    ## 4. Target ambiguity

    Binary and multiclass conflicts are investigated separately. A conflict means
    one observed predictor vector has more than one value of that target. It does
    not establish which label, if any, is wrong.

    The heatmap counts groups containing each unordered pair of classes, shown
    symmetrically. A group containing three classes contributes three pairs;
    heatmap totals therefore must not be read as the number of conflicting groups.
    Representative examples select the largest three groups per split and target,
    with one original row per conflicting value. Full predictor values are saved.
    """)
    code("""
    totals, combos, pairs, examples = [], [], [], []
    for name, df in datasets.items():
        parts = ambiguity_tables(df, groups[name], name)
        for destination, part in zip((totals, combos, pairs, examples), parts):
            destination.append(part)
    ambiguity = save_table(pd.concat(totals, ignore_index=True), "target_ambiguity")
    combinations_table = save_table(pd.concat(combos, ignore_index=True), "conflicting_class_combinations")
    pair_table = save_table(pd.concat(pairs, ignore_index=True), "conflicting_class_pairs")
    example_table = save_table(pd.concat(examples, ignore_index=True), "conflict_examples")
    display(ambiguity)
    display(combinations_table.sort_values("rows", ascending=False).head(20))
    display(example_table[["dataset", "conflict_target", "group_id", "row_position", "id", "label", "attack_cat", "proto", "service", "state"]])
    constraints = []
    for name, group in groups.items():
        for target, counts in (("label", group.binary_counts), ("attack_cat", group.category_counts)):
            minimum_errors = int((counts.sum(axis=1) - counts.max(axis=1)).sum())
            constraints.append({"dataset": name, "target": target, "empirical_minimum_errors_for_identical_vector_prediction": minimum_errors,
                                "empirical_minimum_error_pct": minimum_errors / len(datasets[name]) * 100})
    display(save_table(pd.DataFrame(constraints), "ambiguity_constraints"))
    """)
    code("""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    maximum = max(1, int(pair_table.loc[pair_table.target.eq("attack_cat"), "shared_conflicting_groups"].max()))
    for ax, name in zip(axes, ("train", "test")):
        matrix = pd.DataFrame(0, index=category_order, columns=category_order)
        selected = pair_table.loc[pair_table.dataset.eq(name) & pair_table.target.eq("attack_cat")]
        for row in selected.itertuples(index=False):
            matrix.loc[row.class_a, row.class_b] = row.shared_conflicting_groups
            matrix.loc[row.class_b, row.class_a] = row.shared_conflicting_groups
        values = matrix.to_numpy()
        cmap = plt.colormaps["YlOrRd"].copy(); cmap.set_bad("#f3f3f3")
        im = ax.imshow(np.ma.masked_equal(values, 0), cmap=cmap, norm=LogNorm(vmin=1, vmax=maximum))
        ax.set_xticks(range(len(category_order)), category_order, rotation=60, ha="right")
        ax.set_yticks(range(len(category_order)), category_order)
        ax.set_title(name.title())
        for i, j in zip(*np.nonzero(values)):
            ax.text(j, i, str(values[i,j]), ha="center", va="center", fontsize=8,
                    color="white" if values[i,j] > maximum / 4 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Conflicting groups (log scale)")
    fig.suptitle("Attack categories sharing identical predictors; blank cells = zero")
    show_figure(fig, "target_conflict_heatmap")
    ac = ambiguity.set_index(["dataset", "target"])
    interpret("4. Ambiguity", f"Multiclass conflicts affect {ac.loc[('train','attack_cat'),'affected_rows']:,} train rows in {ac.loc[('train','attack_cat'),'affected_groups']:,} groups and {ac.loc[('test','attack_cat'),'affected_rows']:,} test rows in {ac.loc[('test','attack_cat'),'affected_groups']:,} groups. Binary conflicts affect {ac.loc[('train','label'),'affected_rows']:,}/{ac.loc[('test','label'),'affected_rows']:,} train/test rows.",
              "Omitted event context, label alignment, discretization or extractor issues could explain ambiguity. The empirical minimum-error calculation only constrains deterministic predictions for identical vectors in this sample; it is not a model result or a population error bound.",
              "Keep conflicting versions in the same partition and preserve their labels. Do not replace targets with majority labels.")
    """)
    md("""
    ## 5. Cross-split overlap

    Match exact tuples from independently constructed train and test groups.
    **Consistent** means the union of observed target values across both splits
    contains one value. **Conflicting** means that union contains more than one,
    including groups ambiguous within either split. Binary and multiclass status
    can differ. Comparisons are diagnostics; the supplied files remain separate.

    Masks refer initially to the entire official training file. Section 8 also
    records overlap with the actual fit/model-selection/calibration roles so that
    later reporting does not mislabel all development overlap as fitted-model exposure.
    """)
    code("""
    overlap_groups, test_group_to_train = cross_split_overlap(groups["train"], groups["test"])
    save_table(overlap_groups, "cross_split_group_matches")
    legacy_overlap = duplicate_overlap(train_df, test_df)
    previous_overlap = pd.read_csv(TABLES / "01_cross_split_overlap.csv").iloc[0]
    for metric in ("shared_unique_feature_records", "train_rows_with_test_match", "test_rows_with_train_match",
                   "matching_row_pairs", "shared_feature_records_with_conflicting_targets", "matched_pairs_with_different_targets"):
        assert legacy_overlap[metric] == int(previous_overlap[metric])
    assert len(overlap_groups) == legacy_overlap["shared_unique_feature_records"]
    assert int(overlap_groups.train_rows.sum()) == legacy_overlap["train_rows_with_test_match"]
    assert int(overlap_groups.test_rows.sum()) == legacy_overlap["test_rows_with_train_match"]
    assert int(overlap_groups.category_status.eq("conflicting").sum()) == legacy_overlap["shared_feature_records_with_conflicting_targets"]
    overlap_summary = []
    for prefix in ("binary", "category"):
        for status, table in overlap_groups.groupby(f"{prefix}_status"):
            overlap_summary.append({"target": prefix, "status": status, "vectors": len(table),
                                    "train_rows": int(table.train_rows.sum()), "test_rows": int(table.test_rows.sum())})
    overlap_summary = save_table(pd.DataFrame(overlap_summary), "overlap_summary")
    overlap_classes = save_table(pd.concat([overlap_by_class(df, groups[name], overlap_groups, name)
                                            for name, df in datasets.items()], ignore_index=True), "overlap_by_class")
    display(overlap_summary)
    display(overlap_classes.loc[overlap_classes.target.eq("attack_cat")])
    matched_train_ids = test_group_to_train[groups["test"].row_groups]
    test_masks = pd.DataFrame({"row_position": np.arange(len(test_df)), "csv_line": np.arange(len(test_df)) + 2,
                               "source_id": test_df.id, "test_feature_group_id": groups["test"].row_groups,
                               "matched_official_train_group_id": matched_train_ids,
                               "overlaps_official_train": matched_train_ids >= 0,
                               "nonoverlap_official_train": matched_train_ids < 0,
                               "label": test_df.label, "attack_cat": test_df.attack_cat})
    for prefix in ("binary", "category"):
        lookup = overlap_groups.set_index("test_group_id")[f"{prefix}_status"]
        test_masks[f"{prefix}_overlap_status"] = test_masks.test_feature_group_id.map(lookup).fillna("nonoverlap")
    assert (test_masks.overlaps_official_train ^ test_masks.nonoverlap_official_train).all()
    assert len(test_masks) == len(test_df)
    test_masks.to_csv(SPLITS / "02_test_diagnostic_masks.csv", index=False)
    interpret("5. Cross-split overlap", f"All six prior overlap counts reproduce. {len(overlap_groups):,} exact vectors involve {int(overlap_groups.train_rows.sum()):,} train rows and {int(overlap_groups.test_rows.sum()):,} test rows; {int(test_masks.nonoverlap_official_train.sum()):,} test rows do not match the supplied training file.",
              "Conflicting vectors cannot be explained by a single memorized target. Even nonoverlap by exact vector does not establish host, session or temporal independence.",
              "Preserve all test rows. Report full-test results as primary, and overlapping/non-overlapping populations with their own class support as diagnostics.")
    """)
    md("""
    ## 6. Leakage and feature semantics

    **Verified source definitions:** the dataset creators describe a binary FTP
    login indicator, an FTP-command flow count, a fallback service marker, and
    contextual connection counters. **Observed local values** are checked below.
    **Unresolved:** exact extractor version, causal window boundaries, current-flow
    inclusion, export behavior and source-event provenance.

    Source references and access limitations are documented in
    [02_feature_semantics_sources.md](../docs/02_feature_semantics_sources.md).
    The official source confirms dataset structure; a readable copy of the creators'
    original paper supplies feature descriptions. No unsupported claim of complete
    feature lineage is made.

    The ID plots preserve every row and show ordering, not time. Category codes
    are display-only color indices, never encoded predictor data.
    """)
    code("""
    colors = list(plt.colormaps["tab10"].colors)
    fig, axes = plt.subplots(2, 2, figsize=(13, 5), gridspec_kw={"height_ratios": [1, 1]})
    ordering_rows = []
    for column, (name, df) in enumerate(datasets.items()):
        ordered = df.sort_values("id", kind="stable")
        for row, target in enumerate(("label", "attack_cat")):
            values = ordered[target].to_numpy()
            categories = [0, 1] if target == "label" else category_order
            mapping = {value: i for i, value in enumerate(categories)}
            codes = np.array([mapping[v] for v in values])
            cmap = ListedColormap(["#527b94", "#b66d4b"] if target == "label" else colors)
            axes[row,column].imshow(codes[None,:], aspect="auto", interpolation="nearest", cmap=cmap,
                                    norm=BoundaryNorm(np.arange(len(categories)+1)-0.5, len(categories)),
                                    extent=[ordered.id.min(), ordered.id.max(), 0, 1])
            axes[row,column].set(title=f"{name.title()}: {target}", xlabel="Source ID (not time)", yticks=[])
            ordering_rows.append({"dataset": name, "target": target,
                                  "adjacent_transitions_in_id_order": int((values[1:] != values[:-1]).sum())})
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=colors[i], label=c) for i,c in enumerate(category_order)],
               loc="lower center", ncol=5, bbox_to_anchor=(0.5,-0.1), fontsize=9)
    fig.suptitle("Label ordering by ID; top row: blue Normal / brown Attack")
    show_figure(fig, "target_ordering_by_id")
    ordering = save_table(pd.DataFrame(ordering_rows), "target_ordering")
    display(ordering)
    """)
    code("""
    service_rows, ftp_rows, ftp_value_rows, mismatch_rows = [], [], [], []
    for name, df in datasets.items():
        dash = df.service.eq("-")
        for target in ("label", "attack_cat"):
            table = pd.crosstab(df[target], dash).reindex(columns=[False, True], fill_value=0)
            for value, row in table.iterrows():
                service_rows.append({"dataset": name, "target": target, "class": str(value),
                                     "non_dash_rows": int(row[False]), "dash_rows": int(row[True]),
                                     "dash_pct_within_class": row[True] / row.sum() * 100})
        nonbinary = ~df.is_ftp_login.isin([0, 1])
        unequal = df.is_ftp_login.ne(df.ct_ftp_cmd)
        ftp_rows.append({"dataset": name, "equal_columns": bool(df.is_ftp_login.equals(df.ct_ftp_cmd)),
                         "mismatched_rows": int(unequal.sum()), "nonbinary_login_rows": int(nonbinary.sum()),
                         "nonbinary_login_values": str(sorted(df.loc[nonbinary, "is_ftp_login"].unique().tolist()))})
        for feature in ("is_ftp_login", "ct_ftp_cmd"):
            for value, count in df[feature].value_counts().sort_index().items():
                ftp_value_rows.append({"dataset": name, "feature": feature, "value": value, "rows": int(count)})
        for position in np.flatnonzero(unequal | nonbinary):
            row = df.iloc[position][["id", "proto", "service", "is_ftp_login", "ct_ftp_cmd", "label", "attack_cat"]].to_dict()
            row.update(dataset=name, row_position=int(position))
            mismatch_rows.append(row)
    service_audit = save_table(pd.DataFrame(service_rows), "service_marker_by_class")
    ftp_audit = save_table(pd.DataFrame(ftp_rows), "ftp_field_audit")
    display(ftp_audit)
    display(save_table(pd.DataFrame(ftp_value_rows), "ftp_value_counts"))
    display(save_table(pd.DataFrame(mismatch_rows), "ftp_exception_examples"))
    display(service_audit.loc[service_audit.target.eq("attack_cat")])
    """)
    code("""
    numeric_tables, extreme_examples = [], []
    for name, df in datasets.items():
        table = numeric_statistics(df[predictors])
        table.insert(0, "dataset", name)
        numeric_tables.append(table)
        for feature in ("sload", "rate", "dur", "sbytes", "dbytes", "sjit", "djit"):
            for position in df[feature].nlargest(3, keep="first").index:
                row = df.loc[position, ["id", "dur", "proto", "service", "spkts", "dpkts", "sbytes", "dbytes", "sload", "rate", "label", "attack_cat"]].to_dict()
                row.update(dataset=name, ranked_feature=feature, ranked_value=float(df.loc[position,feature]), row_position=int(position))
                extreme_examples.append(row)
    numeric_audit = save_table(pd.concat(numeric_tables, ignore_index=True), "numeric_semantics_audit")
    save_table(pd.DataFrame(extreme_examples), "extreme_value_examples")
    display(numeric_audit.loc[numeric_audit["Investigation Flags"].ne(""),
        ["dataset", "Column", "min", "max", "Zero %", "Skew", "Investigation Flags"]])
    zero_context = []
    for name, df in datasets.items():
        for proto, table in df.groupby("proto"):
            if len(table) >= 100:
                zero_context.append({"dataset": name, "proto": proto, "rows": len(table),
                                     **{f"{c}_zero_pct": table[c].eq(0).mean() * 100 for c in ("dur", "dbytes", "dpkts", "tcprtt", "response_body_len")}})
    display(save_table(pd.DataFrame(zero_context), "zero_context_by_protocol"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    zero_train = numeric_audit.loc[numeric_audit.dataset.eq("train")].nlargest(10, "Zero %")
    axes[0].barh(zero_train.Column[::-1], zero_train["Zero %"][::-1], color="#527b94")
    axes[0].set(xlim=(0, 100), xlabel="Zero-valued rows (%)", title="Training features with most zeros")
    for name, df in datasets.items():
        values = np.sort(df.sload.to_numpy())
        axes[1].plot(values, np.arange(1, len(values)+1) / len(values), label=name.title())
    axes[1].set_xscale("symlog", linthresh=1)
    axes[1].set(xlabel="sload (bits/s; symlog display only)", ylabel="Empirical cumulative proportion", title="sload range and skew")
    axes[1].legend()
    show_figure(fig, "zeros_and_extreme_values")
    """)
    code("""
    connection_features = {"ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm"}
    feature_availability = []
    for feature in predictors:
        if feature in connection_features:
            evidence, need = "Creator-described cross-flow connection counter", "Verify past-only window, endpoint state, timing and resets"
        elif feature in {"ct_state_ttl", "ct_flw_http_mthd", "ct_ftp_cmd"}:
            evidence, need = "Creator-described additional aggregate", "Verify aggregation scope, lineage and prediction-time availability"
        elif feature == "is_ftp_login":
            evidence, need = "Intended binary indicator; local values violate binary range", "Resolve extractor/version discrepancy before semantic recoding"
        elif feature == "service":
            evidence, need = "Documented service category with fallback marker", "Retain '-' pending extractor-specific interpretation"
        else:
            evidence, need = "CSV observation; full field-by-field semantics not verified here", "Verify units and completed-flow vs early-prediction availability"
        feature_availability.append({"feature": feature, "evidence_status": evidence,
                                     "deployment_availability": "unverified", "requirement": need})
    availability = save_table(pd.DataFrame(feature_availability), "feature_availability_register")
    display(availability.loc[availability.feature.str.startswith("ct_") | availability.feature.isin(["service", "is_ftp_login"])])
    interpret("6. Semantics and leakage", f"Training FTP columns are exactly equal; testing has {int(ftp_audit.set_index('dataset').loc['test','mismatched_rows'])} unequal rows. Nonbinary login counts train/test are {int(ftp_audit.set_index('dataset').loc['train','nonbinary_login_rows'])}/{int(ftp_audit.set_index('dataset').loc['test','nonbinary_login_rows'])}. service='-' counts are {int(train_df.service.eq('-').sum()):,}/{int(test_df.service.eq('-').sum()):,}. ID plots reveal ordered class blocks.",
              "Notebook 01's saved output correctly identified training FTP equality; its research diary incorrectly claimed no exactly equal columns. This is a documentary correction, not a dataset change. Shared history in contextual counters, extreme rates on short flows, and applicability-related zeros remain hypotheses requiring extractor evidence.",
              "Keep all 42 predictors for the grouping contract. Preserve marker and FTP values; do not equate association with leakage or ID with time. Limit the protocol to an offline benchmark until feature causality is established.")
    """)
    md("""
    ## 7. Duplicate and conflict policy

    **Adopted policy:** preserve every original row and label. Partition complete
    exact-feature groups together. This keeps observed class support and avoids
    choosing an unverifiable truth for ambiguous vectors.

    Quantify rejected alternatives as count-only diagnostics: excluding whole
    binary-conflict groups, excluding multiclass-conflict groups, or keeping one
    copy of each exact labeled record. No alternative dataset is written or used.
    Full-test reporting retains original multiplicities. Group-weighted metrics
    may later be a prespecified secondary sensitivity analysis; they must not
    silently replace the primary row-weighted evaluation.
    """)
    code("""
    policy_effects = save_table(pd.concat([policy_class_counts(df, groups[name], name)
                                          for name, df in datasets.items()], ignore_index=True), "policy_class_count_effects")
    display(policy_effects.loc[policy_effects.target.eq("attack_cat")].pivot_table(
        index=["dataset", "class"], columns="policy", values="retained_rows", aggfunc="first"))
    adopted = policy_effects.loc[policy_effects.policy.eq("adopted_keep_all_labels_and_rows")]
    assert adopted.excluded_rows.eq(0).all()
    interpret("7. Policy", "The adopted policy removes zero rows from every class. Alternative exclusion policies disproportionately remove observations from classes that share feature vectors; exact impacts are saved by class.",
              "Retaining repeated rows preserves benchmark frequencies but allows common vectors to carry more evaluation weight. Removing ambiguous groups changes the task and could hide difficult cases.",
              "Retain all labels and multiplicities; use grouped validation and report ambiguity/overlap support. Any later sensitivity experiment must be separately named, training-defined and predeclared.")
    """)
    md("""
    ## 8. Training-only validation

    Use prespecified fold 0 of five-fold `StratifiedGroupKFold`, shuffle enabled,
    seed 42, stratifying by the original multiclass target. Group membership uses
    predictors only; class labels are used solely to balance partitions. No test
    frame, label, overlap flag, model result, seed search or fold search enters
    the splitting function. Targets are never majority-collapsed within a group.

    Within the approximately 20% validation allocation, two grouped folds with
    seed 43 reserve fold 0 for calibration and the other fold for model selection
    and thresholds. This avoids fitting a calibrator on the same observations
    used for early stopping/model choice. These roles are saved now, although
    calibration itself is a later stage. Model-selection scores are development
    results, not an unbiased final performance estimate.

    Class balance is approximate. A 2-percentage-point class-share difference is
    a reporting flag, never a reason to break groups or search seeds. Every rare
    class needs row counts **and** distinct-vector counts. Distinct groups here
    mean unique predictors, not verified independent capture events.
    """)
    code("""
    assert config["validation_n_splits"] == 5 and config["validation_fold"] == 0
    assert config["calibration_n_splits"] == 2 and config["calibration_fold"] == 0
    assignments = grouped_validation_assignments(train_df, groups["train"],
        seed=config["validation_seed"], calibration_seed=config["calibration_seed"])
    split_checks = assert_exact_partition_disjointness(train_df, assignments, predictors)
    repeated_assignments = grouped_validation_assignments(train_df, groups["train"],
        seed=config["validation_seed"], calibration_seed=config["calibration_seed"])
    pd.testing.assert_frame_equal(assignments, repeated_assignments)
    del repeated_assignments
    assert assignments.source_id.to_list() == train_df.id.to_list()
    assert assignments.label.to_list() == train_df.label.to_list()
    assert assignments.attack_cat.to_list() == train_df.attack_cat.to_list()
    assignments.to_csv(SPLITS / "02_train_validation_assignments.csv", index=False)
    balance = save_table(partition_class_balance(assignments), "partition_class_balance")
    partition_sizes = assignments.groupby(["partition", "role"]).agg(rows=("source_id", "size"),
                                               feature_groups=("feature_group_id", "nunique")).reset_index()
    save_table(partition_sizes, "partition_sizes")
    display(partition_sizes)
    display(balance.loc[balance.target.eq("attack_cat") & balance.partition.isin(["train", "validation"])])
    rare_names = train_df.attack_cat.value_counts(normalize=True).loc[lambda s: s < .01].index
    rare_balance = save_table(balance.loc[balance.target.eq("attack_cat") & balance["class"].isin(rare_names)].copy(), "rare_class_partition_support")
    display(rare_balance)
    print("Exact feature intersections:", split_checks)
    print("Deterministic rerun: PASS")
    missing_class_roles = balance.loc[balance.partition.isin(["train", "model_selection", "calibration"]) & balance.rows.eq(0)]
    assert missing_class_roles.empty, "A class is absent from a role; review the protocol without silently changing the seed"
    """)
    code("""
    # Add actual-fit exposure without affecting assignments or looking at test labels.
    train_group_roles = assignments.groupby("feature_group_id").role.first()
    exposure_role = test_masks.matched_official_train_group_id.map(train_group_roles).fillna("none")
    for role in ("train", "model_selection", "calibration"):
        column = "overlaps_fit_train" if role == "train" else f"overlaps_{role}"
        test_masks[column] = exposure_role.eq(role)
    test_masks["overlaps_validation"] = test_masks.overlaps_model_selection | test_masks.overlaps_calibration
    assert (test_masks.overlaps_fit_train | test_masks.overlaps_validation).equals(test_masks.overlaps_official_train)
    assert not (test_masks.overlaps_fit_train & test_masks.overlaps_validation).any()
    test_masks.to_csv(SPLITS / "02_test_diagnostic_masks.csv", index=False)
    mask_support = []
    populations = {"full_test": np.ones(len(test_df), dtype=bool),
                   **{c: test_masks[c].to_numpy() for c in ("overlaps_official_train", "nonoverlap_official_train", "overlaps_fit_train", "overlaps_validation")}}
    for population, mask in populations.items():
        for target in ("label", "attack_cat"):
            for value in sorted(test_df[target].unique(), key=str):
                selected = mask & test_df[target].eq(value).to_numpy()
                mask_support.append({"population": population, "target": target, "class": str(value),
                                     "rows": int(selected.sum()), "feature_vectors": int(np.unique(groups['test'].row_groups[selected]).size)})
    save_table(pd.DataFrame(mask_support), "test_diagnostic_support")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    plot_data = balance.loc[balance.target.eq("attack_cat") & balance.partition.isin(["train", "model_selection", "calibration"])]
    x = np.arange(len(category_order))
    for offset, role in zip((-.25, 0, .25), ("train", "model_selection", "calibration")):
        values = plot_data.loc[plot_data.partition.eq(role)].set_index("class").reindex(category_order)
        ax.bar(x + offset, values.share_of_original_class_pct, width=.25, label=role.replace('_',' ').title())
    ax.set_xticks(x, category_order, rotation=40, ha="right")
    ax.set(ylabel="Share of each original training class (%)", title="Prespecified grouped allocation by class")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1)); show_figure(fig, "grouped_partition_balance")
    max_shift = balance.loc[balance.partition.eq("validation") & balance.target.eq("attack_cat"), "distribution_difference_pp"].abs().max()
    interpret("8. Grouped validation", f"Saved {len(assignments):,} assignments: {int(assignments.role.eq('train').sum()):,} fit rows, {int(assignments.role.eq('model_selection').sum()):,} model-selection rows and {int(assignments.role.eq('calibration').sum()):,} calibration rows. Repeated execution gives identical assignments; every pair of roles has zero exact predictor overlap. Maximum multiclass validation share difference is {max_shift:.6f} percentage points.",
              "Rare-class row support can overstate independent information, and contextual counters can share source history across different vectors. This is an exact-vector-disjoint offline protocol, not a verified temporal or host-disjoint protocol.",
              "Freeze these source-hash-bound assignments. Fit learned preprocessing only on the train role; use model_selection for architecture/early stopping/threshold decisions and calibration for calibration fitting. Do not refit on validation before the documented final test.")
    """)
    md("""
    ## 9. Evaluation protocol and handoff

    The primary multiclass measure is macro F1 over the fixed ten-class vocabulary.
    Always report per-class precision, recall, F1 and support, plus raw and
    row-normalized confusion matrices. Also report accuracy and weighted F1 as
    secondary summaries. For binary detection, report attack recall, precision,
    false-positive rate, false-negative rate, precision–recall curve and average
    precision. Average precision must use continuous attack scores, not thresholded
    labels. For a multiclass detector, attack score is `1 - P(Normal)`; a separate
    binary classifier must be labeled as a different experiment.

    Threshold-dependent metrics use a threshold frozen on the model-selection
    role. Choose the threshold maximizing binary F1 there, breaking ties toward
    the higher threshold; also report the fixed 0.5 threshold as a reference.
    This is a benchmark convention, not an operational security requirement.
    Any alternative cost/FPR operating point must be declared before final testing.

    After model selection, freeze classifier weights. Fit temperature scaling
    later on calibration rows only; select the operating threshold using calibrated
    scores on model-selection rows. Report calibration with Brier score, log loss,
    ECE and reliability plots using prespecified bins. Do not fit calibration or
    choose thresholds on the official test file. Do not use calibration rows for
    early stopping or architecture search.

    Evaluate the frozen pipeline once on the **complete official test set**, then
    report the prespecified overlap/non-overlap diagnostics using the same pipeline
    and thresholds. Include binary-consistent/conflicting and category-consistent/
    conflicting overlap support. Report actual-fit exposure separately from exposure
    to the full official training file. Subsets with no positives/negatives or a
    missing class require explicit unavailable metrics/support; do not silently
    drop classes to inflate macro F1. Use zero_division=0 for defined class F1
    accounting and flag zero-support classes; undefined rate denominators and
    average precision for a single-class subset remain unavailable.

    For uncertainty estimates, a later group bootstrap should resample entire
    exact-feature groups, not assume rows are independent. Even that estimate
    remains conditional on unverified host/session dependencies. A prespecified
    inverse-group-size-weighted score may supplement the primary row-weighted
    scores, but must be separately labeled. Never majority-relabel conflicts.

    **Requirements for Notebook 03:** review feature definitions and the separate
    reference paper, settle prediction-time/causality assumptions, and design the
    preprocessing contract against the frozen assignments. Resolve marker/FTP
    handling using training evidence and provenance. Preserve unknown-category
    handling and target exclusion. If a later justified feature removal or
    transformation collapses additional vectors, audit new cross-partition matches
    before modelling. Do not infer temporal windows from ID or feature ordering.
    Full temporal/DNS architecture and botnet-specific claims remain gated on data
    support. Notebook 03 is not created here.
    """)
    code("""
    protocol = dict(config)
    protocol.update({"source_sha256": {name: sha256(path) for name,path in paths.items()},
                     "notebook01_sha256": protected_before[str(notebook01_path.relative_to(ROOT))],
                     "predictors": predictors, "environment_versions": versions,
                     "class_vocabulary": category_order,
                     "primary_metrics": ["macro_f1", "per_class_precision", "per_class_recall", "per_class_f1", "support", "confusion_matrix"],
                     "binary_metrics": ["attack_recall", "false_positive_rate", "false_negative_rate", "precision", "average_precision", "precision_recall_curve"],
                     "threshold_rule": "maximize binary F1 on model_selection calibrated scores; highest threshold breaks ties; also report 0.5",
                     "calibration_fit_role": "calibration", "calibration_ece_bins": 15,
                     "observed_role_rows": assignments.role.value_counts().to_dict(),
                     "exact_overlap_assertions": split_checks,
                     "assignment_sha256": sha256(SPLITS / "02_train_validation_assignments.csv"),
                     "test_mask_sha256": sha256(SPLITS / "02_test_diagnostic_masks.csv"),
                     "group_ids_are_predictors": False,
                     "test_inspection_scope": "pre-model data audit only; no test-driven model or transform tuning"})
    (SPLITS / "02_split_manifest.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    protected_after = {str(p.relative_to(ROOT)): sha256(p) for p in protected}
    assert protected_before == protected_after, "A protected source artifact changed"
    fresh_train, fresh_test = load_datasets(paths)
    pd.testing.assert_frame_equal(train_df, fresh_train)
    pd.testing.assert_frame_equal(test_df, fresh_test)
    del fresh_train, fresh_test
    integrity = {"before": protected_before, "after": protected_after, "all_protected_files_unchanged": True,
                 "dataframes_unchanged": True, "deterministic_assignments": True,
                 "partition_exact_overlap_checks": split_checks}
    (TABLES / "02_integrity_and_split_verification.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")
    interpret("9. Evaluation and handoff", "The grouped assignments, diagnostic masks and frozen protocol are saved. Source CSVs, Notebook 01 and all protected prior artifacts have identical before/after SHA-256 hashes; loaded frames are unchanged.",
              "Official test labels have been inspected for this declared audit, so provenance must disclose the audit. Future feature/hyperparameter choices must not be tuned using these test observations. Exact grouping does not eliminate contextual dependence or establish deployment causality.",
              "Adopt the documented offline protocol; reserve the full test set for final evaluation. Stop for review. Notebook 03 must address preprocessing and feature-provenance requirements before any modelling.")

    def markdown_table(table):
        data = table.copy().astype(str)
        lines = ["| " + " | ".join(data.columns) + " |", "| " + " | ".join(["---"]*len(data.columns)) + " |"]
        lines += ["| " + " | ".join(row) + " |" for row in data.itertuples(index=False, name=None)]
        return "\\n".join(lines)

    report = "# Notebook 02 — Findings and evaluation protocol\\n\\n" + "\\n\\n".join(sections)
    report += "\\n\\n## Observed repetition\\n\\n" + markdown_table(repetition)
    report += "\\n\\n## Target ambiguity\\n\\n" + markdown_table(ambiguity.round(5))
    report += "\\n\\n## Consistent and conflicting cross-split overlap\\n\\n" + markdown_table(overlap_summary)
    report += "\\n\\n## Frozen development roles\\n\\n" + markdown_table(partition_sizes)
    report += "\\n\\n## Rare-class support\\n\\n" + markdown_table(rare_balance[["partition","class","rows","independent_feature_groups"]])
    report += "\\n\\n## Detailed evaluation protocol and Notebook 03 requirements\\n\\n"
    report += next(c.source for c in nbformat.read(ROOT / "notebooks/02_duplicate_target_ambiguity_and_leakage_investigation.ipynb", as_version=4).cells
                   if c.cell_type == "markdown" and c.source.startswith("## 9. Evaluation protocol"))
    report += "\\n\\n## Provenance and limits\\n\\nSee [feature semantics and primary sources](02_feature_semantics_sources.md), `results/splits/02_split_manifest.json`, and `results/tables/02_integrity_and_split_verification.json`. Exact groups are not verified independent events. Environment is project .venv with inherited system packages; versions are recorded. The reference architecture remains unreviewed. No preprocessing, encoding, resampling, feature selection or model fitting occurred.\\n"
    (ROOT / "docs/02_duplicate_target_ambiguity_and_leakage_protocol.md").write_text(report, encoding="utf-8")
    print("PASS: audit reproduction, label preservation, split reproducibility, zero exact-feature leakage, and source integrity.")
    display(Markdown("**Final decisions:** keep original rows/labels; adopt the frozen grouped roles; use full-test primary reporting with overlap diagnostics; keep causal and botnet-specific claims unresolved. Review before Notebook 03."))
    """)
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata = {"kernelspec": {"display_name": "Python (project .venv)", "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "version": "3.11"}}
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    print(f"Created {path.name}: {len(cells)} cells")


if __name__ == "__main__":
    main()
