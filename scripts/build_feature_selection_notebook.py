"""Readable Notebook 05: predeclared candidates, fitting evidence, verified inputs."""

from pathlib import Path
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]
cells=[]
def md(text): cells.append(nbf.v4.new_markdown_cell(text))
def code(text): cells.append(nbf.v4.new_code_cell(text))

md('''# 05 — Feature Selection

**Question: Which original and engineered features should we carry forward as clearly defined candidates for baseline models?**

Notebook 04 created total bytes, byte balance and packet balance. We now have 42 original features and 3 additions. This notebook prepares a small list of feature sets to test fairly in Notebook 06.

We will define the sets first, study fitting rows, explain the shortlist, and save checked inputs. **We will not name a winning set or compare predictive models here.** The official test data is not loaded.

Feature selection here means deciding **which combinations deserve a model comparison**. A high correlation or label association is evidence to examine, not a rule for automatically deleting a column.''')
md('''## 1. Check the inputs

We use Notebook 02's saved row sets and Notebook 04's inputs. File hashes confirm that we are using the same saved files. We also check the feature names, formulas, row positions and original labels.

| Row set | Use in this notebook |
|---|---|
| `train` | All feature evidence and internal checks |
| `model_selection` | Apply the fixed sets and verify files only |
| `calibration` | Apply the fixed sets and verify files only |

Detailed checks live in `src/`. The short outputs below show whether they passed. The official test file is only hashed to confirm preservation; its records are never loaded.''')
code('''from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

root = next(p for p in (Path.cwd().resolve(), *Path.cwd().resolve().parents)
            if (p / 'results/splits/02_split_manifest.json').is_file())
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from src.data.feature_selection import ENGINEERED
from src.data.feature_selection_audit import (
    setup, define_sets, redundancy_study, relevance_study,
    evidence_figures, relevance_figure, shortlist, save_candidates, finish, figure,
)

state, role_support = setup(root)
display(role_support.rename(columns={
    'role': 'Row set', 'rows': 'Rows', 'original_patterns': 'Different original feature patterns',
}))
print('PASS: source, assignments, Notebook 04 files, feature names, rows and labels checked.')''')
md('''**Provenance note:** Notebook 04's saved code and Markdown match its verified builder, and its processed artifacts match their hashes. Its notebook file was saved with different non-source content after the old execution record, so that old notebook-file hash differs. We record both hashes in `05_notebook04_provenance.json` and keep all earlier files untouched.''')
md('''A different original feature pattern means a different combination of all 42 original measurements. Repeated rows can share a pattern, even if their labels disagree. We keep those rows and labels, and keep each pattern within one row set. This does not prove that patterns come from independent hosts or events.''')
md('''## 2. Define the comparison sets

We write down five comparisons **before this notebook's analysis**. All five keep every original feature. The changes concern only the three engineered additions.

Why these five? The two balances are strongly related in Notebook 04, but we do not know whether they help the same models. We need a reference with no additions, a reference with all additions, and small comparisons that isolate total volume and each balance.

The feature names refer to original source columns. For example, `proto` is one feature, even if encoding creates many columns such as `proto_tcp` and `proto_udp`. We always keep or omit the entire encoded group. We never select one category by itself.''')
code('''comparison_table = define_sets(state)
display(comparison_table.rename(columns={
    'candidate': 'Set name', 'features': 'Source features', 'omitted': 'Omitted additions', 'reason': 'Purpose',
}))
print('All sets retain these 42 original features:')
print(', '.join(state['original']))''')
code('''names = list(state['sets'])
included = np.array([[feature in state['sets'][name]['retained']
                      for feature in ENGINEERED] for name in names], dtype=int)
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.imshow(included, vmin=0, vmax=1, cmap='Blues', aspect='auto')
ax.set_xticks(range(3), ['Total bytes', 'Byte balance', 'Packet balance'])
ax.set_yticks(range(5), names)
for i in range(5):
    for j in range(3):
        ax.text(j, i, 'Keep' if included[i, j] else 'Omit', ha='center', va='center',
                color='white' if included[i, j] else 'black')
ax.set_title('Only these three additions change; all original features stay')
figure(state, fig, 'comparison_sets')''')
md('''This picture is the comparison plan, not a result. The sets will keep these definitions throughout the notebook. Their full feature lists are saved, so Notebook 06 can repeat exactly the same comparisons.''')
md('''## 3. Look for repeated information

We study fitting rows only. We ask four simple questions:

- Are any columns exact copies?
- Do some numeric columns move almost together?
- Does a column almost always contain the same value?
- Are the two balances similar when repeated rows count less?

An exact copy has the same recorded value in every fitting row. A high correlation only describes a relationship; it is not exact equality. A low-variation feature may still describe important rare flows. These findings do not automatically remove features.''')
code('''copies, variation, near_pairs, ftp_values = redundancy_study(state)
display(copies.rename(columns={'feature_a': 'First feature', 'feature_b': 'Second feature',
                               'equal_rows': 'Matching rows', 'equal_original_patterns': 'Matching original patterns'}))
display(variation.sort_values('mode_row_share', ascending=False).head(6)[['feature', 'unique_values', 'most_common_value',
                         'mode_row_share', 'mode_pattern_share', 'low_variation_flag']].rename(columns={
    'feature': 'Feature', 'unique_values': 'Different values', 'most_common_value': 'Most common value',
    'mode_row_share': 'Share of rows at most common value',
    'mode_pattern_share': 'Share of patterns at their most common value',
    'low_variation_flag': 'At least 99.5% of rows share a value',
}))
print('The full copy, variation and correlation tables are saved in results/tables/.')''')
md('''The 99.5% flag is a way to notice a column, not a deletion threshold. We show row frequency and pattern frequency separately because a value may be common mostly through repetitions. One representative per original pattern is safe for these feature-only summaries: original copies have identical predictor values.

### Keep both FTP fields

`is_ftp_login` and `ct_ftp_cmd` match in the supplied training file. Notebook 02 already documented differences elsewhere in the supplied data. We refer to that saved finding; we do not load or analyse official-test records again.

The intended meanings of these fields also differ, and their recorded values need more explanation. **Both stay in every candidate set.** Fitting equality alone is not enough to choose one to delete.''')
code('''display(ftp_values)
display(pd.read_csv(root / 'results/tables/05_prior_ftp_audit_reference.csv'))
print('Decision: retain both FTP fields, including their original nonbinary values.')''')
md('''### Check the two balances

Spearman correlation compares how two numerical columns tend to increase together. A value near +1 means a strong shared ordering; it does not mean equal values. We calculate it once for all fitting rows and once with one row per original pattern.''')
code('''row_corr = state['corr'].loc['fe_byte_balance', 'fe_packet_balance']
pattern_corr = state['pattern_corr'].loc['fe_byte_balance', 'fe_packet_balance']
display(pd.DataFrame([
    {'View': 'All fitting rows', 'Spearman correlation': row_corr},
    {'View': 'One row per original pattern', 'Spearman correlation': pattern_corr},
]))
evidence_figures(state)''')
md('''Read the colour chart as a map of relationships between selected measurements. Near +1 means values tend to move together; near −1 means they tend to move in opposite directions. The full pair table also checks the remaining numeric features.

**Decision:** test byte balance and packet balance separately and together. Their strong relationship gives us a reason for the 44-feature variants, not a reason to declare either balance useless. We also keep all original low-variation and highly correlated fields until a justified model comparison is available.''')
md('''## 4. Look at label associations carefully

A feature may take different values in different attack classes. We describe this with an **association score**, not model accuracy. A score near zero means little association in this view; a score near one means a strong association. A constant observed feature gets an unavailable score.

For a fair internal check, we divide **only the fitting rows** into three temporary folds. Identical original patterns stay together, including conflicting labels. Each fold uses the other two to define numeric value bands, then checks association in its held-out fold. Numeric bands use up to eight training quantile groups; repeated cut values are combined. Categorical columns are checked as whole features using their literal categories.

This is a grouped stability check of descriptive evidence. It does not train a predictive model or use `model_selection` or `calibration` labels. The temporary folds do not replace the frozen development split.''')
code('''association, classes, conflicts = relevance_study(state)
display(classes.rename(columns={'attack_cat': 'Class', 'rows': 'Fitting rows',
                               'original_patterns': 'Different original patterns'}))
display(conflicts.rename(columns={'target': 'Target', 'conflicting_patterns': 'Patterns with conflicting labels',
                                 'affected_rows': 'Affected fitting rows'}))
internal_support = pd.read_csv(root / 'results/tables/05_internal_fold_class_support.csv')
display(internal_support[(internal_support.part == 'inner_holdout') &
                         internal_support.attack_cat.isin(['Worms', 'Shellcode', 'Backdoor'])])''')
md('''Rare classes have fewer examples and patterns, so their associations may be unstable. Conflicting labels mean the same measurements can belong to more than one target class. We preserve them rather than hiding the difficulty.

We calculate two versions of the score: one where every row counts equally, and one where each original pattern has total weight one. In the second version, repeated rows count less; conflicting labels still contribute their original shares. This is a descriptive weighting, not proof of independent observations.''')
code('''chosen = ['sbytes', 'dbytes', 'rate', 'service', 'is_ftp_login', 'ct_ftp_cmd', *ENGINEERED]
display(association[association.target.eq('attack_cat') & association.feature.isin(chosen)][[
    'feature', 'mean_row_score', 'min_row_score', 'max_row_score',
    'mean_pattern_weighted_score', 'defined_folds',
]].rename(columns={
    'feature': 'Feature', 'mean_row_score': 'Mean row score',
    'min_row_score': 'Lowest fold score', 'max_row_score': 'Highest fold score',
    'mean_pattern_weighted_score': 'Mean pattern-weighted score', 'defined_folds': 'Defined folds',
}))
relevance_figure(state)''')
md('''The lines show the range across three folds, not a confidence interval. A feature's score can change when repeated patterns count less.

Both FTP fields have **no score in this check**. Almost all fitting values are zero, so their quantile bands collapse to a single group. This method cannot describe their rare nonzero values properly. An unavailable score is not a zero score or a reason to drop either field; the separate value table above preserves that evidence.

The score is Cramer's V after grouping numeric values into bands. More categories, class imbalance and the chosen bands can affect it. It does not measure the extra value a feature adds when the other columns are already available. A weak individual score can also miss interactions with other features.

**Decision:** save this evidence without using a score cutoff. We do not fit a predictive selector or keep only the highest-ranked features. The internal binning rules are saved for reproducibility and are not applied to the model input files. Method details: [association score](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.contingency.association.html) and [grouped folds](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html).''')
md('''## 5. Choose the shortlist

Carry forward the five predefined sets. We keep the full 42 and 45 as references, and the three smaller engineered variants to answer specific questions. No original predictor is removed on the strength of a ranking or correlation.

| Comparison in Notebook 06 | Question it answers |
|---|---|
| `total_only43` versus `original42` | Does adding total bytes help? |
| `total_and_byte44` versus `total_only43` | Does byte balance add value beyond total bytes? |
| `total_and_packet44` versus `total_only43` | Does packet balance add value beyond total bytes? |
| `all45` versus each 44-feature set | Does the second balance add anything? |
| `all45` versus `original42` | Does the complete engineered addition help? |

This shortlist stays small and easy to interpret. The unanswered question is model benefit, not whether the formulas can be calculated.''')
code('''shortlisted = shortlist(state)
display(shortlisted[['candidate', 'features', 'omitted_original_names', 'reason']].rename(columns={
    'candidate': 'Set', 'features': 'Source features',
    'omitted_original_names': 'Omitted features', 'reason': 'Why carry it forward',
}))
print('Complete retained and omitted lists: results/tables/05_shortlist.csv')
print('No winning feature set has been chosen.')''')
md('''## 6. Save and check each set

We slice the verified Notebook 04 inputs using the saved feature names. There is no new scaling or encoding. All one-hot columns from a categorical feature stay together. Raw feature order follows the source-name lists; prepared matrix order follows Notebook 04 and is saved separately.

Removing columns can make different rows identical. We therefore check exact matches between every pair of row sets, separately for raw, tree and scaled inputs. Any new matches are reported with participating row counts; we do not change the split to hide them.''')
code('''verified, overlaps = save_candidates(state)
display(verified[verified.role.eq('train')][['candidate', 'representation', 'rows', 'columns', 'saved_reload_exact']])
display(overlaps[overlaps.representation.eq('raw')][[
    'candidate', 'role_a', 'role_b', 'shared_vectors',
    'role_a_rows_with_match', 'role_b_rows_with_match',
]])
matches = overlaps[overlaps.shared_vectors.gt(0)]
print(f'Checks with exact cross-role matches: {len(matches)} out of {len(overlaps)}.')
if len(matches):
    display(matches)
print('Frozen row assignments have not changed.')''')
md('''Source features and prepared matrix columns have different counts because one categorical feature creates several one-hot columns. Their grouping is recorded in `05_encoded_column_groups.csv`.

Every set retains all original 42 predictors, which helps preserve the original separation. The actual checks above and in the saved table still verify each prepared representation. Exact separation does not rule out shared network context or near-matches.

| Saved file | Contents |
|---|---|
| `05_<set>_raw_<role>.csv` | Source-name features for one candidate and row set |
| `05_<set>_tree_<role>.npz` | Columns from the frozen Notebook 04 tree inputs |
| `05_<set>_scaled_<role>.npz` | Columns from the frozen Notebook 04 scaled inputs |
| `05_row_links_<role>.csv` | Matching row positions and original labels, stored separately |
| `05_<set>_features.json` | Complete retained and omitted source-name lists |

Inputs are under `data/processed/05/`; lists and matrix-column mappings are under `models/preprocessing/05/`. Corresponding files share row order. Detailed saved/reloaded checks remain in `results/tables/` rather than crowding the notebook.''')
md('''## 7. Hand over to Notebook 06

Notebook 06 should compare the five sets with a small, declared group of baseline models. For example, compare a regularized linear model using the scaled inputs and a tree-based model using the tree inputs. Give every feature set the same tuning budget, training rows and reproducible settings. Keep unweighted comparisons; any class-weighted experiment should be named separately and applied consistently across sets.

Fit only on `train`. Use `model_selection` to compare sets and models; keep `calibration` for later confidence adjustment. Do not evaluate the official test while choosing the baseline. Follow Notebook 02's metrics, including fixed-ten-class macro F1, each class's results and support, confusion matrices, and binary attack measures. Choose any binary decision threshold on model-selection rows only.

To decide whether engineered features help, we need **a better model-selection result under a fair matched comparison**, with the actual difference, rare-class tradeoffs and stability reported. A high association score here is not that result. If uncertainty is estimated later, compare sets using the same resampled original groups, while keeping the limits of unknown network dependence explicit.''')
code('''checks = finish(state)
display(checks)
print('Report: docs/05_feature_selection_findings.md')
print('Output hashes: data/processed/05/05_artifact_manifest.json')
print('PASS: original labels, saved values, whole categorical groups and earlier files verified.')''')
md('''## What we decided

**Test these sets:** original 42, all 45, total bytes plus byte balance, total bytes plus packet balance, and total bytes alone.

**Why:** preserve clear references and separate the contribution of volume and each balance. Keep all original columns, including both FTP fields, because fitting similarity does not settle their meaning or usefulness.

**Still uncertain:** which set works best for which model, whether either balance adds value, and how rare classes, repeated patterns, conflicting labels and shared context affect the comparison.

**What would settle the feature question:** reproducible model-selection evidence from Notebook 06, with matched baselines and transparent class results. No set is called the winner in this notebook.''')

nb=nbf.v4.new_notebook(cells=cells,metadata={
    'kernelspec':{'name':'botnet-research','display_name':'Botnet research (.venv)','language':'python'},
    'language_info':{'name':'python','version':'3.11.9'}})
nbf.validate(nb)
nbf.write(nb,ROOT/'notebooks/05_feature_selection.ipynb')
print(f'Built {len(cells)} cells.')
