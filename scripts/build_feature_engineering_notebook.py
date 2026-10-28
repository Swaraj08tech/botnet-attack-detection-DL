"""Build a readable feature-engineering notebook with visible formulas."""

from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []
def md(text): cells.append(nbf.v4.new_markdown_cell(text))
def code(text): cells.append(nbf.v4.new_code_cell(text))

md('''# 04 — Feature Engineering

**Goal: create three new columns that describe the size and direction of a network flow.**

We already have bytes and packet counts in each direction. We will use them to make **total bytes**, **byte balance**, and **packet balance**. You will see the formulas, the code that creates these columns, and examples from the data.

At the end, the data will have **42 original columns + 3 new columns = 45 feature columns**. We will save these for the next notebook. We do not train a model here, so we cannot yet say these features improve detection.''')
md('''## What is feature engineering?

A **feature** is a column that describes a flow. For example, `sbytes` describes how many bytes travelled from source to destination.

**Feature engineering means calculating a useful new description from measurements we already have.** If we have bytes in both directions, we can add them to describe total volume, or compare them to describe direction. The new description becomes another column.

In this notebook, we do three kinds of work:

| Step | What we do | Why |
|---|---|---|
| Create | Calculate total bytes and two balances | Make flow size and direction explicit |
| Check | Read examples, ranges, zeros and simple plots | Confirm that the calculations behave as intended |
| Save | Add the same columns to the saved row sets | Make the results ready for later comparisons |

The formulas **do not add information that was absent from the original measurements**. They may make a useful relationship easier for a particular model to use. Whether that happens must be tested later.

This is different from choosing which columns a model should use. We keep all 42 original features. We also do not decide that a flow is an attack just because its new feature is large or positive.''')
md('''## 1. Load the data

Use the same groups of rows saved in Notebook 02:

- `train`: rows we use to study and choose the new features.
- `model_selection`: rows reserved for later comparisons.
- `calibration`: rows reserved for later confidence adjustment.

The checks below confirm that the files and row assignments have not changed. We do not load the official test data. Detailed file checks run in the background so we can focus on feature engineering.''')
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

from src.data.feature_engineering import RETAINED, retained_features
from src.data.feature_engineering_audit import (
    starting_point, proposals, fitting_investigation, decisions,
    apply_development_roles, finish, save_table, save_figure,
)

state, role_counts = starting_point(root)
fit_rows = state['source'].iloc[state['positions']['train']]
predictors = state['predictors']
display(role_counts.rename(columns={'role': 'Row set', 'rows': 'Rows',
                                   'distinct_feature_groups': 'Different original feature patterns'}))
print('File checks passed. We will study only the train rows.')''')
md('''## 2. Create the new features

These are the four existing columns we need:

| Existing column | Meaning |
|---|---|
| `sbytes` | Bytes sent from source to destination |
| `dbytes` | Bytes sent back from destination to source |
| `spkts` | Packets sent from source to destination |
| `dpkts` | Packets sent back from destination to source |

Source and destination are the directions recorded in the dataset. They do not always mean client and server. We use completed flows, because we need their final byte and packet counts.

The calculations use only these measurements. `id` identifies a row; `label` and `attack_cat` are answers we want a future model to predict. They are never ingredients in a feature formula.

We choose the formulas using fitting rows only. No average, threshold or class label is needed to calculate these three features.''')
md('''### Feature 1: Total bytes

**Formula: `sbytes + dbytes`**

This tells us how much data moved in both directions together. Its unit is **bytes**, like the source columns. It cannot be negative when the original counts are nonnegative.

Example: `900 + 100 = 1,000` bytes.

Why propose it? Two flows can send different amounts forward and back but have the same overall volume. A total gives a direct description of size. It is not a transfer rate, and it is not necessarily application payload size. Large volume alone does not identify an attack.

The next cell adds the first feature to the real fitting rows.''')
code('''train_features = fit_rows[predictors].copy()
counts = train_features[['sbytes', 'dbytes', 'spkts', 'dpkts']].astype(float)

train_features['fe_total_bytes'] = counts['sbytes'] + counts['dbytes']

display(train_features[['sbytes', 'dbytes', 'fe_total_bytes']].head(5))''')
md('''Read each row from left to right: add the two byte counts to get the new total. The original columns are still present.

### Feature 2: Byte balance

**Formula: `(sbytes - dbytes) / (sbytes + dbytes)`**

This tells us which direction carried more bytes. We divide the difference by the total so the value measures **directional balance**, rather than just the size of the flow. It has no unit and stays between −1 and +1.

Example: `(900 − 100) / (900 + 100) = 800 / 1,000 = 0.8`.

Why divide by the total? A flow with 9,000 bytes forward and 1,000 back also has a balance of 0.8, even though it is ten times larger. Total bytes captures size; byte balance captures the split between directions.

| Balance | Meaning when the total is positive |
|---|---|
| +1 | All bytes went from source to destination |
| Between 0 and +1 | More bytes went forward than back |
| 0 | Equal byte counts in both directions |
| Between −1 and 0 | More bytes went back than forward |
| −1 | All bytes went from destination to source |

If both counts are zero, division would be undefined. We store zero in that case. Keeping the original counts lets us distinguish no recorded bytes from equal amounts of bytes.''')
code('''total_bytes = counts['sbytes'] + counts['dbytes']

train_features['fe_byte_balance'] = np.divide(
    counts['sbytes'] - counts['dbytes'], total_bytes,
    out=np.zeros(len(counts)), where=total_bytes > 0,
)

display(train_features[['sbytes', 'dbytes', 'fe_byte_balance']].head(5))''')
md('''`np.divide` performs the division. `where` limits it to positive totals; `out=np.zeros(...)` supplies zero for the other rows. We do not add a small artificial number to the denominator.

### Feature 3: Packet balance

**Formula: `(spkts - dpkts) / (spkts + dpkts)`**

This tells us which direction carried more packets. It uses the same −1 to +1 scale and zero-total rule as byte balance, but it compares **packet counts**, not bytes.

Example: `(6 − 4) / (6 + 4) = 2 / 10 = 0.2`.

Why keep a separate packet balance? Packets can carry different amounts of data. A flow may carry most of its bytes forward while its packet counts are more even. Our example has byte balance 0.8 and packet balance 0.2. These describe different aspects of the same flow.

This feature does not establish packet loss, response success or malicious behaviour.''')
code('''total_packets = counts['spkts'] + counts['dpkts']

train_features['fe_packet_balance'] = np.divide(
    counts['spkts'] - counts['dpkts'], total_packets,
    out=np.zeros(len(counts)), where=total_packets > 0,
)

display(train_features[['spkts', 'dpkts', 'fe_packet_balance']].head(5))
print(f'Before: {len(predictors)} columns. After: {train_features.shape[1]} columns.')''')
md('''### Put the formulas in one function

We have now created each feature separately. The function below collects the same three formulas so we can apply them consistently to other rows. It copies the input rather than changing the original data.''')
code('''def add_flow_features(data):
    """Keep the existing columns and add our three flow summaries."""
    result = data.copy()
    counts = data[['sbytes', 'dbytes', 'spkts', 'dpkts']].astype(float)
    assert np.isfinite(counts.to_numpy()).all() and counts.ge(0).all().all()

    total_bytes = counts['sbytes'] + counts['dbytes']
    total_packets = counts['spkts'] + counts['dpkts']

    result['fe_total_bytes'] = total_bytes
    result['fe_byte_balance'] = np.divide(
        counts['sbytes'] - counts['dbytes'], total_bytes,
        out=np.zeros(len(data)), where=total_bytes > 0,
    )
    result['fe_packet_balance'] = np.divide(
        counts['spkts'] - counts['dpkts'], total_packets,
        out=np.zeros(len(data)), where=total_packets > 0,
    )
    return result

# Confirm the combined function gives exactly the values we just created.
pd.testing.assert_frame_equal(train_features, add_flow_features(fit_rows[predictors]))
display(train_features[['sbytes', 'dbytes', 'spkts', 'dpkts', *RETAINED]].head(5))''')
md('''**The three features have now been created.** The table shows the original counts beside the new values. `fe_` means “feature engineering”; it helps us recognise the columns we added.

### Worked examples

The next rows are **made-up examples for learning**, not observations from the dataset. They let us check different directions, equal traffic, larger volume and zero totals.''')
code('''examples = pd.DataFrame({
    'sbytes': [900.0, 100.0, 500.0, 9000.0, 0.0],
    'dbytes': [100.0, 900.0, 500.0, 1000.0, 0.0],
    'spkts': [6.0, 4.0, 5.0, 60.0, 0.0],
    'dpkts': [4.0, 6.0, 5.0, 40.0, 0.0],
}, index=['More forward', 'More back', 'Equal amounts', 'Ten times larger', 'No recorded traffic'])
worked = add_flow_features(examples)
display(worked)
save_table(state, worked.rename_axis('Example').reset_index(), 'worked_examples')''')
md('''**How to read the examples:**

- “More back” reverses the balance signs but keeps the same total bytes.
- “Equal amounts” has zero balances and a positive total.
- “Ten times larger” has a larger total but the same balances as “More forward”.
- “No recorded traffic” also has zero balances, but its original counts and total are zero.

This is why we keep both volume and balance, and why we keep the original measurements alongside them.''')
md('''## 3. Check the new features

We check the ranges, zeros and unusually large values. We also count different original feature patterns, because some rows have identical measurements. These counts do not prove the flows are independent events.

The detailed checks also examine three alternatives: packet rate and average bytes per packet in each direction. Only fitting rows are used. The short results below show what matters for our three new features.''')
code('''proposals(state)
stats, redundancy, class_support = fitting_investigation(state)

# Make sure the notebook formulas match the reusable code used when saving.
pd.testing.assert_frame_equal(train_features[RETAINED], retained_features(fit_rows[predictors]))

summary = stats[stats.feature.isin(RETAINED)].copy()
display(summary[['feature', 'min', 'median', 'p99', 'max', 'zero_rows']].rename(columns={
    'feature': 'Feature', 'min': 'Smallest', 'median': 'Middle value',
    'p99': '99% of rows are at or below', 'max': 'Largest', 'zero_rows': 'Zero rows',
}))
print(f'Training support: {len(fit_rows):,} rows and {state["fit_groups"].nunique():,} different original feature patterns.')
print('Rows with a zero total for either balance:', int(summary.denominator_zero_rows.sum()))''')
md('''### Read the value checks

“Middle value” means the median: half the rows are at or below it. The 99% column helps us compare typical high values with the largest value. It is **not a rule for deleting rows**.

Zero means different things for different features: zero total bytes means no recorded bytes, while zero balance can mean equal amounts or a zero-total placeholder. The total-zero count tells us whether that second case occurs in these fitting rows.

We keep unusual values. Their size alone does not prove that they are wrong.''')
code('''direction_rows = []
groups = state['fit_groups']
for feature in RETAINED[1:]:
    values = train_features[feature]
    for meaning, mask in [
        ('More forward', values.gt(0)),
        ('Zero balance', values.eq(0)),
        ('More back', values.lt(0)),
    ]:
        direction_rows.append({
            'Feature': feature, 'Direction': meaning, 'Rows': int(mask.sum()),
            'Different original feature patterns': groups[mask].nunique(),
            'Share of fitting rows (%)': round(100 * mask.mean(), 2),
        })
direction_counts = pd.DataFrame(direction_rows)
display(direction_counts)
save_table(state, direction_counts, 'balance_direction_support')''')
md('''Some patterns occur in many rows. The row count describes their frequency in this dataset; the pattern count describes how many different full 42-feature patterns are represented. Neither number confirms independent hosts or events.

### Look at the distributions

The plots show how often each value occurs. Total bytes uses `log1p` only on the display so small and large flows can be seen together. The saved raw total is still measured in bytes.''')
code('''fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
axes[0].hist(np.log1p(train_features['fe_total_bytes']), bins=40, color='#527b94')
axes[0].set(title='Total bytes', xlabel='log1p(bytes), for display only', ylabel='Number of rows')
for ax, name, title in zip(axes[1:], RETAINED[1:], ['Byte balance', 'Packet balance']):
    ax.hist(train_features[name], bins=np.linspace(-1, 1, 31), color='#527b94')
    ax.set(title=title, xlabel='−1: back  |  0: equal  |  +1: forward', ylabel='Number of rows')
save_figure(state, fig, 'feature_distributions')''')
code('''s = summary.set_index('feature')
display(Markdown(
    f"The middle total-byte value is **{s.loc['fe_total_bytes', 'median']:,.0f} bytes**, "
    f"but the largest is **{s.loc['fe_total_bytes', 'max']:,.0f} bytes**. "
    "The log display lets us see small and large flows together; it does not change the saved raw feature. "
    "Balances stay between −1 and +1. A balance near +1 means most traffic went forward; "
    "it does not by itself mean an attack."
))''')
md('''### A quick look at the labels

The table below shows the middle feature value for normal and attack rows. Labels are used only to describe the results, never to calculate a feature. A difference between these groups does not prove the new features will improve a model.''')
code('''relations = state['relationships']
binary = relations[relations.target.eq('label') & relations.feature.isin(RETAINED)].copy()
binary['Traffic'] = binary['target_class'].map({'0': 'Normal', '1': 'Attack'})
display(binary[['Traffic', 'feature', 'rows', 'distinct_feature_groups', 'median']].rename(columns={
    'feature': 'Feature', 'rows': 'Rows', 'distinct_feature_groups': 'Different original feature patterns',
    'median': 'Middle value',
}))

fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
for ax, name, title in zip(axes, RETAINED[1:], ['Byte balance', 'Packet balance']):
    values = binary[binary.feature.eq(name)]
    ax.bar(values['Traffic'], values['median'], color=['#527b94', '#b66d4b'])
    ax.set(title=title, ylabel='Middle value', ylim=(-1, 1))
save_figure(state, fig, 'balance_by_class')''')
code('''middle_values = binary.pivot(index='Traffic', columns='feature', values='median')
display(Markdown(
    f"In these fitting rows, the middle byte balance is **{middle_values.loc['Normal', 'fe_byte_balance']:.3f}** "
    f"for normal traffic and **{middle_values.loc['Attack', 'fe_byte_balance']:.3f}** for attack traffic. "
    "This describes the sample; individual flows can differ from these middle values. "
    "We have not tested a detection rule, set a threshold, or measured model performance."
))''')
md('''### Are the two balances similar?

They use different counts, but they might often move together. The next plot shows a fixed random sample of fitting rows. A point on the grey line has the same byte and packet balance. A point away from it has different balances. The full-data correlation is calculated separately; the plot is only a small view.''')
code('''sample = train_features[RETAINED[1:]].sample(n=min(3000, len(train_features)), random_state=42)
fig, ax = plt.subplots(figsize=(5, 4))
ax.scatter(sample['fe_byte_balance'], sample['fe_packet_balance'], s=8, alpha=0.18)
ax.plot([-1, 1], [-1, 1], color='grey', linestyle='--', label='Equal balances')
ax.set(xlabel='Byte balance', ylabel='Packet balance', title='Do the balances move together?',
       xlim=(-1.05, 1.05), ylim=(-1.05, 1.05))
ax.legend()
save_figure(state, fig, 'balance_relationship')

balance_correlation = state['candidate_corr'].loc['fe_byte_balance', 'fe_packet_balance']
print(f'Full fitting-data Spearman correlation: {balance_correlation:.4f}')''')
md('''A high positive correlation means the balances often increase together. It does not mean they are identical or that we can safely remove one. We keep both for later comparison, without choosing features using a model here.''')
md('''### Why keep only these three additions?

We keep total bytes and the two balances because their meanings are clear and they do not divide by flow duration. We leave out the rate and average-packet-size alternatives because existing columns already cover similar information. No original column is removed.''')
code('''comparison = redundancy.copy()
comparison['Agreement (%)'] = 100 * comparison['close_fraction']
display(comparison[['candidate', 'comparator', 'valid_rows', 'valid_groups', 'Agreement (%)']].rename(columns={
    'candidate': 'Formula checked', 'comparator': 'Existing feature',
    'valid_rows': 'Rows with positive denominator', 'valid_groups': 'Different original feature patterns',
}))''')
md('''The bytes-per-packet comparisons allow half a byte of rounding difference, plus a small relative tolerance. The rate comparisons use a much smaller absolute tolerance and the same small relative tolerance. All counts use positive denominators; detailed tolerances are saved with the results.

The ordinary packet-rate formula does not exactly reproduce the existing `rate`. The alternative `(spkts + dpkts - 1) / dur` often comes closer. That is a reason to avoid adding a second uncertain rate definition, rather than replacing the supplied column. The average packet-size formulas closely match `smean` and `dmean`, so we leave them out of this small set.''')
code('''decisions(state)
choice = pd.DataFrame([
    ['Total bytes', 'Keep as a candidate', 'A simple measure of overall flow size.'],
    ['Byte balance', 'Keep as a candidate', 'Shows which direction carries more bytes.'],
    ['Packet balance', 'Keep as a candidate', 'Shows which direction carries more packets.'],
    ['Packet rate', 'Leave out', 'Similar to rate; its exact calculation needs more checking.'],
    ['Source bytes per packet', 'Leave out', 'Closely matches the existing smean column.'],
    ['Destination bytes per packet', 'Leave out', 'Closely matches the existing dmean column.'],
], columns=['Feature idea', 'Choice', 'Reason'])
save_table(state, choice, 'simple_feature_choices')
display(choice)

balance_correlation = state['candidate_corr'].loc['fe_byte_balance', 'fe_packet_balance']
print(f'The two balances are closely related: correlation = {balance_correlation:.2f}.')''')
md('''**Both balances may not be needed later.** For now, we save both as candidates. The next notebook can study which features are worth using. None of these three additions is a new measurement; each is a summary of columns we already have.''')
md('''## 4. Add the features and save

Apply the same formulas to all three saved row sets. Keep all original features, rows and labels. The cell below creates the new columns for each set and then saves checked copies.

We also save copies that work with Notebook 03's existing prepared inputs. Any extra scaling uses only `train` rows; the old prepared inputs stay unchanged. Detailed file and overlap checks run in the background.''')
code('''development_features = {}
for role, positions in state['positions'].items():
    original = state['source'].iloc[positions][predictors]
    development_features[role] = add_flow_features(original)
    pd.testing.assert_frame_equal(
        development_features[role][RETAINED], retained_features(original),
    )

# Save the new inputs and check every saved row and value.
shapes, overlap = apply_development_roles(state)
saved_rows = pd.DataFrame([
    {'Row set': role, 'Rows': len(data), 'Original columns': 42, 'New columns': 3,
     'Total feature columns': data.shape[1]}
    for role, data in development_features.items()
])
display(saved_rows)

# Show only the check for the complete 45-column inputs here.
display(overlap[overlap.representation.eq('raw45')][['role_a', 'role_b', 'shared_vectors']].rename(
    columns={'role_a': 'First row set', 'role_b': 'Second row set',
             'shared_vectors': 'Identical complete feature patterns'}
))''')
md('''The complete inputs still keep the row sets apart: the table reports any identical 45-column patterns across them. We do not change the saved split. Using only the three new columns would lose much of the original information and create many matches; that separate check is saved in the detailed tables.

Files are saved under `data/processed/04/`. For each row set, `04_raw_augmented_*.csv` contains the 45 feature columns. `04_row_links_*.csv` stores row positions and original labels separately. The original 42-column data and Notebook 03 inputs remain available for comparison.''')
md('''### What is in each saved file?

| File | What it contains |
|---|---|
| `04_raw_augmented_<role>.csv` | All 42 original predictors and the 3 new columns |
| `04_engineered_<role>.csv` | Only the 3 new columns, for inspection |
| `04_row_links_<role>.csv` | Row positions, original labels and group information; never model features |
| `04_tree_augmented_<role>.npz` | Notebook 03's tree inputs plus the 3 raw new columns |
| `04_scaled_augmented_<role>.npz` | Notebook 03's scaled inputs plus separately scaled new columns |

`<role>` means `train`, `model_selection` or `calibration`. Rows in corresponding files are in the same order. The input files do not contain the answers a model should predict.

For the scaled copy only, total bytes uses `log1p` and the new columns use a training-fitted RobustScaler. No old pipeline is refitted. The raw 45-column files remain the clearest view of the feature engineering itself.

To read CSV values exactly as saved, use pandas `float_precision='round_trip'` and `float64` for the three new columns. These reading settings preserve stored numbers; they are not new features.''')
code('''checks = finish(state)
print('Saved the feature files and the short findings report.')
print('Saved values and original labels checked successfully.')
print(f'Earlier files unchanged: {checks["protected_files_unchanged"]}.')
print('Full details: docs/04_feature_engineering_findings.md')''')
md('''## 5. What we did

We created and saved **three new features: total bytes, byte balance and packet balance**. We now have **45 feature columns instead of 42**, without deleting any original columns.

The new features describe flow size and direction. We checked their values, looked at a few simple plots, and kept the original rows and labels.

**Still unknown:** whether these additions help a model, and whether both balances are useful. They do not establish botnet-specific labels or early detection.

We have not created DNS information or a time sequence. These would need measurements that this subset does not provide. Existing shared-context counters and conflicting labels remain open research issues; our three formulas do not resolve them.

**Next:** Notebook 05 should compare which original and new features are worth using, followed by baseline model experiments. Keep the same saved row sets and use the official test only for final evaluation.''')

nb = nbf.v4.new_notebook(cells=cells, metadata={
    'kernelspec': {'display_name': 'Botnet research (.venv)', 'language': 'python', 'name': 'botnet-research'},
    'language_info': {'name': 'python', 'version': '3.11.9'},
})
nbf.validate(nb)
nbf.write(nb, ROOT / 'notebooks/04_feature_engineering.ipynb')
print(f'Built {len(cells)} cells with visible feature formulas.')
