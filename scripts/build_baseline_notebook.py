"""Create a readable Notebook 06; calculations stay in src/."""

from pathlib import Path
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]
cells=[]
def md(text): cells.append(nbf.v4.new_markdown_cell(text))
def code(text): cells.append(nbf.v4.new_code_cell(text))

md('''# 06 — Baseline Models

**Question: Do Notebook 04's engineered features improve simple intrusion-detection models compared with the original 42 features?**

We compare five feature sets with two simple models. Every comparison uses the same fitting rows, model settings, seed and model-selection rows. Only the engineered additions change within each model family.

The result will be a measured feature comparison and a **provisional** model choice. It will not be a final test score. We keep calibration for later, and do not load official-test records.''')
md('''## 1. Check the saved inputs

We check file hashes, feature names, matrix shapes, row order, original labels and the frozen row assignments. Each prepared matrix must be an exact column slice of the verified Notebook 04 inputs.

`id`, targets and row metadata stay outside the model matrices. The categorical columns keep their whole one-hot groups. No new preprocessing or split is created.

Calibration files receive integrity checks only. Models never predict their rows here. The official test file is only hashed to confirm preservation.''')
code('''from pathlib import Path
import sys
import pandas as pd
from IPython.display import display, Markdown

root = next(p for p in (Path.cwd().resolve(), *Path.cwd().resolve().parents)
            if (p / 'configs/02_validation_protocol.json').is_file())
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from src.baseline_models import SETS, MODELS, CLASSES
from src.baseline_audit import (
    setup, declare_plan, fit_family, compare_results, result_figures,
    detailed_figures, findings_text, finish,
)

state, role_support = setup(root)
display(role_support.rename(columns={
    'role': 'Row set', 'rows': 'Rows', 'original_patterns': 'Different original patterns', 'use': 'Use here',
}))
shapes = pd.read_csv(root / 'results/tables/06_input_verification.csv')
display(shapes[shapes.role.eq('train')][['feature_set', 'representation', 'source_features', 'columns']])
print('PASS: hashes, target mapping, row links, finite values and full categorical groups verified.')''')
md('''**Earlier provenance note:** Notebooks 04 and 05 have saved notebook-file hashes that differ from their old execution records. Their code and Markdown match the manifest-verified builders, and all listed artifacts verify. We record the discrepancy and both Notebook 05 hashes in `06_input_integrity.json`, while preserving every earlier file and execution record.

Original patterns stay in one role, even when their labels conflict. Previous exact-overlap checks are hash-verified and show zero cross-role matches for these inputs. This does not establish independent hosts, sessions or times.''')
md('''## 2. Write down the experiment plan

| Set | Original features | Engineered additions |
|---|---:|---|
| `original42` | All 42 | None |
| `total_only43` | All 42 | Total bytes |
| `total_and_byte44` | All 42 | Total bytes + byte balance |
| `total_and_packet44` | All 42 | Total bytes + packet balance |
| `all45` | All 42 | All three |

We fit an **L2-regularized logistic classifier** on the saved scaled inputs. It learns one class-versus-rest classifier for each of the ten classes, then provides multiclass predictions. L2 discourages large coefficients. We also fit a **decision tree** on the saved tree inputs. It learns branching rules, limited to depth 16 with at least 5 fitting rows in each leaf.

Both models accept sparse matrices. We use unweighted fitting, seed 42, and one fixed configuration per model across all five sets: ten experiments in total, with no tuning search. The linear solver has a maximum of 500 iterations per class; any convergence warning is recorded, and that candidate cannot become the provisional choice. We do not retry it based on its score.''')
code('''plan = declare_plan(state)
display(plan)
print('The complete settings and comparison rules were saved before fitting: configs/06_baseline_plan.json')''')
md('''### Decide what counts as better

**Main score: macro F1 across all ten classes, including Normal.** It calculates each class's F1 and averages the ten values equally. Accuracy and weighted F1 give more weight to common classes, so they are supporting measures.

**Tie rule, declared before fitting:** keep eligible candidates within 0.001 macro F1 of the highest score. That is 0.1 percentage point. Prefer fewer source features; if that ties, prefer the linear model; then higher macro F1 and the declared feature-set order. This is a practical simplicity rule, not a significance test.

We will show each class's precision (how often its predictions are right), recall (how many of its actual rows it finds), F1 and support. There are only **13 model-selection Worms rows**. One changed correct prediction shifts Worms recall by **7.69 percentage points**, so small changes are uncertain.''')
md('''## 3. Fit on the training rows

The only data used to fit model weights is `train`. The target is `attack_cat`, using the frozen ten-class mapping. We preserve repeated rows and conflicting labels. Row positions and group IDs link results back to records; they are not predictors.

For every experiment we save the model, exact settings and model-selection predictions. We reload the model and confirm that it gives exactly the same predictions and probabilities. Detailed loops live in `src/`; the progress below shows the actual fits.''')
code('''linear_results = fit_family(state, 'logistic')
display(linear_results[['feature_set', 'fit_seconds', 'max_iterations', 'convergence_warning', 'eligible']])''')
code('''tree_results = fit_family(state, 'decision_tree')
display(tree_results[['feature_set', 'fit_seconds', 'convergence_warning', 'eligible']])''')
md('''An eligible fit passed the finite-probability and model-reload checks without a convergence warning. A finite score from a limited fit would still be saved and shown, but excluded from the development choice. Timing is a diagnostic, not the selection score. The fixed plan is unchanged after fitting.''')
md('''## 4. Compare on model-selection rows

All ten experiments predict the same 17,534 rows. We keep the ten-class vocabulary fixed when calculating macro F1, including any class with no predicted rows. Undefined class precision is recorded as zero under the frozen protocol, with predicted support shown in the detailed table.

These are development results. Calibration and official-test outcomes are not part of the comparison.''')
code('''results, differences, decision = compare_results(state)
display(results[['model', 'feature_set', 'macro_f1', 'accuracy', 'weighted_f1', 'eligible']].round(6))
result_figures(state)''')
md('''The first chart compares macro F1 within each model family. The second shows the difference from that model's original-42 reference; bars above zero improved on these rows. The Worms chart gives actual correct counts out of 13. An overall gain does not imply that every class improves.''')
md('''### Look closely at the classes

The F1 tables below show all ten classes for every set. We then show precision, recall, F1, support and correct counts for rare classes. The complete 100-row table contains these measures for every class in every experiment and is saved in `results/tables/06_per_class_metrics.csv`.''')
code('''for family in MODELS:
    display(Markdown(f'**{family}: per-class F1**'))
    per_class = state['classes'][state['classes'].model.eq(family)]
    display(per_class.pivot(index='feature_set', columns='attack_cat', values='f1')
            .reindex(index=SETS, columns=CLASSES).round(4))

rare = state['classes'][state['classes'].attack_cat.isin(['Worms', 'Shellcode', 'Backdoor', 'Analysis'])]
display(rare[['model', 'feature_set', 'attack_cat', 'precision', 'recall', 'f1',
              'support', 'predicted_support', 'correct']].round(4))''')
md('''### Check Attack versus Normal

For binary decisions, an attack is detected whenever the multiclass prediction is not Normal. Attack recall is the share of attacks detected; false-positive rate is the share of Normal rows flagged as attacks; attack precision is the share of flagged rows that are attacks.

We also use **`1 − P(Normal)`** as a continuous attack score. Average precision and the precision–recall curves describe its ranking across possible thresholds. These probabilities have not been calibrated. We do not select a final threshold, and the binary decision derived from the multiclass prediction is not the same as thresholding this score at 0.5.''')
code('''display(results[['model', 'feature_set', 'attack_recall', 'false_positive_rate',
                 'attack_precision', 'false_negative_rate', 'attack_average_precision']].round(5))
detailed_figures(state)''')
md('''The displayed confusion matrices belong to the provisional choice: rows are actual classes and columns are predictions. The count matrix shows how many rows go to each class; the normalized matrix shows the share of each actual class. Read the Worms row with its small support in mind. Both matrices and a figure are saved for every experiment.

Precision–recall curves use raw probability scores. Better ranking here does not demonstrate reliable confidence or a safe operating threshold.''')
md('''## 5. Answer the feature question

Every difference below is **candidate minus reference, using the same model**. Multiply a score difference by 100 to read percentage points. Positive macro F1 is an observed gain; lower false-positive rate is an improvement for that particular binary measure.

We compare all engineered sets with `original42`, each balance with total bytes alone, and the second balance with the corresponding 44-feature set. This tests incremental model benefit rather than reusing Notebook 05's association scores as proof.''')
code('''view = differences[['model', 'reference', 'candidate', 'delta_macro_f1',
                    'delta_accuracy', 'delta_worms_f1', 'delta_worms_correct',
                    'delta_false_positive_rate', 'both_eligible']].copy()
for column in ['delta_macro_f1', 'delta_accuracy', 'delta_worms_f1', 'delta_false_positive_rate']:
    view[column] *= 100
display(view.rename(columns={
    'delta_macro_f1': 'Macro F1 change (pp)', 'delta_accuracy': 'Accuracy change (pp)',
    'delta_worms_f1': 'Worms F1 change (pp)', 'delta_worms_correct': 'Change in correct Worms rows',
    'delta_false_positive_rate': 'FPR change (pp)',
}).round(4))

tradeoffs = state['class_differences']
display(Markdown('**Largest class gains and losses when comparing engineered sets with original42**'))
shown = []
for family in MODELS:
    rows = tradeoffs[(tradeoffs.model == family) & (tradeoffs.reference == 'original42')]
    shown.append(pd.concat([rows.nsmallest(3, 'delta_f1'), rows.nlargest(3, 'delta_f1')]).drop_duplicates())
display(pd.concat(shown)[['model', 'candidate', 'attack_cat', 'support',
                          'delta_f1', 'delta_recall', 'delta_correct']].round(5))''')
md('''The class changes explain where a gain comes from and what it costs. These are paired point comparisons on the same rows, not confidence intervals. A positive result for one fixed model need not generalize to another model or seed. The engineered columns summarize existing measurements, so the benefit, if any, comes from making a relationship easier for that model to use.''')
md('''## 6. Make a provisional development choice

We apply the declared macro-F1 and tie rule. The chosen candidate is a handoff for review, not a final deployment decision. We show its full class results rather than hiding rare-class weaknesses behind an average.''')
code('''display(pd.DataFrame([decision]))
chosen_classes = state['classes'][state['classes'].experiment.eq(decision['experiment'])]
display(chosen_classes[['attack_cat', 'precision', 'recall', 'f1', 'support',
                        'predicted_support', 'correct']].round(5))
print('Full findings appear at the end of the notebook and in the saved report.')''')
md('''**Why the selected score is optimistic:** we used the same model-selection rows to compare ten candidates and choose one. The chosen development score therefore benefits from selection. Final performance is still unknown.

We do not refit on model-selection or calibration rows. Weak rare-class results remain a reason for investigation even when the declared rule selects a stronger overall score. Model settings, seed stability, repeated patterns, conflicting labels and shared network context limit the conclusion.''')
md('''## 7. Save the handoff

| Saved item | Location |
|---|---|
| Fixed plan, exact model settings and trained models | `configs/06_baseline_plan.json`, `models/baselines/06/` |
| Row-linked model-selection predictions and probabilities | `data/processed/06/` |
| Metrics, class tradeoffs, confusion matrices and PR points | `results/tables/06_*` |
| Comparison and diagnostic figures | `results/figures/06_*` |
| Findings and next steps | `docs/06_baseline_models_findings.md` |
| Hashes for the saved artifacts | `data/processed/06/06_artifact_manifest.json` |

Notebook 07 should investigate errors and rare-class tradeoffs first. If needed, declare a separate equal-budget class-weighted or seed-stability study using only train and model-selection rows. When classifier choice is frozen, investigate confidence calibration on the reserved calibration role under Notebook 02. Keep final threshold selection and official-test evaluation for the documented later stages.''')
code('''checks = finish(state)
display(checks)
print('PASS: earlier notebooks, raw sources, assignments and saved artifacts stayed unchanged.')
print('No calibration predictions, confidence calibration, threshold selection, refit or official-test evaluation.')
print('Findings: docs/06_baseline_models_findings.md')''')
md('''## What we learned

The measured findings below state **what improved, what did not, and the provisional choice** using actual outputs. The full comparison and per-class tables preserve every result.

The unanswered questions are stability, rare-class detection, confidence quality and final performance. Any next experiment must be clearly named and keep the frozen roles. These are offline intrusion-detection results; the dataset has no literal Botnet class and does not establish botnet-specific or temporal ground truth.''')
code('''display(Markdown(findings_text(state)))
display(Markdown('**Next:** investigate errors and rare-class tradeoffs; freeze classifier choice before using the reserved calibration role. Final threshold selection and official-test evaluation remain later steps.'))''')

nb=nbf.v4.new_notebook(cells=cells,metadata={
    'kernelspec':{'name':'botnet-research','display_name':'Botnet research (.venv)','language':'python'},
    'language_info':{'name':'python','version':'3.11.9'}})
nbf.validate(nb)
nbf.write(nb,ROOT/'notebooks/06_baseline_models.ipynb')
print(f'Built {len(cells)} cells.')
