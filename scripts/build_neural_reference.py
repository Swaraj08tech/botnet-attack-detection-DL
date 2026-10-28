"""Create only Notebook 07; never rebuild earlier notebooks."""
from pathlib import Path
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]
cells=[]
def md(text): cells.append(nbf.v4.new_markdown_cell(text))
def code(text): cells.append(nbf.v4.new_code_cell(text))

md('''# 07. A simple neural reference

**Our question:** Can a straightforward neural network classify these recorded flows well enough to be a fair reference for later Conv1D and attention experiments?

We train a small **MLP**, a network that learns combinations of a row's features. We compare **only two inputs**: the original 42 features, and those features plus total bytes and byte balance. We then compare its results with the saved LightGBM models from Notebook 06.

**Inputs:** Frozen fitting and model-selection rows, saved scaled inputs and column maps, and the existing three fitting-only grouped folds.

**Expected result:** A checked neural reference, an honest comparison, and a clear plan for Notebook 08. A neural model is not guaranteed to beat a boosted tree. Calibration targets and official-test records stay reserved.''')
md('''## 1. Keep the task clear

**Why this matters:** Our project title and the reference paper must not make us claim more than our data support.

The local paper uses network and DNS measurements aligned into endpoint-linked time windows, followed by Conv1D branches, attention and BiLSTM. We inspected its architecture on PDF page 9 and its alignment description on pages 9–10. See [the paper](../docs/reference_paper/Deep%20Learning%20Project%20Reference%20Paper.pdf) and [the existing handoff](../docs/revision_v2/architecture_handoff.md).

Our CSVs have flow summaries and ten intrusion classes. They do **not** have a verified Botnet class, DNS query records, endpoint-linked timestamps or genuine event sequences. `service='dns'` is a category, not a second measured stream.

**Decision:** First establish an ordinary tabular MLP reference. Later Conv1D work will be called a **tabular adaptation**. Column order is not time. An exact reproduction of the paper is currently impossible.''')
md('''## 2. Check our inputs

**Question:** Are we continuing the same experiment with the same rows?

We verify earlier artifact hashes, frozen assignments, original-pattern groups, feature maps and matrix values. We load fitting and model-selection labels. Calibration matrices receive integrity checks only; their labels are not loaded. The official test file is hashed, never parsed.

**Why:** A misplaced row or changed feature order would make the comparison unreliable. The output below gives the actual support and verification result.''')
code('''from pathlib import Path
import sys
import pandas as pd
from IPython.display import display, Markdown
ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p/'src').is_dir() and (p/'notebooks').is_dir())
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from src import neural_reference as neural
s = neural.start(ROOT)
display(pd.read_csv(s['out']['tables']/'07_input_checks.csv'))
display(pd.read_csv(s['out']['tables']/'07_exact_role_overlap.csv'))''')
md('''**Interpretation:** Counts refer to rows and distinct recorded patterns, which are different amounts of evidence. Conflicting labels are preserved. Zero cross-role matches mean the same recorded input cannot appear on both sides of this comparison; this does not establish independence of hosts or capture times.

**Decision:** Use the existing assignments and representations without changing them. Any pre-existing notebook-save differences from the earlier seal are logged; their current bytes are protected during this run.''')
md('''## 3. Understand the MLP

**Question:** What is the network actually doing?

| Part | Simple meaning |
|---|---|
| Input | One flow becomes 193 or 195 numeric values. Category indicators account for the extra columns. |
| Hidden layer 1 | Learn weighted combinations of the inputs, then apply ReLU: keep positive values, turn negative values into zero. |
| Hidden layer 2 | Combine those learned values again. These are learned numbers, not newly measured network traffic. |
| Output | Produce ten class scores. Softmax converts them into ten nonnegative numbers that sum to one. |
| Prediction | Pick the class with the largest number. Normal gives a Normal decision; any other class gives an Attack decision. |
| Loss | Cross-entropy penalizes a low probability for the actual class: `-log(P(actual class))`. L2 adds a penalty for large weights. |
| Training | Adam adjusts weights using small batches. One epoch means a pass through all fitting rows. |

All original categorical columns stay as complete groups. IDs, targets and split metadata never enter the network.

**Example:** The next table uses two invented inputs and invented weights to show one hidden calculation and the ten-class output. It is a teaching example, not a result from the dataset.''')
code('neural.worked_example(s)')
md('''**Interpretation:** A confidently wrong prediction has a large loss. The network learns weights from fitting labels; it does not put those labels into the input.

**Decision:** Use two hidden layers, ReLU, a ten-class softmax output and unweighted cross-entropy with L2 regularization. Probability scores will remain uncalibrated.''')
md('''## 4. Write the small experiment plan

**Question:** How will we choose settings without using model-selection to keep tweaking the network?

We predeclare two capacity/L2 combinations and test each on the **same three grouped folds inside fitting**, using original42. The mean ten-class macro F1 chooses one setting; a difference within 0.001 prefers the smaller setting. Capacity and regularization change together, so this search chooses a configuration rather than isolating their separate effects.

We freeze that setting across both inputs and train each with seeds **42, 43 and 44**. Every run uses **30 epochs**, batch size 512 and Adam at learning rate 0.001. There is no early stopping and no epoch selection from model-selection. No class weighting, resampling or new scaling is added. L2 is the regularizer; this MLP has no dropout.

**Budget:** Six inner fits and six final fits. It matches the saved LightGBM's six-fit search count, although tree and neural training algorithms and costs differ.

PyTorch is installed but Windows Application Control blocks its DLL (WinError 4551). We use the available CPU `scikit-learn MLPClassifier` implementation. This is a real MLP experiment; the environment file records the limitation.''')
code('neural.declare(s)')
md('''## 5. Choose settings inside fitting

**Why:** Slicing a matrix scaled using the whole fitting role would expose inner holdout information during tuning.

For each inner fold, we clone and refit Notebook 03's scaler and category encoder on that fold's training rows. Identical original patterns, including conflicting-label versions, stay together. Each model then predicts the inner holdout. Model-selection and calibration do not take part.

**Read the result:** Higher mean macro F1 is better. The fold range shows variation across these three held-out pattern groups, not a confidence interval.''')
code('neural.tune(s)')
md('''## 6. Train the frozen neural reference

**Question:** What happens with the chosen settings on all fitting rows?

We now use the exact saved, train-fitted scaled inputs from Notebook 05. Each feature set gets the same chosen network settings, three seeds and 30 epochs. Only the input width changes from 193 to 195. We save and reload every model before reporting its model-selection predictions.

**Read the result:** Macro F1 treats all ten classes equally. Accuracy counts all correct rows together, so frequent classes matter more. Worms correct counts must be read out of only 13 rows.''')
code('''reference_spec = neural.make_model(s['config'], seed=42)
display(reference_spec)
neural.final_fits(s)''')
md('''## 7. Compare with the saved trees

**Question:** Does a neural model improve the existing benchmark, and do the two added features help?

We reload the exact Notebook 06 LightGBM models and reproduce their predictions and metrics on the same model-selection rows. LightGBM receives its saved tree inputs; the MLP receives its saved scaled inputs. We do not retrain or retune the trees.

Compare each MLP with the tree on the **matching feature set**, and compare 44 versus 42 within each MLP seed. All three neural seeds are reported; none is chosen because it scored best. Seed 42 is the predeclared reference checkpoint for figures and later ablations.

**Why:** An aggregate gain can hide a class becoming much worse. We show Analysis, Backdoor and DoS being predicted as Exploits, and show Worms counts. One extra correct Worms row changes recall by 7.69 percentage points.''')
code('''neural.compare(s)
from src.neural_reference_review import explain_results
explain_results(s)''')
md('''### Read all classes and binary decisions

The table below shows all ten classes for the fixed seed-42 reference on both inputs. Full results for every seed are also saved. **Precision** asks how many predictions of a class were correct; **recall** asks how many actual rows of that class were found. F1 balances the two.

Binary Attack/Normal decisions are derived from the multiclass prediction. We report attack recall, attack precision and the fraction of Normal rows falsely flagged as attacks. Average precision summarizes the precision–recall curve using `1 - P(Normal)`; no operating threshold is selected here.''')
code('''display(s['per_class'].query('seed == 42')[['feature_set','attack_cat','precision','recall','f1','correct','support']])
binary_columns = ['model','feature_set','seed','attack_recall','attack_precision','false_positive_rate','attack_average_precision','false_positive','false_negative']
display(pd.concat([s['metrics'][binary_columns], s['baseline'][binary_columns]], ignore_index=True))''')
md('''### Check the cost

**Question:** How much computation does this reference need?

Training times include the final fit, not the inner search. LightGBM training times come from Notebook 06. Prediction times are newly measured as the median of three warmed-up passes on the same computer, with two library threads. Milliseconds per row are batch averages, not end-to-end live-network latency. Model bytes include the saved estimator state, which differs across libraries.''')
code('''cost = ['model','feature_set','seed','fit_seconds','inference_seconds','inference_ms_per_row']
display(pd.concat([s['metrics'][cost],s['baseline'][cost]],ignore_index=True))
display(s['metrics'][['feature_set','seed','parameters','model_bytes']])''')
md('''**Interpretation:** A fair reference includes both performance and cost. Several seeds can show that a small score difference is unstable, but cannot prove statistical significance. Identical recorded inputs with conflicting labels still constrain a deterministic model; a larger network cannot invent missing measurements.

**Decision:** Keep the full class-level results and both matching tree controls. Defer class weighting until it can be tested as a separate intervention with a weighted tabular control.''')
md('''## 8. See one real prediction

**Question:** What does a trained model return for a single flow?

We take the first model-selection row in its existing order and show the ten scores from the predeclared seed-42 checkpoint. The actual class is shown for explanation, not supplied to the network. This example is not chosen for being correct.''')
code('neural.real_example(s)')
md('''## 9. Check the saved results

**Why:** A result must survive saving and reloading, with the same inputs and class order.

We reload all six final models and row-linked prediction files. We check input width, exact feature order, class order, labels, probabilities, predictions and recomputed metrics. We verify the fitting-only preprocessor provenance and unchanged earlier files, including both raw CSVs.

**Decision:** The next notebook may use these artifacts only if these checks pass.''')
code('neural.verify_saved(s)')
md('''## 10. Make the next step clear

**Question:** What does this experiment justify?

The summary below states the actual outcome. It chooses the MLP feature set by mean macro F1 across three seeds, preferring original42 within 0.001. It keeps seed 42 as the checkpoint rather than selecting the most favourable seed.

**Notebook 08:** Test a clearly labelled **tabular Conv1D adaptation** on these same two representations. Keep frozen rows, fitting-only grouped tuning, class order, unweighted training, fixed seeds and matching MLP/LightGBM controls. Predeclare source-feature order and order permutations, keeping categorical blocks intact. Declare settings, training budget and any deviations before fitting. Compare class-level changes and costs as well as macro F1. Column adjacency is not time; a DNS or temporal branch needs actual additional measured data.

**Limit:** Model-selection has already been used to compare earlier candidates. These are optimistic development scores. Calibration and final-test work remain for later notebooks.''')
code('neural.finish(s)')

nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Botnet research (.venv)','language':'python','name':'botnet-research'},'language_info':{'name':'python','version':'3.11.9'}})
path=ROOT/'notebooks/07_neural_reference_model.ipynb'
if path.exists(): raise RuntimeError('Notebook 07 already exists; do not overwrite executed evidence with a builder')
nbf.write(nb,path)
print('Created',path,len(cells),'cells')
