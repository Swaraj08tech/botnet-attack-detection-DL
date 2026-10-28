"""Author the reviewed Notebook 07; does not fit or select any model."""
from pathlib import Path
import nbformat
ROOT=Path(__file__).resolve().parents[1]
cells=[]
def md(s):cells.append(nbformat.v4.new_markdown_cell(s))
def code(s):cells.append(nbformat.v4.new_code_cell(s))

md('''# 07. A bounded neural reference

**Question:** Can a small MLP provide a fair neural reference for later tabular Conv1D experiments?

**Answer:** Yes, as a checked reference. It is **not the best-performing model**. Three-seed mean macro F1 is **0.5198** with original42 and **0.5241** with total_and_byte44. Matching saved LightGBM scores are **0.6307** and **0.6320**. All six MLP runs find **0/13 Worms** rows.

We keep the original two feature sets, frozen roles, three fitting-only grouped folds, selected 128/64-unit MLP, L2 alpha 0.01, **30 epochs** and seeds **42/43/44**. This review reloads the saved experiments; it does not retune or retrain them. Calibration labels and official-test records are unused.''')
md('''## 1. Know what the data support

We classify completed flow records into ten `attack_cat` classes. The CSVs contain 42 original flow predictors, but no verified Botnet class, raw DNS query transactions, endpoint-linked timestamps or genuine event sequences. ID and column order are not time.

The [local reference paper](../docs/reference_paper/Deep%20Learning%20Project%20Reference%20Paper.pdf), particularly PDF pages 9–10, needs measured network/DNS streams aligned into endpoint-linked windows. Those streams cannot be reproduced here. See the [architecture handoff](../docs/revision_v2/architecture_handoff.md).

**Decision:** Use an ordinary tabular MLP now. Notebook 08 will be a clearly labelled **tabular Conv1D adaptation**. DNS, temporal and botnet-specific claims require additional measured data.''')
md('''## 2. Check the frozen inputs

We check raw and assignment hashes, row links, class order, original predictor groups, feature names and matrix values. The categorical indicators explain why 42/44 source features become **193/195 encoded inputs**. IDs, targets and audit metadata are excluded.

Calibration inputs receive alignment and overlap checks only, with no calibration labels. Zero exact overlap protects against matching recorded inputs across roles; it does not prove host or capture-time independence.''')
code('''from pathlib import Path
import sys
import pandas as pd
from IPython.display import display, Markdown, Image
ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p/'src').is_dir() and (p/'notebooks').is_dir())
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from src import neural_reference_finish as review
from src import neural_reference as neural
s = review.load(ROOT)''')
md('''## 3. Understand the MLP

| Part | What happens |
|---|---|
| Input | One recorded flow supplies 193 or 195 scaled numbers, including category indicators. |
| Hidden layer 1 | 128 units learn weighted combinations. ReLU keeps positive values and replaces negative values with zero. |
| Hidden layer 2 | 64 units combine those learned values again. |
| Output | Ten scores become ten softmax probabilities, in the fixed class order. |
| Prediction | The largest score chooses the class. Any class other than Normal gives an Attack decision. |
| Loss | Cross-entropy penalizes low probability for the actual class; L2 discourages large weights. |
| Training | Unweighted Adam, learning rate 0.001, batch 512, exactly 30 passes through fitting rows. |

Labels supervise training but never enter the predictor matrix. Probability scores are **not calibrated confidence**. The invented two-input example below illustrates weighted sums and softmax; real models have two hidden layers and all encoded inputs.''')
code('neural.worked_example(s)')
md('''## 4. Keep the original small search

Only two jointly declared capacity/L2 settings were searched: **64/32 units with alpha 0.0001**, and **128/64 with alpha 0.01**. Each used the same three existing grouped folds inside fitting, original42, seed 42 and 30 epochs. Capacity and regularization change together, so this is not a separate causal test of either.

Preprocessing was refitted only on each inner training fold. Conflicting-label versions of identical original patterns stayed together. Mean ten-class macro F1 selected setting 1; the declared 0.001 tie rule preferred the smaller setting. The setting was then frozen for both feature sets.

The next cell audits saved scores and provenance. It refits only the fold-local preprocessors to verify their saved transforms. **No predictive model is fitted or settings changed.**''')
code('review.audit_search(s)')
md('''### Important training limit

This is a **two-configuration, 30-epoch reference**, not exhaustive MLP optimization. A falling training loss proves neither generalization nor convergence. Completing 30 epochs is different from convergence.

We will not add epochs or search settings after seeing model-selection. If a stronger MLP is worth investigating later, create a **separately versioned, predeclared experiment**. Choose capacity, regularization, epochs or stopping rules only in fitting-only grouped folds, with fold-local preprocessing. Any early-stopping rows must also come from fitting and keep original patterns together. Report its new budget and compare it as a separate experiment.''')
md('''## 5. Reload all six neural runs

The original fits used the full fitting role, the same chosen settings and seeds **42, 43 and 44** on each input. Seed 42 was declared as the reference checkpoint; we do not select the seed with the highest score.

We now reload every model, reproduce its probabilities and class predictions, and recompute its metrics, per-class counts and confusion matrices on the existing model-selection rows. The original loss curves and 30-epoch sample counts must also match. All saved models and experimental result files remain byte-exact.''')
code('review.verify_models(s)')
md('''## 6. Verify the matching tree controls

The controls are the exact saved, unweighted Notebook 06 LightGBM models on original42 and total_and_byte44. Their own saved tree inputs are used; the MLP uses scaled inputs. We check both representations' source names and the same row/class order. We neither fit nor retune the trees.

Higher ten-class macro F1 is better. Accuracy and weighted F1 are supporting measures; they can hide missed minority classes.''')
code('review.verify_controls(s)')
md('''## 7. Read the evidence

The tables retain **all six MLP runs and all ten classes**, including correct counts and support. They show paired seed differences, errors toward Exploits, binary false alarms and misses, and costs. All raw/normalized confusion matrices and precision–recall points remain in [the preserved result tables](../results/tables/neural_reference_v1/).

Attack/Normal decisions come from the multiclass prediction. Average precision uses `1 - P(Normal)`; no operating threshold is selected.

**Costs below are the existing recorded costs**, not new timings from this replay. Earlier VS Code history shows different inference timings; benchmarks vary with runtime/background activity. A batch average is not live-network latency. Replaying predictions here does not replace those historical measurements.

The three training seeds vary initialization and shuffle order. Their range is not a confidence interval, and the tree controls have one saved seed.''')
code('review.show_results(s)')
md('''### What improved, and what did not?

The two added features raise mean MLP macro F1 by **0.44 percentage points**. The paired changes are **+1.73, -1.20 and +0.78 points**: the sign reverses across seeds. This does not establish a stable improvement on new traffic.

For 44 features and fixed seed 42, the MLP correctly finds **271/1,226 DoS** rows versus **86/1,226** for LightGBM. But it finds fewer Analysis (**21 vs 30**), Backdoor (**15 vs 26**), Exploits (**2,942 vs 3,192**), Fuzzers (**1,304 vs 1,391**) and Worms (**0 vs 9**). A benefit for DoS is not a benefit everywhere.

All six MLP runs find **0/13 Worms** rows. One changed Worms prediction would move recall by 7.69 percentage points, so this tiny support limits rare-class conclusions. Repeated or conflicting recorded patterns also limit what deterministic models can separate.

**Decision:** LightGBM remains the stronger overall development benchmark. Keep the MLP as a useful bounded neural reference, and keep the full class tradeoffs visible. Model-selection has already compared earlier candidates, so these scores are optimistic development evidence.''')
md('''## 8. Check the runtime for Notebook 08

Notebook 07 succeeded with scikit-learn on CPU. Its earlier PyTorch probe recorded **WinError 4551**. This is an environment issue, separate from MLP quality.

The new check below uses only a synthetic two-row tensor. It verifies CPU Conv1D forward/backward computation, a single optimizer step, finite values and checkpoint reload. It never accesses project records. No Windows security control was changed. See [the runtime investigation and next steps](../docs/neural_reference_review_v2/07_runtime_readiness.md).''')
code('''from src.research_revision.common import read_json
display(pd.DataFrame([read_json(ROOT/'docs/neural_reference_review_v2/07_runtime_check.json')]))''')
md('''## 9. Handoff to Notebook 08

Use **original42 and total_and_byte44**, the same 193/195-column maps, saved scaled values, ten-class order, fitting/model-selection rows and fitting-only grouped folds. Keep all categorical blocks intact. Declare a semantic source-feature order and at least two order permutations before experiments; adjacency is not time.

Compare the tabular Conv1D adaptation with both saved seed-42 MLP checkpoints and the matching LightGBM controls. Use seeds 42/43/44 for matched neural repeats, unweighted training and a predeclared bounded search/epoch budget. Any dtype, preprocessing or budget change must be declared and checked; learned preprocessing must fit only inner training during tuning. Report fixed-ten-class macro F1, every class/count, binary false alarms and misses, seed variation and costs. A future attention ablation needs a matched backbone and budget. Weighting needs a separate matched weighted tabular control.

Keep calibration labels and official-test outcomes reserved. This is a tabular flow-classification experiment, not a measured DNS/event-time or botnet-specific reproduction.''')
md('''## 10. Save an honest verification record

The next cell saves a concise findings report and review checks. Its in-kernel checks are followed by a fresh-kernel execution record and the **independent verifier**.

A seal applies only to the recorded saved bytes. A later VS Code save may change metadata, formatting or outputs and make that seal stale. **This notebook cannot certify its own future saves.** Use the read-only live check and refresh procedure in [the verification guide](../docs/neural_reference_review_v2/07_verification_guide.md). The old seal and starting evidence were archived before this review.

After any later save: run the Notebook 07 tests, then `scripts/run_neural_reference.py`, then `scripts/verify_neural_reference.py`. Never update a hash by hand to make a check pass.''')
code('review.finish(s)')
nb=nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Botnet research (.venv)','language':'python','name':'botnet-research'},'language_info':{'name':'python','version':'3.11.9'}})
path=ROOT/'notebooks/07_neural_reference_model.ipynb';nbformat.write(nb,path)
print('Authored reviewed Notebook 07:',len(cells),'cells; no models fitted.')
