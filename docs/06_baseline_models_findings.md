# Notebook 06 — Baseline models

**logistic:** original42 macro F1 = 0.4169. Highest eligible score: total_and_byte44 = 0.4242.

- `total_only43`: 0.068 percentage points higher; Worms correct 0/13 versus 0/13.

- `total_and_byte44`: 0.728 percentage points higher; Worms correct 0/13 versus 0/13.

- `total_and_packet44`: 0.102 percentage points higher; Worms correct 0/13 versus 0/13.

- `all45`: 0.679 percentage points higher; Worms correct 0/13 versus 0/13.

**What each balance adds:**

- `total_and_byte44` versus `total_only43`: +0.661 macro-F1 percentage points; +0 correct Worms rows.

- `total_and_packet44` versus `total_only43`: +0.034 macro-F1 percentage points; +0 correct Worms rows.

- `all45` versus `total_and_byte44`: -0.049 macro-F1 percentage points; +0 correct Worms rows.

- `all45` versus `total_and_packet44`: +0.577 macro-F1 percentage points; +0 correct Worms rows.

For all45 versus original42, class F1 changes range from -0.011 points for DoS to +4.694 points for Shellcode.

**decision_tree:** original42 macro F1 = 0.6093. Highest eligible score: original42 = 0.6093.

- `total_only43`: 2.376 percentage points lower; Worms correct 4/13 versus 7/13.

- `total_and_byte44`: 2.359 percentage points lower; Worms correct 4/13 versus 7/13.

- `total_and_packet44`: 2.278 percentage points lower; Worms correct 4/13 versus 7/13.

- `all45`: 2.622 percentage points lower; Worms correct 4/13 versus 7/13.

**What each balance adds:**

- `total_and_byte44` versus `total_only43`: +0.017 macro-F1 percentage points; +0 correct Worms rows.

- `total_and_packet44` versus `total_only43`: +0.099 macro-F1 percentage points; +0 correct Worms rows.

- `all45` versus `total_and_byte44`: -0.262 macro-F1 percentage points; +0 correct Worms rows.

- `all45` versus `total_and_packet44`: -0.344 macro-F1 percentage points; +0 correct Worms rows.

For all45 versus original42, class F1 changes range from -20.238 points for Worms to +0.724 points for Backdoor.


These are observed development differences for these fixed models, not proof of a general improvement. All three additions summarize existing measurements; they add no new measured information. Positive and negative per-class changes, including rare classes, are saved in the class-tradeoff table.


**Provisional choice:** `decision_tree_original42`, macro F1 0.6093. Its gap to the highest eligible score is 0.000 percentage points. The declared tie rule prefers fewer features within 0.1 percentage point, then the linear model. This tolerance is a practical preference, not evidence of statistical equivalence.


**Rare-class caution:** the chosen model detects 7/13 Worms rows. One changed prediction shifts Worms recall by 7.69 percentage points. A good overall macro F1 can coexist with weak rare-class recall; this decision does not override that tradeoff.


**Still uncertain:** seed stability, sensitivity to model settings and class imbalance, performance on new network context, calibration, and the final official-test score. No uncertainty interval or significance claim is made from this single frozen split and seed. Repeated patterns and conflicting labels are retained, and exact patterns are not verified independent events.


**Selection optimism:** the same model-selection rows compared ten candidates and informed the choice. The selected development score is optimistic; final performance is unknown. Accuracy, weighted F1 and binary results are secondary and do not replace the declared macro-F1 rule.

## What the experiments did

Ten unweighted experiments fitted the frozen train rows only: an L2 one-vs-rest logistic classifier on scaled inputs and a depth-limited decision tree on tree inputs, each with all five predeclared feature sets. Settings, seeds and budgets were identical within a model family. No new split, preprocessing fit, class resampling, confidence calibration, threshold selection, refit or official-test evaluation occurred. Calibration matrices and metadata were checked only for integrity; no model predicted their rows.

The main score is row-weighted macro F1 over the fixed ten classes. All class precision/recall/F1 and support, raw and row-normalized confusion matrices, accuracy and weighted F1 are saved. Binary decisions come from the predicted class being different from Normal. Attack average precision and PR curves use 1 − P(Normal). These uncalibrated probabilities are ranking scores, not validated confidence. Multiclass argmax is not equivalent to an attack-score threshold of 0.5, and no final operating threshold was chosen.

## Notebook 07

First review misclassifications and class tradeoffs, especially Worms, Backdoor and Analysis. Inspect how the chosen model confuses attack categories versus Normal. A separately declared equal-budget, class-weighted sensitivity comparison or seed-stability study may be justified; use train and model_selection only and disclose further selection optimism. Do not silently change the present ten-experiment plan or refit on development rows.

When classifier choice and weights are frozen, investigate confidence calibration on the reserved calibration role under Notebook 02. Later choose an operating threshold using calibrated model-selection scores, document its purpose and freeze it before any official-test evaluation. Keep threshold selection and final full-test plus overlap/non-overlap reporting for their documented later stages.

All five sets preserve the original 42 predictors. The two balances were highly correlated, but only the matched model results here address their observed incremental benefit. No Botnet class, botnet-specific ground truth, temporal validation or verified deployment causality is established. Contextual feature availability and unknown host/session dependence remain limitations.

## Files and verification

Models and exact settings: `models/baselines/06/`. Model-selection predictions with source row positions, original labels and ten probability columns: `data/processed/06/`. Metrics, all class tradeoffs and confusion matrices: `results/tables/06_*`. Figures: `results/figures/06_*`. The fixed plan and decision are saved separately. Detailed input, execution and independent model-reload checks accompany the results.

Training-source hash: `bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa`.

Frozen assignment hash: `2c01281b021a7be536bfd0268d53707a3ea74858730fb577db095d618436b168`.

Notebook 05 manifest hash: `970e635c8929ee68b412f0466c615708e4f53bfae2034f9ff0205da491e343c4`.

Declared plan hash: `937c879231ca21f68fb275fde26fc77d7b29321f4ea8b95d076b3185e377e7e3`.

Prior provenance limitation: Notebook 04 and Notebook 05 have old execution-record notebook hashes that differ from current non-source content. Their cell sources match the manifest-verified builders, and all listed artifacts verify. Both hashes for Notebook 05 are recorded in `06_input_integrity.json`; earlier notebooks and execution records remain unchanged.

Methods: [logistic regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html), [one-vs-rest](https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html), [decision tree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html). Exact installed versions are in the plan.
