# Current baseline results

All numbers below use the same frozen 17,534 model-selection rows. They are development evidence, not final performance. The role previously served ten experiments. All models are unweighted, with family settings chosen by three grouped fitting-only folds and then frozen across sets.

| Model | Feature set | Macro F1 | Accuracy | Worms correct / 13 | Fit seconds |
|---|---|---:|---:|---:|---:|
| logistic | original42 | 0.4169 | 0.7744 | 0 | 33.7 |
| logistic | total_only43 | 0.4176 | 0.7747 | 0 | 39.2 |
| logistic | total_and_byte44 | 0.4242 | 0.7768 | 0 | 40.0 |
| logistic | total_and_packet44 | 0.4180 | 0.7749 | 0 | 73.1 |
| logistic | all45 | 0.4237 | 0.7769 | 0 | 70.7 |
| decision_tree | original42 | 0.6097 | 0.8146 | 6 | 11.6 |
| decision_tree | total_only43 | 0.6059 | 0.8123 | 6 | 12.0 |
| decision_tree | total_and_byte44 | 0.6160 | 0.8136 | 8 | 12.3 |
| decision_tree | total_and_packet44 | 0.6130 | 0.8124 | 7 | 13.1 |
| decision_tree | all45 | 0.6160 | 0.8129 | 8 | 13.4 |
| lightgbm | original42 | 0.6307 | 0.8366 | 8 | 14.2 |
| lightgbm | total_only43 | 0.6207 | 0.8363 | 7 | 15.2 |
| lightgbm | total_and_byte44 | 0.6320 | 0.8356 | 9 | 14.5 |
| lightgbm | total_and_packet44 | 0.6189 | 0.8356 | 7 | 14.5 |
| lightgbm | all45 | 0.6186 | 0.8349 | 7 | 15.0 |

**Provisional choice:** lightgbm with `total_and_byte44`, macro F1 **0.6320**, accuracy 83.56%. The declared .001 tie rule was applied. It prefers fewer source features, then simpler declared families within the tolerance. This tolerance is not statistical equivalence.

For the ensemble, total bytes plus byte balance improves original42 macro F1 by **+0.123 percentage points**. Worms alone contributes +0.243 points, while the other nine classes together contribute -0.120 points. The small net gain does not establish robust feature usefulness. Original42 stays a close comparison reference. Packet balance does not improve this ensemble; all45 is worse than original42. Logistic gets a modest byte-balance gain but still detects no Worms. The tuned tree gets a higher macro F1 with balances, largely alongside rare-class changes; accuracy does not improve.

## Important class tradeoffs

| Class | Precision | Recall | F1 | Correct / support |
|---|---:|---:|---:|---:|
| Analysis | 0.566 | 0.150 | 0.237 | 30 / 200 |
| Backdoor | 0.542 | 0.149 | 0.233 | 26 / 175 |
| DoS | 0.667 | 0.070 | 0.127 | 86 / 1226 |
| Exploits | 0.622 | 0.956 | 0.754 | 3192 / 3340 |
| Fuzzers | 0.751 | 0.765 | 0.758 | 1391 / 1818 |
| Generic | 0.999 | 0.981 | 0.990 | 3923 / 4000 |
| Normal | 0.946 | 0.916 | 0.930 | 5127 / 5600 |
| Reconnaissance | 0.926 | 0.743 | 0.824 | 779 / 1049 |
| Shellcode | 0.715 | 0.779 | 0.746 | 88 / 113 |
| Worms | 0.750 | 0.692 | 0.720 | 9 / 13 |

Binary Attack/Normal: recall 0.9753, precision 0.9609, Normal false-positive rate 0.0845, average precision 0.9967. These cannot replace attack-category metrics: an attack assigned to the wrong category can still count as a binary success. Scores are uncalibrated; no operational threshold is chosen.

DoS→Exploits accounts for 1,108 errors, 964 in category-conflicting original patterns. Analysis→Exploits has 143 errors and Backdoor→Exploits 140. The full confusion/ambiguity tables preserve every count. This locates missing distinctions in recorded inputs and model errors; it does not prove annotation corruption or causal error mechanisms.

On model-selection, one deterministic prediction per original vector has an empirical **accuracy** ceiling of 90.72% from conflicting labels (1,627 unavoidable recorded-row disagreements). The chosen model makes 2,883 errors. This is not a macro-F1 ceiling or a population-independent Bayes bound. Three engineered formulas cannot separate identical original inputs; additional measured context could.

## Search and practical limits

- Two configurations per family × three grouped fitting folds = 18 inner fits. Logistic C=.1/1; tree depth16/leaf5 versus depth24/leaf2; LightGBM 31/63 leaves, 120 fixed rounds. No search expansion after scores. Final experiments: 15 matched comparisons, with five protected logistic fits reused because exact settings match.
- Current selected settings: {"decision_tree": {"max_depth": 24, "min_samples_leaf": 2}, "lightgbm": {"num_leaves": 31}, "logistic": {"C": 1.0}}.
- Single fixed seed; no claim of significance or repeat-seed robustness. Only 13 Worms rows; one correct row changes recall by 7.69 percentage points.
- Timing depends on this environment and concurrent load. Logistic timings come from the preserved exact earlier fits. Complexity counts (coefficients, tree nodes, boosting leaves) have different meanings; bytes are also recorded.
- LightGBM/scikit-learn emits a feature-name warning for unnamed CSR prediction input. Frozen input order, mapping, shape, input hashes and exact reloaded predictions are independently verified; the warning is retained in the notebook record.
- The environment inherits system packages and is not a completely isolated dependency lock. Installed relevant versions are recorded. No dependency prevented completion.
- Group separation addresses recorded-vector overlap, not host or temporal independence. Context-counter causality, literal botnet ground truth, DNS streams and time windows remain unverified/unavailable.
- No calibration fitting, reserved-role refit, final operating threshold or official-test model evaluation. Selected development results are optimistic.

Exact inputs: `data/processed/05/05_total_and_byte44_tree_{role}.npz`; row links: `05_row_links_{role}.csv`; source/encoded maps: `models/preprocessing/05/05_total_and_byte44_tree_columns.json`. Model: `models/revision_v2/06_lightgbm_total_and_byte44.joblib`. Full tables, matched deltas, predictions, settings and manifests are in the corresponding `revision_v2` folders.