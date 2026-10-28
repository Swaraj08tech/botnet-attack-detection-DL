# Notebooks 01–21, version 3 (17–18 October 2026)

All twenty-one notebooks were rewritten so that the **main work is visible in the notebook itself**. Each one has simple headings,
plain-language explanations, worked examples, a "What we see" note after every result, and ends with *What we learned /
What is still uncertain / What comes next*. Results are unchanged: every notebook checks that it reproduces the earlier
frozen split, inputs and model predictions exactly.

| Notebook | Question | Main result |
|---|---|---|
| 01 Data understanding | What is inside the two files? | 42 predictors, clean data, very unbalanced classes (Worms 130 rows), no Botnet class, no DNS-query data or timestamps |
| 02 Repeats, conflicts, fair split | Can repeats or overlap mislead us? | 17% of training rows have conflicting labels; split keeps identical patterns together (140,273 / 17,534 / 17,534), identical to the frozen split |
| 03 EDA and preprocessing | What separates the classes, and how do we prepare inputs? | Analysis/Backdoor/DoS look alike; tree and scaled inputs fitted on train rows only |
| 04 Feature engineering | Can simple new columns describe flows better? | Total bytes, byte balance, packet balance |
| 05 Feature selection | Which features go forward? | Keep all 42 + total bytes + byte balance; drop packet balance (near-copy) → sets `original42` and `total_and_byte44` |
| 06 Baseline models | How good are standard models, and do the new features help? | LightGBM best (0.632 `total_and_byte44`, 0.631 `original42`; identical with any seed). Decision tree 0.612/0.611 as a 3-seed mean (seed-sensitive). Feature gain clear only for logistic regression |
| 07 Neural reference (MLP) | How well does a plain neural network do? | Mean macro F1 0.520 / 0.524 over 3 seeds, below LightGBM; 0/13 Worms; PyTorch Conv1D works on CPU |
| 08 Main model: Conv1D + attention | Does the paper-style network (tabular adaptation) beat the references? | 0.535 ± 0.012 (3 seeds): below LightGBM for every seed, not reliably above the MLP. Attention clearly helps (vs 0.515 without, all seeds, all arms trained to early stopping); BiLSTM and feature order give no reliable gain. Final all-model table: `results/tables/v3/all_models_results.csv` |
| 09 Leakage study | Does a random split inflate scores? | Hardly (+0.004); deduplication biases scores down (−0.013); label conflicts dominate |
| 10 Rare-class handling | Do class weights or SMOTE help? | sqrt weights best: LightGBM 0.632→0.666, network 0.513→0.587; SMOTE no help |
| 11 Multi-scale Conv1D | Kernels 3/5/7? | +0.006, seed-dependent; mainly more stable |
| 12 Cross-attention | Flow ↔ context fusion? | +0.002, seed-dependent; fewer false alarms |
| 13 TCN-BiLSTM | Sequence block over feature tokens? | +0.007, seed-dependent |
| 14 Proposed model | All parts together | 0.603 ± 0.011: beats MLP / reference network, below LightGBM (sqrt) |
| 15 Ablation | Which parts matter? | Only class weighting clearly (−0.066 without); final detector = LightGBM (sqrt) |
| 16 SHAP and LIME | Why each alert? | Exact, faithful SHAP (top column 17× random); top drivers sttl, sbytes, service |
| 17 Calibration and conformal | Trustworthy confidence? | Already calibrated; class-wise sets 0.928 coverage; 95% of conflicting flows get multi-class sets |
| 18 RAG reports | Verified incident reports? | With retrieval 99% of claims, 97-100% of ATT&CK IDs verified; without sources the LLM invents IDs |
| 19 Final results | Official test file, once | Macro F1 0.518, attack recall 98.8%, FPR 32.5% (sttl shift between files); coverage 0.80 |
| 20 Shift-aware detector v2 | Can we adapt to the new traffic without its labels? | Strong shift (AUC 0.896, mostly recent-flow counters); v2 = importance-weighted, tuned LightGBM |
| 21 Final results v2 | Is v2 better on the test file? | Macro F1 0.530 vs 0.518 (clearly better), FPR 30.5% vs 32.5%; label-free selection picked the test-best candidate |
| 22 Detector v3 and score ceiling | Can v2 improve without test labels, and how should the scores be read? | Class offsets fitted on calibration rows, chosen by the importance-weighted rule: macro F1 0.541 (+0.011, 95% CI +0.007 to +0.017), accuracy 74.5%, attack F1 88.8%, false alarms 27.2%, top-2 91.2%. Ceiling of the test file: accuracy 93.6%, macro F1 0.765 (third, disclosed use of the test labels) |
| 23 Final improvement round | Can label-free adaptation beat v3? | Self-training (43,927 pseudo-labelled test flows), two-stage detection, LightGBM + XGBoost ensemble and their combination: none beat v3 on the label-free rule (test macro F1 0.516-0.543 vs 0.541), so v3 is kept and frozen. Per-class ceiling: Analysis 0.16, Backdoor 0.19 even for a model that knows the labels (fourth, final, disclosed use of the test labels) |

## Where things are

- Notebook sources (readable `.py`): `scripts/notebook_sources_v3/`. Build: `scripts/build_v3_notebooks.py`; execute: `scripts/run_v3_notebooks.py`.
- Shared plumbing only (paths, saving, plot style): `src/notebook_helpers.py`.
- Outputs: `results/tables/v3/`, `results/figures/v3/`, `data/processed/v3/`, `models/v3/`.
- Previous notebook versions: `snapshots/notebooks_before_v3_20261017/` (with SHA-256 list). Earlier artifacts in other folders are untouched. The old Notebook 07 verification seal (`scripts/verify_neural_reference.py`) refers to the old notebook bytes and is now historical; the new 07 retrains the six MLPs and checks they match the earlier predictions exactly.

```powershell
.venv\Scripts\python.exe scripts/build_v3_notebooks.py
.venv\Scripts\python.exe scripts/run_v3_notebooks.py
```

## Finding made while rebuilding

LightGBM needs `reg_lambda=1.0` (part of the frozen settings). With the library default of 0 it becomes unstable on this data
(macro F1 falls to about 0.46 after 120 trees). Notebook 06 states this setting explicitly.

## Software versions used for the v3 runs

| Package | Version |
|---|---|
| Python | 3.11.9 |
| OS | Windows-10-10.0.26200-SP0 |
| numpy | 2.4.6 |
| pandas | 3.0.3 |
| scipy | 1.17.1 |
| scikit-learn | 1.9.0 |
| lightgbm | 4.6.0 |
| torch | 2.14.0 |
| matplotlib | 3.10.9 |
| nbclient | 0.11.0 |

All training ran on CPU. Seeds: split 42/43 (Notebook 02), models 42 (plus 43 and 44 for the neural networks).

## Results table for the paper

`results/tables/v3/all_models_results.csv` and `results/figures/v3/08_all_models.png` hold all 13 models (Notebooks 06–08), scored identically on the 17,534 `model_selection` rows, with mean ± std over seeds where a model is seed-sensitive.

`scripts/pretrain_08_arm.py ARM SEED` trains one Notebook 08 network outside the notebook (same code) so several can train in parallel; the notebook then loads the checkpoints.

## Extra requirements for notebooks 10-19

- GPU PyTorch in `.venv` (`torch 2.14.0+cu130`); shared network code `src/flow_networks.py`, final detector `src/final_detector.py`, explanations `src/explain.py`, RAG `src/rag.py`.
- Notebook 18: Ollama running with `qwen2.5:7b` (`ollama pull qwen2.5:7b`), and the knowledge base from `scripts/fetch_knowledge_base.py`.
- Trained networks are cached in `models/v3/nn_checkpoints/`; a full rerun of all 19 notebooks takes about 40 minutes with the cache.
