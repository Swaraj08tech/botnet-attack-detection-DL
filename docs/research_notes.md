# Research Notes

## 2026-10-11 — Project initialization

**Project:** Botnet Attack Detection.

**Base Research:** Hybrid Network Flow + DNS deep-learning IDS, as described in the supplied project instructions. The actual reference methodology is pending PDF review.

**Reference Model:** Conv1D + Feature Fusion + Multi-Head Attention + BiLSTM + SHAP/LIME, provisionally described by the project plan; not yet verified from the paper.

**Proposed Detection Direction:** Multi-Scale Conv1D + Cross-Attention + TCN-BiLSTM + Temporal Attention. This is a research hypothesis, with no claimed performance benefit.

**Later Extension:** Calibrated Confidence + XAI Evidence + Threat Intelligence RAG + Evidence Verification + Incident Reporting.

**Purpose:** Initialize the workspace and preserve source assets before beginning research.

**Inspected:** The project root and all existing subdirectories. The only source file found was `research paper source/Deep Learning Project Reference Paper.pdf` (5,705,220 bytes). No CSV files were present.

**Changes:** Created the planned directory structure and initial README, dependency list, Git exclusions, and research diary. Relocated the original PDF to `docs/reference_paper/Deep Learning Project Reference Paper.pdf` without renaming it. Saved before/after SHA-256 values in `docs/initialization_manifest.json`.

**Dataset:** Official UNSW-NB15 training and testing CSVs are missing from the workspace. `data/raw/` remains empty. No dataset contents, column names, or split membership have been inspected or modified.

**Preprocessing / Model / Hyperparameters / Metrics / Results:** Not applicable; no experiments or implementations have begun.

**Verification:** Confirmed the PDF hash is unchanged and all planned directories exist. No notebook files or Python implementation files were created.

**Problems:** Dataset identification and relocation cannot be completed until the two source CSVs are available. DNS feature availability and temporal sequence construction remain unresolved and require evidence from the paper and dataset.

**Decision:** Stop after initialization and wait for approval. Do not preprocess, train, or implement RAG.

**Next Step:** After approval and dataset availability, verify official dataset provenance and file identities, review the reference paper, then agree on a data-understanding stage and assess whether its modalities and temporal design can be reproduced with the available data.

**Suggested Git commit:** `chore: initialize botnet detection research workspace`

## 2026-10-11 — Dataset placement

**Purpose:** Place the two user-provided CSVs in the raw data directory.

**Files:** `data/raw/UNSW_NB15_training-set.csv` (32,293,018 bytes) and `data/raw/UNSW_NB15_testing-set.csv` (15,380,800 bytes).

**Verification:** Copied from the provided Downloads paths, preserving filenames. Each destination SHA-256 matches its source. Downloads originals remain in place. Updated the initialization manifest and README to record availability.

**Scope:** Placement only. No preprocessing, schema inspection, training, or independent provenance verification performed. The earlier missing-dataset issue is resolved.

**Next Step:** Await approval for paper review and data understanding.

## Experiment 01 — Dataset Understanding and Validation

**Date:** 2026-10-11.

**Purpose:** Inspect the supplied official-split CSVs separately, audit quality and evaluation risks, and determine the next research question before preprocessing.

**Dataset:** User-provided UNSW-NB15 files in `data/raw/`; hashes match the placement manifest. Filename and shape are consistent with the intended split, but official source provenance has not been independently verified.

**Environment:** Designed for VS Code + Local Python Virtual Environment. Executed with Python 3.11.9 through Jupyter/nbclient in project `.venv`, not through the VS Code UI. The environment inherits existing system site packages; missing notebook dependencies were installed locally. Recorded versions: pandas 3.0.3, NumPy 2.4.6, matplotlib 3.10.9, ipykernel 7.4.0, nbformat 5.11.1, nbclient 0.11.0. See `results/tables/01_environment_versions.csv`.

**Train Dataset:** `UNSW_NB15_training-set.csv`, 175,341 rows and 45 columns.

**Test Dataset:** `UNSW_NB15_testing-set.csv`, 82,332 rows and 45 columns.

**Key Findings:** Both splits have 42 candidate predictors: 39 numeric and three categorical (`proto`, `service`, `state`), plus `id` and targets `label` and `attack_cat`. There are two binary classes and ten attack categories including Normal. Neither split has missing cells, infinite values, negative numerical values, or constant columns. Target/category consistency checks found zero contradictions. No schema, column-order, or dtype mismatches were observed. No exactly redundant columns were found.

**Potential Problems:** The `service` placeholder `-` occurs 94,168 times in training and 47,153 times in testing; it is not pandas NaN, and its meaning requires source documentation. Test-only states are `ACC` and `CLO`; training-only states are `ECO`, `PAR`, `URN`, and `no`. Protocols `icmp` and `rtp` occur only in training. Test FTP-related fields `is_ftp_login` and `ct_ftp_cmd` have at least 99% dominant values. Many numerical fields are strongly skewed or contain very large values; the maximum training `sload` is about 5.988e9, requiring unit/context review rather than automatic outlier removal. Largest standardized numerical mean differences include `tcprtt`, `dttl`, `synack`, `swin`, and `ackdat`.

**Duplicate and Target Ambiguity Findings:** Full-row duplicate counts are zero because IDs are unique. Excluding only ID gives 67,601/26,387 excess duplicate labeled rows in train/test. Excluding ID and both targets gives 74,301/28,386 excess duplicate feature rows. Within-split feature groups with binary target conflicts number 229 in training and six in testing; attack-category conflicts number 1,772 and 575. Across splits, exact tuple verification found 1,302 shared distinct feature vectors involving 10,279 training rows and 8,541 testing rows. Of those shared vectors, 548 have conflicting target combinations. These counts do not prove that underlying network events are identical.

**Leakage Risks:** Including either target as a predictor would expose class information. `attack_cat` deterministically maps to the binary target in these data. IDs are unique, sequential and monotonic within each split; ID-label association is about 0.727 in training and -0.387 in testing. Only four adjacent binary-label transitions occur in test file order, compared with 8,171 in training. Treat row ordering as unestablished, not as verified temporal order. Cross-split repeated vectors and target conflicts require an explicit evaluation protocol. Low-cardinality mapping diagnostics do not establish feature provenance or prove leakage.

**Class Imbalance:** Attack share is 68.06% in training and 55.06% in testing. Multiclass majority/minority ratios are 430.77:1 and 840.91:1. Worms has 130/44 records; Shellcode 1,133/378; Backdoor 1,746/583. Analysis has 2,000/677 records and is below 1% in test. Rare categories will need class-wise reporting and careful validation design.

**Preprocessing Requirements:** Decide task/target policy and validation protocol before fitting anything. Fit any future transforms and encoders on the training partition only, handle unknown categorical values, review placeholder meanings, and investigate feature repeats and target ambiguity before cleaning decisions. No scaling, encoding, rebalancing, row deletion, feature selection, windowing, or modelling was performed.

**Architecture Questions:** No columns are named for DNS/domain/query/entropy, and no explicit start/end timestamps are present under the checked names. DNS modality and temporal sequence construction are therefore unresolved. This is a name-based observation, not a complete semantic audit. There is no literal Botnet attack category, so botnet-specific ground truth is not established by this audit. The reference paper still requires methodological review before reproduction.

**Validation:** All 15 code cells executed without errors using the local Jupyter kernel, from both `notebooks/` and the project root. Raw SHA-256 hashes and both in-memory dataframes remained unchanged. Three targeted tests passed for overlap counting, exact verification under hash collisions, and matching values under differing numeric dtypes. Figures were visually reviewed. Artifact verification reconciled class-count totals and percentages against split sizes, checked notebook format and successful cell outputs, and confirmed that interim/processed directories remain empty with only Notebook 01 present. The initial sandboxed kernel run stalled; the completed kernel executions ran outside the sandbox with approval.

**Artifacts:** `notebooks/01_data_understanding.ipynb`; reusable helpers in `src/data/` and `src/utils/`; required `feature_summary.csv`, `dataset_summary.csv`, and both `class_distribution_*.csv` tables; additional audit tables and execution/integrity records in `results/tables/`; three `01_*.png` figures; and `docs/01_dataset_findings.md`. README and requirements were updated for local execution.

**Decisions:** Preserve the official split and all original rows. Treat duplicate vectors and target conflicts as investigation findings, not automatic cleaning instructions. Do not use this test audit to tune later models. Stop for review after Notebook 01.

**Next Recommended Step:** Notebook 02 — Duplicate, Target Ambiguity, and Leakage Investigation, alongside reference-paper and feature-provenance review. Establish defensible evaluation and modality assumptions before preprocessing. Notebook 02 has not been created.

**Suggested Git Commit:** `research: complete UNSW-NB15 data understanding and validation`. No commit performed.

## Experiment 02 — Duplicate, Target Ambiguity, and Leakage Investigation

**Date:** 2026-10-11.

**Purpose:** Reproduce Notebook 01's evidence, investigate label ambiguity and feature semantics, and establish a reproducible validation/evaluation protocol before preprocessing.

**Dataset and Environment:** The original training/testing CSVs, loaded separately. Python 3.11.9 in the existing project `.venv`, Jupyter/nbclient execution. Exact versions, source hashes, configuration and output checksums are saved in `results/splits/02_split_manifest.json`. No VS Code UI execution was observed.

**Source Review:** Inspected Notebook 01's code and stored outputs, its findings, saved tables and reusable code. All 18 within-split audit checks and all six prior cross-split counts reproduced. Primary dataset documentation and the creators' paper were consulted for intended feature definitions; access limitations and unverified implementation details are recorded in `docs/02_feature_semantics_sources.md`.

**Correction to Experiment 01 Diary:** Its statement that no exactly redundant columns were found was incorrect. Notebook 01's actual output shows `is_ftp_login` and `ct_ftp_cmd` are exactly equal in training. Notebook 02 confirms that equality and finds eight unequal rows in testing. Notebook 01 and its outputs remain unchanged.

**Duplicates:** Training contains 101,040 distinct predictor vectors, with 84,049 rows participating in 9,748 repeated groups and 74,301 excess rows. Testing contains 53,946 vectors, with 34,704 rows participating in 6,318 repeated groups and 28,386 excess rows. Largest group sizes are 415 and 88. Class-wise counts distinguish rows, distinct vectors and excess repetitions.

**Target Ambiguity:** Binary conflicts affect 940 training rows in 229 groups and 14 testing rows in six groups. Multiclass conflicts affect 30,528 training rows in 1,772 groups and 8,809 testing rows in 575 groups. Conflicting combinations, pairwise group counts, original-row examples and a heatmap are saved. No conflict was removed or majority-relabeled.

**Cross-split Overlap:** Exact tuple comparisons identify 1,302 shared vectors. For multiclass labels, 754 vectors are consistent (7,624 training/3,564 testing rows), and 548 are conflicting (2,655 training/4,977 testing rows). For binary labels, 1,023 vectors are consistent and 279 conflicting. The full test set has 8,541 overlapping and 73,791 non-overlapping rows relative to the official training file. Actual fit-role overlap covers 6,783 test rows; another 1,758 match validation roles (930 model-selection and 828 calibration). These are diagnostic populations, not revised test sets.

**Semantics and Leakage:** There are 22 nonbinary `is_ftp_login` rows in training and four in testing. `service='-'` appears 94,168/47,153 times. Source definitions describe a fallback service marker and intended binary login indicator, but extraction/version behavior remains unverified. Several contextual counters require cross-flow information; no endpoint/timestamp replay or past-only causality check is possible from these subset CSVs alone. ID plots show class ordering, not verified temporal order. Zeros, skew and extreme-value examples were examined without transformation. Strong predictive association is not treated as proof of leakage. No literal Botnet class exists.

**Policy and Class Effects:** Retain every original row and label; remove zero rows from every class. Quantified rejected alternatives are saved in `02_policy_class_count_effects.csv`. For example, removing all multiclass-conflict groups would remove 1,441 of 2,000 Analysis training rows and 1,340 of 1,746 Backdoor rows, demonstrating a substantial change in the task's class composition. Those exclusions were not applied.

**Frozen Assignments:** All 42 predictors define exact groups without targets or ID. Prespecified fold 0 of five-fold StratifiedGroupKFold, shuffle=True, seed=42, creates 140,273 fit rows (80,848 groups) and 35,068 validation rows. Prespecified fold 0 of two grouped validation folds, seed=43, reserves 17,534 calibration rows (9,934 groups); 17,534 rows (10,258 groups) are reserved for model selection and thresholds. Targets are used only for stratification, not grouping; no test data or overlap mask is passed to the splitter. No seed/fold search or performance tuning occurred.

**Rare-Class Support:** Worms has 104 fit rows across 102 distinct vectors, 13 model-selection rows across 13 vectors, and 13 calibration rows across 12 vectors. All ten categories are present in each role. The maximum multiclass validation distribution difference is about 0.001707 percentage points. Distinct vectors do not prove independence of network events or contextual windows.

**Validation:** The 16 code cells executed successfully with zero errors. Seven targeted unit tests passed. Two calls to the split function yielded identical assignments; independent exact-value checks found zero intersections between every pair of roles. Raw files, Notebook 01 and all protected prior tables/figures/findings retained their hashes, and reloaded dataframes equal the originals. Six figures were visually reviewed. No interim/processed dataset or model was created.

**Evaluation Decisions:** Fit preprocessing and classifiers only on the fit role. Use model-selection rows for architecture, early stopping and threshold choice; reserve calibration rows for later temperature scaling. Report fixed-vocabulary macro F1, per-class precision/recall/F1/support and confusion matrices; include binary attack recall, FPR, FNR, precision, PR curve and average precision. Evaluate the complete supplied test set with a frozen pipeline, and report overlap/non-overlap diagnostics without test-driven tuning. Do not refit on validation before this final evaluation.

**Next Recommended Step:** Review Notebook 02, then Notebook 03 should establish feature provenance/prediction-time requirements and a preprocessing contract using the saved assignments. Resolve service/FTP semantics, unknown-category handling and numeric treatment using training evidence; inspect the separate reference architecture before choosing modalities. Re-audit exact overlaps if later justified transformations collapse vectors. Notebook 03 has not been created.

**Artifacts:** `notebooks/02_duplicate_target_ambiguity_and_leakage_investigation.ipynb`; `docs/02_duplicate_target_ambiguity_and_leakage_protocol.md`; `docs/02_feature_semantics_sources.md`; configuration in `configs/`; diagnostic tables and six figures prefixed `02_`; assignments, test masks and manifest in `results/splits/`; reusable grouping/splitting modules and meaningful invariant tests.

**Suggested Git Commit:** `research: investigate duplicates and freeze grouped validation protocol`. No commit performed.

## Template for future experiments

- Experiment:
- Purpose:
- Dataset:
- Preprocessing:
- Model:
- Hyperparameters:
- Metrics:
- Results:
- Observations:
- Problems:
- Decision:
- Next Step:

Create numbered notebooks only when a research stage begins. Once completed, clean the notebook, retain explanatory markdown, promote stable reusable code to `src/`, save figures and metrics outside notebook outputs, update this diary, and suggest a Git commit message.


## 2026-10-12 — Research workflow revision v2

Reviewed the local reference paper and recorded data/label/sequence gaps before implementation. Preserved a verified 379-file snapshot and all earlier artifacts. Expanded 02–06 with transparent examples and fitting-only investigations; reproduced transforms and feature/association evidence. Added 18 grouped inner fits and an unweighted ensemble to the five-set baseline experiment. Provisional development choice: lightgbm/total_and_byte44, macro F1 0.6320; small engineered gain is rare-class-sensitive. Calibration and final test remain reserved. Current reports, manifests and independent verification are in revision_v2 folders. See workflow_map.md, change_log.md, baseline_summary.md and architecture_handoff.md there.
