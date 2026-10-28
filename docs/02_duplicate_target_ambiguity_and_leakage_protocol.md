# Notebook 02 — Findings and evaluation protocol

### 1. Integrity — interpretation

**Observation:** All 18 initial audit checks reproduce Notebook 01; source and prior notebook hashes are recorded.

**Hypothesis / limitation:** Matching the local placement manifest proves preservation, not independent official-file authentication.

**Decision:** Use the supplied files separately. Recheck all protected hashes at completion; cross-split counts are independently confirmed in section 5.

### 2. Targets — interpretation

**Observation:** Both files have binary labels and ten attack categories including Normal; no literal Botnet label and no Normal/binary inconsistency are observed.

**Hypothesis / limitation:** An attack category cannot be renamed Botnet without event-level provenance. Intended feature definitions do not establish extractor correctness or causal availability.

**Decision:** Use an offline IDS task description. Preserve both labels as outcomes only; exclude ID from prediction and grouping.

### 3. Repetition — interpretation

**Observation:** Train: 101,040 unique vectors; 84,049 participating rows and 74,301 excess rows. Test: 53,946 unique vectors; 34,704 participating rows and 28,386 excess rows.

**Hypothesis / limitation:** Repeated aggregate measurements can arise from separate events, feature quantization or extraction behavior; identity of source events is unproven.

**Decision:** Retain rows. Use complete feature groups as the unit of partitioning and report both row support and distinct-vector support.

### 4. Ambiguity — interpretation

**Observation:** Multiclass conflicts affect 30,528 train rows in 1,772 groups and 8,809 test rows in 575 groups. Binary conflicts affect 940/14 train/test rows.

**Hypothesis / limitation:** Omitted event context, label alignment, discretization or extractor issues could explain ambiguity. The empirical minimum-error calculation only constrains deterministic predictions for identical vectors in this sample; it is not a model result or a population error bound.

**Decision:** Keep conflicting versions in the same partition and preserve their labels. Do not replace targets with majority labels.

### 5. Cross-split overlap — interpretation

**Observation:** All six prior overlap counts reproduce. 1,302 exact vectors involve 10,279 train rows and 8,541 test rows; 73,791 test rows do not match the supplied training file.

**Hypothesis / limitation:** Conflicting vectors cannot be explained by a single memorized target. Even nonoverlap by exact vector does not establish host, session or temporal independence.

**Decision:** Preserve all test rows. Report full-test results as primary, and overlapping/non-overlapping populations with their own class support as diagnostics.

### 6. Semantics and leakage — interpretation

**Observation:** Training FTP columns are exactly equal; testing has 8 unequal rows. Nonbinary login counts train/test are 22/4. service='-' counts are 94,168/47,153. ID plots reveal ordered class blocks.

**Hypothesis / limitation:** Notebook 01's saved output correctly identified training FTP equality; its research diary incorrectly claimed no exactly equal columns. This is a documentary correction, not a dataset change. Shared history in contextual counters, extreme rates on short flows, and applicability-related zeros remain hypotheses requiring extractor evidence.

**Decision:** Keep all 42 predictors for the grouping contract. Preserve marker and FTP values; do not equate association with leakage or ID with time. Limit the protocol to an offline benchmark until feature causality is established.

### 7. Policy — interpretation

**Observation:** The adopted policy removes zero rows from every class. Alternative exclusion policies disproportionately remove observations from classes that share feature vectors; exact impacts are saved by class.

**Hypothesis / limitation:** Retaining repeated rows preserves benchmark frequencies but allows common vectors to carry more evaluation weight. Removing ambiguous groups changes the task and could hide difficult cases.

**Decision:** Retain all labels and multiplicities; use grouped validation and report ambiguity/overlap support. Any later sensitivity experiment must be separately named, training-defined and predeclared.

### 8. Grouped validation — interpretation

**Observation:** Saved 175,341 assignments: 140,273 fit rows, 17,534 model-selection rows and 17,534 calibration rows. Repeated execution gives identical assignments; every pair of roles has zero exact predictor overlap. Maximum multiclass validation share difference is 0.001707 percentage points.

**Hypothesis / limitation:** Rare-class row support can overstate independent information, and contextual counters can share source history across different vectors. This is an exact-vector-disjoint offline protocol, not a verified temporal or host-disjoint protocol.

**Decision:** Freeze these source-hash-bound assignments. Fit learned preprocessing only on the train role; use model_selection for architecture/early stopping/threshold decisions and calibration for calibration fitting. Do not refit on validation before the documented final test.

### 9. Evaluation and handoff — interpretation

**Observation:** The grouped assignments, diagnostic masks and frozen protocol are saved. Source CSVs, Notebook 01 and all protected prior artifacts have identical before/after SHA-256 hashes; loaded frames are unchanged.

**Hypothesis / limitation:** Official test labels have been inspected for this declared audit, so provenance must disclose the audit. Future feature/hyperparameter choices must not be tuned using these test observations. Exact grouping does not eliminate contextual dependence or establish deployment causality.

**Decision:** Adopt the documented offline protocol; reserve the full test set for final evaluation. Stop for review. Notebook 03 must address preprocessing and feature-provenance requirements before any modelling.

## Observed repetition

| dataset | rows | unique_vectors | repeated_groups | participating_rows | excess_duplicate_rows | largest_group |
| --- | --- | --- | --- | --- | --- | --- |
| train | 175341 | 101040 | 9748 | 84049 | 74301 | 415 |
| test | 82332 | 53946 | 6318 | 34704 | 28386 | 88 |

## Target ambiguity

| dataset | target | affected_groups | affected_rows | affected_row_pct |
| --- | --- | --- | --- | --- |
| train | label | 229 | 940 | 0.5361 |
| train | attack_cat | 1772 | 30528 | 17.41065 |
| test | label | 6 | 14 | 0.017 |
| test | attack_cat | 575 | 8809 | 10.69936 |

## Consistent and conflicting cross-split overlap

| target | status | vectors | train_rows | test_rows |
| --- | --- | --- | --- | --- |
| binary | conflicting | 279 | 795 | 526 |
| binary | consistent | 1023 | 9484 | 8015 |
| category | conflicting | 548 | 2655 | 4977 |
| category | consistent | 754 | 7624 | 3564 |

## Frozen development roles

| partition | role | rows | feature_groups |
| --- | --- | --- | --- |
| train | train | 140273 | 80848 |
| validation | calibration | 17534 | 9934 |
| validation | model_selection | 17534 | 10258 |

## Rare-class support

| partition | class | rows | independent_feature_groups |
| --- | --- | --- | --- |
| official_train | Backdoor | 1746 | 1535 |
| official_train | Shellcode | 1133 | 1091 |
| official_train | Worms | 130 | 127 |
| train | Backdoor | 1397 | 1227 |
| train | Shellcode | 907 | 874 |
| train | Worms | 104 | 102 |
| validation | Backdoor | 349 | 308 |
| validation | Shellcode | 226 | 217 |
| validation | Worms | 26 | 25 |
| model_selection | Backdoor | 175 | 153 |
| model_selection | Shellcode | 113 | 108 |
| model_selection | Worms | 13 | 13 |
| calibration | Backdoor | 174 | 155 |
| calibration | Shellcode | 113 | 109 |
| calibration | Worms | 13 | 12 |

## Detailed evaluation protocol and Notebook 03 requirements

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

## Provenance and limits

See [feature semantics and primary sources](02_feature_semantics_sources.md), `results/splits/02_split_manifest.json`, and `results/tables/02_integrity_and_split_verification.json`. Exact groups are not verified independent events. Environment is project .venv with inherited system packages; versions are recorded. The reference architecture remains unreviewed. No preprocessing, encoding, resampling, feature selection or model fitting occurred.
