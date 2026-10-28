# Research alignment — recorded before implementation

Revision: 12 October 2026. Preserved starting point: `snapshots/workflow_20261012T143405Z` (379 files, independently verified copies).

The defensible task is **offline, completed-flow intrusion classification** on the supplied UNSW-NB15 development file. Predict the ten `attack_cat` classes, including Normal, from 42 original recorded predictors. Binary Attack/Normal is a second view of this task: `label=1` means an attack in this dataset, not verified botnet activity. Primary model comparison: macro F1 over all ten fixed classes. Each class contributes equally; accuracy is secondary. Preserve all labels, conflicting patterns and frozen roles. Keep the official test separate and calibration reserved.

The reference paper is Mohan et al., *Enhancing botnet attack detection using a hybrid deep learning model combining network flow and DNS query features with explainable AI for cybersecurity*, Scientific Reports, accepted manuscript, DOI 10.1038/s41598-026-65732-w. We reviewed the local 29-page PDF, including the architecture and result tables. Its architecture requires two measured streams: network-flow features and DNS-transaction features aligned into time windows by source/destination pair. It then uses Conv1D branches, fusion, attention and BiLSTM. Our CSVs contain neither the transaction-level DNS measurements nor the timestamps and endpoint identifiers needed to reproduce that windowing. A `service='dns'` value is a service category, not a DNS transaction stream. Row order and ID are not time.

We can test a clearly labelled **tabular adaptation** of the architecture against strong tabular and non-convolutional baselines. A sequence of column positions is an ordered feature representation, not a measured sequence of network events. TCN or BiLSTM applied to it cannot establish temporal detection. Engineered byte and packet balances are deterministic summaries of existing counts; they add a representation, not new information.

Botnet-specific ground truth requires independently established infection/C2 labels, capture/scenario provenance and a defensible mapping to each flow. A real DNS branch requires measured query names, types, timestamps, responses and endpoint identifiers; query entropy, lengths, rates, domain diversity and failure ratios must be computed from those transactions. Genuine sequence models additionally need documented past-only windows, endpoint grouping, ordering, flow/window labels and host/time-separated evaluation. These are requirements for additional measured data, not features we will invent.

Earlier official-test audits disclosed duplicates, class distributions, target conflicts and overlap. They remain historical diagnostics; they must not drive transformation or model choices. The model-selection role has already served ten baseline experiments. Further results there are development evidence with selection optimism. Exact-pattern-disjoint roles prevent recorded-vector overlap but do not prove independent hosts, events or times.

## What is preserved and what changes

Preserve the exact duplicate calculations, frozen group assignments, train-fitted preprocessing, three label-free engineered features, five predeclared feature variants, and ten original unweighted baseline experiments. Clarify their meaning and expand EDA and worked examples. Correct stale workflow documentation and distinguish distinct patterns from independent events. Add bounded grouped tuning and an unweighted boosted-tree ensemble, if available. Tune model settings on the original 42-feature representation inside fitting folds, then freeze those settings across all five variants to isolate representation effects. No calibration, operational threshold selection, reserved-role refitting or official-test evaluation occurs in this revision.

## Reference-paper checks and limits

Page numbers below refer to PDF pages, not printed article pagination.

| Evidence | What the paper says | Consequence here |
|---|---|---|
| pp. 8, 15: feature tables | Five NET features (packet rate, duration, size, inter-arrival time, protocol distribution) and five DNS features (entropy, rate, length, frequency, type distribution) | Flow counters/timing are available; per-window protocol distribution needs actual windows. DNS columns are unavailable. |
| pp. 9–10: architecture and alignment | Two Conv1D branches; fusion, attention, BiLSTM; aligned fixed time windows per source/destination pair | Cannot reproduce extraction/alignment from these CSVs. Clearly identify any tabular adaptation. |
| p. 14: data extraction | 68,661 DNS records obtained by TCP/UDP port-53 filtering | The supplied 42 predictors lack port identifiers and query records. Service is an insufficient substitute. |
| pp. 6–8: preprocessing | Ridge selection, imputation, scaling and SMOTE; scaling equation is standardization but prose also describes [0,1] scaling | Split order, extractor details and scaling interpretation need clarification. Do not inherit possible leakage or ambiguous transformations. |
| p. 16: settings | Batch 512, dropout .2, 50 epochs, learning rate .001, 4 attention heads, 2 Conv layers, 128 neurons, 5-second windows, Adam | Exact tensor shapes, meaning of “filter size 16”, numerical L2 penalty and some layer details remain insufficiently specified. |
| pp. 16–17, 20–21: evaluation | Very high reported accuracy, ten-fold validation, highest-accuracy-fold confusion matrices and ablations | Not directly comparable to our fixed-ten-class macro F1, original-pattern grouping, different inputs and reserved final test. |
| p. 16: Table 6 | Several reported F1 values do not equal the harmonic mean of their printed precision and recall (e.g., Normal 98.24/98.67 implies about 98.45, printed F1 97.05) | Need raw predictions or an explanation of averaging. Preserve our independently computed metrics; do not use this table as a performance target. |
| p. 15: Table 2 | Worms test support 44 of 82,332; printed percentage .53 | The arithmetic is approximately .0534%; derive counts/percentages ourselves. |

The accepted manuscript explicitly notes that it is unedited. These ambiguities limit exact reproduction; they do not establish why the discrepancies occurred. Obtain author code, extraction definitions and split/metric details before calling a later model a faithful reproduction. Explainability can describe a model, but cannot turn broad intrusion labels into botnet ground truth.
