# Handoff to the main model research

## The common task and performance bar

Use offline completed-flow classification into 10 fixed `attack_cat` classes (including Normal). Primary metric: fixed-ten-class macro F1. Report every class, binary attack/normal metrics, PR/AP and confusion matrices. The current bounded tabular bar is **lightgbm with total_and_byte44**, macro F1 0.6320 on model-selection. Final performance is unknown. Original42 ensemble is 0.6307 and remains a close reference; the engineered gain is small, Worms-sensitive and not proven robust.

Exact model: `models/revision_v2/06_lightgbm_total_and_byte44.joblib`. Exact input: `data/processed/05/05_total_and_byte44_tree_{role}.npz`, with `05_row_links_{role}.csv`. Original source names and encoded column order are in `models/preprocessing/05/05_total_and_byte44_tree_columns.json`. Settings are in `models/revision_v2/06_lightgbm_total_and_byte44_settings.json`; preprocessing comes from protected Notebooks 03/04 artifacts. The revised manifests bind every dependency to its hash. The separate scaled inputs are available for linear/neural models; do not silently feed scaled values to a model trained on tree inputs.

## Available measurements and missing streams

`available_feature_groups.json` maps all 42 predictors exactly once. It groups flow volume/load, timing summaries, protocol/state/header values, service counters and contextual counts. These are recorded tabular measurements, not separate NET/DNS event streams. Intended definitions and local extraction remain distinct claims; contextual connection counts require history whose past-only implementation cannot be replayed from this subset.

| Reference requirement | CSV evidence | What we can honestly say |
|---|---|---|
| Packet rate, duration, size, inter-arrival information | `rate`, `dur`, directional packet/byte counters, `smean/dmean`, `sinpkt/dinpkt` | Related per-flow fields exist. Exact extractor formula/units need documented mapping; a per-flow summary is not the paper's time-window extraction. |
| Protocol distribution per aligned window | `proto` per flow | A measured category exists, but no endpoint/time window for a protocol distribution. |
| DNS query entropy, rate, length, frequency, type distribution | No query names, transaction types, timestamps or raw responses | Unavailable. Do not approximate query entropy/length from total flow bytes. |
| DNS indication | `service='dns'` category | A recorded service marker, already represented by one-hot encoding. It is not raw DNS transaction evidence or a second stream. |
| Additional engineered summaries | Total bytes and two direction balances | Deterministic count summaries with documented formulas and zero rules; no added measured information. |
| Source/destination-pair aligned time windows | No endpoint identifiers or timestamps | Cannot reproduce the paper's alignment, replay ct counters or establish genuine temporal samples. |
| Botnet/C2 infection labels | Binary Attack and broad intrusion categories | Botnet-specific detection is unsupported without separately established per-event/scenario infection/C2 ground truth. |

The reviewed paper uses aligned NET/DNS windows, Conv1D branches, fusion, attention and BiLSTM. Exact reproduction needs author extraction code, input shapes, splitting/averaging details and the missing measured records. Some paper preprocessing/settings/results details are ambiguous; see `research_alignment.md` with PDF page locators. Do not compare our macro F1 directly with its reported accuracy or highest-fold figures.

## A concrete first architecture experiment

1. Freeze a tabular task/input contract and a small training-only grouped search plan before fitting. Start with original42 and total_and_byte44; keep the five-set baseline record. Any additional representation is a newly declared candidate, not a hidden change.
2. Train a fair non-convolutional MLP comparator using the same fitting rows, targets, fitted scaling/category groups and a bounded search budget. It helps determine whether a benefit is from general neural capacity rather than convolution.
3. Test a **tabular Conv1D adaptation**, explicitly labelled as such. Declare a semantic source-feature order and respect complete categorical blocks or defined feature tokens. Column position is not network time. Compare against the MLP and stronger ensemble.
4. Predeclare several frozen column-order permutations using fitting-only choices. Keep rows/settings/budget fixed. A gain that depends strongly on arbitrary adjacency is an ordering-sensitive tabular finding, not temporal botnet detection.
5. Add attention in a matched ablation: same input, backbone and budget, with/without the block. Describe attention over feature tokens as feature interaction. A flow/context two-branch adaptation is possible only if labelled that way; do not call the contextual branch DNS.
6. Defer genuine TCN/BiLSTM event-sequence claims until measured timestamps/endpoints/windows exist. If experimenting over feature positions, label the ordered-feature adaptation and compare with/without it. It does not demonstrate temporal dependence. Bidirectional processing also requires an offline-complete-window availability contract.
7. Compare fixed seeds across all arms; after a modest declared search, use matched repeat seeds to examine stability. Report macro F1, all classes, support, confusion tradeoffs, fit/prediction cost and complexity. Small Worms gains are unstable; DoS/Analysis/Backdoor→Exploits confusion is an important target for improvement. Their ambiguous recorded inputs cannot be fixed by deterministic formulas alone.
8. If class weighting is proposed for neural training, include a separate matched weighted tabular control and declare its rule/search beforehand. Do not compare a weighted neural model only against unweighted tabular models and attribute the difference to architecture.
9. Keep model-selection exposure explicit; do not choose architecture, thresholds or features from official-test outcomes. Calibration remains reserved. After development choices are frozen, plan calibration/operating policy under the existing protocol, then evaluate the full official test once with specified full/overlap/non-overlap reporting. Name whether overlap refers to supplied training or actual fitted rows.

Each component must improve a matched comparator at acceptable class-level cost. No model is promised to win. The final test, rather than repeated development-score inspection, must determine the eventual performance claim.

## Data required for an authentic DNS/temporal or botnet claim

Obtain measured flow start/end times and endpoint/session identifiers; DNS query names/types/times/responses and endpoint linkage; documented completed-flow extraction; past-only window state/reset/tie rules; and independently verified botnet infection/C2/scenario labels aligned to records. Preserve capture provenance and group/evaluate by hosts/capture periods where possible. Define how a flow or window receives a target before training. A different dataset would need a newly documented compatible protocol, not a fabricated label mapping.
