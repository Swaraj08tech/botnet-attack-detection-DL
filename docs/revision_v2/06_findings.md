# Notebook 06: revised findings


## Findings
logistic: original42 macro F1 0.4169; highest observed set total_and_byte44 0.4242 (difference +0.728 percentage points).

decision_tree: original42 macro F1 0.6097; highest observed set all45 0.6160 (difference +0.636 percentage points).

lightgbm: original42 macro F1 0.6307; highest observed set total_and_byte44 0.6320 (difference +0.123 percentage points).

Provisional choice: lightgbm, total_and_byte44. Every result uses the same 17,534 model-selection rows; Worms support is 13.


## Decisions
Use the declared provisional candidate as the current tabular performance bar. Preserve all earlier controls, all matched feature variants and the frozen roles. Carry the same ten-class metric and input contracts into architecture ablations.


## Limitations
Model-selection has prior and current candidate exposure; the chosen score is optimistic. Only 13 Worms rows support rare-class comparisons. Three folds/two configurations are a bounded search, not exhaustive optimization. No confidence calibration, final threshold, reserved-role refit or official-test evaluation occurred. DNS transactions, temporal windows and botnet labels remain unavailable.


## Saved outputs
- `results/tables/revision_v2/06_role_class_support.csv`
- `results/tables/revision_v2/06_preserved_06_model_selection_metrics.csv`
- `results/tables/revision_v2/06_candidate_verification.csv`
- `results/tables/revision_v2/06_candidate_overlap.csv`
- `configs/revision_v2_baseline_plan.json`
- `models/revision_v2/06_inner_scaled_fold0_pipeline.joblib`
- `models/revision_v2/06_inner_scaled_fold0_provenance.json`
- `models/revision_v2/06_inner_tree_fold0_pipeline.joblib`
- `models/revision_v2/06_inner_tree_fold0_provenance.json`
- `models/revision_v2/06_inner_scaled_fold1_pipeline.joblib`
- `models/revision_v2/06_inner_scaled_fold1_provenance.json`
- `models/revision_v2/06_inner_tree_fold1_pipeline.joblib`
- `models/revision_v2/06_inner_tree_fold1_provenance.json`
- `models/revision_v2/06_inner_scaled_fold2_pipeline.joblib`
- `models/revision_v2/06_inner_scaled_fold2_provenance.json`
- `models/revision_v2/06_inner_tree_fold2_pipeline.joblib`
- `models/revision_v2/06_inner_tree_fold2_provenance.json`
- `results/tables/revision_v2/06_grouped_tuning_folds.csv`
- `results/tables/revision_v2/06_grouped_tuning_summary.csv`
- `results/tables/revision_v2/06_inner_class_support.csv`
- `models/revision_v2/06_frozen_family_settings.json`
- `data/processed/revision_v2/06_logistic_original42_predictions.csv`
- `results/tables/revision_v2/06_logistic_original42_confusion_raw.csv`
- `results/tables/revision_v2/06_logistic_original42_confusion_normalized.csv`
- `results/tables/revision_v2/06_logistic_original42_precision_recall.csv`
- `models/revision_v2/06_logistic_original42.joblib`
- `models/revision_v2/06_logistic_original42_settings.json`
- `data/processed/revision_v2/06_logistic_total_only43_predictions.csv`
- `results/tables/revision_v2/06_logistic_total_only43_confusion_raw.csv`
- `results/tables/revision_v2/06_logistic_total_only43_confusion_normalized.csv`
- `results/tables/revision_v2/06_logistic_total_only43_precision_recall.csv`
- `models/revision_v2/06_logistic_total_only43.joblib`
- `models/revision_v2/06_logistic_total_only43_settings.json`
- `data/processed/revision_v2/06_logistic_total_and_byte44_predictions.csv`
- `results/tables/revision_v2/06_logistic_total_and_byte44_confusion_raw.csv`
- `results/tables/revision_v2/06_logistic_total_and_byte44_confusion_normalized.csv`
- `results/tables/revision_v2/06_logistic_total_and_byte44_precision_recall.csv`
- `models/revision_v2/06_logistic_total_and_byte44.joblib`
- `models/revision_v2/06_logistic_total_and_byte44_settings.json`
- `data/processed/revision_v2/06_logistic_total_and_packet44_predictions.csv`
- `results/tables/revision_v2/06_logistic_total_and_packet44_confusion_raw.csv`
- `results/tables/revision_v2/06_logistic_total_and_packet44_confusion_normalized.csv`
- `results/tables/revision_v2/06_logistic_total_and_packet44_precision_recall.csv`
- `models/revision_v2/06_logistic_total_and_packet44.joblib`
- `models/revision_v2/06_logistic_total_and_packet44_settings.json`
- `data/processed/revision_v2/06_logistic_all45_predictions.csv`
- `results/tables/revision_v2/06_logistic_all45_confusion_raw.csv`
- `results/tables/revision_v2/06_logistic_all45_confusion_normalized.csv`
- `results/tables/revision_v2/06_logistic_all45_precision_recall.csv`
- `models/revision_v2/06_logistic_all45.joblib`
- `models/revision_v2/06_logistic_all45_settings.json`
- `data/processed/revision_v2/06_decision_tree_original42_predictions.csv`
- `results/tables/revision_v2/06_decision_tree_original42_confusion_raw.csv`
- `results/tables/revision_v2/06_decision_tree_original42_confusion_normalized.csv`
- `results/tables/revision_v2/06_decision_tree_original42_precision_recall.csv`
- `models/revision_v2/06_decision_tree_original42.joblib`
- `models/revision_v2/06_decision_tree_original42_settings.json`
- `data/processed/revision_v2/06_decision_tree_total_only43_predictions.csv`
- `results/tables/revision_v2/06_decision_tree_total_only43_confusion_raw.csv`
- `results/tables/revision_v2/06_decision_tree_total_only43_confusion_normalized.csv`
- `results/tables/revision_v2/06_decision_tree_total_only43_precision_recall.csv`
- `models/revision_v2/06_decision_tree_total_only43.joblib`
- `models/revision_v2/06_decision_tree_total_only43_settings.json`
- `data/processed/revision_v2/06_decision_tree_total_and_byte44_predictions.csv`
- `results/tables/revision_v2/06_decision_tree_total_and_byte44_confusion_raw.csv`
- `results/tables/revision_v2/06_decision_tree_total_and_byte44_confusion_normalized.csv`
- `results/tables/revision_v2/06_decision_tree_total_and_byte44_precision_recall.csv`
- `models/revision_v2/06_decision_tree_total_and_byte44.joblib`
- `models/revision_v2/06_decision_tree_total_and_byte44_settings.json`
- `data/processed/revision_v2/06_decision_tree_total_and_packet44_predictions.csv`
- `results/tables/revision_v2/06_decision_tree_total_and_packet44_confusion_raw.csv`
- `results/tables/revision_v2/06_decision_tree_total_and_packet44_confusion_normalized.csv`
- `results/tables/revision_v2/06_decision_tree_total_and_packet44_precision_recall.csv`
- `models/revision_v2/06_decision_tree_total_and_packet44.joblib`
- `models/revision_v2/06_decision_tree_total_and_packet44_settings.json`
- `data/processed/revision_v2/06_decision_tree_all45_predictions.csv`
- `results/tables/revision_v2/06_decision_tree_all45_confusion_raw.csv`
- `results/tables/revision_v2/06_decision_tree_all45_confusion_normalized.csv`
- `results/tables/revision_v2/06_decision_tree_all45_precision_recall.csv`
- `models/revision_v2/06_decision_tree_all45.joblib`
- `models/revision_v2/06_decision_tree_all45_settings.json`
- `data/processed/revision_v2/06_lightgbm_original42_predictions.csv`
- `results/tables/revision_v2/06_lightgbm_original42_confusion_raw.csv`
- `results/tables/revision_v2/06_lightgbm_original42_confusion_normalized.csv`
- `results/tables/revision_v2/06_lightgbm_original42_precision_recall.csv`
- `models/revision_v2/06_lightgbm_original42.joblib`
- `models/revision_v2/06_lightgbm_original42_settings.json`
- `data/processed/revision_v2/06_lightgbm_total_only43_predictions.csv`
- `results/tables/revision_v2/06_lightgbm_total_only43_confusion_raw.csv`
- `results/tables/revision_v2/06_lightgbm_total_only43_confusion_normalized.csv`
- `results/tables/revision_v2/06_lightgbm_total_only43_precision_recall.csv`
- `models/revision_v2/06_lightgbm_total_only43.joblib`
- `models/revision_v2/06_lightgbm_total_only43_settings.json`
- `data/processed/revision_v2/06_lightgbm_total_and_byte44_predictions.csv`
- `results/tables/revision_v2/06_lightgbm_total_and_byte44_confusion_raw.csv`
- `results/tables/revision_v2/06_lightgbm_total_and_byte44_confusion_normalized.csv`
- `results/tables/revision_v2/06_lightgbm_total_and_byte44_precision_recall.csv`
- `models/revision_v2/06_lightgbm_total_and_byte44.joblib`
- `models/revision_v2/06_lightgbm_total_and_byte44_settings.json`
- `data/processed/revision_v2/06_lightgbm_total_and_packet44_predictions.csv`
- `results/tables/revision_v2/06_lightgbm_total_and_packet44_confusion_raw.csv`
- `results/tables/revision_v2/06_lightgbm_total_and_packet44_confusion_normalized.csv`
- `results/tables/revision_v2/06_lightgbm_total_and_packet44_precision_recall.csv`
- `models/revision_v2/06_lightgbm_total_and_packet44.joblib`
- `models/revision_v2/06_lightgbm_total_and_packet44_settings.json`
- `data/processed/revision_v2/06_lightgbm_all45_predictions.csv`
- `results/tables/revision_v2/06_lightgbm_all45_confusion_raw.csv`
- `results/tables/revision_v2/06_lightgbm_all45_confusion_normalized.csv`
- `results/tables/revision_v2/06_lightgbm_all45_precision_recall.csv`
- `models/revision_v2/06_lightgbm_all45.joblib`
- `models/revision_v2/06_lightgbm_all45_settings.json`
- `results/tables/revision_v2/06_model_selection_metrics.csv`
- `results/tables/revision_v2/06_per_class_metrics.csv`
- `results/tables/revision_v2/06_feature_comparisons.csv`
- `results/tables/revision_v2/06_class_tradeoffs.csv`
- `results/figures/revision_v2/06_matched_models.png`
- `results/tables/revision_v2/06_historical_controls.csv`
- `models/revision_v2/06_provisional_choice.json`
- `results/figures/revision_v2/06_logistic_original42_confusions.png`
- `results/figures/revision_v2/06_logistic_total_only43_confusions.png`
- `results/figures/revision_v2/06_logistic_total_and_byte44_confusions.png`
- `results/figures/revision_v2/06_logistic_total_and_packet44_confusions.png`
- `results/figures/revision_v2/06_logistic_all45_confusions.png`
- `results/figures/revision_v2/06_decision_tree_original42_confusions.png`
- `results/figures/revision_v2/06_decision_tree_total_only43_confusions.png`
- `results/figures/revision_v2/06_decision_tree_total_and_byte44_confusions.png`
- `results/figures/revision_v2/06_decision_tree_total_and_packet44_confusions.png`
- `results/figures/revision_v2/06_decision_tree_all45_confusions.png`
- `results/figures/revision_v2/06_lightgbm_original42_confusions.png`
- `results/figures/revision_v2/06_lightgbm_total_only43_confusions.png`
- `results/figures/revision_v2/06_lightgbm_total_and_byte44_confusions.png`
- `results/figures/revision_v2/06_lightgbm_total_and_packet44_confusions.png`
- `results/figures/revision_v2/06_lightgbm_all45_confusions.png`
- `results/figures/revision_v2/06_precision_recall.png`
- `results/figures/revision_v2/06_chosen_class_recall.png`
- `results/tables/revision_v2/06_ambiguity_error_diagnostic.csv`
- `results/tables/revision_v2/06_class_ambiguity_errors.csv`
- `results/tables/revision_v2/06_important_confusions.csv`
- `results/tables/revision_v2/06_chosen_gain_contributions.csv`
- `results/figures/revision_v2/06_ambiguity_and_gain.png`


## Next question
Can a fair non-convolutional neural baseline, tabular Conv1D and attention adaptation improve the strongest bounded tabular benchmark without relying on arbitrary column order or unsupported data claims?


## Section evidence
**Evidence:** The bounded search uses only fitting rows. Fold-local category vocabularies and scaling are refitted before each group-held-out prediction.

**Decision:** Freeze one setting per family and use it unchanged for all five representations. Do not use model-selection results to expand this search.

**Still uncertain:** Three folds and two configurations are a modest benchmark budget, not exhaustive tuning. Pattern groups are not verified independent hosts; fold ranges are not confidence intervals.

**Evidence:** The declared rule selects lightgbm with total_and_byte44 (macro F1 0.6320). This selection uses previously exposed model-selection rows.

**Decision:** Carry this exact model/input contract as the performance bar. Retain class-level tradeoffs and negative engineered-feature findings.

**Still uncertain:** Scores are development point estimates, not final performance or statistical significance. Thirteen Worms rows cannot establish a stable rare-class advantage; raw scores are not calibrated confidence.

**Evidence:** The chosen set changes macro F1 by +0.123 percentage points versus original42 within lightgbm; Worms alone contributes +0.243 points. The printed confusion table quantifies errors toward Exploits, and the subgroup table locates conflicting recorded patterns.

**Decision:** Keep the provisional rule unchanged. Carry original42 as a close reference and describe the engineered gain as uncertain, with class-level losses retained. Use the error diagnosis to define future matched ablations, not to start another search here.

**Still uncertain:** Clear and conflicting subgroups have different class mixtures; their recalls are descriptive, not causal attribution. The deterministic-vector ceiling concerns accuracy on this population, not macro F1, a theoretical Bayes limit or unseen data. Extra truly measured context could separate identical recorded flows; deterministic features cannot.