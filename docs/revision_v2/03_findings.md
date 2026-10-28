# Notebook 03: revised findings


## Findings
Systematic fitting EDA exposes class imbalance, repeated patterns, zero-heavy fields and numeric tails. Fresh train-only preprocessing exactly reproduces the prior six matrices; round trips and cross-role overlap checks pass.


## Decisions
Keep the raw-numeric tree and log/robust-scaled representations. Preserve all 42 source predictors, protocol markers, nonbinary FTP values, zeros and extremes. Keep categorical groups intact.


## Limitations
The extractor and timing semantics are not independently verified. Scaling does not remove outliers, create information or establish performance. Unknown categories lack learned category-specific effects; rare-class support remains small.


## Saved outputs
- `results/tables/revision_v2/03_role_class_support.csv`
- `results/tables/revision_v2/03_fitting_support.csv`
- `results/figures/revision_v2/03_class_support.png`
- `results/tables/revision_v2/03_numeric_summary.csv`
- `results/figures/revision_v2/03_zeros_and_skew.png`
- `results/tables/revision_v2/03_categorical_support.csv`
- `results/figures/revision_v2/03_categorical_values.png`
- `results/tables/revision_v2/03_protocol_zero_context.csv`
- `results/tables/revision_v2/03_target_relationships.csv`
- `results/figures/revision_v2/03_target_relationships.png`
- `results/tables/revision_v2/03_preserved_03_preprocessing_decisions.csv`
- `results/tables/revision_v2/03_scaling_worked_example.csv`
- `models/revision_v2/03_tree_pipeline.joblib`
- `data/processed/revision_v2/03_tree_train.npz`
- `data/processed/revision_v2/03_tree_model_selection.npz`
- `data/processed/revision_v2/03_tree_calibration.npz`
- `models/revision_v2/03_tree_names.json`
- `models/revision_v2/03_scaled_pipeline.joblib`
- `data/processed/revision_v2/03_scaled_train.npz`
- `data/processed/revision_v2/03_scaled_model_selection.npz`
- `data/processed/revision_v2/03_scaled_calibration.npz`
- `models/revision_v2/03_scaled_names.json`
- `results/tables/revision_v2/03_transform_verification.csv`
- `results/tables/revision_v2/03_transformed_overlap.csv`
- `results/tables/revision_v2/03_unknown_categories.csv`
- `results/figures/revision_v2/03_before_after.png`


## Next question
Can simple count summaries make recorded information easier for models to use?


## Section evidence
**Evidence:** Fitting has 140,273 rows but 80,848 distinct original patterns. Worms has 104 rows and 102 patterns; several numeric fields are mostly zero or highly skewed.

**Decision:** Use fitting distributions to justify representation changes. Keep zeros, extremes and protocol markers unless a verified measurement error is established.

**Still uncertain:** A tail value can be valid network traffic. Histogram shape is not evidence that it should be clipped or deleted.

**Evidence:** Protocol-specific zero frequencies differ, and class medians show associations between recorded measurements and outcomes.

**Decision:** Preserve zero as a measured or extractor-dependent value. Use associations to describe the dataset, not to claim causal detection or discard features.

**Still uncertain:** Repeated patterns and capture-specific TTL/context relationships can inflate apparent relevance; prospective availability remains unverified.

**Evidence:** Fresh train-only fits exactly reproduce all six earlier matrices, feature order and saved/reloaded transformations. Transformed role pairs have zero exact overlap.

**Decision:** Keep raw numeric values for trees and train-fitted log/robust scaling for linear or later neural models. Ignore unknown categories with documented all-zero one-hot blocks.

**Still uncertain:** Robust scaling centres by the fitting median and divides by its interquartile range; it does not eliminate outliers or create information. Some unknown categories have no learned category-specific effect.