# Notebook 03 — Training EDA and preprocessing findings



The 42-predictor offline IDS contract and Notebook 02 grouped roles were retained. All descriptive feature/target analysis used fitting rows only.



Fit/model-selection/calibration rows: 140,273 / 17,534 / 17,534.

Categorical columns: proto, service, state. Numerical columns: 39.

Prespecified nonnegative, high-skew columns log-transformed only in the scaled baseline: dur, spkts, dpkts, sbytes, dbytes, sload, sloss, dloss, sinpkt, dinpkt, sjit, djit, response_body_len.

Tree baseline retains original numeric values. Scaled baseline uses fitting-only RobustScaler and categorical one-hot encoding.

The service '-' marker, original FTP values, zeros, rows and target conflicts were preserved.



## Fit-only evidence and decisions

See `results/tables/03_numeric_fit_summary.csv`, `03_categorical_fit_frequencies.csv`, `03_fit_target_relationships.csv`, `03_preprocessing_decisions.csv`, and the linked figures.

Distinct feature groups are not verified independent network events. Correlations and class associations do not prove leakage.

## Transformed cross-role overlap

```csv
representation,role_a,role_b,shared_transformed_vectors,role_b_rows_with_match
tree,train,model_selection,0,0
tree,train,calibration,0,0
tree,model_selection,calibration,0,0
scaled,train,model_selection,0,0
scaled,train,calibration,0,0
scaled,model_selection,calibration,0,0

```

## Notebook 04 handoff

Use only frozen roles and saved pipelines; compare unweighted and candidate weighted baselines. Use model_selection for selection, calibration only for calibration fitting, and the full official test at final evaluation under Notebook 02 protocol.

## Limits

No DNS field, timestamp-based sequence, literal Botnet class, or verified early-detection availability is established. The reference architecture needs separate review.
