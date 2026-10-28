# Notebook 05 — Feature Selection

## What we will compare

Question: Which original and engineered features should be carried forward as candidates for baseline models?

- **original42 (42 features):** Reference: no engineered additions. Omitted: fe_total_bytes, fe_byte_balance, fe_packet_balance.
- **all45 (45 features):** Reference: all three engineered additions. Omitted: none.
- **total_and_byte44 (44 features):** Test total bytes and byte direction without packet balance. Omitted: fe_packet_balance.
- **total_and_packet44 (44 features):** Test total bytes and packet direction without byte balance. Omitted: fe_byte_balance.
- **total_only43 (43 features):** Test volume alone, without either directional balance. Omitted: fe_byte_balance, fe_packet_balance.

All five sets retain every original predictor, including both FTP fields. These are named candidates, not a winning set. Full retained/omitted lists are in `results/tables/05_shortlist.csv` and `models/preprocessing/05/05_*_features.json`. Each categorical feature includes its complete group of one-hot columns. Source-feature counts differ from encoded-matrix column counts. The definitions were saved before this notebook's analysis and did not change afterwards.

## Evidence and limits

Only fitting rows informed evidence: 140,273 rows with 80,848 different original patterns. Row/pattern support, exact copies, most-common-value shares, correlations and near-redundancy flags are saved in the evidence tables. A >=99.5% most-common-value share and absolute Spearman >=0.95 are diagnostic flags, not removal rules. Correlation between the balances is 0.9734 for rows and 0.8940 for one row per original pattern.

Both FTP columns match exactly in fitting rows. Notebook 02's saved audit already records differences in the supplied data. This is referenced provenance; no new official-test records were read or used to rank, prune or add candidates. Unresolved field meanings are a reason to preserve both. Low variation and correlation alone do not establish uselessness or label leakage.

Label associations were checked in three grouped internal folds (seed 52), entirely inside the frozen train role. Numeric features use up to eight quantile bins with cuts fitted on each inner training partition, then applied unchanged to that partition's holdout. Categories use literal values, not one-hot fragments. Constant observed features produce unavailable scores. Scores are Cramer's V summaries; the fractional-weight version gives each original pattern total weight one while retaining every conflicting label. No independence test or p-value is reported. High cardinality, binning and class frequencies can bias these descriptive scores; their fold ranges are not confidence intervals or model-performance estimates. They do not measure incremental information beyond the other columns.

Rare-class and conflict support are reported for fitting rows and internal folds. Identical original vectors stay together, including conflicting labels. Exact patterns are not verified independent hosts or events. No majority relabeling, deduplication or resampling occurs. No predictive selector, model comparison or confidence calibration was fitted. Saved internal binning is an audit tool only and is not used in the candidate model matrices.

Both FTP fields have unavailable association scores: their nearly all-zero values make the quantile bands collapse to one group. This diagnostic cannot describe those rare nonzero values properly. It is not evidence of zero relevance; the separate FTP value audit remains important.

## Saved inputs and checks

Each candidate has raw, tree and scaled inputs for train, model_selection and calibration. All are fixed column slices of verified Notebook 04 inputs; no preprocessing is refitted. Source-name lists define raw-column order; encoded-column order follows Notebook 04 and is separately saved. Shared row-link files contain original labels, source positions and pattern IDs; these are not predictors.

Saved/reloaded raw values and sparse matrices match exactly, with finite entries and preserved rows, labels and whole categorical groups. Every candidate/representation has an exact cross-role overlap check with counts for both roles. Checks compare full tuples or canonical sparse index/value bytes after dictionary lookup, not hash equality alone. Representation/role-pair checks with matches: 0. Any new matches must be reported before modelling; no split changes occur here. All earlier notebooks, raw files, assignments, scripts, utilities and saved artifacts have unchanged hashes. Official-test bytes were only hashed for preservation.

Provenance note: Notebook 04's current file hash matches its old execution record: False. Its code and Markdown were independently rebuilt in memory from the manifest-verified builder and match exactly; all 47 listed artifacts match their hashes and the saved notebook has no error outputs. The discrepancy is in non-source notebook content. Both hashes and this limitation are recorded in `05_notebook04_provenance.json`; no earlier file was changed or execution record rewritten.

## Notebook 06

Use these five sets with both frozen tree and scaled representations. Prespecify a small baseline family, such as one regularized linear classifier and one tree-based classifier, and give sets the same tuning budget, fitting rows, seeds and validation rules. Keep unweighted baselines; any use of Notebook 03's candidate class weights must be separately named and applied equally across sets. Keep the all-original 42 and all-45 comparisons.

Train only on train, select models on model_selection, and keep calibration for later confidence adjustment. Do not refit on validation or run the official test during baseline selection. Compare fixed-ten-class macro F1, every class's precision/recall/F1 and support, confusion matrices, and Notebook 02's binary measures. For binary threshold-dependent results, choose thresholds only on model_selection. Use paired differences and, if estimating uncertainty, resample entire original groups with the same group draws across sets; this remains conditional on unknown context dependence.

Compare total_only43 against original42 for volume; total_and_byte44 and total_and_packet44 against total_only43 for each balance; all45 against both 44-feature sets for the second balance; all45 against original42 for the complete addition. Engineered features help only if model_selection results improve under a fair matched comparison, with rare-class tradeoffs and stability made explicit. No winner is declared here. Reserve full-test and overlap/non-overlap reporting for the later frozen final pipeline under Notebook 02.

## Hashes and sources

- Training source: `bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa`
- Frozen assignments: `2c01281b021a7be536bfd0268d53707a3ea74858730fb577db095d618436b168`
- Notebook 04 artifact manifest: `36c98f1754192e76ecfd5180910f4a6cbe2194a6157b4813f1c8daf1404f4f13`
- Predeclared feature sets: `fd581cea17cf1e3a76a70f6f218fb05804027f52109b662bb5b0d45b271c3671`

All deliverable hashes are in `data/processed/05/05_artifact_manifest.json`. Method references: [Cramer's V in SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.contingency.association.html) and [StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html). The weighted extension is our descriptive calculation, not an inferential SciPy test. Dataset semantics and deployment limits remain as documented in Notebooks 02–04.
