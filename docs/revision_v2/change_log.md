# What changed and why

Starting point: `snapshots/workflow_20261012T143405Z`, created before editing existing work. All 379 copied files were independently hash-verified. Raw files were fingerprinted, not rewritten. Snapshot restore instructions are in its manifest; recover individual project-relative files from the corresponding snapshot path after checking its recorded hash. No automatic restore or deletion is performed.

| Work | Status | Explanation |
|---|---|---|
| Notebook 01, raw CSVs, frozen assignments, all earlier source/config/model/data/table/figure artifacts | Preserved | Protected hashes are checked before/after execution. The earlier experimental record remains available in place and in the snapshot. |
| Exact-vector audit and keep-all policy | Preserved and freshly verified | Group by all original predictors, exclude ID/targets, resolve hashes with full tuple equality; retain conflicting versions and every label. |
| “Independent feature groups” terminology | Corrected | Display distinct original patterns. Independent physical events/hosts/times are not established. |
| Conflict constraints and overlap definitions | Expanded | Distinguish binary/category ambiguity, affected rows/groups, consistent versus conflicting target unions and empirical deterministic-vector accuracy limits. |
| Data explanations and EDA | Expanded | Plain questions, reasons, methods, evidence, decisions and uncertainties; worked examples; range/zero/skew/category/protocol views; protected full fitting-EDA gallery retained. |
| Tree/log-robust preprocessing | Preserved and rerun | Fresh fitting-only clones exactly reproduce previous six role matrices. No imputation, clipping or new transform search. |
| Three engineered features | Preserved and explained | Direct formulas, units, zero sentinel, prediction scope, count examples, distributions, extreme support and rejected size/rate alternatives. Verify augmented values against the saved named-column transformer, including its internal log operation. |
| Feature “selection” claim | Clarified | Notebook 05 defines five representation candidates, not a winning subset of original42. No predictive selector is invented. All categorical blocks and original columns remain. |
| Descriptive association stability | Preserved and rerun | Repeat three fitting-internal grouped folds and fold-fitted quantile bins; retain undefined constant-feature scores and pattern-weighted diagnostics. |
| Original ten baseline experiments | Preserved as historical controls | Recompute predictions from an earlier fit only when final settings match exactly; otherwise fit a new model in a new folder. Earlier reports are not overwritten. |
| Baseline performance bar | Expanded | Add unweighted LightGBM; two configurations per family, three grouped fitting folds, inner-fold preprocessing refits, frozen settings across five sets. No uncontrolled search or feature-specific retuning. |
| Metric interpretation | Expanded | Fixed ten-class macro F1, every class’s diagnostics, binary metrics/AP, matched deltas, rare-class counts, fit time and family-specific complexity. |
| Notebook presentation | Simplified | Simple headings and meaningful examples; implementation lives in `src/research_revision`. Preserve useful evidence instead of imposing a length limit. |
| Old workflow status in README | Superseded | It incorrectly said only 01–02 existed and the reference was unread. Revised status points to real execution/verification records and paper review. |
| Earlier Notebook 04/05 execution-container hashes | Historical limitation documented | Existing sources matched their protected builders; non-source notebook container drift was already documented. Keep the old execution records; generate fresh byte hashes for revised notebooks. |
| Reference-model claim | Qualified | Paper requires measured DNS/window alignment unavailable here. Later tabular adaptations must be labelled; source extraction and some paper metric/settings details remain unresolved. |

## File mapping

No renumbering is needed. The current 02–06 filenames retain their research boundaries:

- `02_duplicate_target_ambiguity_and_leakage_investigation.ipynb` → revised data and evaluation design.
- `03_training_eda_and_leakage_safe_preprocessing.ipynb` → expanded fitting EDA and reproduced preprocessing.
- `04_feature_engineering.ipynb` → direct three-feature calculations and evidence.
- `05_feature_selection.ipynb` → honest candidate shortlist and group-aware descriptive stability.
- `06_baseline_models.ipynb` → historical controls plus bounded stronger tabular benchmarking.

Old builders/runners remain historical implementation artifacts. Use `scripts/build_research_revision.py` (optionally a notebook number) and `scripts/run_research_revision.py <number>` for current notebooks. Use `scripts/verify_research_revision.py` for independent reload checks. The new manifests index protected earlier inputs as dependencies instead of silently replacing them. Calibration is only aligned/transformed for integrity; no calibration or official-test model predictions are made.

## Scope deliberately left for later

Confidence calibration, operational thresholds, final official-test reporting, any reserved-role refit, multi-seed neural comparisons and authentic DNS/temporal extraction are separate future experiments. The current ensemble search is a credible bounded bar, not exhaustive optimization. Historical test audits and model-selection exposure are part of the research record, not erased by this revision.
