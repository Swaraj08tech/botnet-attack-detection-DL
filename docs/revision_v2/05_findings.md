# Notebook 05: revised findings


## Findings
Fitting redundancy and three-fold descriptive association calculations reproduce the earlier evidence. FTP fields are fitting copies; balances are related. All five role-linked candidate representations verify and have zero cross-role exact matches.


## Decisions
Shortlist original42, total_only43, total_and_byte44, total_and_packet44 and all45. Keep whole categorical groups and every original predictor. Fit no predictive selector and claim no winner.


## Limitations
This is a representation shortlist, not original-feature subset discovery. Rankings, correlation and low variation do not establish improved detection. Grouping recorded patterns does not establish host/time independence.


## Saved outputs
- `results/tables/revision_v2/05_role_class_support.csv`
- `results/tables/revision_v2/05_exact_copies.csv`
- `results/tables/revision_v2/05_near_redundancy.csv`
- `results/tables/revision_v2/05_variation.csv`
- `results/figures/revision_v2/05_redundancy.png`
- `results/tables/revision_v2/05_association_folds.csv`
- `results/tables/revision_v2/05_internal_fold_support.csv`
- `results/tables/revision_v2/05_association_summary.csv`
- `results/figures/revision_v2/05_association.png`
- `results/tables/revision_v2/05_shortlist.csv`
- `models/revision_v2/05_feature_sets.json`
- `results/tables/revision_v2/05_candidate_verification.csv`
- `results/tables/revision_v2/05_candidate_overlap.csv`
- `results/tables/revision_v2/05_verified_input_index.csv`


## Next question
Do matched simple and stronger tabular models benefit from any engineered representation under the frozen development protocol?


## Section evidence
**Evidence:** FTP fields are exact copies in fitting rows. Several low-variation fields still have nonzero minority pattern support. Balance correlation repeats the engineering finding.

**Decision:** Do not drop original predictors on these descriptive tests. Preserve both FTP fields because intended meanings differ and prior audit found supplied-data disagreement.

**Still uncertain:** High correlation need not mean interchangeable predictive use. Low variation can conceal meaningful rare protocol behaviour.

**Evidence:** Three group-held-out association runs reproduce the earlier evidence. Numeric quantile edges are fitted on each inner-training fold; one pattern has total weight one in the weighted diagnostic.

**Decision:** Use this as descriptive stability evidence only. Keep all five engineered ablations and all 42 original predictors for fair model comparisons.

**Still uncertain:** Cramér’s V measures association, not model improvement. Fold ranges are not confidence intervals. Rare classes, repeated patterns and conflicting targets limit rankings; constant FTP bins have undefined scores, not zero relevance.

**Evidence:** All five source-name definitions keep whole categorical groups. Every saved role-linked input matches its protected predecessor and has zero cross-role exact matches.

**Decision:** This notebook shortlists representation variants; it does not identify a winning subset. Each variant answers volume, byte direction, packet direction or joint contribution.

**Still uncertain:** No predictive selector is fitted. Removing columns in future may create new identical rows; audit and report any new matches without silently redrawing roles.