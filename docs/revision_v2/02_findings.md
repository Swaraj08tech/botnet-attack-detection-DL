# Notebook 02: revised findings


## Findings
Critical counts reproduce. Training has 101,040 patterns and 1,772 category-conflicting groups. The official splits share 1,302 vectors. ID ordering and special values need careful interpretation.


## Decisions
Keep all original rows and labels. Preserve the frozen group-disjoint development roles. Use ten-class macro F1 and class-level diagnostics; preserve the complete official test.


## Limitations
Distinct recorded vectors do not establish distinct events or host/time independence. Flow-extractor causality and botnet/DNS/temporal ground truth are unresolved. The test was historically audited; model-selection rows have prior experiment exposure.


## Saved outputs
- `results/tables/revision_v2/02_role_class_support.csv`
- `results/tables/revision_v2/02_audit_reproduction.csv`
- `results/tables/revision_v2/02_groups_train.csv`
- `results/tables/revision_v2/02_groups_test.csv`
- `results/figures/revision_v2/02_group_sizes.png`
- `results/tables/revision_v2/02_repetition.csv`
- `results/tables/revision_v2/02_class_repetition.csv`
- `results/figures/revision_v2/02_class_repetition.png`
- `results/tables/revision_v2/02_conflicts.csv`
- `results/tables/revision_v2/02_conflict_combinations.csv`
- `results/tables/revision_v2/02_conflict_pairs.csv`
- `results/tables/revision_v2/02_conflict_examples.csv`
- `results/tables/revision_v2/02_ambiguity_constraints.csv`
- `results/figures/revision_v2/02_conflict_heatmap.png`
- `results/tables/revision_v2/02_cross_split_exact_matches.csv`
- `results/tables/revision_v2/02_cross_split_summary.csv`
- `results/tables/revision_v2/02_cross_split_by_class.csv`
- `results/tables/revision_v2/02_test_diagnostic_masks.csv`
- `results/tables/revision_v2/02_test_overlap_populations.csv`
- `results/figures/revision_v2/02_id_ordering.png`
- `results/tables/revision_v2/02_semantics_checks.csv`
- `results/tables/revision_v2/02_feature_availability.csv`
- `results/tables/revision_v2/02_policy_effects.csv`
- `results/tables/revision_v2/02_frozen_role_overlap.csv`
- `results/tables/revision_v2/02_evaluation_roles.csv`


## Next question
What does fitting-only EDA justify about numeric and categorical preprocessing?


## Section evidence
**Evidence:** All 14 core Notebook 01 audit counts reproduce; original-pattern IDs also reproduce the frozen assignment.

**Decision:** Keep both source files and all frozen roles. Test access here only repeats the previously disclosed audit.

**Still uncertain:** Preservation hashes authenticate local bytes, not the dataset creator or capture process.

**Evidence:** Training has 101,040 distinct patterns, 84,049 rows in repeated groups, and 74,301 excess rows; these are different quantities.

**Decision:** Keep all rows but partition complete original patterns together. Report row support and pattern support.

**Still uncertain:** Two identical recorded vectors can describe different events. Distinct patterns are not verified independent events.

**Evidence:** Training contains 1,772 category-conflicting groups (30,528 rows) and 229 binary-conflicting groups (940 rows). Binary and category ambiguity differ.

**Decision:** Retain original labels and keep each whole pattern in one role. Do not apply majority-label replacement.

**Still uncertain:** Missing context, quantization or annotation differences may explain conflicts; these data do not distinguish them. The empirical accuracy ceiling applies only to one deterministic prediction per recorded vector on the audited population; additional genuinely measured inputs or another population can change it. It is not a majority-relabeling instruction.

**Evidence:** Full-tuple equality confirms 1,302 shared patterns and 8,541 participating test rows. Category overlap is consistent for 754 patterns and conflicting for 548.

**Decision:** Retain the full supplied test. Preserve masks relative both to supplied training and to actual fitting rows; name the reference population in every later diagnostic. Never resplit it.

**Still uncertain:** Dictionary hashing only locates candidates; full tuple equality determines a match. Disjoint exact vectors do not prove host/time independence.

**Evidence:** Class blocks follow ID ordering. FTP fields match throughout supplied training but differ on eight previously audited test rows; login values include 2 and 4.

**Decision:** Exclude ID, preserve the literal service marker and both numeric FTP fields. Restrict claims to offline completed-flow use.

**Still uncertain:** Contextual ct_* counts require history. We cannot verify past-only extraction, timing or real-time availability. Predictive association alone is not proof of leakage.

**Evidence:** All original-pattern groups belong to one frozen role; every pair has zero exact overlap. Model selection has 13 Worms rows in 13 patterns.

**Decision:** Adopt keep-all policy. Reuse frozen assignments rather than draw a more favourable split. Calibration remains reserved.

**Still uncertain:** Stratification cannot guarantee perfect balance with uneven/conflicting groups. Pattern support is only an upper bound on effective independent information.