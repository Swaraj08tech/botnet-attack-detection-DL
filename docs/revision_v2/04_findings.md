# Notebook 04: revised findings


## Findings
Three transparent completed-flow summaries reproduce the earlier engineered inputs. Byte and packet balances are strongly related but not identical. Extra packet-size/rate ideas overlap existing measurements.


## Decisions
Retain total bytes, byte balance and packet balance as candidates; preserve all 42 original predictors and both preprocessing baselines. Use the same zero sentinel and formulas in every role.


## Limitations
No new measurements, botnet ground truth, DNS transactions or time windows were created. Descriptive target association is not evidence of model improvement. The zero sentinel needs original counts for interpretation.


## Saved outputs
- `results/tables/revision_v2/04_role_class_support.csv`
- `results/tables/revision_v2/04_proposals.csv`
- `results/tables/revision_v2/04_worked_example.csv`
- `results/tables/revision_v2/04_candidate_summary.csv`
- `results/figures/revision_v2/04_engineered_distributions.png`
- `results/tables/revision_v2/04_balance_redundancy.csv`
- `results/figures/revision_v2/04_balance_relationship.png`
- `results/tables/revision_v2/04_target_relationships.csv`
- `results/tables/revision_v2/04_existing_information.csv`
- `results/tables/revision_v2/04_decisions.csv`
- `results/tables/revision_v2/04_zero_denominators.csv`
- `results/tables/revision_v2/04_extreme_examples.csv`
- `results/tables/revision_v2/04_preserved_04_redundancy_checks_fit.csv`
- `results/figures/revision_v2/04_original_and_engineered.png`
- `data/processed/revision_v2/04_engineered_train.csv`
- `data/processed/revision_v2/04_row_links_train.csv`
- `data/processed/revision_v2/04_engineered_model_selection.csv`
- `data/processed/revision_v2/04_row_links_model_selection.csv`
- `data/processed/revision_v2/04_engineered_calibration.csv`
- `data/processed/revision_v2/04_row_links_calibration.csv`
- `results/tables/revision_v2/04_role_verification.csv`
- `results/tables/revision_v2/04_representation_overlap.csv`


## Next question
Which focused combinations should be carried forward, with redundancy and stability evidence?


## Section evidence
**Evidence:** Byte/packet balances correlate 0.9734 by row and 0.8940 by distinct original pattern. They are related, not exact copies.

**Decision:** Retain total bytes and both balances as predeclared model candidates. Reject extra average-size/rate fields from this small set.

**Still uncertain:** These formulas add no new measured information. Only matched model experiments can show whether this representation helps; source/destination are not proven client/server.

**Evidence:** No fitting row has a zero total-byte or total-packet denominator. The zero rule is still tested with toy flows. Extreme examples expose which source counts create the totals; rate alternatives closely track existing rate but differ in formula.

**Decision:** Keep the transparent three-feature contract and the negative redundancy evidence. Do not clip high byte counts or call a rounded average a new measurement.

**Still uncertain:** Class medians summarize heterogeneous traffic. A high correlation, central trend or extreme example cannot predict a model benefit; the engineered-three-only overlap diagnostic discards original information and is not a model input set.

**Evidence:** The same three stateless formulas reproduce saved engineered values in every development role, with exact CSV round trips and frozen row links.

**Decision:** Use all original features as the comparison baseline. Use previously train-fitted scaling of engineered columns, verified by the protected manifests.

**Still uncertain:** Calibration receives the fixed transformation for integrity only. No labels occur in formulas, and no validation outcome chooses a formula.