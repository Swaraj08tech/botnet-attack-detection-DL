# Key Dataset Findings

1. **Training size:** 175,341 rows, 45 columns (`UNSW_NB15_training-set.csv`).
2. **Testing size:** 82,332 rows, 45 columns (`UNSW_NB15_testing-set.csv`).
3. **Feature count:** 42 candidate predictors after separately identifying ID and the two targets. No columns were removed.
4. **Numerical features:** 39 candidate predictors. Counts of numeric columns including ID and targets differ from predictor counts.
5. **Categorical features:** proto, service, state; no actual boolean dtypes in either split.
6. **Targets:** label (binary) and attack_cat (10 observed categories, including Normal). Normal/label=0 contradictions: {'train': 0, 'test': 0}.
7. **Data quality:** missing cells train/test 0/0; infinite cells 0/0. Constant columns: {'train': [], 'test': []}. Dash placeholders are counted separately from NaN.
8. **Duplicates:** full-row train/test 0/0; excluding ID and targets 74,301/28,386 excess rows. Shared unique feature vectors: 1,302; participating train/test rows: 10,279/8,541. Shared vectors with conflicting target combinations: 548. Matching identical features do not establish identical source events.
9. **Imbalance:** attack share is 68.06% in train vs 55.06% in test. Multiclass majority/minority ratio is 430.77:1 in train and 840.91:1 in test.
10. **Rare attack categories (<1%):** train: Backdoor: 1,746 (0.996%); Shellcode: 1,133 (0.646%); Worms: 130 (0.074%). Test: Analysis: 677 (0.822%); Backdoor: 583 (0.708%); Shellcode: 378 (0.459%); Worms: 44 (0.053%). These limited counts make later class-wise estimates sensitive to a few errors.
11. **Numerical flags:** negative-value columns {'train': [], 'test': []}; largest observed train maximum is sload=5.988e+09. Investigation flags affect dur, spkts, dpkts, sbytes, dbytes, rate, sload, dload, sloss, dloss, sinpkt, dinpkt, sjit, djit, stcpb, dtcpb, synack, ackdat, trans_depth, response_body_len, is_ftp_login, ct_ftp_cmd, ct_flw_http_mthd, is_sm_ips_ports. These flags do not establish invalid values.
12. **Train/test differences:** missing columns []; extra columns []; same order True; dtype mismatches {}. Test-only category values: state: ['ACC', 'CLO']. Largest numeric mean differences: tcprtt, dttl, synack, swin, ackdat.
13. **Leakage risks:** both targets must be excluded from predictors. attack_cat carries binary-label information. Unique sequential ID may reflect target ordering. Repeated vectors across splits can inflate apparent generalization, and conflicting targets warrant provenance and ambiguity investigation. Low-cardinality purity is a screening diagnostic, not proof of leakage.
14. **Future preprocessing requirements:** establish target policy and a validation protocol; fit categorical encoders and numeric transforms only on the training partition; support unknown categorical values; investigate placeholders, skew and repeated/conflicting records before deciding how to handle them. No preprocessing choice has been fitted here.
15. **Open questions:** independently verify source provenance and feature meanings. DNS/domain/query/entropy-named columns: none; explicit start/end timestamp-named columns: none. Their absence by name does not prove all DNS-related information is absent, but DNS modality and temporal ordering remain unestablished. Attack categories contain a literal Botnet class: False. The broad IDS dataset does not by itself establish botnet-specific ground truth.

## Conclusion
The source files pass the recorded integrity, missingness, and schema checks. Repeated feature vectors, target conflicts, ID ordering and split differences require investigation before preprocessing or modelling. No records were changed or removed.

**Recommended Notebook 02:** Duplicate, Target Ambiguity, and Leakage Investigation. Review within-split target conflicts, cross-split overlap, ID ordering, and feature provenance alongside the reference paper. Document an evaluation protocol while retaining the official test file intact. Do not create Notebook 02 until review.
