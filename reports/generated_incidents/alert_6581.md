### Incident report: alert 6581 (Fuzzers)

The detector flags this flow as **Fuzzers** (calibrated probability 0.64). It is confident: the prediction set contains only this class.

**Evidence**
- [E1] Predicted class: Fuzzers with calibrated probability 0.64
- [E2] Conformal prediction set (90% coverage): Fuzzers
- [E3] Column sbytes = 1780 supports Fuzzers (SHAP +0.128)
- [E4] Column synack = 0.066007 supports Fuzzers (SHAP +0.113)
- [E5] Column sttl = 254 supports Fuzzers (SHAP +0.090)
- [E6] Column ct_dst_src_ltm = 11 supports Fuzzers (SHAP +0.059)
- [E7] Column service = - supports Fuzzers (SHAP +0.053)

**Verified findings**
- The alert predicts a fuzzer attack with a calibrated probability of 0.64. ['E1']
- The prediction set includes only fuzzer attacks, with a 90% coverage probability. ['E2']
- The column 'sbytes' supports the prediction, with a SHAP value of +0.128. ['E3']
- The column 'synack' also supports the prediction, with a SHAP value of +0.113. ['E4']
- The column 'sttl' supports the prediction, with a SHAP value of +0.090. ['E5']

**ATT&CK techniques**
- T1595.002 Vulnerability Scanning ['S1']

**Recommended actions**

**Sources**
- [S1] MITRE ATT&CK, T1595.002
- [S2] MITRE ATT&CK, T1499.004
- [S3] MITRE ATT&CK, T1204.001
- [S4] CISA IR Playbooks, page 31
- [S5] CISA IR Playbooks, page 31
- [S6] CISA IR Playbooks, page 14

*After one repair round, 2 unsupported item(s) were still removed by the verifier.*