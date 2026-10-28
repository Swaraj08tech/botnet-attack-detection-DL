### Incident report: alert 7905 (Worms)

The detector flags this flow as **Worms** (calibrated probability 0.83). It is confident: the prediction set contains only this class.

**Evidence**
- [E1] Predicted class: Worms with calibrated probability 0.83
- [E2] Conformal prediction set (90% coverage): Worms
- [E3] Column sload = 359111104.0 supports Worms (SHAP +0.383)
- [E4] Column dur = 9e-06 supports Worms (SHAP +0.157)
- [E5] Column ct_dst_src_ltm = 1 supports Worms (SHAP +0.152)
- [E6] Column sbytes = 808 supports Worms (SHAP +0.060)
- [E7] Column sttl = 254 supports Worms (SHAP +0.043)

**Verified findings**
- The network alert predicted a Worms attack with a calibrated probability of 0.83. ['E1']
- The prediction set confirms that the alert is within the Worms category. ['E2']
- Column sload = 359111104.0 supports the Worms prediction with a SHAP value of +0.383. ['E3']
- Column dur = 9e-06 supports the Worms prediction with a SHAP value of +0.157. ['E4']
- Column ct_dst_src_ltm = 1 supports the Worms prediction with a SHAP value of +0.152. ['E5']

**ATT&CK techniques**
- T1210 Exploitation of Remote Services ['S1']

**Recommended actions**

**Sources**
- [S1] MITRE ATT&CK, T1210
- [S2] MITRE ATT&CK, T1570
- [S3] MITRE ATT&CK, T1105
- [S4] CISA IR Playbooks, page 31
- [S5] CISA IR Playbooks, page 31
- [S6] CISA IR Playbooks, page 14

*After one repair round, 2 unsupported item(s) were still removed by the verifier.*