### Incident report: alert 5323 (Worms)

The detector flags this flow as **Worms** (calibrated probability 0.98). It is confident: the prediction set contains only this class.

**Evidence**
- [E1] Predicted class: Worms with calibrated probability 0.98
- [E2] Conformal prediction set (90% coverage): Worms
- [E3] Column service = http supports Worms (SHAP +0.399)
- [E4] Column sbytes = 1296 supports Worms (SHAP +0.309)
- [E5] Column smean = 130 supports Worms (SHAP +0.112)
- [E6] Column dbytes = 354 supports Worms (SHAP +0.068)
- [E7] Column ackdat = 0.061552 supports Worms (SHAP +0.027)

**Verified findings**
- The predicted class is Worms with a calibrated probability of 0.98, indicating a high likelihood of a Worms attack. ['E1']
- The prediction set for this alert is Worms with 90% coverage, suggesting that the model is confident in this prediction. ['E2']
- The column 'service = http' supports the prediction of Worms with a SHAP value of +0.399, indicating its importance in the prediction. ['E3']
- The column 'sbytes = 1296' also supports the prediction of Worms with a SHAP value of +0.309, highlighting its significance. ['E4']

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