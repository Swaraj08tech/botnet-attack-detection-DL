### Incident report: alert 4751 (Backdoor)

The detector flags this flow as **Backdoor** (calibrated probability 0.90). It is confident: the prediction set contains only this class.

**Evidence**
- [E1] Predicted class: Backdoor with calibrated probability 0.90
- [E2] Conformal prediction set (90% coverage): Backdoor
- [E3] Column ct_src_ltm = 11 supports Backdoor (SHAP +0.247)
- [E4] Column proto = mtp supports Backdoor (SHAP +0.116)
- [E5] Column dbytes = 0 supports Backdoor (SHAP +0.091)
- [E6] Column service = - supports Backdoor (SHAP +0.048)
- [E7] Column sttl = 254 supports Backdoor (SHAP +0.044)

**Verified findings**
- The predicted class is Backdoor with a calibrated probability of 0.90. ['E1']
- The prediction set confirms the Backdoor prediction with 90% coverage. ['E2']
- The column ct_src_ltm = 11 supports the Backdoor prediction with a SHAP value of +0.247. ['E3']
- The column proto = mtp supports the Backdoor prediction with a SHAP value of +0.116. ['E4']

**ATT&CK techniques**
- T1571 Non-Standard Port ['S1']

**Recommended actions**
- Isolate affected systems and networks to prevent further spread of the backdoor. ['S5']

**Sources**
- [S1] MITRE ATT&CK, T1571
- [S2] MITRE ATT&CK, T1205
- [S3] MITRE ATT&CK, T1132.002
- [S4] CISA IR Playbooks, page 31
- [S5] CISA IR Playbooks, page 31
- [S6] CISA IR Playbooks, page 15

*After one repair round, 0 unsupported item(s) were still removed by the verifier.*