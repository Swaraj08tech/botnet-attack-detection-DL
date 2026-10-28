### Incident report: alert 4764 (Backdoor)

The detector flags this flow as **Backdoor** (calibrated probability 0.68). It is confident: the prediction set contains only this class.

**Evidence**
- [E1] Predicted class: Backdoor with calibrated probability 0.68
- [E2] Conformal prediction set (90% coverage): Backdoor
- [E3] Column dbytes = 268 supports Backdoor (SHAP +0.244)
- [E4] Column proto = tcp supports Backdoor (SHAP +0.134)
- [E5] Column sloss = 2 supports Backdoor (SHAP +0.133)
- [E6] Column smean = 62 supports Backdoor (SHAP +0.119)
- [E7] Column sbytes = 624 supports Backdoor (SHAP +0.048)

**Verified findings**
- The predicted class is Backdoor with a calibrated probability of 0.68, indicating a high likelihood of malicious activity. ['E1']
- The prediction set (90% coverage) confirms the presence of a Backdoor, aligning with the initial prediction. ['E2']
- Column dbytes = 268 supports the Backdoor prediction, contributing +0.244 to the SHAP value. ['E3']
- Column proto = tcp also supports the Backdoor prediction, adding +0.134 to the SHAP value. ['E4']

**ATT&CK techniques**
- T1571 Non-Standard Port ['S1']

**Recommended actions**
- Isolate affected systems and networks to prevent further spread of the backdoor. ['S5']
- Change system admin passwords and rotate private keys to mitigate potential compromise. ['S5']

**Sources**
- [S1] MITRE ATT&CK, T1571
- [S2] MITRE ATT&CK, T1205
- [S3] MITRE ATT&CK, T1132.002
- [S4] CISA IR Playbooks, page 31
- [S5] CISA IR Playbooks, page 31
- [S6] CISA IR Playbooks, page 15

*After one repair round, 0 unsupported item(s) were still removed by the verifier.*