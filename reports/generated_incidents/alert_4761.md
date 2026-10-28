### Incident report: alert 4761 (Backdoor)

The detector flags this flow as **Backdoor** (calibrated probability 0.44). The attack type is uncertain: the prediction set contains Analysis, Backdoor.

**Evidence**
- [E1] Predicted class: Backdoor with calibrated probability 0.44
- [E2] Conformal prediction set (90% coverage): Analysis, Backdoor
- [E3] Column proto = unas supports Backdoor (SHAP +0.194)
- [E4] Column ct_src_ltm = 12 supports Backdoor (SHAP +0.115)
- [E5] Column ct_dst_ltm = 3 supports Backdoor (SHAP +0.098)
- [E6] Column smean = 100 argues against Backdoor (SHAP -0.093)
- [E7] Column sloss = 0 supports Backdoor (SHAP +0.059)

**Verified findings**
- The alert predicts a backdoor with a 44% probability, based on the SHAP values of the columns involved. ['E1']
- The prediction set includes Analysis and Backdoor, indicating a high likelihood of malicious activity. ['E2']
- Column proto = unas supports the prediction of a backdoor, with a SHAP value of +0.194. ['E3']
- Column ct_src_ltm = 12 also supports the backdoor prediction, with a SHAP value of +0.115. ['E4']

**ATT&CK techniques**
- T1571 Non-Standard Port ['S1']

**Recommended actions**
- Isolate affected systems and networks to prevent further spread of the potential backdoor. ['S5']
- Monitor for signs of threat actor response and gather additional evidence. ['S4']

**Sources**
- [S1] MITRE ATT&CK, T1571
- [S2] MITRE ATT&CK, T1205
- [S3] MITRE ATT&CK, T1132.002
- [S4] CISA IR Playbooks, page 31
- [S5] CISA IR Playbooks, page 31
- [S6] CISA IR Playbooks, page 15

*After one repair round, 0 unsupported item(s) were still removed by the verifier.*