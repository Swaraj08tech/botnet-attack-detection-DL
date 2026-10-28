### Incident report: alert 6392 (Reconnaissance)

The detector flags this flow as **Reconnaissance** (calibrated probability 1.00). It is confident: the prediction set contains only this class.

**Evidence**
- [E1] Predicted class: Reconnaissance with calibrated probability 1.00
- [E2] Conformal prediction set (90% coverage): Reconnaissance
- [E3] Column sbytes = 564 supports Reconnaissance (SHAP +0.422)
- [E4] Column dbytes = 354 supports Reconnaissance (SHAP +0.196)
- [E5] Column sttl = 254 supports Reconnaissance (SHAP +0.146)
- [E6] Column ct_dst_src_ltm = 1 supports Reconnaissance (SHAP +0.102)
- [E7] Column state = FIN argues against Reconnaissance (SHAP -0.064)

**Verified findings**
- The alert has a predicted class of Reconnaissance with a calibrated probability of 1.00, indicating a very high likelihood of this activity. ['E1']
- The prediction set includes only Reconnaissance, suggesting a high confidence in this classification. ['E2']
- The column sbytes = 564 supports Reconnaissance, with a SHAP value of +0.422, indicating its importance in the prediction. ['E3']
- The column dbytes = 354 also supports Reconnaissance, with a SHAP value of +0.196, further reinforcing the prediction. ['E4']
- The column sttl = 254 supports Reconnaissance, with a SHAP value of +0.146, contributing to the overall prediction. ['E5']

**ATT&CK techniques**
- T1595 Active Scanning ['S1']

**Recommended actions**
- Monitor for signs of threat actor response to containment activities and report updated findings to CISA. ['S4']
- Isolate affected systems and networks, including perimeter and internal network containment, to prevent further damage. ['S5']

**Sources**
- [S1] MITRE ATT&CK, T1595
- [S2] MITRE ATT&CK, T1590
- [S3] MITRE ATT&CK, T1007
- [S4] CISA IR Playbooks, page 31
- [S5] CISA IR Playbooks, page 31
- [S6] CISA IR Playbooks, page 14

*After one repair round, 0 unsupported item(s) were still removed by the verifier.*