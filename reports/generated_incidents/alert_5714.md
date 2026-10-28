### Incident report: alert 5714 (Reconnaissance)

The detector flags this flow as **Reconnaissance** (calibrated probability 1.00). It is confident: the prediction set contains only this class.

**Evidence**
- [E1] Predicted class: Reconnaissance with calibrated probability 1.00
- [E2] Conformal prediction set (90% coverage): Reconnaissance
- [E3] Column sbytes = 168 supports Reconnaissance (SHAP +0.419)
- [E4] Column smean = 84 supports Reconnaissance (SHAP +0.185)
- [E5] Column dbytes = 0 supports Reconnaissance (SHAP +0.109)
- [E6] Column dmean = 0 supports Reconnaissance (SHAP +0.077)
- [E7] Column ct_srv_src = 5 supports Reconnaissance (SHAP +0.037)

**Verified findings**
- The alert was classified with a 100% probability of being a reconnaissance activity. ['E1']
- The prediction set includes only reconnaissance activities, indicating a high confidence in this classification. ['E2']
- The column sbytes, which measures 168 bytes, supports the prediction of reconnaissance with a SHAP value of +0.419. ['E3']
- The column smean, measuring 84 bytes, also supports the prediction of reconnaissance with a SHAP value of +0.185. ['E4']
- The column dbytes, measuring 0 bytes, supports the prediction of reconnaissance with a SHAP value of +0.109. ['E5']

**ATT&CK techniques**
- T1595 Active Scanning ['S1']

**Recommended actions**
- Monitor for signs of threat actor response to containment activities and report updated findings to CISA. ['S4']
- Isolate affected systems and networks, including perimeter and internal network containment. ['S5']

**Sources**
- [S1] MITRE ATT&CK, T1595
- [S2] MITRE ATT&CK, T1590
- [S3] MITRE ATT&CK, T1007
- [S4] CISA IR Playbooks, page 31
- [S5] CISA IR Playbooks, page 31
- [S6] CISA IR Playbooks, page 14

*After one repair round, 0 unsupported item(s) were still removed by the verifier.*