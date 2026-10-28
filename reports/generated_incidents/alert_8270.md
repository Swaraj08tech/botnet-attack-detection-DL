### Incident report: alert 8270 (Analysis)

The detector flags this flow as **Analysis** (calibrated probability 1.00). It is confident: the prediction set contains only this class.

**Evidence**
- [E1] Predicted class: Analysis with calibrated probability 1.00
- [E2] Conformal prediction set (90% coverage): Analysis
- [E3] Column ct_flw_http_mthd = 30 supports Analysis (SHAP +0.369)
- [E4] Column ackdat = 0.187635 supports Analysis (SHAP +0.159)
- [E5] Column response_body_len = 156 supports Analysis (SHAP +0.132)
- [E6] Column proto = tcp supports Analysis (SHAP +0.125)
- [E7] Column sttl = 62 supports Analysis (SHAP +0.100)

**Verified findings**
- The alert has a predicted class of Analysis with a calibrated probability of 1.00, indicating a high confidence level. ['E1']
- The prediction set includes only the class 'Analysis', suggesting this is the primary focus of the alert. ['E2']
- The column 'ct_flw_http_mthd' with a value of 30 supports the Analysis prediction, contributing +0.369 to the SHAP value. ['E3']
- The column 'response_body_len' with a value of 156 also supports the Analysis prediction, adding +0.132 to the SHAP value. ['E5']

**ATT&CK techniques**
- T1007 System Service Discovery ['S1']
- T1595 Active Scanning ['S2']

**Recommended actions**

**Sources**
- [S1] MITRE ATT&CK, T1007
- [S2] MITRE ATT&CK, T1595
- [S3] MITRE ATT&CK, T1046
- [S4] CISA IR Playbooks, page 31
- [S5] CISA IR Playbooks, page 2
- [S6] CISA IR Playbooks, page 31

*After one repair round, 2 unsupported item(s) were still removed by the verifier.*