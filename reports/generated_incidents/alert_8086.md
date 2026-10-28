### Incident report: alert 8086 (Analysis)

The detector flags this flow as **Analysis** (calibrated probability 0.98). It is confident: the prediction set contains only this class.

**Evidence**
- [E1] Predicted class: Analysis with calibrated probability 0.98
- [E2] Conformal prediction set (90% coverage): Analysis
- [E3] Column ct_flw_http_mthd = 16 supports Analysis (SHAP +0.332)
- [E4] Column response_body_len = 132 supports Analysis (SHAP +0.211)
- [E5] Column sttl = 62 supports Analysis (SHAP +0.108)
- [E6] Column proto = tcp supports Analysis (SHAP +0.091)
- [E7] Column dbytes = 1262 supports Analysis (SHAP +0.084)

**Verified findings**
- The network alert indicates a high probability of Analysis activity, with a 98% confidence level, based on the presence of specific network traffic patterns [E1]. ['E1']
- The prediction set includes Analysis with 90% coverage, suggesting that the alert is part of a broader set of similar alerts [E2]. ['E2']
- The column ct_flw_http_mthd = 16, which supports Analysis (SHAP +0.332), is the most influential factor in the prediction [E3]. ['E3']
- The column response_body_len = 132, which supports Analysis (SHAP +0.211), is the second most influential factor in the prediction [E4]. ['E4']
- The column sttl = 62, which supports Analysis (SHAP +0.108), is the third most influential factor in the prediction [E5]. ['E5']

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