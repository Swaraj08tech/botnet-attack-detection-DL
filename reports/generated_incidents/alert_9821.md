### Incident report: alert 9821 (Analysis)

The detector flags this flow as **Analysis** (calibrated probability 0.68). It is confident: the prediction set contains only this class.

**Evidence**
- [E1] Predicted class: Analysis with calibrated probability 0.68
- [E2] Conformal prediction set (90% coverage): Analysis
- [E3] Column sbytes = 828 supports Analysis (SHAP +0.241)
- [E4] Column dload = 10501.60449 supports Analysis (SHAP +0.144)
- [E5] Column stcpb = 2738155303 supports Analysis (SHAP +0.095)
- [E6] Column response_body_len = 127 supports Analysis (SHAP +0.077)
- [E7] Column ct_dst_src_ltm = 2 supports Analysis (SHAP +0.037)

**Verified findings**
- The network alert indicates a high probability of Analysis, with a calibrated probability of 0.68, and is part of a 90% coverage prediction set. ['E1', 'E2']
- The column sbytes = 828 supports Analysis (SHAP +0.241). ['E3']
- The column dload = 10501.60449 supports Analysis (SHAP +0.144). ['E4']
- The column stcpb = 2738155303 supports Analysis (SHAP +0.095). ['E5']

**ATT&CK techniques**
- T1007 System Service Discovery ['S1']
- T1595 Active Scanning ['S2']

**Recommended actions**
- Isolate affected systems and networks, including perimeter and internal network containment, and host-based/endpoint containment. ['S4', 'S6']
- Change system admin passwords, rotate private keys and service/application account secrets where compromise is suspected, and revoke privileged access. ['S6']

**Sources**
- [S1] MITRE ATT&CK, T1007
- [S2] MITRE ATT&CK, T1595
- [S3] MITRE ATT&CK, T1046
- [S4] CISA IR Playbooks, page 31
- [S5] CISA IR Playbooks, page 2
- [S6] CISA IR Playbooks, page 31

*After one repair round, 0 unsupported item(s) were still removed by the verifier.*