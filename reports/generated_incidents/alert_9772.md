### Incident report: alert 9772 (Exploits)

The detector flags this flow as **Exploits** (calibrated probability 0.69). The attack type is uncertain: the prediction set contains DoS, Exploits, Worms.

**Evidence**
- [E1] Predicted class: Exploits with calibrated probability 0.69
- [E2] Conformal prediction set (90% coverage): DoS, Exploits, Worms
- [E3] Column dttl = 60 supports Exploits (SHAP +0.155)
- [E4] Column sttl = 254 supports Exploits (SHAP +0.148)
- [E5] Column smean = 1014 supports Exploits (SHAP +0.085)
- [E6] Column ct_dst_src_ltm = 1 supports Exploits (SHAP +0.055)
- [E7] Column ct_srv_dst = 1 supports Exploits (SHAP +0.033)

**Verified findings**
- The network alert indicates a high probability of an exploit being attempted, with a 69% likelihood, based on the SHAP values of specific columns. ['E1']
- The prediction set includes DoS, Exploits, and Worms, with Exploits being the most likely outcome. ['E2']
- The column dttl = 60 supports the prediction of Exploits, with a SHAP value of +0.155. ['E3']
- The column sttl = 254 also supports the prediction of Exploits, with a SHAP value of +0.148. ['E4']
- The column smean = 1014 supports the prediction of Exploits, with a SHAP value of +0.085. ['E5']

**ATT&CK techniques**
- T1588.005 Exploits ['S1']

**Recommended actions**
- Monitor for signs of threat actor response to containment activities and report updated findings to CISA. ['S4']
- Isolate affected systems and networks, including perimeter and internal network containment, and close specific ports and mail servers. ['S5']

**Sources**
- [S1] MITRE ATT&CK, T1588.005
- [S2] MITRE ATT&CK, T1587.004
- [S3] MITRE ATT&CK, T1190
- [S4] CISA IR Playbooks, page 31
- [S5] CISA IR Playbooks, page 31
- [S6] CISA IR Playbooks, page 14

*After one repair round, 0 unsupported item(s) were still removed by the verifier.*