### Incident report: alert 9216 (DoS)

The detector flags this flow as **DoS** (calibrated probability 0.37). The attack type is uncertain: the prediction set contains Analysis, Backdoor, DoS, Reconnaissance.

**Evidence**
- [E1] Predicted class: DoS with calibrated probability 0.37
- [E2] Conformal prediction set (90% coverage): Analysis, Backdoor, DoS, Reconnaissance
- [E3] Column sbytes = 200 supports DoS (SHAP +0.099)
- [E4] Column sttl = 254 supports DoS (SHAP +0.085)
- [E5] Column proto = idrp supports DoS (SHAP +0.079)
- [E6] Column smean = 100 supports DoS (SHAP +0.059)
- [E7] Column dinpkt = 0.0 argues against DoS (SHAP -0.019)

**Verified findings**
- The network alert predicted a DoS attack with a 37% probability, based on the analysis of network traffic features. ['E1']
- The prediction set includes DoS, along with other tactics such as Analysis, Backdoor, and Reconnaissance. ['E2']
- The column sbytes = 200 supports the DoS prediction with a SHAP value of +0.099. ['E3']
- The column sttl = 254 also supports the DoS prediction with a SHAP value of +0.085. ['E4']
- The column proto = idrp supports the DoS prediction with a SHAP value of +0.079. ['E5']

**ATT&CK techniques**
- T1499 Endpoint Denial of Service ['S1']
- T1498 Network Denial of Service ['S2']

**Recommended actions**

**Sources**
- [S1] MITRE ATT&CK, T1499
- [S2] MITRE ATT&CK, T1498
- [S3] MITRE ATT&CK, T1499.003
- [S4] CISA IR Playbooks, page 31
- [S5] CISA IR Playbooks, page 31
- [S6] CISA IR Playbooks, page 14

*After one repair round, 2 unsupported item(s) were still removed by the verifier.*