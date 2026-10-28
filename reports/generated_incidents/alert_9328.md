### Incident report: alert 9328 (Shellcode)

The detector flags this flow as **Shellcode** (calibrated probability 0.78). It is confident: the prediction set contains only this class.

**Evidence**
- [E1] Predicted class: Shellcode with calibrated probability 0.78
- [E2] Conformal prediction set (90% coverage): Shellcode
- [E3] Column ct_dst_src_ltm = 1 supports Shellcode (SHAP +0.140)
- [E4] Column dinpkt = 207.869578 supports Shellcode (SHAP +0.125)
- [E5] Column tcprtt = 0.1479 supports Shellcode (SHAP +0.114)
- [E6] Column sttl = 254 supports Shellcode (SHAP +0.109)
- [E7] Column smean = 50 supports Shellcode (SHAP +0.072)

**Verified findings**
- The alert predicts a 78% probability of shellcode execution. ['E1']
- The prediction set confirms that shellcode is the most likely outcome. ['E2']
- The column ct_dst_src_ltm, which measures the destination source long-term connection, supports the prediction with a SHAP value of +0.140. ['E3']
- The column dinpkt, which measures the average packet size, supports the prediction with a SHAP value of +0.125. ['E4']
- The column tcprtt, which measures the TCP round-trip time, supports the prediction with a SHAP value of +0.114. ['E5']

**ATT&CK techniques**
- T1055 Process Injection ['S1']

**Recommended actions**
- Isolate affected systems and networks to prevent further damage. ['S5']
- Change system admin passwords and rotate private keys to mitigate potential compromise. ['S5']

**Sources**
- [S1] MITRE ATT&CK, T1055
- [S2] MITRE ATT&CK, T1204.001
- [S3] MITRE ATT&CK, T1203
- [S4] CISA IR Playbooks, page 31
- [S5] CISA IR Playbooks, page 31
- [S6] CISA IR Playbooks, page 14

*After one repair round, 0 unsupported item(s) were still removed by the verifier.*