### Incident report: alert 5242 (Shellcode)

The detector flags this flow as **Shellcode** (calibrated probability 0.84). The attack type is uncertain: the prediction set contains Shellcode, Worms.

**Evidence**
- [E1] Predicted class: Shellcode with calibrated probability 0.84
- [E2] Conformal prediction set (90% coverage): Shellcode, Worms
- [E3] Column service = - supports Shellcode (SHAP +0.227)
- [E4] Column ct_src_dport_ltm = 1 supports Shellcode (SHAP +0.104)
- [E5] Column proto = udp supports Shellcode (SHAP +0.092)
- [E6] Column spkts = 2 supports Shellcode (SHAP +0.078)
- [E7] Column dmean = 0 supports Shellcode (SHAP +0.058)

**Verified findings**
- The alert predicts a 0.84 probability of Shellcode execution. ['E1']
- The prediction set includes Shellcode and Worms, with Shellcode being the most likely outcome. ['E2']
- The column 'service' supports the prediction of Shellcode with a SHAP value of +0.227. ['E3']
- The column 'ct_src_dport_ltm' also supports the prediction of Shellcode with a SHAP value of +0.104. ['E4']
- The column 'proto' supporting Shellcode has a SHAP value of +0.092. ['E5']

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