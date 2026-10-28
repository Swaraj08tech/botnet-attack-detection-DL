### Incident report: alert 6810 (Shellcode)

The detector flags this flow as **Shellcode** (calibrated probability 0.95). It is confident: the prediction set contains only this class.

**Evidence**
- [E1] Predicted class: Shellcode with calibrated probability 0.95
- [E2] Conformal prediction set (90% coverage): Shellcode
- [E3] Column service = - supports Shellcode (SHAP +0.248)
- [E4] Column sbytes = 148 supports Shellcode (SHAP +0.206)
- [E5] Column ct_src_dport_ltm = 1 supports Shellcode (SHAP +0.173)
- [E6] Column proto = udp supports Shellcode (SHAP +0.085)
- [E7] Column ct_dst_src_ltm = 1 supports Shellcode (SHAP +0.077)

**Verified findings**
- The alert predicts Shellcode with a calibrated probability of 0.95, indicating a high likelihood of malicious activity. ['E1']
- The prediction set confirms that the activity is within the expected range of Shellcode, with 90% coverage. ['E2']
- The column 'service' supports the prediction of Shellcode with a SHAP value of +0.248, indicating its importance in the prediction. ['E3']
- The column 'sbytes' also supports the prediction of Shellcode with a SHAP value of +0.206, highlighting its significance in the detection. ['E4']
- The column 'ct_src_dport_ltm' supports the prediction of Shellcode with a SHAP value of +0.173, further emphasizing its relevance. ['E5']

**ATT&CK techniques**
- T1055 Process Injection ['S1']

**Recommended actions**
- Isolate affected systems and networks to prevent further damage and reduce the impact of the incident. ['S5']
- Change system admin passwords and rotate private keys to mitigate potential compromise. ['S5']

**Sources**
- [S1] MITRE ATT&CK, T1055
- [S2] MITRE ATT&CK, T1204.001
- [S3] MITRE ATT&CK, T1203
- [S4] CISA IR Playbooks, page 31
- [S5] CISA IR Playbooks, page 31
- [S6] CISA IR Playbooks, page 14

*After one repair round, 0 unsupported item(s) were still removed by the verifier.*