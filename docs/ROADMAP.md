# Final project roadmap (agreed 18 October 2026)

**Goal:** a research-paper-quality network attack detector with trustworthy, explained and evidence-backed alerts, delivered as a
professional, deployable web application.

## Decisions

| Topic | Decision |
|---|---|
| Botnet dataset CTU-13 | **Not included** |
| LLM for incident reports | **Free, local model via Ollama** (no API costs, flow data stays on the machine) |
| Tech stack | **FastAPI** backend · **React + TypeScript** (Vite, Tailwind CSS, shadcn/ui components, Recharts) frontend · **PostgreSQL** · **Ollama** · **Docker + docker-compose** · **GitHub Actions** CI · Kubernetes manifests in `deployment/kubernetes/` |
| Dataset | **The official training and testing CSVs only** (175,341 + 82,332 flows; SHA-256 confirmed identical to the official UNSW files). The full 2.54M-record version is downloaded to `data/raw/unsw_nb15_full/` but **not used**; it is kept only as possible future work on time windows |
| Notebook numbering | Existing files 01–08 keep their names; new notebooks continue from 09 |

## Research notebooks

| No. | Notebook | Topic | Status |
|---|---|---|---|
| 01 | Data understanding | What is in the UNSW-NB15 files | ✅ Done |
| 02 | Repeats, conflicts, fair split | Duplicates, label conflicts, leakage-safe split | ✅ Done |
| 03 | EDA and preprocessing | Class patterns, model inputs | ✅ Done |
| 04 | Feature engineering | Total bytes, byte balance, packet balance | ✅ Done |
| 05 | Feature selection | Which features go forward | ✅ Done |
| 06 | Baseline models | Logistic regression, decision tree, LightGBM | ✅ Done |
| 07 | Neural reference | MLP | ✅ Done |
| 08 | Reference model | Paper's Conv1D + attention on single flows | ✅ Done |
| 09 | Leakage study | Random split vs drop-duplicates split vs leakage-safe split: measured score inflation (C1) | ✅ Done: random split inflates only +0.004; dedup biases −0.013; label conflicts dominate |
| 10 | Rare-class handling | Class weighting and SMOTE, tested on LightGBM **and** the neural network with matched rules | ✅ sqrt class weights best for both families: LightGBM 0.632→0.666, network 0.513→0.587; SMOTE does not help |
| 11 | Multi-scale Conv1D | Parallel convolutions with kernel sizes 3, 5 and 7 over the ordered flow features | ✅ kept (+0.006, seed-dependent); mainly more stable across seeds |
| 12 | Cross-attention fusion | Flow-measurement features attend to the context-counter and protocol/service/state features, and back | ✅ kept (+0.002, seed-dependent); fewer false alarms |
| 13 | TCN-BiLSTM | Tested honestly as a block over the feature sequence (not time); kept only if the ablation shows it helps | ✅ kept (+0.007, seed-dependent); feature sequence, not time |
| 14 | Proposed model | Best combination from 10–13: multi-scale Conv1D + cross-attention (+ rare-class handling) | ✅ proposed network 0.603 ± 0.011: beats MLP and reference network for every seed, below LightGBM (sqrt) 0.666 |
| 15 | Ablation study | Remove each part; 3 seeds; bootstrap intervals; compared with LightGBM, MLP and the reference model (C4) | ✅ only class weighting is clearly essential (−0.066 without); blocks ≤0.009; final detector = LightGBM (sqrt) |
| 16 | SHAP and LIME | Explanations of the final model, plus a faithfulness check | ✅ SHAP exact and faithful (top column 17× random); top drivers sttl, sbytes, service |
| 17 | Calibration and conformal sets | Temperature scaling on the reserved calibration rows; conflict-aware prediction sets; abstention (C2, C3) | ✅ already calibrated (T = 1.001); class-wise sets 0.928 coverage; 95% of conflicting flows get multi-class sets |
| 18 | RAG evidence and verified reports | ATT&CK / NIST / CISA knowledge base, local LLM, automatic claim checking (C5) | ✅ with retrieval 99% of claims / 97-100% of ATT&CK IDs verified; without sources the LLM invents IDs (7-8 fake) |
| 19 | Final results | One-time evaluation on the official testing CSV, with "seen in training" vs "new" rows; all paper tables and figures | ✅ test macro F1 0.518 (accuracy 0.728, attack recall 98.8%, FPR 32.5%); shift in sttl between official files causes most false alarms; coverage 0.80 |
| 20 | Shift-aware detector (v2) | Label-free shift diagnosis (adversarial AUC 0.896), 8 candidates, importance-weighted selection and tuning; no test labels | ✅ v2 = LightGBM, all columns, sqrt × importance weights, 400 trees |
| 21 | Final results of v2 | Second (disclosed) test evaluation; weighted conformal; check of the label-free selection | ✅ test macro F1 0.530 (v1 0.518, clearly better), FPR 30.5%; dropping TTL does not help; weighted conformal does not restore coverage |

## Application (Phase E)

| Step | Content |
|---|---|
| E1 Backend | ✅ FastAPI: upload + validation, staged analysis jobs, one analysis object (confidence, severity, risk rules in one module), paginated flow predictions, TreeSHAP explanations, PDF security reports, optional verified LLM explanations, saved test-set evaluation; serving bundle verified against detector v2 |
| E2 Frontend | ✅ React + TypeScript: Overview, Analyze Flows (validation + real stages), Results (traffic donut, risk, attack distribution, confidence, Sankey, flow table), flow detail, Model (metrics, confusion matrix, comparison, rules), Reports (PDF preview / download), About |
| E3 Testing and security | ✅ 38 backend tests + 10 frontend tests; full official test file analysed end to end reproduces the saved test metrics; bcrypt + JWT, login rate limit, upload validation |
| E4 Deployment | ✅ Dockerfiles, docker-compose, Kubernetes manifests, GitHub Actions CI + release; images not yet built locally (Docker not installed) |
| E5 Documentation | ✅ README (results + start steps), docs/ARCHITECTURE.md (diagrams), docs/USER_GUIDE.md, docs/DEPLOYMENT.md, in-app model card; one-click `setup-app.cmd` / `start-app.cmd`. A demo video is left to record by hand |

The app can be built in parallel with notebooks 09–19, starting from today's LightGBM through a versioned model interface,
and upgraded to the final model when it exists.

## Planned contributions

| # | Contribution |
|---|---|
| C1 | Measured effect of the split method on UNSW-NB15: duplicate leakage barely inflates scores (+0.004), the common deduplication fix biases them downward (−0.013), and label conflicts, not leakage, are the dominant problem (Notebook 09) |
| C2 | Label-conflict ceiling: the highest overall accuracy any model can reach (90.8% training file, 93.6% test file), plus per-class majority-vote references |
| C3 | Conflict-aware conformal prediction sets and abstention |
| C4 | Proposed flow model: multi-scale Conv1D + cross-attention between feature groups, with rare-class handling, proven by ablation against strong baselines. A win is not assumed; a well-measured negative result is still reported. Needs a deeper literature check before claiming novelty |
| C5 | Verified, evidence-grounded incident reports (each claim checked against SHAP features and retrieved ATT&CK / NIST / CISA text) |

## Research framing (paper point of view)

**Working title:** *Trustworthy Network Attack Detection on UNSW-NB15: Leakage-Safe Evaluation, Label-Conflict-Aware Confidence and
Verified Evidence-Grounded Incident Reports*

### Research questions (fixed before the experiments)

| RQ | Question | Notebooks |
|---|---|---|
| RQ1 | How much do duplicate and conflicting records inflate reported scores on UNSW-NB15? | 02, 09 |
| RQ2 | What is the highest score any model can reach, given label conflicts, and how close do models get? | 02, 06–08, 19 |
| RQ3 | Does a multi-scale Conv1D + cross-attention model beat strong baselines on per-flow data, and which parts matter? | 10–15 |
| RQ4 | Can conformal prediction sets express the label ambiguity honestly while keeping coverage? | 17 |
| RQ5 | Can every claim in an LLM incident report be automatically checked against the model's evidence and trusted sources? | 16, 18 |

### Hypotheses

- **H1:** a random split inflates macro F1 compared with the leakage-safe split. → **Not supported** for LightGBM (Notebook 09): +0.004, within noise.
- **H2:** the proposed model beats LightGBM on the leakage-safe split. → **Not supported** (0.603 vs 0.666 on model_selection; 0.474 vs 0.518 on test). If it does not, we report that honestly as a negative result.
- **H3:** conformal sets reach their target coverage, and their multi-class sets concentrate on the conflicting classes. → **Supported on model_selection** (0.928; 95% vs 12%); **not under the test-file shift** (0.80).
- **H4:** verified reports contain fewer unsupported claims than unverified RAG reports. → **Supported** (Notebook 18).

### Evaluation standards for every result

- **Main metric** is macro F1 over the 10 classes, plus per-class precision, recall and F1, confusion matrices, attack recall and false-alarm rate. Accuracy is never the headline.
- **Uncertainty:** 3 seeds (mean ± std), bootstrap 95% intervals and seed-paired comparisons. A difference counts only if it holds across seeds.
- **No tuning on the official test file.** It is used once, in notebook 19.
- **Fair baselines:** same rows, same budget, same rules for every compared model.
- **Every number in the notebook text is checked against the actual outputs.**
- **Ceiling-aware reporting:** results are shown next to the label-conflict ceiling.

### Novelty safeguards

- Before writing claims: a **systematic literature review** (last 3 years, IEEE / ACM / Springer / Elsevier / arXiv) with a novelty matrix comparing our contributions with the closest papers. C4 needs this before it is claimed.
- Contributions are phrased so they stay true even if the proposed model does not win (C1, C2, C3 and C5 do not depend on it).

### Threats to validity (reported in the paper)

- UNSW-NB15 is a testbed dataset; TTL-like features may be testbed-specific.
- Very small classes (Worms: 13 model-selection rows) make per-class results unstable.
- There is no DNS query-level data and no timestamps; the model works on single completed flows, and its sequence-style blocks (Conv1D, TCN-BiLSTM) run over the ordered features, not over time.
- Distinct records are not proven to be independent network events.

### Paper deliverables

| Item | Content |
|---|---|
| P1 | Literature review and novelty matrix (before notebook 11) |
| P2 | Paper draft: introduction, related work, method, experiments (all tables and figures from notebook 19), discussion, threats to validity |
| P3 | Reproducibility package: code, frozen splits, seeds, environment versions, one command to rerun |

## Environment notes (checked 18 Oct 2026)

- Laptop: i7-14650HX, 16 GB RAM, **NVIDIA RTX 4060 Laptop GPU (8 GB)**, 585 GB free on D:.
- **Ollama is installed.** An 8-billion-parameter model (for example Llama 3.1 8B or Qwen2.5 7B, quantised) fits the GPU.
- PyTorch 2.14.0+cu130 (CUDA) is installed in `.venv` and verified on the RTX 4060; notebooks 11–15 train on the GPU. Its CPU kernels are slower than the old CPU build, so timings are always printed from the run itself.
- Node.js and Git are installed; **Docker is not installed** (needed for deployment).

## Quality checklist (applied to every notebook before it is called done)

"100% accuracy" for this project means every analysis, diagnosis, number and conclusion is correct, complete and verifiable.

1. **Correct data handling:** raw files fingerprinted; row order, labels and role assignments checked; no test or calibration leakage.
2. **Correct code:** results reproduce exactly when rerun (fixed seeds); key outputs compared with earlier saved results where they exist.
3. **Correct numbers:** every number written in the text is checked against the actual output; no rounded or remembered numbers.
4. **Correct diagnosis:** every "why" claim (for example "these classes fail because of label conflicts") is backed by a measurement in the notebook, not by assumption.
5. **Fair comparisons:** same rows, budget, seeds and rules for every compared model; limitations (such as epoch caps) removed or stated.
6. **Uncertainty shown:** seed spread and bootstrap intervals; small differences are called "could be chance" when they are.
7. **Plots checked visually:** every figure is opened and its description matches what it shows.
8. **Honest scope:** what the data cannot support (no DNS queries, testbed-specific features, tiny classes) is stated, never hidden.
9. **Clean run:** the notebook runs top to bottom with 0 errors from a fresh kernel.
10. **Simple language:** a student or reviewer can follow the question, method, result and decision of every section.
