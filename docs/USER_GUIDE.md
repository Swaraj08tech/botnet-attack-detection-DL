# User guide

How to use the web application. To install and start it, see the [README](../README.md#start-the-application).

The application analyses **datasets of completed network flows** (UNSW-NB15 format). Its flow is:

**Analyze Flows → (validate) → analysis → Results → Security Report**

## Navigation

| Page | What it is for |
|---|---|
| **Overview** | What the system does, and totals over all analysed files: files analysed, total flows, total threats, the model's test accuracy, safe vs malicious traffic and detected attacks by type |
| **Analyze Flows** | Upload a CSV, see it validated, start the analysis |
| **Results** | One analysed dataset: traffic classification, risk, attacks, confidence, threat-flow diagram, every flow |
| **Model** | The model, its official test-set performance, confusion matrix, model comparison, and the rules for confidence, severity and risk |
| **Reports** | Every analysed dataset with its PDF security report (generate, preview, download) |
| **About** | The project, dataset, ML pipeline and limitations |

The bottom of the sidebar shows whether the **Detection Model** and the **Report Generator** are ready. The optional AI flow explanations
(local LLM) are shown as online or offline. If the server cannot be reached, a red bar appears at the top. You stay signed in, and the
page works again as soon as the server is back.

## 1. Analyze Flows

1. Drag a `.csv` file onto the upload box, or click **Browse File**.
   - **No file at hand?** Click **Use sample dataset**: 300 real flows from the official UNSW-NB15 test file, 30 of each class, with the
     true class in `ground_truth`.
   - **Download sample CSV** saves the same kind of file.
2. The file is uploaded (with a percentage) and **validated** before anything is predicted. The checks:

   | Check | Detail |
   |---|---|
   | File type | Must be `.csv`; the file must be readable |
   | Not empty | At least one flow |
   | Size | 50 MB and 200,000 flows at most |
   | Required features | All 42 UNSW-NB15 features must be present. Missing ones are listed with their official description |
   | Values | Numeric features must contain finite, non-negative numbers |
   | Warnings | Unknown protocol, service or state values are treated as unknown; extra columns are ignored; true labels are found (`attack_cat` or `ground_truth`) |

3. The file card shows the name, size, number of flows and columns, and **Dataset validated successfully** (or what failed).
   **Analyze Network Traffic** is only enabled when validation succeeded. *Remove / change file* starts again.
4. The analysis shows the real processing stages:
   - Dataset uploaded
   - Dataset validated
   - Preparing model features
   - Running attack detection
   - Calculating prediction confidence
   - Preparing security analytics

   Each stage is shown as pending, processing, completed or failed. A 300-flow file takes about a second; the full official test file
   (82,332 flows) takes about 16 seconds.
5. **Analysis Complete** appears, and the Results page opens automatically.

## 2. Results

The header shows the dataset name, analysis time, number of flows and processing time, with these buttons:
- **Analyze Another File**;
- **Generate Security Report**;
- a selector to switch between analyses.

| Section | Meaning |
|---|---|
| **Total flows / Safe traffic / Malicious traffic** | Counts and percentages. Safe = predicted Normal; malicious = predicted as one of the nine attack types |
| **Average confidence** | Mean confidence of all predictions (see below) |
| **Traffic Classification** | Donut chart: safe vs malicious |
| **Overall Network Status** | LOW / MODERATE / HIGH / CRITICAL from fixed rules (see the Model page), with the counts per severity |
| **Summary cards** | Most detected attack, number of attack types, highest-confidence attack, flows requiring review |
| **Attack Distribution** | Malicious flows per attack type; bar colour = severity |
| **Prediction Confidence** | High (≥ 90%), Medium (70–89.9%), Low (< 70%) |
| **Attack details** | Per attack type: flows, % of malicious traffic, average confidence, severity, flows needing review |
| **Network Threat Flow Visualization** | Sankey diagram: flows → detection model → safe / malicious → Normal and each attack type, with the real counts |
| **Check against the labels in this file** | Only when the file has true labels. It compares the predictions with them for this file (attacks caught, false alarms, exact class), separate from the model's official test results |
| **Detailed Flow Analysis** | Every flow, searchable (flow number or attack type) and filterable (traffic, attack type, severity, confidence), 25/50/100 per page |

**Terms used**

- **Confidence**: the model's calibrated probability for the class it predicts. It is *not* the model's accuracy.
- **Needs review**: confidence below 70%.
- **Attack type uncertain**: predicted as an attack, but the 90% shortlist (conformal prediction set) contains more than one attack type.
- **Severity**: a fixed rule per attack type, not a model output:
  - Critical: Backdoor, Shellcode, Worms
  - High: Exploits, DoS
  - Medium: Fuzzers, Generic
  - Low: Reconnaissance, Analysis

The flow files contain no IP addresses or ports, so flows are identified by their row number.

### Flow detail

Click a row to open one flow. The page shows:
- the prediction, severity and confidence;
- the shortlist and the class probabilities;
- **Why this prediction?**: exact TreeSHAP contributions (red pushes towards the predicted class, blue away from it);
- the key measurements (hover a name for its meaning).

**Explain this flow** (optional) writes a short incident explanation with the local LLM. It links the flow to MITRE ATT&CK techniques and
recommended actions and checks every statement against its source. Without Ollama, only the evidence and sources are shown.

## 3. Security Report

Click **Generate Security Report** on Results, or **Generate** on the Reports page. The report answers one question: *what did the
detector predict, what evidence supports it, what remains uncertain, and what should an analyst check next?* The web report, the PDF
and the JSON / Markdown exports are all made from the same report record, so they always agree
(see [report_schema_v2.md](report_schema_v2.md)).

The panel shows the real stages (the middle four repeat for each illustrative flow, and the line under the active step names the flow):
1. Summarising detections and uncertainty
2. Explaining illustrative flows (TreeSHAP)
3. Retrieving sources (MITRE ATT&CK, NIST, CISA)
4. Local LLM writing explanations from the evidence
5. Checking every generated statement against its citation
6. Rendering the PDF from the same report record

With the local LLM it takes about 30–100 s for 300 flows; without it, a few seconds, and the report says it is evidence-only.
When it is done, click **Open interactive report** (or **Preview PDF** / **Download PDF**). Generating again creates a new version;
old versions stay in the list.

### The interactive report (`/reports/{id}`)

A switch at the top chooses the view; the app remembers your choice:
- **Summary** (default): for students and general readers. Six short sections: overall flow summary, what was detected, why the
  traffic was flagged (one explanation of 3-5 bullets), what to do, how to check the outcome, limits and sources. **View evidence** and
  **Open detailed report** lead to the supporting detail.
- **Detailed**: for analysts and reviewers, with the tabs below.

Both views show the same analysis and numbers; switching keeps the selected flow, notes and action statuses. **Download Summary PDF**
and **Download Detailed PDF** give the same two views as files.


- **Component status** at the top: detection, XAI, retrieval (RAG), local LLM, automated checks and PDF, each *available*,
  *unavailable*, *not run* or *failed*, with the reason. **Retry** generates the report again.
- **Questions**: *Why this prediction?*, *Could it be Normal?*, *What should I check first?* are answered from the report record
  (no new model or LLM call).
- **Exports**: View PDF, Download PDF, JSON (the full record), Markdown, and copy as Markdown.
- Tabs (arrow keys, Home and End move between them):
  - **Assessment**: the main findings (the same bullets as the Summary), predicted Normal / predicted attack, needs review, average confidence, the **batch triage priority** (a heuristic,
    not a model output), the most important uncertainty and what to check first.
  - **Detections**: every predicted class (click a class to list its flows), the uncertainty groups (attack *status* uncertain:
    Normal is still on the 90% shortlist; attack *type* uncertain: several attack types), and the flows to look at first
    (click one for a quick view).
  - **Why this prediction**: one card per illustrative flow, with the rule that chose it, the probabilities, the measured values
    compared with the training flows, the **TreeSHAP** bars with the exact decomposition (base value + contributions = model score;
    score / temperature + class offset = deployed class score), the evidence facts, the **LLM explanation** that passed the checks,
    and the statements the checks rejected (with the reason). LIME is marked *not generated*.
  - **Sources**: the source documents (publisher, version, dates, link, SHA-256) and every cited passage. Click a citation chip
    anywhere (for example `S3` or `F79.E2`) to see the exact passage with the supporting sentence highlighted, or the exact evidence
    fact. Passage text is untrusted reference text: it is shown, never followed.
  - **Actions**: Verify → Contain if confirmed → Recover, each with its reason, citations, when it applies, owner, disruption and how to
    validate it. Nothing is executed by the app. Record a status (open, reviewed, planned, completed by analyst, not applicable) and a
    note; every change is kept with your name and the time.
  - **Limits & model**: what the report cannot establish, the origin legend, the comparison with labels in the file (kept separate),
    the model benchmark, provenance hashes and timings.

Every statement carries an **origin** tag: observed in file, batch statistic, model output, XAI (TreeSHAP), retrieved source,
LLM · passed checks, proposed next step, or heuristic.

### The PDFs

- **Summary PDF** (2-4 pages): the six Summary sections in plain language, a few clickable references.
- **Detailed PDF**: 1 executive assessment, 2 detection results and uncertainty, 3 evidence and model explanation (group findings and
  finding cards with the TreeSHAP chart), 4 retrieved context and source support, 5 response and mitigation plan, 6 limitations and
  provenance; then Appendix A (all measured values and contributions), B (evidence given to the LLM, its accepted wording and rejected
  statements), C (sources), D (hashes and technical metadata), E (model benchmark and confusion matrix), F (labels in the uploaded file,
  if any).

Both have bookmarks, clickable citations and source links, embedded fonts, the report type, analysis, version and generation time in
every header and footer. Reports created before this version still open.

### How the AI text is checked

Every LLM statement must cite evidence facts (`F<flow>.E<n>`) or retrieved passages (`S<n>`). Rule-based checks remove anything that
fails (one repair round first) and list it with the reason:
- **Explanation of the prediction (cites evidence only):** every number, column and class must be in the cited facts; each
  `feature = value` and each contribution must belong to that feature; supports / argues against must match; the probability must be
  the flow's own; the class's flow count, share and average confidence must match the dataset fact; the 90% shortlist must be described
  as it is; no value judgements and no causal or certainty words ("caused", "proves"). If every value is right but a citation is
  missing, the checker adds it and says so.
- **Techniques:** real ATT&CK IDs that appear in a cited passage.
- **Context and actions:** most of their content words must appear in the cited passage (the matching sentence is recorded and
  highlighted). This is a lexical check, not a check of meaning.

## 4. Model

- **About the model**: name, problem type, dataset, number of input features, classes, short description.
- **Performance on the official test set**: accuracy, precision, recall and F1, switchable between macro average, weighted average and
  attack vs normal, each explained in one line. These numbers were measured once on 82,332 labelled test flows; they are never
  calculated from your uploads.
- **Confusion matrix**: where the model was right (green diagonal) and wrong (red).
- **Results per class** and **Model comparison**: every model from the research notebooks on the same test set; the deployed model is highlighted.
- **How predictions are turned into results**: the confidence levels, conformal prediction (the 90% shortlist), the severity rules and
  the risk rules.

## 5. Reports

A table of every analysed dataset with its date, flows, threats, risk and report status. You can preview, download, generate or
regenerate a report. Click a dataset name to open its results.
