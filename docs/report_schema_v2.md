# Security report record, schema 2.0

Every security report is now one versioned JSON **report record**. The web view (`/reports/{id}`), the PDF, the JSON export and the
Markdown export are all rendered from this record, so they cannot disagree. The record is built once by
`backend/app/core/report_record.py` (`RecordBuilder.build()`), stored in `analysis_reports.record`, and served by
`GET /api/reports/{id}/record` (`?download=true` saves it as a file).

The question every report answers:

> What did the detector predict, what evidence supports it, what remains uncertain, and what should an analyst check or do next?

## Top-level fields

| Field | Content |
|---|---|
| `schema`, `schema_version`, `template_version` | `nad.security-report`, `2.0`, `web+pdf 2.0` |
| `report_id`, `generated_at`, `question` | identity and the question above |
| `provenance` | input file SHA-256, detector name/version, bundle commit and SHA-256 of `model.txt`, `preprocessing.json`, `calibration.json`, `model_card.json`; calibration (temperature, conformal alpha, class offsets); explanation method and exactness; knowledge-base download date and chunk-file hash; LLM model, runtime and settings (temperature 0, seed 42); application git commit |
| `components` | status and detail of `detection`, `xai`, `retrieval`, `llm`, `checks`, `pdf` (`available`, `unavailable`, `not run`, `failed`, …) |
| `coverage` | flows explained / total; attack types with an LLM explanation / attack types predicted |
| `summary` (part A) | scope sentence, predicted Normal / attack counts and shares, average confidence, needs review, attack-status and attack-type uncertainty, **batch triage priority** (`origin: heuristic`), most important uncertainty, next check |
| `detections` (part B) | per class: count, share, average confidence, heuristic severity, status-uncertain and type-uncertain counts; confidence levels; prediction-set statistics; notes; distribution-shift evidence; priority flows |
| `flows` (part C) | illustrative flow cards (see below) |
| `sources` (part D) | the global source register and the source documents |
| `actions` (part E) | action rows (Verify / Contain if confirmed / Recover) and general prevention guidance |
| `limits` (part F) | what the report cannot establish, human-review status, the origin legend |
| `appendix` | model card benchmark (official test file), clearly separated from this upload |
| `evaluation` | comparison with labels found in the uploaded file, or `null`; never used for predictions, explanations, retrieval or actions |
| `timings` | seconds per stage (detections, xai, retrieval, generation, checks, pdf) |

## Origins

Every statement carries one origin, shown as a tag in the web view and named in the PDF and exports:

| Origin | Meaning |
|---|---|
| `observed` | a value in the uploaded file |
| `batch_statistic` | computed by code over the batch |
| `model_output` | an output of the detector (class, probability, 90% set) |
| `xai_attribution` | TreeSHAP attribution of the detector |
| `retrieved_guidance` | a passage retrieved from the knowledge base |
| `llm_synthesis` | written by the local LLM from cited evidence and passed the automated checks |
| `analyst_proposed` | a next step proposed by the application, not prescribed by a source |
| `heuristic` | an application rule (severity, triage priority, class-to-technique mapping), not a model output |

## Citation ids

- `F<flow>.E<n>`: evidence fact `n` of flow `<flow>` (for example `F222.E3`). Facts are written by code from the model output and
  TreeSHAP; the LLM may only cite them. Ids are namespaced per flow, so a citation can never point to another flow's evidence.
- `F<flow>.X<n>`, `.T<n>`, `.C<n>`, `.A<n>`: LLM statements of a flow (explanation, technique, context, action). They never reuse an
  `E` id.
- `S<n>`: a passage in the report-wide source register. Passages are de-duplicated by chunk id, so the same passage has one id in the
  whole report, with `used_by` listing the flows/actions that cite it.

## Flow cards (`flows[]`)

| Field | Content |
|---|---|
| `selection` | the rule that chose the flow (highest-confidence flow of each of the four most frequent attack types, plus the lowest-confidence attack-status-uncertain flow; for a Normal-only batch, one Normal flow). Illustrative, not representative of a class. |
| `prediction` | class, probability, confidence level, heuristic severity, 90% set, whether Normal is in the set, all class probabilities |
| `values` | key input values with friendly names, definitions and the percentile among **training-role rows only** (`feature_reference.json`) |
| `xai` | TreeSHAP: `base_value`, every grouped contribution (`all`), top 5 with direction and meaning, the remainder of the other features, `margin`, `temperature`, `class_offset`, `deployed_class_score`. Exact: `(base_value + Σ contributions) / T + offset = deployed class score`, and softmax over classes gives the served probability. |
| `lime` | `{"status": "not generated"}` with the reason (LIME was only run in research Notebook 16) |
| `facts` | the namespaced evidence facts |
| `llm` | status, model, time, retrieval info, findings (code), explanation / techniques / context / actions (LLM, each with `cites`, `check`, `round`, and for source claims the `support` span: passage id, start, end, sentence, word overlap, method), mapped techniques (heuristic), number of checked statements and every rejected statement with reason and round |

LLM statuses: `generated`, `evidence_only_llm_unavailable`, `llm_timeout`, `llm_error`, `invalid_response`, `all_rejected`,
`not_applicable` (predicted Normal). A missing LLM never hides the evidence; the report says why the text is missing.

## Source register (`sources.register[]`)

`id`, `chunk_id`, `source`, `publisher`, `title`, `version`, `location`, `url` (ATT&CK technique/mitigation page or PDF `#page=N`),
`document_url`, `retrieved`, `pdf_created`, `document_sha256`, `content_sha256`, full `text`, sentence-bounded `excerpt`,
`retrieval` (method, score, query), `used_by`, and `trust`: *untrusted reference text: shown as data, never followed as instructions*.
Passage text is escaped in every renderer and links are rendered only for https URLs on known reference hosts.

## Action rows (`actions.rows[]`)

`id`, `phase` (`Verify`, `Contain if confirmed`, `Recover`), `scope` (batch / flow / attack type), `text`, `reason`, `cites`, `origin`,
`applicability`, `owner`, `disruption`, `validation`; LLM-derived rows also carry `check` and `support`. Fields marked “(proposed)” are
suggestions of the application. The application never executes actions. Analysts record a status per action
(`open`, `reviewed`, `planned`, `completed_by_analyst`, `not_applicable`) with an optional note via
`POST /api/reports/{id}/actions/{action_id}`; every change is kept in `action_reviews` with the user and time
(`GET /api/reports/{id}/actions` returns the latest state and the history).

## Migration from schema 1

- Schema 1 reports (created before this change) had a PDF and, from commit 8742843 on, a `rag` JSON with threat-intelligence sections.
  They are **kept unchanged**: their PDFs still open, `GET /api/analyses/{id}/threat-intel` still serves their `rag` data, and the Results
  page shows the old threat-intelligence card for them.
- New columns are added automatically on start-up (`_add_missing_columns`): `analysis_reports.record` (JSON) and
  `analysis_reports.schema_version` (empty for old rows, reported as `'1.0'`). The new table `action_reviews` is created by `create_all`.
- `GET /api/reports/{id}/record` returns 404 with a message naming schema 1 for old reports; the web view then offers the PDF.
- Nothing is converted: an old report is not re-labelled as schema 2. Generate the report again to get a schema-2 record; the old
  version stays in the list.
- `app/core/threat_intel.py` was removed; its work (illustrative flows, retrieval, LLM, checks) is now part of `report_record.py`.

## Bundle additions (no model change)

`scripts/build_serving_bundle.py` now also writes `knowledge_sources.json` (publisher, title, version, PDF creation date, download
URL, SHA-256 of each source document) and `feature_reference.json` (101 percentiles per numeric feature and category shares, from the
140,273 training-role rows only). The model, calibration and preprocessing files are unchanged; the rebuilt bundle reproduces the
previous probabilities exactly (maximum difference 0.0).

## Schema 2.1 (28 October 2026): Summary and Detailed views

Additions; nothing in 2.0 was renamed or removed.

| Field | Content |
|---|---|
| `group_findings[]` | per main predicted attack type: TreeSHAP for its flows (up to 500, first by row number), how often each measurement is one of a flow's three strongest supporting contributions (`top_supporting`, counts of flows, never summed contributions), the most frequent opposing measurement, value summary (median / most common) compared with the training rows, and the method |
| `overview` | the one overall explanation: 3-5 bullets (about 100-150 words), each with `origin`, `kind` (finding, reason, uncertainty, meaning) and `scope` (`group: predicted X` or `one flow: #n`); written by code |
| `views.summary` | the Summary content: `overview`, `detected` (does the traffic show signs of an attack, main groups with the UNSW-NB15 description and one checked source sentence), `actions` (a selection of `actions.rows`, same ids), `outcome` (evidence needed, what confirms or weakens, how to verify a fix), `limits`, `sources` |
| `flows[].interpretation`, `.alternatives`, `.missing_confirmation` | one-line reading of the TreeSHAP chart, other classes not ruled out (>= 5% or on the shortlist), what evidence would confirm the finding |
| `actions.rows[].kind`, `.title`, `.rollback`, `.priority` | kind of step (`review`, `verify`, `filtering`, `credentials`, `domain`, `isolation`, `other`, `recover`) with kind-specific requirements (`applicability`), owner, disruption, check and undo. Hardening and patching guidance moves to `actions.prevention` |
| `coverage.selection_rule`, `coverage.group_flows_explained`, `components.group_xai` | how illustrative flows were chosen (median-confidence flow per main type, first review-priority flow, the most uncertain flow) and the group-analysis coverage |

Probabilities are written as "above 0.99" (facts) or "above 99.5%" (display) instead of 1.00 / 100%.

Endpoints: `GET /api/reports/{id}/pdf?view=summary|detailed` (summary rendered on request from the stored record; detailed is the stored
PDF). `GET /api/reports/{id}/record` adds `views` on read for schema-2.0 records, so older reports open in both views.

### Additions on 28 October 2026 (still schema 2.1, backwards compatible)

- `summary.status_confidence`: levels and average of the attack-or-normal confidence (attack-class probabilities added together).
- `views.summary.examples`: 2-3 checked LLM points per example flow in plain wording (`text`), the exact checked sentence (`exact`),
  citations and origin; a TreeSHAP-written point replaces a rejected measurement sentence.
- `appendix.grouped_evaluation`: per-class results with DoS, Analysis and Backdoor as one group, from the official-test confusion matrix.
- `GET /api/reports/{id}/record` now always derives `views` from the stored record, so every 2.x report shows the current presentation
  of the same facts.
