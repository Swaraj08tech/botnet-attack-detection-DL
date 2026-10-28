# Summary and Detailed reports: verification (28 October 2026)

Example: `samples/test_flows.csv` (300 flows), detector v3, local `qwen2.5:7b`. Both PDFs were produced from ONE report record through the
API of the same report (`GET /api/reports/{id}/pdf?view=summary` and `?view=detailed`).

| File | Content |
|---|---|
| `summary_report_test_flows_20261028.pdf` | Summary report, 4 pages, six sections |
| `detailed_report_test_flows_20261028.pdf` | Detailed report, 20 pages: 6 main sections (pages 1-10), appendices A-F |
| `report_record_test_flows_20261028.json` | the shared report record (schema 2.1) both PDFs were rendered from |
| `screenshots/` | web views: desktop (1440 px, headless Chrome, report #9 in the running app) and mobile (375 px phone emulation) |

The earlier example `C:\Users\Swaraj\Downloads\20e3773c-...pdf` (report #8, schema 2.0) and every older report are unchanged; schema-2.0
records still open in both views (the views are derived on read).

## The concise overall explanation (real output, 131 words)

- 266 of 300 flows (88.7%) were predicted as attacks and 34 as Normal. The largest predicted groups are DoS (70), Exploits (45) and Generic (30).
- For the 70 DoS predictions, the most frequent reason was the average packet size sent by the source (a top-three reason in 62 of 70 flows).
- For the 45 Exploits predictions, the most frequent reason was the number of recent connections to the same service and destination (a top-three reason in 37 of 45 flows).
- 151 predictions have confidence below 70%, and 133 predicted attacks could be more than one attack type. For 7 flows, Normal traffic is still a possible answer.
- These are predictions about completed flows, not confirmed incidents or a forecast of future attacks. Check the flagged flows against logs before blocking anything.

Bullets 2 and 3 come from the **group findings**: TreeSHAP of the predicted class for every flow of that type (70 of 70, 45 of 45),
counting in how many flows a measurement is one of the three strongest supporting contributions. Contributions are never summed across
flows. Without a group analysis a bullet is written as "In the inspected example (flow #n) ...".

## How the two views share their evidence

- The record is built once (`RecordBuilder.build`). The overall explanation (`overview`) and the Summary pieces (`views.summary`: what was
  detected, prioritised actions, outcome checks, key limits, references) are computed from it by `app/core/report_views.py`, in code, with
  no model, retrieval or LLM call, and stored in the record.
- The Summary PDF, the Detailed PDF, the web Summary view, the web Detailed view, and the Markdown export read those same stored values.
  The Detailed view's "Main findings" are the same bullets; the Summary's actions are a selection of the Detailed action rows (same ids).
- Switching views in the web app changes only the presentation: the URL keeps the report, tab, selected flow and filters; review notes
  and action statuses are stored per report and shown in both views.

## Checks

| Check | Result |
|---|---|
| Detector outputs on the approved fixture (probabilities, sets, explanations, counts, triage level) | identical to the baseline hashes |
| Official-test evaluation | not run (unchanged model card is shown as context) |
| Backend tests (pytest) | 57 passed; ruff clean |
| Frontend tests (vitest) | 30 passed; `tsc` and ESLint clean; production build succeeds |
| Same findings in both PDFs (text extraction) | test `test_summary_and_detailed_views_share_the_same_facts` |
| Summary length | 4 pages; explanation 5 bullets, 131 words; no SHAP scores or feature codes |
| Record integrity | no dangling or cross-flow citations; support spans equal `passage[start:end]`; TreeSHAP sums exact |
| LLM statements | 24 checked, 0 rejected; each accepted sentence compared by hand with its facts |
| Access control | PDF and record endpoints return 401 without a token; unknown view 422; schema-1 summary 404 with reason |
| PDF quality | bookmarks, internal citation links, clickable source links, only embedded fonts, repeated table headers, no blank pages |
| Visual inspection | every page of both PDFs rendered and inspected; web desktop and mobile inspected |
| Mobile (375 px) | no element wider than the screen in the Summary view or any Detailed tab; view switch works with the arrow keys |

## Problems found during the visual review and fixed

- A batch-level action appeared after the per-flow verification table; the five per-flow verification steps repeated the same
  requirement, owner and check text. Now one table with the shared conditions stated once.
- The LLM's numeric paragraph repeated the chart in each finding card; it now sits in Appendix B next to the facts it cites, and the card
  shows the chart plus a one-line interpretation, alternatives and what is missing.
- Rounded probabilities ("1.00 confident", "100%") are no longer possible: facts say "above 0.99" and the checker rejects 1.00 / 100%.
- A footnote number ("20 See NIST SP 800-184 ...") at the start of a retrieved passage excerpt.
- "Ensure unnecessary ports and services are closed" was listed as containment; it is now general prevention.
- Bullet points used a non-embedded font; fixed.

## Reader tests (judgement, not measured with users)

- **Summary:** the six sections answer, in order, what was found, why, how uncertain it is, and what to do first; the explanation is
  five sentences without scores or codes. A student can read it in about two minutes.
- **Detailed:** each action names its evidence ids (`F284.E1`, `S7`); each id links to the finding card or the source register; the
  finding card links to Appendix A (all contributions) and Appendix B (LLM evidence). A reviewer can trace "filter the confirmed DoS
  traffic" -> A7 -> S7 (MITRE M1037) and -> F284 -> group finding for DoS without searching.

## Limitations

- Source support is checked by word overlap, not meaning. Containment steps carry explicit prerequisites, but the passages themselves
  are generic guidance.
- The class-to-technique mapping is a hypothesis (for example Generic -> T1600 Weaken Encryption); both reports say so.
- Detailed explanations cover 6 of 300 flows; group findings cover the four largest predicted types (172 flows). LIME is not generated.
- Full-length mobile screenshots were not captured (the pane captures the top of the page; headless Chrome cannot go below ~500 px wide).
- Reader tests are the author's judgement; no user study was run.
