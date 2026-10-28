# Final report version: verification (28 October 2026)

Example: `samples/test_flows.csv` (300 flows), detector v3 (unchanged), local `qwen2.5:7b`.

| File | Content |
|---|---|
| `summary_report_test_flows.pdf` | Summary report, 4 pages |
| `detailed_report_test_flows.pdf` | Detailed report, 24 pages (main sections, then appendices A-F) |
| `report_record_test_flows.json` | the record both PDFs were rendered from |
| `screenshots/` | Results, Model, report Summary view, report Detailed view, About (desktop, report #11 in the running app) |

## What changed in this version

- **Two confidences.** Attack or normal (attack-class probabilities added together): 263 high, 26 medium, 11 low. Exact type: 92 / 57 /
  151. Same calibrated probabilities; no model or threshold change. Older analyses are filled in from their stored flows on first open.
- **Grouped view** of the per-class results (DoS, Analysis and Backdoor as one group), computed from the existing official-test
  confusion matrix: accuracy 77.0%, macro F1 68.5%, group F1 57.3%. The 10-class scores remain the official ones.
- **AI explanations as 2-3 separate checked points** per example flow, shown in a labelled box in the Detailed report and in plain
  wording ("Examples explained by the AI") in the Summary.
- **Stricter checks.** The direction of every column in a list ("while A and B argue against it") and the sign of every written
  contribution must match the evidence. The end-to-end run found one wrong sentence of this kind (proto marked as arguing against DoS
  although it supports it); it is now rejected, with a regression test.
- Site wording simplified (reading guide on Results, confidence explanations, grouped-view explanation on the Model page).

## Checks

| Check | Result |
|---|---|
| Detector outputs on the fixture | identical to the baseline hashes |
| Backend tests / ruff | 63 passed / clean |
| Frontend tests / tsc / ESLint / build | 33 passed / clean / clean / succeeds |
| Example run | all components available; 59 LLM statements checked, 7 rejected; accepted sentences compared by hand with their facts |
| PDFs | every page rendered and inspected |
| Web | desktop screenshots of every changed page; at 375 px no element wider than the screen on Results, Model (both views) and the report |

## Limitations

The source check is lexical, not semantic. Group findings cover the four largest predicted types. Analysis and Backdoor cannot be
separated from DoS on this data (best possible F1 15.8% and 19.0%); the grouped view shows this, it does not fix it.

## Update: 5-point reasoning (28 October 2026, evening)

Each example flow now has five reasoning points (how sure, main reasons, how typical the values are in training flows, why not the
runner-up class, this file). New evidence: per-class typical values from the 140,273 training rows (`class_reference.json`, built by
`scripts/build_class_reference.py`; the test file is never read) and TreeSHAP of the runner-up class. New checks: comparison direction,
the attack-or-normal probability and the runner-up probability are each accepted only for their own subject. Final run: 75 LLM statements
checked, 6 rejected; all 29 reasoning points reviewed by hand against their evidence. Backend 68 tests, frontend 34 tests, build clean.
