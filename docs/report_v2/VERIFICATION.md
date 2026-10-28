# Security report v2: verification (26 October 2026)

Demo: `security_report_test_flows_schema2_20261026.pdf` (17 pages) and its record `.json`, generated from `samples/test_flows.csv`
with detector v3 and local `qwen2.5:7b`. The original demo (`security_report_test_flows_analysis_4.pdf`, report #4) is untouched.

## Why the old demo had no RAG
Report #4 was generated on 2026-10-19 16:06 UTC, before the RAG commit 8742843 (17:05 UTC); its `rag` field is empty and it was made by
detector v2. Reports did not record a code version then; schema 2 now records the git commit, bundle hashes and input hash.

## Results
- Detector unchanged: probabilities, prediction sets and explanations hash-identical to the pre-change baseline (34 Normal / 266 attack,
  151 review, 133 type-uncertain, mean confidence 0.7299, CRITICAL). No retraining, recalibration or test-set evaluation.
- Components: detection, XAI, retrieval, LLM, checks, PDF all available. 5 illustrative flows; 4 LLM explanations accepted;
  22 statements checked, 1 rejected (a fake ATT&CK ID). Timings: XAI 0.40 s, retrieval 0.05 s, generation 33.7 s, checks 0.02 s, PDF 0.35 s.
- Record integrity: no dangling or cross-flow citations, every support span equals `passage[start:end]`, TreeSHAP sums exact (<1e-6).
- PDF: 21 bookmarks, 96 links (59 internal citation links), only embedded fonts, no blank pages.
- Tests: backend 51 passed, ruff clean; frontend 23 passed, typecheck, lint and build clean.

## Bugs found during verification and fixed
1. LLM statement ids reused evidence ids (`F222.E1`) -> now `X/T/C/A` prefixes.
2. Support-span offsets were off by the leading space -> highlight now exact.
3. "average confidence N" was checked as the flow probability: 3 correct sentences were rejected and, in the repair round, a sentence with a
   wrong value (0.99 instead of 0.67) was accepted. New typed check binds count, share and average confidence to the dataset fact.
4. Table-of-contents passages (dot leaders, 25 of 1,353 chunks) could be retrieved -> filtered.
5. Missing-glyph boxes and non-embedded default fonts in the PDF -> fixed.

## Web screenshots (`screenshots/`)
Desktop (1440 px, headless Chrome against the running app, report #7): one per tab. Mobile (375 px phone emulation): the summary and the
citation drill-down. At 375 px no element extends past the screen on any tab (checked in the browser); this check found the status
list overflowing and it was fixed. Headless Chrome cannot render narrower than about 500 px, so full-length mobile pages were not captured.

## Limitations
- Source checks are lexical overlap, not semantic entailment; an LLM action can be generic (e.g. DNS guidance for Generic).
- Illustrative flows cover 5 of 300 flows; LIME is not generated.
