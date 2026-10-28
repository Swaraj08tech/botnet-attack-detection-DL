# Our research workflow

**One task:** classify a completed network flow into the ten supplied intrusion classes, including Normal. Binary Attack/Normal is a second view. These files do not establish botnet-specific labels, raw DNS transactions or chronological sequences.

| Notebook | Plain question | Evidence and output | What follows |
|---|---|---|---|
| 01 — unchanged | What is in the supplied files? | Original schema and quality audit | Preserve source bytes. |
| 02 — data and evaluation | Can repeats, conflicts and overlap mislead us? | Exact-pattern audits, special-value checks, frozen role verification and evaluation rules | Keep labels and whole groups; disclose limitations. |
| 03 — EDA and preprocessing | How should recorded values be represented? | Fitting-only class/range/zero/category/protocol views; freshly reproduced tree and scaled inputs | Fit learned operations only on fitting rows. |
| 04 — engineering | Can three simple formulas expose useful structure? | Total bytes, byte balance, packet balance; worked flows, redundancy, zero rules and role checks | Treat them as representations to test, not new measurements. |
| 05 — shortlist | Which comparisons isolate each idea? | Five source-name sets; grouped descriptive association; verified input index and overlap checks | No final subset or winner yet. |
| 06 — baselines | Do these features help matched models? | Linear, single-tree and stronger ensemble; bounded grouped tuning, class metrics, errors and provisional choice | Establish a credible bar for architecture experiments. |

**Keep the data roles separate.** Fitting: 140,273 rows. Model selection: 17,534 rows. Calibration: 17,534 rows, reserved. Identical original patterns—including conflicting labels—stay in one role. Distinct vectors are not proven independent events. The official 82,332-row test stays separate; its earlier audits are disclosed, and its outcomes never tune this revision.

**Judge performance clearly.** Ten-class macro F1 gives each class equal weight. Also inspect precision, recall, F1, support and confusion matrices for every class. Accuracy and weighted F1 are secondary. Report binary attack recall, precision, false-positive rate and average precision from `1 − P(Normal)`. Thirteen model-selection Worms rows make small changes unstable. Model-selection already served ten earlier experiments; selected scores remain optimistic development evidence.

**Then test architecture ideas fairly.** Compare the strongest bounded tabular candidate, an MLP and a clearly labelled tabular Conv1D adaptation. Test attention and convolution with matched ablations and frozen column-order sensitivity checks. Real DNS/temporal branches require additional measured records, timestamps, endpoints and matching labels. A complex model must demonstrate a gain; no gain is assumed.

**Where to look:** Current notebook names remain unchanged. New tables, figures, models, inputs and reports use `revision_v2` folders. Older artifacts remain in place; the full recoverable snapshot is `snapshots/workflow_20261012T143405Z`. See the change log, research alignment, baseline summary and architecture handoff beside this map. Execution and independent verification records are in `results/tables/revision_v2`.
