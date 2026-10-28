# Research history (earlier README, kept for the record)

This was the project README before the application phase. It describes the revision_v2 workflow for notebooks 02-06. The current notebooks are listed in [v3/README.md](v3/README.md).

## Current workflow

**Update 17 Oct 2026:** Notebooks 01–06 were rewritten in a simpler, self-explaining style (v3) with identical results. See [docs/v3/README.md](v3/README.md). Run them with `scripts/build_v3_notebooks.py` and `scripts/run_v3_notebooks.py`. The notes below describe the earlier revision_v2 record.

Notebook 01 remains unchanged. Notebooks 02–06 have been revised with simple questions, worked examples, full evidence, reproducible checks and versioned artifacts. See the [one-page workflow](revision_v2/workflow_map.md), [research alignment and paper review](revision_v2/research_alignment.md), [change log](revision_v2/change_log.md), [baseline results](revision_v2/baseline_summary.md) and [architecture handoff](revision_v2/architecture_handoff.md).

The provisional baseline is lightgbm with total_and_byte44 (development macro F1 0.6320). Its small engineered-feature gain is uncertain and includes rare-class tradeoffs. Final performance is unknown. The same model-selection role previously served ten experiments; further selected scores are optimistic development evidence.

## Preserved data and experimental history

Raw CSVs remain in `data/raw`, unchanged. Frozen development roles are 140,273 fitting, 17,534 model-selection and 17,534 calibration rows, with identical original patterns kept together. Calibration receives integrity/transformation checks only; it is not predicted or used for fitting/selection in this revision. The 82,332-row official test remains separate. It was historically inspected in data audits, which are disclosed; its outcomes are not used for improvements.

All earlier source code, assignments, generated inputs, models, reports, tables and figures remain in place. A recoverable pre-edit snapshot of 379 files is in `snapshots/workflow_20261012T143405Z`, with a verified manifest and raw hashes. Current 02–06 notebook filenames are unchanged; their earlier contents are recoverable from that snapshot. New outputs use `results/tables/revision_v2`, `results/figures/revision_v2`, `data/processed/revision_v2`, `models/revision_v2` and `docs/revision_v2`.

## Execute and verify

Select `.venv/Scripts/python.exe` as interpreter/kernel. Use the current runner, one notebook at a time:

```powershell
.venv\Scripts\python.exe scripts/run_research_revision.py 02
.venv\Scripts\python.exe scripts/run_research_revision.py 03
.venv\Scripts\python.exe scripts/run_research_revision.py 04
.venv\Scripts\python.exe scripts/run_research_revision.py 05
.venv\Scripts\python.exe scripts/run_research_revision.py 06
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe scripts/verify_research_revision.py
```

The runner registers a project-local Jupyter kernel, saves executed outputs, checks protected files and records notebook/artifact hashes. Kernel communication may require local execution permission. The baseline runner reuses only hash-bound unchanged tuning checkpoints and exact saved model settings; no hidden new search is added. `scripts/build_research_revision.py` recreates unexecuted notebook sources (optionally specify a number); rebuilding clears saved outputs, so execute afterward. Old builders/runners are historical and should not overwrite the revised notebook files.

`results/tables/revision_v2/final_verification.json` records independent artifact reload, fold-local scaler/vocabulary verification, regenerated metrics/confusion matrices and exact saved-model predictions. Each notebook has execution and artifact manifests. `docs/revision_v2/environment_versions.json` records installed versions. The Python 3.11 project environment inherits system packages and is not a full isolated lock; the original `requirements.txt` remains unchanged. LightGBM is available locally. No neural, XAI, calibration or RAG training is included in this phase.

## Rules for later work

Exclude IDs, targets and audit metadata. Fit learned preprocessing/selection/tuning inside fitting data with original-pattern groups. Keep original42 as a reference; engineered features are representations of the same measurements. Compare architecture components with matched baselines/ablations and inspect rare-class errors. Column order is not time, a DNS service marker is not a query stream, and Attack is not verified Botnet. Confidence calibration, operating thresholds, reserved-role refits and final official-test reporting are later declared steps.
