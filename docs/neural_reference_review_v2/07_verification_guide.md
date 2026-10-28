# How to keep Notebook 07 verification honest

A successful seal is a **historical statement about exact saved bytes**, not a permanent “fully verified” badge. VS Code can change notebook metadata, formatting or outputs on save. Saved execution counts alone do not prove the file still matches its verified version.

## Check without changing anything

From `D:\projects\Botnet attack detection`:

```powershell
.venv\Scripts\python.exe scripts/verify_neural_reference.py --check-seal
```

This reads and compares recorded hashes; it never updates the execution record or reseals changed files. A mismatch fails explicitly as `STALE seal`. Read the `verified_utc` and `verified_notebook_sha256` fields in the latest deliverable manifest together with the live check.

## Refresh after a later save

Finish and save all intended notebook/source/report edits first. Close the editor's unsaved notebook buffer so an old in-memory copy cannot overwrite the executed file. Then run these commands **in order**:

```powershell
.venv\Scripts\python.exe scripts/run_neural_reference_tests.py
.venv\Scripts\python.exe scripts/run_neural_reference.py
.venv\Scripts\python.exe scripts/verify_neural_reference.py
```

The test runner binds passing tests to source hashes. The notebook runner launches a fresh `.venv` kernel, replays the frozen saved experiment without fitting or retuning predictive models, saves all outputs, and records the actual notebook SHA-256 in `results/tables/neural_reference_v1/07_execution.json` and the versioned review record. A failed/partial run cannot receive a successful seal. The independent verifier runs **last**, reloads models, recomputes metrics, checks protected artifacts and source/test hashes, and seals those saved bytes. Do not edit execution or manifest hashes manually.

After this sequence, open the notebook without saving it again. If another save occurs, rerun the live check and refresh if stale. The tooling cannot prevent an editor save or automatically change a historical JSON record at that moment; the live check is the authority for current validity.

The latest review manifest is in `docs/neural_reference_review_v2/07_deliverable_manifest.json`, with the existing `docs/neural_reference_v1/07_deliverable_manifest.json` as the latest-seal entry point. The v1 scientific artifacts and their archived prior seal remain historical evidence. The reviewed artifact manifest is in `data/processed/neural_reference_review_v2/07_artifact_manifest.json`; the original v1 artifact manifest is preserved and binds its historical reports, not the revised findings aliases.

Original models, scientific metrics, predictions, costs, loss curves and figures stay unchanged. Revised presentation aliases are documented and their original bytes are in the review snapshot. If scientific inputs/results actually change, stop and version a new experiment instead of refreshing this reference silently.
