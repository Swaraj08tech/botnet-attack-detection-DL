# Why the old seal was stale

The starting Notebook 07 SHA-256 was `cb23f9a5ed1cd193bb7f8af2eae4c42000af53f9e8e5bf529837037d31122c8d`. The execution record and old deliverable seal named `6caf8d714fd6544718c2c012dbbc859c1ff4675bc2e73ac0e09baae9e114acf0`. Running the original independent verifier reproduced its assertion failure at the execution/notebook hash check. No hash was manually corrected.

The old seal also differed from the current `07_artifact_manifest.json`, `07_model_selection_metrics.csv`, `07_verified_lightgbm_controls.csv`, and `07_environment.json`. The six MLP checkpoint files and saved prediction CSVs still matched the old seal. The existing source modules matched it too. The reported classification results remain the same; they are independently recomputed in the final review.

VS Code local history has an entry for this notebook (`History/-291e6dd9/DaHg.ipynb`, entry timestamp 2026-10-12 17:56:07.733 UTC). Every cell source matches the starting notebook. The notebook's current kernel metadata says `.venv (3.11.9)` / `python3`, whereas the runner used `Botnet research (.venv)` / `botnet-research`. Normal editor saves can serialize changed metadata and formatting.

The history copy and current file have different outputs in the comparison and cost cells. The history comparison cell has no outputs; the current cell has results. The cost-cell inference times also differ: for example, the first MLP row is 0.261968 seconds in history versus 0.073022 seconds in the current display. This is evidence of output/benchmark refresh, consistent with selective re-execution as well as editor saving. It is not just a harmless whitespace difference.

We could not reconstruct the exact old sealed notebook from the available copy by restoring kernel metadata, selecting either output history, or adjusting the final newline. The evidence therefore does not establish the exact action sequence, the person who saved it, or every byte-level cause. We do not attribute the entire mismatch to a normal VS Code save alone.

Before editing, 89 current Notebook 07 files and six cached search results were copied and hash-checked into the directory recorded by `.cache/neural_review_snapshot_path.txt`. It also contains the VS Code history copy and the previous seal/record. The preserved snapshot manifest lists the five stale-seal differences. Earlier Notebooks 01–06 and their evidence are not edited.

## The repair

The revised notebook replays the frozen experiment without new predictive fitting or retuning. Original model, prediction, metric, figure, loss, settings and cost files stay byte-exact. New review checks use `neural_reference_review_v2`. Only the notebook presentation, concise findings aliases, runner and verifier are updated; the archived copies retain the historical versions.

The runner executes the final notebook from a fresh project kernel, saves it, and computes SHA-256 from those actual bytes. The independent verifier then checks those bytes, all six MLP models, both LightGBM controls, inputs, rows, classes, preprocessing provenance, recomputed metrics and test evidence. It seals only the final validated files. Any later save can invalidate that historical claim; see [the refresh guide](07_verification_guide.md).
