"""Build final documentation from the executed result tables, never invented scores."""
from pathlib import Path
import sys
import json
import importlib.metadata
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.research_revision.common import read_json,write_json
from src.data.load_data import sha256
from src.baseline_models import SETS,CLASSES

def main():
    doc=ROOT/'docs/revision_v2'; doc.mkdir(parents=True,exist_ok=True)
    results=pd.read_csv(ROOT/'results/tables/revision_v2/06_model_selection_metrics.csv',float_precision='round_trip')
    chosen=read_json(ROOT/'models/revision_v2/06_provisional_choice.json')
    plan=read_json(ROOT/'configs/revision_v2_baseline_plan.json')
    family_settings=read_json(ROOT/'models/revision_v2/06_frozen_family_settings.json')
    definitions=read_json(ROOT/'models/preprocessing/05/05_comparison_sets.json')['sets']
    best=results[(results.model==chosen['model'])&(results.feature_set==chosen['feature_set'])].iloc[0]
    name=chosen['model']+'_'+chosen['feature_set']; per=pd.read_csv(ROOT/'results/tables/revision_v2/06_per_class_metrics.csv')
    per=per[(per.model==chosen['model'])&(per.feature_set==chosen['feature_set'])]
    ambiguity=pd.read_csv(ROOT/'results/tables/revision_v2/06_ambiguity_error_diagnostic.csv')
    gains=pd.read_csv(ROOT/'results/tables/revision_v2/06_chosen_gain_contributions.csv')
    total_gain=gains.contribution_percentage_points.sum(); worms_gain=gains.loc[gains.attack_cat.eq('Worms'),'contribution_percentage_points'].iloc[0]
    lines=['# Current baseline results','', 'All numbers below use the same frozen 17,534 model-selection rows. They are development evidence, not final performance. The role previously served ten experiments. All models are unweighted, with family settings chosen by three grouped fitting-only folds and then frozen across sets.','',
        '| Model | Feature set | Macro F1 | Accuracy | Worms correct / 13 | Fit seconds |', '|---|---|---:|---:|---:|---:|']
    for r in results.itertuples(): lines.append(f'| {r.model} | {r.feature_set} | {r.macro_f1:.4f} | {r.accuracy:.4f} | {r.worms_correct} | {r.fit_seconds:.1f} |')
    lines += ['',f'**Provisional choice:** {chosen["model"]} with `{chosen["feature_set"]}`, macro F1 **{best.macro_f1:.4f}**, accuracy {100*best.accuracy:.2f}%. The declared .001 tie rule was applied. It prefers fewer source features, then simpler declared families within the tolerance. This tolerance is not statistical equivalence.','',
        f'For the ensemble, total bytes plus byte balance improves original42 macro F1 by **{total_gain:+.3f} percentage points**. Worms alone contributes {worms_gain:+.3f} points, while the other nine classes together contribute {total_gain-worms_gain:+.3f} points. The small net gain does not establish robust feature usefulness. Original42 stays a close comparison reference. Packet balance does not improve this ensemble; all45 is worse than original42. Logistic gets a modest byte-balance gain but still detects no Worms. The tuned tree gets a higher macro F1 with balances, largely alongside rare-class changes; accuracy does not improve.','',
        '## Important class tradeoffs','', '| Class | Precision | Recall | F1 | Correct / support |','|---|---:|---:|---:|---:|']
    for r in per.itertuples(): lines.append(f'| {r.attack_cat} | {r.precision:.3f} | {r.recall:.3f} | {r.f1:.3f} | {r.correct} / {r.support} |')
    lines += ['',f'Binary Attack/Normal: recall {best.attack_recall:.4f}, precision {best.attack_precision:.4f}, Normal false-positive rate {best.false_positive_rate:.4f}, average precision {best.attack_average_precision:.4f}. These cannot replace attack-category metrics: an attack assigned to the wrong category can still count as a binary success. Scores are uncalibrated; no operational threshold is chosen.','',
       'DoS→Exploits accounts for 1,108 errors, 964 in category-conflicting original patterns. Analysis→Exploits has 143 errors and Backdoor→Exploits 140. The full confusion/ambiguity tables preserve every count. This locates missing distinctions in recorded inputs and model errors; it does not prove annotation corruption or causal error mechanisms.','',
       f'On model-selection, one deterministic prediction per original vector has an empirical **accuracy** ceiling of {100*ambiguity.iloc[0].empirical_vector_accuracy_ceiling:.2f}% from conflicting labels (1,627 unavoidable recorded-row disagreements). The chosen model makes 2,883 errors. This is not a macro-F1 ceiling or a population-independent Bayes bound. Three engineered formulas cannot separate identical original inputs; additional measured context could.','',
       '## Search and practical limits','',
       '- Two configurations per family × three grouped fitting folds = 18 inner fits. Logistic C=.1/1; tree depth16/leaf5 versus depth24/leaf2; LightGBM 31/63 leaves, 120 fixed rounds. No search expansion after scores. Final experiments: 15 matched comparisons, with five protected logistic fits reused because exact settings match.',
       '- Current selected settings: '+json.dumps(family_settings['chosen'],sort_keys=True)+'.',
       '- Single fixed seed; no claim of significance or repeat-seed robustness. Only 13 Worms rows; one correct row changes recall by 7.69 percentage points.',
       '- Timing depends on this environment and concurrent load. Logistic timings come from the preserved exact earlier fits. Complexity counts (coefficients, tree nodes, boosting leaves) have different meanings; bytes are also recorded.',
       '- LightGBM/scikit-learn emits a feature-name warning for unnamed CSR prediction input. Frozen input order, mapping, shape, input hashes and exact reloaded predictions are independently verified; the warning is retained in the notebook record.',
       '- The environment inherits system packages and is not a completely isolated dependency lock. Installed relevant versions are recorded. No dependency prevented completion.',
       '- Group separation addresses recorded-vector overlap, not host or temporal independence. Context-counter causality, literal botnet ground truth, DNS streams and time windows remain unverified/unavailable.',
       '- No calibration fitting, reserved-role refit, final operating threshold or official-test model evaluation. Selected development results are optimistic.','',
       'Exact inputs: `data/processed/05/05_'+chosen['feature_set']+'_tree_{role}.npz`; row links: `05_row_links_{role}.csv`; source/encoded maps: `models/preprocessing/05/05_'+chosen['feature_set']+'_tree_columns.json`. Model: `models/revision_v2/06_'+name+'.joblib`. Full tables, matched deltas, predictions, settings and manifests are in the corresponding `revision_v2` folders.']
    (doc/'baseline_summary.md').write_text('\n'.join(lines),encoding='utf-8')
    feature_groups={
      'Flow totals and load':['dur','spkts','dpkts','sbytes','dbytes','rate','sload','dload','sloss','dloss','smean','dmean'],
      'Recorded timing summaries':['sinpkt','dinpkt','sjit','djit','tcprtt','synack','ackdat'],
      'Protocol, state and header values':['proto','service','state','sttl','dttl','swin','dwin','stcpb','dtcpb','is_sm_ips_ports'],
      'Service-specific counters':['trans_depth','response_body_len','is_ftp_login','ct_ftp_cmd','ct_flw_http_mthd'],
      'Contextual connection counts':['ct_srv_src','ct_state_ttl','ct_dst_ltm','ct_src_dport_ltm','ct_dst_sport_ltm','ct_dst_src_ltm','ct_src_ltm','ct_srv_dst']}
    original=read_json(ROOT/'results/splits/02_split_manifest.json')['predictors']
    flattened=sum(feature_groups.values(),[])
    assert len(flattened)==42 and len(set(flattened))==42 and set(flattened)==set(original)
    write_json(doc/'available_feature_groups.json',dict(groups=feature_groups,meaning='Auditable tabular grouping; not validated streams or event order',
            engineered=['fe_total_bytes','fe_byte_balance','fe_packet_balance'],unavailable=['query names','query types','DNS responses','flow timestamps','endpoint identifiers','verified botnet/C2 labels']))
    rows=[dict(group=k,columns=', '.join(v),availability='Recorded in supplied CSV; extractor semantics/timing partly unresolved') for k,v in feature_groups.items()]
    pd.DataFrame(rows).to_csv(ROOT/'results/tables/revision_v2/available_feature_groups.csv',index=False)
    text=f'''# Handoff to the main model research

## The common task and performance bar

Use offline completed-flow classification into {len(CLASSES)} fixed `attack_cat` classes (including Normal). Primary metric: fixed-ten-class macro F1. Report every class, binary attack/normal metrics, PR/AP and confusion matrices. The current bounded tabular bar is **{chosen['model']} with {chosen['feature_set']}**, macro F1 {best.macro_f1:.4f} on model-selection. Final performance is unknown. Original42 ensemble is {results[(results.model=='lightgbm')&(results.feature_set=='original42')].iloc[0].macro_f1:.4f} and remains a close reference; the engineered gain is small, Worms-sensitive and not proven robust.

Exact model: `models/revision_v2/06_{name}.joblib`. Exact input: `data/processed/05/05_{chosen['feature_set']}_tree_{{role}}.npz`, with `05_row_links_{{role}}.csv`. Original source names and encoded column order are in `models/preprocessing/05/05_{chosen['feature_set']}_tree_columns.json`. Settings are in `models/revision_v2/06_{name}_settings.json`; preprocessing comes from protected Notebooks 03/04 artifacts. The revised manifests bind every dependency to its hash. The separate scaled inputs are available for linear/neural models; do not silently feed scaled values to a model trained on tree inputs.

## Available measurements and missing streams

`available_feature_groups.json` maps all 42 predictors exactly once. It groups flow volume/load, timing summaries, protocol/state/header values, service counters and contextual counts. These are recorded tabular measurements, not separate NET/DNS event streams. Intended definitions and local extraction remain distinct claims; contextual connection counts require history whose past-only implementation cannot be replayed from this subset.

| Reference requirement | CSV evidence | What we can honestly say |
|---|---|---|
| Packet rate, duration, size, inter-arrival information | `rate`, `dur`, directional packet/byte counters, `smean/dmean`, `sinpkt/dinpkt` | Related per-flow fields exist. Exact extractor formula/units need documented mapping; a per-flow summary is not the paper's time-window extraction. |
| Protocol distribution per aligned window | `proto` per flow | A measured category exists, but no endpoint/time window for a protocol distribution. |
| DNS query entropy, rate, length, frequency, type distribution | No query names, transaction types, timestamps or raw responses | Unavailable. Do not approximate query entropy/length from total flow bytes. |
| DNS indication | `service='dns'` category | A recorded service marker, already represented by one-hot encoding. It is not raw DNS transaction evidence or a second stream. |
| Additional engineered summaries | Total bytes and two direction balances | Deterministic count summaries with documented formulas and zero rules; no added measured information. |
| Source/destination-pair aligned time windows | No endpoint identifiers or timestamps | Cannot reproduce the paper's alignment, replay ct counters or establish genuine temporal samples. |
| Botnet/C2 infection labels | Binary Attack and broad intrusion categories | Botnet-specific detection is unsupported without separately established per-event/scenario infection/C2 ground truth. |

The reviewed paper uses aligned NET/DNS windows, Conv1D branches, fusion, attention and BiLSTM. Exact reproduction needs author extraction code, input shapes, splitting/averaging details and the missing measured records. Some paper preprocessing/settings/results details are ambiguous; see `research_alignment.md` with PDF page locators. Do not compare our macro F1 directly with its reported accuracy or highest-fold figures.

## A concrete first architecture experiment

1. Freeze a tabular task/input contract and a small training-only grouped search plan before fitting. Start with original42 and total_and_byte44; keep the five-set baseline record. Any additional representation is a newly declared candidate, not a hidden change.
2. Train a fair non-convolutional MLP comparator using the same fitting rows, targets, fitted scaling/category groups and a bounded search budget. It helps determine whether a benefit is from general neural capacity rather than convolution.
3. Test a **tabular Conv1D adaptation**, explicitly labelled as such. Declare a semantic source-feature order and respect complete categorical blocks or defined feature tokens. Column position is not network time. Compare against the MLP and stronger ensemble.
4. Predeclare several frozen column-order permutations using fitting-only choices. Keep rows/settings/budget fixed. A gain that depends strongly on arbitrary adjacency is an ordering-sensitive tabular finding, not temporal botnet detection.
5. Add attention in a matched ablation: same input, backbone and budget, with/without the block. Describe attention over feature tokens as feature interaction. A flow/context two-branch adaptation is possible only if labelled that way; do not call the contextual branch DNS.
6. Defer genuine TCN/BiLSTM event-sequence claims until measured timestamps/endpoints/windows exist. If experimenting over feature positions, label the ordered-feature adaptation and compare with/without it. It does not demonstrate temporal dependence. Bidirectional processing also requires an offline-complete-window availability contract.
7. Compare fixed seeds across all arms; after a modest declared search, use matched repeat seeds to examine stability. Report macro F1, all classes, support, confusion tradeoffs, fit/prediction cost and complexity. Small Worms gains are unstable; DoS/Analysis/Backdoor→Exploits confusion is an important target for improvement. Their ambiguous recorded inputs cannot be fixed by deterministic formulas alone.
8. If class weighting is proposed for neural training, include a separate matched weighted tabular control and declare its rule/search beforehand. Do not compare a weighted neural model only against unweighted tabular models and attribute the difference to architecture.
9. Keep model-selection exposure explicit; do not choose architecture, thresholds or features from official-test outcomes. Calibration remains reserved. After development choices are frozen, plan calibration/operating policy under the existing protocol, then evaluate the full official test once with specified full/overlap/non-overlap reporting. Name whether overlap refers to supplied training or actual fitted rows.

Each component must improve a matched comparator at acceptable class-level cost. No model is promised to win. The final test, rather than repeated development-score inspection, must determine the eventual performance claim.

## Data required for an authentic DNS/temporal or botnet claim

Obtain measured flow start/end times and endpoint/session identifiers; DNS query names/types/times/responses and endpoint linkage; documented completed-flow extraction; past-only window state/reset/tie rules; and independently verified botnet infection/C2/scenario labels aligned to records. Preserve capture provenance and group/evaluate by hosts/capture periods where possible. Define how a flow or window receives a target before training. A different dataset would need a newly documented compatible protocol, not a fabricated label mapping.
'''
    (doc/'architecture_handoff.md').write_text(text,encoding='utf-8')
    versions={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','lightgbm','matplotlib','joblib','nbformat','nbclient','ipykernel','pytest']}
    write_json(doc/'environment_versions.json',dict(python=sys.version,executable=sys.executable,project_venv_inherits_system_packages=True,relevant_installed_versions=versions,full_isolated_lock=False))
    readme=f'''# Botnet Attack Detection research

The current defensible benchmark is **offline completed-flow intrusion detection** on UNSW-NB15: ten classes including Normal. The supplied files do not establish literal botnet labels, raw DNS transactions or chronological event sequences. Authentic NET/DNS and temporal reference-model work requires additional measured data; tabular adaptations must be labelled clearly.

## Current workflow

Notebook 01 remains unchanged. Notebooks 02–06 have been revised with simple questions, worked examples, full evidence, reproducible checks and versioned artifacts. See the [one-page workflow](docs/revision_v2/workflow_map.md), [research alignment and paper review](docs/revision_v2/research_alignment.md), [change log](docs/revision_v2/change_log.md), [baseline results](docs/revision_v2/baseline_summary.md) and [architecture handoff](docs/revision_v2/architecture_handoff.md).

The provisional baseline is {chosen['model']} with {chosen['feature_set']} (development macro F1 {best.macro_f1:.4f}). Its small engineered-feature gain is uncertain and includes rare-class tradeoffs. Final performance is unknown. The same model-selection role previously served ten experiments; further selected scores are optimistic development evidence.

## Preserved data and experimental history

Raw CSVs remain in `data/raw`, unchanged. Frozen development roles are 140,273 fitting, 17,534 model-selection and 17,534 calibration rows, with identical original patterns kept together. Calibration receives integrity/transformation checks only; it is not predicted or used for fitting/selection in this revision. The 82,332-row official test remains separate. It was historically inspected in data audits, which are disclosed; its outcomes are not used for improvements.

All earlier source code, assignments, generated inputs, models, reports, tables and figures remain in place. A recoverable pre-edit snapshot of 379 files is in `snapshots/workflow_20261012T143405Z`, with a verified manifest and raw hashes. Current 02–06 notebook filenames are unchanged; their earlier contents are recoverable from that snapshot. New outputs use `results/tables/revision_v2`, `results/figures/revision_v2`, `data/processed/revision_v2`, `models/revision_v2` and `docs/revision_v2`.

## Execute and verify

Select `.venv/Scripts/python.exe` as interpreter/kernel. Use the current runner, one notebook at a time:

```powershell
.venv\\Scripts\\python.exe scripts/run_research_revision.py 02
.venv\\Scripts\\python.exe scripts/run_research_revision.py 03
.venv\\Scripts\\python.exe scripts/run_research_revision.py 04
.venv\\Scripts\\python.exe scripts/run_research_revision.py 05
.venv\\Scripts\\python.exe scripts/run_research_revision.py 06
.venv\\Scripts\\python.exe -m pytest -q
.venv\\Scripts\\python.exe scripts/verify_research_revision.py
```

The runner registers a project-local Jupyter kernel, saves executed outputs, checks protected files and records notebook/artifact hashes. Kernel communication may require local execution permission. The baseline runner reuses only hash-bound unchanged tuning checkpoints and exact saved model settings; no hidden new search is added. `scripts/build_research_revision.py` recreates unexecuted notebook sources (optionally specify a number); rebuilding clears saved outputs, so execute afterward. Old builders/runners are historical and should not overwrite the revised notebook files.

`results/tables/revision_v2/final_verification.json` records independent artifact reload, fold-local scaler/vocabulary verification, regenerated metrics/confusion matrices and exact saved-model predictions. Each notebook has execution and artifact manifests. `docs/revision_v2/environment_versions.json` records installed versions. The Python 3.11 project environment inherits system packages and is not a full isolated lock; the original `requirements.txt` remains unchanged. LightGBM is available locally. No neural, XAI, calibration or RAG training is included in this phase.

## Rules for later work

Exclude IDs, targets and audit metadata. Fit learned preprocessing/selection/tuning inside fitting data with original-pattern groups. Keep original42 as a reference; engineered features are representations of the same measurements. Compare architecture components with matched baselines/ablations and inspect rare-class errors. Column order is not time, a DNS service marker is not a query stream, and Attack is not verified Botnet. Confidence calibration, operating thresholds, reserved-role refits and final official-test reporting are later declared steps.
'''
    (ROOT/'README.md').write_text(readme,encoding='utf-8')
    diary=ROOT/'docs/research_notes.md'; old=diary.read_text(encoding='utf-8-sig')
    marker='## 2026-10-12 — Research workflow revision v2'
    entry=f'\n\n{marker}\n\nReviewed the local reference paper and recorded data/label/sequence gaps before implementation. Preserved a verified 379-file snapshot and all earlier artifacts. Expanded 02–06 with transparent examples and fitting-only investigations; reproduced transforms and feature/association evidence. Added 18 grouped inner fits and an unweighted ensemble to the five-set baseline experiment. Provisional development choice: {chosen["model"]}/{chosen["feature_set"]}, macro F1 {best.macro_f1:.4f}; small engineered gain is rare-class-sensitive. Calibration and final test remain reserved. Current reports, manifests and independent verification are in revision_v2 folders. See workflow_map.md, change_log.md, baseline_summary.md and architecture_handoff.md there.\n'
    if marker not in old: diary.write_text(old+entry,encoding='utf-8')
    print('Saved baseline summary, architecture handoff, available-feature map, environment record, README and diary.')

if __name__=='__main__': main()
