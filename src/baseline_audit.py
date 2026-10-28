"""Notebook 06 provenance, experiment execution, figures and frozen handoff."""

from pathlib import Path
import ast
import importlib.metadata
import json
import time
import warnings

import joblib
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from src.data.load_data import sha256
from src.data.feature_selection import comparison_sets, encoded_sources
from src.baseline_models import (SETS, MODELS, CLASSES, NORMAL, SEED, TIE_TOLERANCE,
    make_model, exact_settings, evaluate, choose_candidate, comparisons)

NEW_FILES={'baseline_models.py','baseline_audit.py','build_baseline_notebook.py',
    'run_baseline_models.py','verify_baseline_models.py','test_baseline_models.py'}


def protected_files(root):
    files=[]
    for folder in ('data','models','results','docs','configs','src','notebooks','scripts','tests'):
        for p in (root/folder).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and not p.name.startswith('06_') and p.name not in NEW_FILES:
                files.append(p)
    files += [root/p for p in ('README.md','requirements.txt') if (root/p).is_file()]
    return sorted(files)


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def table(state,frame,name):
    frame.to_csv(state['root']/f'results/tables/06_{name}.csv',index=False,float_format='%.17g')
    return frame


def figure(state,fig,name,show=True):
    fig.tight_layout()
    fig.savefig(state['root']/f'results/figures/06_{name}.png',dpi=150,bbox_inches='tight')
    if show: plt.show()
    plt.close(fig)


def setup(root):
    root=Path(root)
    protected={p.relative_to(root).as_posix():sha256(p) for p in protected_files(root)}
    read=lambda name:json.loads((root/name).read_text(encoding='utf-8'))
    m2=read('results/splits/02_split_manifest.json')
    m3=read('data/processed/03/03_artifact_manifest.json')
    m4=read('data/processed/04/04_artifact_manifest.json')
    m5=read('data/processed/05/05_artifact_manifest.json')
    run5=read('results/tables/05_notebook_execution.json')
    assert sha256(root/'data/processed/05/05_artifact_manifest.json')==run5['artifact_manifest_sha256']
    assert m5['manifest04_sha256']==sha256(root/'data/processed/04/04_artifact_manifest.json')
    assert m4['source_manifest03_sha256']==sha256(root/'data/processed/03/03_artifact_manifest.json')
    for manifest,key in [(m3,'artifacts_sha256'),(m4,'artifact_sha256'),(m5,'artifact_sha256')]:
        for path,expected in manifest[key].items(): assert sha256(root/path)==expected,path
    # The notebook container can change when an editor saves outputs/metadata.
    # Verify every cell source against its already hash-verified builder in memory.
    builder=root/'scripts/build_feature_selection_notebook.py'
    parsed=ast.parse(builder.read_text(encoding='utf-8'))
    def omitted(node):
        if not isinstance(node,ast.Expr) or not isinstance(node.value,ast.Call): return False
        function=node.value.func
        return ((isinstance(function,ast.Name) and function.id=='print') or
                (isinstance(function,ast.Attribute) and isinstance(function.value,ast.Name) and
                 function.value.id=='nbf' and function.attr=='write'))
    parsed.body=[node for node in parsed.body if not omitted(node)]
    scope={'__file__':str(builder)}
    exec(compile(parsed,str(builder),'exec'),scope)
    actual05=nbformat.read(root/'notebooks/05_feature_selection.ipynb',4); expected05=scope['nb']
    assert len(actual05.cells)==len(expected05.cells)
    assert all(a.cell_type==b.cell_type and a.source==b.source for a,b in zip(actual05.cells,expected05.cells))
    provenance05=dict(actual_sha256=sha256(root/'notebooks/05_feature_selection.ipynb'),
        recorded_execution_sha256=run5['notebook_sha256'],
        file_hash_matches_old_execution_record=sha256(root/'notebooks/05_feature_selection.ipynb')==run5['notebook_sha256'],
        all_code_and_markdown_match_manifest_verified_builder=True,verified_artifacts=len(m5['artifact_sha256']),
        explanation='The saved notebook differs only in non-source content; all source and manifest artifacts verify. Earlier files and execution records were preserved.')
    for role,token in [('train','training'),('test','testing')]:
        assert sha256(root/f'data/raw/UNSW_NB15_{token}-set.csv')==m2['source_sha256'][role]
    for m in (m3,m4,m5): assert m['assignment_sha256']==m2['assignment_sha256']
    assert m5['source_train_sha256']==m4['source_train_sha256']==m2['source_sha256']['train']
    assert sha256(root/'results/splits/02_train_validation_assignments.csv')==m2['assignment_sha256']
    assert sha256(root/'models/preprocessing/05/05_comparison_sets.json')==m5['set_definition_sha256']
    # The Notebook 04 non-source hash drift was already verified and documented in 05.
    provenance=read('results/tables/05_notebook04_provenance.json')
    assert sha256(root/'notebooks/04_feature_engineering.ipynb')==provenance['actual_sha256']
    assert provenance['all_code_and_markdown_match_manifest_verified_builder']
    notebook_sources={}
    for p in sorted((root/'notebooks').glob('0[2-5]*.ipynb')):
        n=nbformat.read(p,4); nbformat.validate(n)
        code=[c for c in n.cells if c.cell_type=='code']
        assert all(c.execution_count is not None for c in code)
        assert not any(o.output_type=='error' for c in code for o in c.outputs)
        notebook_sources[p.name]=sha256(p)
    original=m2['predictors']; sets=comparison_sets(original)
    assert m5['sets']==sets==read('models/preprocessing/05/05_comparison_sets.json')['sets']
    for candidate in SETS: assert read(f'models/preprocessing/05/05_{candidate}_features.json')==sets[candidate]
    assert read('models/preprocessing/03/03_target_mapping.json')=={c:i for i,c in enumerate(CLASSES)}
    assignment=pd.read_csv(root/'results/splits/02_train_validation_assignments.csv')
    # Training-file targets only. No official-test records are parsed.
    source_targets=pd.read_csv(root/'data/raw/UNSW_NB15_training-set.csv',usecols=['id','label','attack_cat'])
    assert assignment.row_position.equals(pd.Series(np.arange(len(assignment)),name='row_position'))
    for a,b in [('source_id','id'),('label','label'),('attack_cat','attack_cat')]:
        assert np.array_equal(assignment[a],source_targets[b])
    assert assignment.groupby('feature_group_id').role.nunique().max()==1
    assert set(assignment.role)=={'train','model_selection','calibration'}
    links={}; role_support=[]; shapes=[]; mappings={}
    for role in ('train','model_selection','calibration'):
        link=pd.read_csv(root/f'data/processed/05/05_row_links_{role}.csv')
        expected=assignment[assignment.role.eq(role)].reset_index(drop=True)
        for column in ('row_position','source_id','feature_group_id','role','label','attack_cat'):
            assert np.array_equal(link[column],expected[column])
        assert np.array_equal(link.input_row,np.arange(len(link)))
        assert np.array_equal(link.label,link.attack_cat.ne('Normal').astype(int))
        assert set(link.attack_cat)==set(CLASSES)
        links[role]=link
        role_support.append(dict(role=role,rows=len(link),original_patterns=link.feature_group_id.nunique(),
            use='Fit only' if role=='train' else 'Evaluate and compare' if role=='model_selection' else 'Integrity only; no model predictions'))
        for representation in ('scaled','tree'):
            full=sparse.load_npz(root/f'data/processed/04/04_{representation}_augmented_{role}.npz').tocsr()
            full_names=read(f'models/preprocessing/04/04_{representation}_augmented_feature_names.json')
            for candidate in SETS:
                mapping=read(f'models/preprocessing/05/05_{candidate}_{representation}_columns.json')
                kept=sets[candidate]['retained']; names=mapping['encoded_names']; ix=mapping['indices_in_04']
                assert mapping['retained_original_names']==kept
                assert names==[full_names[i] for i in ix]
                sources=encoded_sources(names,original)
                assert sources==mapping['original_source_for_each_encoded_column']
                assert set(sources)==set(kept) and not set(sources)&{'id','label','attack_cat','row_position','feature_group_id','role'}
                full_sources=encoded_sources(full_names,original)
                assert ix==[i for i,s in enumerate(full_sources) if s in kept]
                matrix=sparse.load_npz(root/f'data/processed/05/05_{candidate}_{representation}_{role}.npz').tocsr()
                assert matrix.shape==(len(link),len(names)) and np.isfinite(matrix.data).all()
                assert (matrix!=full[:,ix]).nnz==0
                mappings[(candidate,representation)]=mapping
                shapes.append(dict(feature_set=candidate,representation=representation,role=role,
                    rows=matrix.shape[0],columns=matrix.shape[1],source_features=len(kept),finite=True,exact_04_slice=True))
    # Use the hash-verified prior exact overlap audit, including calibration, without scoring that role.
    overlaps=pd.read_csv(root/'results/tables/05_candidate_cross_role_overlap.csv')
    assert len(overlaps)==45 and overlaps.shared_vectors.eq(0).all()
    for folder in ('results/tables','results/figures','models/baselines/06','data/processed/06','docs','configs'):
        (root/folder).mkdir(parents=True,exist_ok=True)
    state=dict(root=root,protected=protected,m2=m2,m5=m5,sets=sets,links=links,mappings=mappings,
        notebooks=notebook_sources,provenance=provenance,provenance05=provenance05,experiment_metrics=[],per_class=[],fit_records=[])
    support=table(state,pd.DataFrame(role_support),'role_support')
    table(state,pd.DataFrame(shapes),'input_verification')
    class_support=[]
    for role in ('train','model_selection'):
        for target in CLASSES:
            subset=links[role][links[role].attack_cat.eq(target)]
            class_support.append(dict(role=role,attack_cat=target,rows=len(subset),original_patterns=subset.feature_group_id.nunique()))
    table(state,pd.DataFrame(class_support),'class_support')
    assert links['model_selection'].attack_cat.eq('Worms').sum()==13
    write_json(root/'results/tables/06_input_integrity.json',dict(source_sha256=m2['source_sha256'],
        assignment_sha256=m2['assignment_sha256'],manifest05_sha256=sha256(root/'data/processed/05/05_artifact_manifest.json'),
        verified05_artifacts=len(m5['artifact_sha256']),notebook_sha256=notebook_sources,
        prior_notebook04_provenance=provenance,notebook05_provenance=provenance05,
        calibration_use='Integrity only',official_test_parsed=False))
    return state,support


def declare_plan(state):
    models={m:dict(representation='scaled' if m=='logistic' else 'tree',settings=exact_settings(make_model(m))) for m in MODELS}
    plan=dict(version='06-v1',question='Do engineered features improve these simple models?',
        feature_sets=list(SETS),source_feature_lists=state['sets'],models=models,seed=SEED,
        budget='One fixed fit per family per feature set; 10 experiments; no hyperparameter search',
        weights='Unweighted only; class_weight=None; no sample weights or resampling',
        fit_role='train',evaluation_role='model_selection',calibration_role='Integrity only; no predictions',
        official_test='Hash only; no record parsing',primary_metric='macro F1 over fixed ten classes',
        classes=list(CLASSES),probability_columns=list(CLASSES),binary_decision='Multiclass prediction is not Normal',
        attack_score='1 - P(Normal); raw uncalibrated scores; average precision and PR curve only',
        threshold_selection=False,calibration_fitting=False,refitting=False,
        tie_rule=dict(tolerance=TIE_TOLERANCE,units='Absolute macro F1 (0.1 percentage point)',
            within_best='Prefer fewer source features, then logistic, then higher macro F1, then declared set order',
            meaning='Practical preference, not statistical equivalence'),
        eligible_rule='Finite predictions, probability checks and saved reload checks; no ConvergenceWarning',
        convergence_rule='Keep results of limited fits but flag and exclude them from provisional choice; do not increase budget based on selection scores',
        uncertainty='Point comparisons from one seed and one frozen role; no statistical significance or bootstrap claim',
        assignment_sha256=state['m2']['assignment_sha256'],manifest05_sha256=sha256(state['root']/'data/processed/05/05_artifact_manifest.json'),
        train_row_links_sha256=sha256(state['root']/'data/processed/05/05_row_links_train.csv'),
        versions={p:importlib.metadata.version(p) for p in ('numpy','pandas','scipy','scikit-learn','joblib','nbformat','nbclient')})
    path=state['root']/'configs/06_baseline_plan.json'; write_json(path,plan)
    state['plan']=plan; state['plan_sha256']=sha256(path)
    return pd.DataFrame([dict(model=m,representation=models[m]['representation'],
        main_settings='L2; C=1; one-vs-rest; liblinear; max 500 iterations per class; tol=0.0001' if m=='logistic' else 'Gini; depth <=16; leaf >=5 rows; all columns considered',
        seed=SEED,feature_sets=5,fits_per_set=1,class_weights='None') for m in MODELS])


def fit_family(state,family):
    root=state['root']; representation=state['plan']['models'][family]['representation']
    links=state['links']; mapping={c:i for i,c in enumerate(CLASSES)}
    y=links['train'].attack_cat.map(mapping).to_numpy()
    truth=links['model_selection'].attack_cat.map(mapping).to_numpy()
    for candidate in SETS:
        assert sha256(root/'configs/06_baseline_plan.json')==state['plan_sha256']
        experiment=f'{family}_{candidate}'; model=make_model(family)
        assert exact_settings(model)==state['plan']['models'][family]['settings']
        train=sparse.load_npz(root/f'data/processed/05/05_{candidate}_{representation}_train.npz').tocsr()
        selection=sparse.load_npz(root/f'data/processed/05/05_{candidate}_{representation}_model_selection.npz').tocsr()
        print(f'Fitting {experiment}: {train.shape[0]:,} rows, {train.shape[1]} prepared columns',flush=True)
        start=time.perf_counter()
        with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
            warnings.simplefilter('always')
            model.fit(train,y)
        seconds=time.perf_counter()-start
        assert np.array_equal(model.classes_,np.arange(10))
        probabilities=model.predict_proba(selection); predicted=model.predict(selection)
        metrics,classes,cm,norm,pr,attack_score=evaluate(truth,predicted,probabilities)
        warning_log=[dict(category=w.category.__name__,message=str(w.message)) for w in caught]
        convergence=any(issubclass(w.category,ConvergenceWarning) for w in caught)
        fitted_iterations={CLASSES[i]:int(est.n_iter_.max()) for i,est in enumerate(model.estimators_)} if family=='logistic' else {}
        model_path=root/f'models/baselines/06/06_{experiment}.joblib'
        joblib.dump(model,model_path,compress=3)
        reloaded=joblib.load(model_path)
        assert np.array_equal(predicted,reloaded.predict(selection))
        assert np.array_equal(probabilities,reloaded.predict_proba(selection))
        record=dict(experiment=experiment,model=family,feature_set=candidate,representation=representation,
            source_features=len(state['sets'][candidate]['retained']),matrix_columns=train.shape[1],
            train_rows=len(y),model_selection_rows=len(truth),fit_seconds=seconds,
            convergence_warning=convergence,eligible=not convergence,saved_model_predictions_exact=True,
            max_iterations=max(fitted_iterations.values()) if fitted_iterations else 0,**metrics)
        state['experiment_metrics'].append(record)
        classes.insert(0,'feature_set',candidate); classes.insert(0,'model',family); classes.insert(0,'experiment',experiment)
        state['per_class'].append(classes)
        metadata=dict(experiment=experiment,model=family,feature_set=candidate,representation=representation,
            settings=exact_settings(model),fit_role='train',fit_rows=len(y),fit_row_links_sha256=state['plan']['train_row_links_sha256'],
            plan_sha256=state['plan_sha256'],assignment_sha256=state['m2']['assignment_sha256'],
            source_features=state['sets'][candidate]['retained'],encoded_columns=state['mappings'][(candidate,representation)]['encoded_names'],
            probability_columns=list(CLASSES),warnings=warning_log,n_iter_by_class=fitted_iterations,
            eligible=not convergence,fit_seconds=seconds,calibration_predicted=False)
        write_json(root/f'models/baselines/06/06_{experiment}_settings.json',metadata)
        prediction=links['model_selection'].copy()
        prediction['experiment']=experiment
        prediction['predicted_attack_cat']=[CLASSES[i] for i in predicted]
        prediction['predicted_binary_attack']=(predicted!=NORMAL).astype(int)
        prediction['attack_score']=attack_score
        for i,target in enumerate(CLASSES): prediction[f'p_{target}']=probabilities[:,i]
        destination=root/f'data/processed/06/06_{experiment}_model_selection_predictions.csv'
        prediction.to_csv(destination,index=False,float_format='%.17g')
        saved=pd.read_csv(destination,float_precision='round_trip')
        pd.testing.assert_frame_equal(saved,prediction,check_dtype=False,check_exact=True)
        for name,values in [('raw',cm),('normalized',norm)]:
            pd.DataFrame(values,index=CLASSES,columns=CLASSES).rename_axis('actual_class').to_csv(
                root/f'results/tables/06_{experiment}_confusion_{name}.csv',float_format='%.17g')
        pr.to_csv(root/f'results/tables/06_{experiment}_precision_recall.csv',index=False,float_format='%.17g')
        state['fit_records'].append(dict(experiment=experiment,warning_log=json.dumps(warning_log),
            n_iter_by_class=json.dumps(fitted_iterations),eligible=not convergence))
        print(f'  macro F1={metrics["macro_f1"]:.4f}; Worms={metrics["worms_correct"]}/{metrics["worms_support"]}; '
              f'fit={seconds:.1f}s; converged={not convergence}',flush=True)
        table(state,pd.DataFrame(state['experiment_metrics']),'model_selection_metrics')
        table(state,pd.concat(state['per_class'],ignore_index=True),'per_class_metrics')
        table(state,pd.DataFrame(state['fit_records']),'fit_diagnostics')
    return pd.DataFrame(state['experiment_metrics']).query('model == @family')


def compare_results(state):
    results=pd.DataFrame(state['experiment_metrics']); classes=pd.concat(state['per_class'],ignore_index=True)
    assert len(results)==10 and len(classes)==100
    differences,class_differences=comparisons(results,classes)
    table(state,differences,'feature_comparisons'); table(state,class_differences,'class_tradeoffs')
    decision=choose_candidate(results,{k:len(v['retained']) for k,v in state['sets'].items()})
    write_json(state['root']/'models/baselines/06/06_provisional_decision.json',decision)
    state.update(results=results,classes=classes,differences=differences,class_differences=class_differences,decision=decision)
    return results,differences,decision


def result_figures(state):
    results=state['results']; delta=state['differences']
    fig,axes=plt.subplots(1,2,figsize=(12,4),sharey=True)
    for ax,family in zip(axes,MODELS):
        rows=results[results.model.eq(family)].set_index('feature_set').loc[list(SETS)]
        bars=ax.barh(range(5),rows.macro_f1,color=['#64748b','#0d9488','#0d9488','#0d9488','#0d9488'])
        ax.set_yticks(range(5),SETS); ax.invert_yaxis(); ax.set_xlim(0,1)
        for bar,value in zip(bars,rows.macro_f1): ax.text(value+.01,bar.get_y()+bar.get_height()/2,f'{value:.4f}',va='center',fontsize=9)
        ax.set_title('Logistic classifier' if family=='logistic' else 'Decision tree'); ax.set_xlabel('Ten-class macro F1')
    fig.suptitle('Same rows and settings; model-selection results only')
    figure(state,fig,'macro_f1_comparison')
    fig,ax=plt.subplots(figsize=(8,4)); width=.35; x=np.arange(4)
    for index,family in enumerate(MODELS):
        rows=delta[(delta.model==family)&(delta.reference=='original42')].set_index('candidate').loc[list(SETS[1:])]
        ax.bar(x+(index-.5)*width,100*rows.delta_macro_f1,width,label=family)
    ax.axhline(0,color='black',linewidth=.8); ax.set_xticks(x,SETS[1:])
    ax.set(ylabel='Macro F1 difference (percentage points)',title='Engineered additions versus the original 42')
    ax.legend(); figure(state,fig,'engineered_differences')
    fig,axes=plt.subplots(1,2,figsize=(12,4),sharey=True)
    for ax,family in zip(axes,MODELS):
        rows=results[results.model.eq(family)].set_index('feature_set').loc[list(SETS)]
        ax.barh(range(5),rows.worms_recall,color='#b45309'); ax.set_yticks(range(5),SETS); ax.invert_yaxis(); ax.set_xlim(0,1)
        for i,row in enumerate(rows.itertuples()): ax.text(row.worms_recall+.02,i,f'{row.worms_correct}/13',va='center')
        ax.set_title(family); ax.set_xlabel('Worms recall')
    fig.suptitle('Only 13 Worms rows: one correct prediction changes recall by 7.69 points')
    figure(state,fig,'worms_recall')


def confusion_figure(state,experiment,show=True):
    root=state['root']; matrices=[pd.read_csv(root/f'results/tables/06_{experiment}_confusion_{kind}.csv',index_col=0).to_numpy() for kind in ('raw','normalized')]
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,matrix,kind in zip(axes,matrices,('Counts','Share of each actual class')):
        plot=ax.imshow(matrix,cmap='Blues',vmin=0,vmax=1 if kind!='Counts' else None)
        ax.set_xticks(range(10),CLASSES,rotation=65,ha='right'); ax.set_yticks(range(10),CLASSES)
        ax.set(xlabel='Predicted class',ylabel='Actual class',title=kind)
        for i in range(10):
            for j in range(10):
                value=matrix[i,j]
                if value: ax.text(j,i,f'{value:.2f}' if kind!='Counts' else f'{int(value)}',ha='center',va='center',fontsize=7,
                    color='white' if value>matrix.max()*.55 else 'black')
        fig.colorbar(plot,ax=ax,fraction=.035,pad=.02)
    fig.suptitle(f'{experiment}: model selection')
    figure(state,fig,f'{experiment}_confusion',show=show)


def detailed_figures(state):
    chosen=state['decision']['experiment']
    for experiment in state['results'].experiment: confusion_figure(state,experiment,show=experiment==chosen)
    fig,axes=plt.subplots(1,2,figsize=(12,4),sharex=True,sharey=True)
    for ax,family in zip(axes,MODELS):
        for candidate in SETS:
            experiment=f'{family}_{candidate}'
            pr=pd.read_csv(state['root']/f'results/tables/06_{experiment}_precision_recall.csv')
            metric=state['results'].set_index('experiment').loc[experiment,'attack_average_precision']
            ax.step(pr.recall,pr.precision,where='post',label=f'{candidate}: AP {metric:.4f}')
        prevalence=state['links']['model_selection'].label.mean()
        ax.axhline(prevalence,color='gray',linestyle=':',label=f'Attack share {prevalence:.3f}')
        ax.set(title=family,xlabel='Attack recall',ylabel='Attack precision',xlim=(0,1),ylim=(0,1.02)); ax.legend(fontsize=7)
    fig.suptitle('Precision–recall from 1 − P(Normal); no operating threshold selected')
    figure(state,fig,'attack_precision_recall')


def findings_text(state):
    rows=state['results']; diffs=state['differences']; decision=state['decision']; lines=[]
    for family in MODELS:
        r=rows[rows.model.eq(family)].set_index('feature_set')
        original=r.loc['original42']; eligible=r[r.eligible]
        best=eligible.macro_f1.idxmax() if len(eligible) else None
        lines.append(f'**{family}:** original42 macro F1 = {original.macro_f1:.4f}. '+
            (f'Highest eligible score: {best} = {r.loc[best,"macro_f1"]:.4f}.' if best else 'No eligible converged result.'))
        for candidate in SETS[1:]:
            difference=float(r.loc[candidate,'macro_f1']-original.macro_f1)
            direction='higher' if difference>0 else 'lower' if difference<0 else 'unchanged'
            lines.append(f'- `{candidate}`: {abs(100*difference):.3f} percentage points {direction}; '
                f'Worms correct {int(r.loc[candidate,"worms_correct"])}/13 versus {int(original.worms_correct)}/13.')
        incremental=diffs[(diffs.model==family)&(diffs.reference!='original42')]
        lines.append('**What each balance adds:**')
        for row in incremental.itertuples():
            lines.append(f'- `{row.candidate}` versus `{row.reference}`: '
                f'{100*row.delta_macro_f1:+.3f} macro-F1 percentage points; '
                f'{row.delta_worms_correct:+.0f} correct Worms rows.')
        tradeoffs=state['class_differences']
        complete=tradeoffs[(tradeoffs.model==family)&(tradeoffs.reference=='original42')&(tradeoffs.candidate=='all45')]
        lower=complete.loc[complete.delta_f1.idxmin()]; upper=complete.loc[complete.delta_f1.idxmax()]
        lines.append(f'For all45 versus original42, class F1 changes range from '
            f'{100*lower.delta_f1:+.3f} points for {lower.attack_cat} to '
            f'{100*upper.delta_f1:+.3f} points for {upper.attack_cat}.')
    lines.append('\nThese are observed development differences for these fixed models, not proof of a general improvement. '
        'All three additions summarize existing measurements; they add no new measured information. '+
        'Positive and negative per-class changes, including rare classes, are saved in the class-tradeoff table.')
    lines.append(f'\n**Provisional choice:** `{decision["experiment"]}`, macro F1 {decision["macro_f1"]:.4f}. '
        f'Its gap to the highest eligible score is {100*decision["gap_to_best"]:.3f} percentage points. '
        'The declared tie rule prefers fewer features within 0.1 percentage point, then the linear model. '
        'This tolerance is a practical preference, not evidence of statistical equivalence.')
    chosen=state['classes'][state['classes'].experiment.eq(decision['experiment'])].set_index('attack_cat')
    lines.append(f'\n**Rare-class caution:** the chosen model detects {int(chosen.loc["Worms","correct"])}/13 Worms rows. '
        'One changed prediction shifts Worms recall by 7.69 percentage points. '
        'A good overall macro F1 can coexist with weak rare-class recall; this decision does not override that tradeoff.')
    limited=rows[~rows.eligible]
    if len(limited): lines.append('\n**Fit limitation:** '+', '.join(limited.experiment)+
        ' had convergence warnings. Their scores remain visible, but they were excluded from the provisional choice; no score-guided retry occurred.')
    lines.append('\n**Still uncertain:** seed stability, sensitivity to model settings and class imbalance, '
        'performance on new network context, calibration, and the final official-test score. '
        'No uncertainty interval or significance claim is made from this single frozen split and seed. '
        'Repeated patterns and conflicting labels are retained, and exact patterns are not verified independent events.')
    lines.append('\n**Selection optimism:** the same model-selection rows compared ten candidates and informed the choice. '
        'The selected development score is optimistic; final performance is unknown. '
        'Accuracy, weighted F1 and binary results are secondary and do not replace the declared macro-F1 rule.')
    return '\n\n'.join(lines)


def finish(state):
    root=state['root']
    after={p.relative_to(root).as_posix():sha256(p) for p in protected_files(root)}
    assert after==state['protected'],'Earlier file changed'
    assert sha256(root/'configs/06_baseline_plan.json')==state['plan_sha256']
    report='# Notebook 06 — Baseline models\n\n'+findings_text(state)+'''

## What the experiments did

Ten unweighted experiments fitted the frozen train rows only: an L2 one-vs-rest logistic classifier on scaled inputs and a depth-limited decision tree on tree inputs, each with all five predeclared feature sets. Settings, seeds and budgets were identical within a model family. No new split, preprocessing fit, class resampling, confidence calibration, threshold selection, refit or official-test evaluation occurred. Calibration matrices and metadata were checked only for integrity; no model predicted their rows.

The main score is row-weighted macro F1 over the fixed ten classes. All class precision/recall/F1 and support, raw and row-normalized confusion matrices, accuracy and weighted F1 are saved. Binary decisions come from the predicted class being different from Normal. Attack average precision and PR curves use 1 − P(Normal). These uncalibrated probabilities are ranking scores, not validated confidence. Multiclass argmax is not equivalent to an attack-score threshold of 0.5, and no final operating threshold was chosen.

## Notebook 07

First review misclassifications and class tradeoffs, especially Worms, Backdoor and Analysis. Inspect how the chosen model confuses attack categories versus Normal. A separately declared equal-budget, class-weighted sensitivity comparison or seed-stability study may be justified; use train and model_selection only and disclose further selection optimism. Do not silently change the present ten-experiment plan or refit on development rows.

When classifier choice and weights are frozen, investigate confidence calibration on the reserved calibration role under Notebook 02. Later choose an operating threshold using calibrated model-selection scores, document its purpose and freeze it before any official-test evaluation. Keep threshold selection and final full-test plus overlap/non-overlap reporting for their documented later stages.

All five sets preserve the original 42 predictors. The two balances were highly correlated, but only the matched model results here address their observed incremental benefit. No Botnet class, botnet-specific ground truth, temporal validation or verified deployment causality is established. Contextual feature availability and unknown host/session dependence remain limitations.

## Files and verification

Models and exact settings: `models/baselines/06/`. Model-selection predictions with source row positions, original labels and ten probability columns: `data/processed/06/`. Metrics, all class tradeoffs and confusion matrices: `results/tables/06_*`. Figures: `results/figures/06_*`. The fixed plan and decision are saved separately. Detailed input, execution and independent model-reload checks accompany the results.
'''
    report+=f'\nTraining-source hash: `{state["m2"]["source_sha256"]["train"]}`.\n\nFrozen assignment hash: `{state["m2"]["assignment_sha256"]}`.\n\nNotebook 05 manifest hash: `{state["plan"]["manifest05_sha256"]}`.\n\nDeclared plan hash: `{state["plan_sha256"]}`.\n'
    report+='\nPrior provenance limitation: Notebook 04 and Notebook 05 have old execution-record notebook hashes that differ from current non-source content. Their cell sources match the manifest-verified builders, and all listed artifacts verify. Both hashes for Notebook 05 are recorded in `06_input_integrity.json`; earlier notebooks and execution records remain unchanged.\n'
    report+='\nMethods: [logistic regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html), [one-vs-rest](https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html), [decision tree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html). Exact installed versions are in the plan.\n'
    (root/'docs/06_baseline_models_findings.md').write_text(report,encoding='utf-8')
    write_json(root/'results/tables/06_integrity_verification.json',dict(protected_before=state['protected'],
        protected_after=after,protected_unchanged=True,plan_unchanged=True,calibration_predicted=False,
        official_test_parsed=False,refit=False,threshold_selected=False,calibrator_fitted=False))
    deliverables=[*root.glob('results/tables/06_*'),*root.glob('results/figures/06_*'),
        *root.glob('models/baselines/06/06_*'),*root.glob('data/processed/06/06_*'),root/'docs/06_baseline_models_findings.md',root/'configs/06_baseline_plan.json']
    deliverables += [root/f'src/{p}' for p in ('baseline_models.py','baseline_audit.py')]
    deliverables += [root/f'scripts/{p}' for p in ('build_baseline_notebook.py','run_baseline_models.py','verify_baseline_models.py')]
    deliverables += [root/'tests/test_baseline_models.py']
    deliverables=[p for p in deliverables if p.name not in ('06_artifact_manifest.json','06_notebook_execution.json','06_independent_verification.json')]
    manifest=dict(version='06-v1',plan_sha256=state['plan_sha256'],assignment_sha256=state['m2']['assignment_sha256'],
        manifest05_sha256=state['plan']['manifest05_sha256'],models=state['plan']['models'],feature_sets=list(SETS),
        classes=list(CLASSES),fit_role='train',evaluation_role='model_selection',decision=state['decision'],
        notebook05_provenance=state['provenance05'],
        calibration_predicted=False,official_test_parsed=False,protected_unchanged=True,
        artifact_sha256={p.relative_to(root).as_posix():sha256(p) for p in sorted(deliverables)})
    write_json(root/'data/processed/06/06_artifact_manifest.json',manifest)
    return dict(experiments=10,saved_models=10,protected_files=len(after),earlier_files_unchanged=True,
        calibration_predictions=0,official_test_evaluations=0,declared_plan_unchanged=True,artifacts=len(deliverables))
