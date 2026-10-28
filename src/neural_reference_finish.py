"""Read-only replay of the frozen experiment plus versioned review evidence.

No estimator is fitted here. Fold-local preprocessing is refitted only to verify
its saved provenance, never to change the inputs or select a model.
"""
from pathlib import Path
import hashlib
import warnings
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import clone
from IPython.display import display, Markdown, Image
from src.baseline_models import CLASSES, evaluate
from src.data.load_data import sha256
from src.data.feature_engineering import canonical_csr, sparse_row_keys, overlap_counts
from src.research_revision.common import read_json, write_json
from src.neural_reference import SETS, SEEDS, EPOCHS, labels, assert_groups, predict, make_model, current_protected

REVIEW='neural_reference_review_v2'
ORIGINAL='neural_reference_v1'


def preserved_files(root):
    protected,_=current_protected(root)
    # The training module, plan, checkpoints and scientific evidence stay byte-exact.
    for base in ('data/processed','models','results/figures','results/tables'):
        for p in (root/base/ORIGINAL).rglob('*'):
            if p.is_file() and p.name!='07_execution.json':
                protected[p.relative_to(root).as_posix()]=sha256(p)
    for name in ('src/neural_reference.py','configs/07_neural_reference_v1_plan.json',
                 'configs/07_neural_reference_v1_requirements.txt',
                 'docs/neural_reference_v1/07_environment.json','docs/neural_reference_v1/07_resource_context.json'):
        protected[name]=sha256(root/name)
    return protected


def verify_hashes(root, expected):
    for name,h in expected.items():
        if sha256(root/name)!=h: raise RuntimeError(f'Protected scientific evidence changed: {name}')


def review_table(s, frame, name):
    if not s.get('record_outputs',True): return frame
    path=s['root']/f'results/tables/{REVIEW}/07_{name}.csv';path.parent.mkdir(parents=True,exist_ok=True)
    frame.to_csv(path,index=False,float_format='%.17g');s['outputs'].append(path);return frame


def load(root, record=True):
    root=Path(root); protected=preserved_files(root)
    pd.set_option('display.max_rows',100);pd.set_option('display.max_columns',16)
    snap=root/(root/'.cache/neural_review_snapshot_path.txt').read_text().strip()
    snapshot=read_json(snap/'snapshot_manifest.json')
    # Baseline is archived before this review; do not silently accept later changes.
    for name,h in protected.items():
        if name in snapshot['files']: assert h==snapshot['files'][name],name
    for name,h in snapshot['files'].items(): assert sha256(snap/name)==h,name
    plan=read_json(root/'configs/07_neural_reference_v1_plan.json')
    frozen=read_json(root/'models/neural_reference_v1/07_frozen_settings.json')
    assert frozen['plan_sha256']==sha256(root/'configs/07_neural_reference_v1_plan.json')
    assert plan['implementation_sha256']==sha256(root/'src/neural_reference.py')
    assert frozen['configuration']=={'hidden_layer_sizes':[128,64],'alpha':.01}
    assert frozen['epochs']==EPOCHS==30 and frozen['final_seeds']==list(SEEDS)
    assert plan['sets']==list(SETS) and plan['classes']==list(CLASSES)
    assignment=pd.read_csv(root/'results/splits/02_train_validation_assignments.csv',
        usecols=['row_position','source_id','feature_group_id','role'])
    assert sha256(root/'results/splits/02_train_validation_assignments.csv')==plan['assignment_sha256']
    assert assignment.groupby('feature_group_id').role.nunique().max()==1
    links={};matrices={};maps={};audit=[];overlap=[]
    for role in ('train','model_selection','calibration'):
        cols=None if role!='calibration' else ['input_row','row_position','source_id','feature_group_id','role']
        link=pd.read_csv(root/f'data/processed/05/05_row_links_{role}.csv',usecols=cols)
        expected=assignment[assignment.role.eq(role)].reset_index(drop=True)
        assert np.array_equal(link.input_row,np.arange(len(link)))
        for col in expected: assert np.array_equal(link[col],expected[col]),(role,col)
        if role!='calibration': labels(link)
        links[role]=link
    assert read_json(root/'models/preprocessing/03/03_target_mapping.json')=={c:i for i,c in enumerate(CLASSES)}
    for candidate in SETS:
        mapping=read_json(root/f'models/preprocessing/05/05_{candidate}_scaled_columns.json')
        source=mapping['retained_original_names'];names=mapping['encoded_names']
        assert len(source)==(42 if candidate=='original42' else 44)
        assert not set(source)&{'id','label','attack_cat','role','source_id','row_position','feature_group_id'}
        assert len(names)==len(set(names)) and set(mapping['original_source_for_each_encoded_column'])==set(source)
        parent_names=read_json(root/'models/preprocessing/04/04_scaled_augmented_feature_names.json')
        assert names==[parent_names[i] for i in mapping['indices_in_04']]
        maps[candidate]=mapping;matrices[candidate]={};keys={}
        for role in links:
            x=canonical_csr(sparse.load_npz(root/f'data/processed/05/05_{candidate}_scaled_{role}.npz'))
            parent=canonical_csr(sparse.load_npz(root/f'data/processed/04/04_scaled_augmented_{role}.npz'))
            assert x.shape==(len(links[role]),len(names)) and (x!=parent[:,mapping['indices_in_04']]).nnz==0
            keys[role]=sparse_row_keys(x)
            if role!='calibration':matrices[candidate][role]=x
            audit.append(dict(feature_set=candidate,role=role,rows=x.shape[0],encoded_columns=x.shape[1],finite=True,parent_equal=True))
        for a,b in (('train','model_selection'),('train','calibration'),('model_selection','calibration')):
            overlap.append(dict(feature_set=candidate,role_a=a,role_b=b,**overlap_counts(keys[a],keys[b])))
    folds=pd.read_csv(root/'data/processed/05/05_internal_folds.csv');assert_groups(folds,links['train'])
    assert sha256(root/'data/processed/05/05_internal_folds.csv')==plan['inner_folds_sha256']
    raw=pd.read_csv(root/'data/processed/05/05_original42_raw_train.csv',float_precision='round_trip')
    assert list(raw)==maps['original42']['retained_original_names'] and len(raw)==len(links['train'])
    actual=raw.groupby(list(raw),dropna=False,sort=False,observed=True).ngroup()
    check=pd.DataFrame({'actual':actual,'frozen':links['train'].feature_group_id})
    assert check.groupby('actual').frozen.nunique().max()==1 and check.groupby('frozen').actual.nunique().max()==1
    s=dict(root=root,protected=protected,outputs=[],links=links,matrices=matrices,maps=maps,folds=folds,raw=raw,plan=plan,frozen=frozen,record_outputs=record)
    display(review_table(s,pd.DataFrame(audit),'input_checks'))
    o=review_table(s,pd.DataFrame(overlap),'exact_role_overlap');assert o.shared_vectors.eq(0).all();display(o)
    support=pd.read_csv(root/'results/tables/neural_reference_v1/07_class_support.csv')
    display(support)
    print(f'Protected {len(protected)} earlier/scientific files. No calibration labels or official-test records loaded.')
    return s


def audit_search(s):
    root=s['root'];scores=pd.read_csv(root/'results/tables/neural_reference_v1/07_inner_fold_scores.csv',float_precision='round_trip')
    assert len(scores)==6 and set(zip(scores.setting,scores.fold))=={(i,f) for i in (0,1) for f in (0,1,2)}
    assert scores.epochs.eq(30).all() and scores.plan_sha256.eq(s['frozen']['plan_sha256']).all()
    means=scores.groupby('setting').macro_f1.mean();selected=int(means[means.ge(means.max()-.001)].index.min())
    assert selected==s['frozen']['chosen_setting']==1
    rows=[]
    for fold in range(3):
        fit=~s['folds'].internal_holdout_fold.eq(fold).to_numpy()
        assert not set(s['folds'].loc[fit,'feature_group_id'])&set(s['folds'].loc[~fit,'feature_group_id'])
        provenance=read_json(root/f'models/neural_reference_v1/07_inner_fold{fold}_provenance.json')
        expected=hashlib.sha256(s['folds'].loc[fit,'row_position'].to_numpy(dtype=np.int64).tobytes()).hexdigest()
        assert provenance['fit_positions_sha256']==expected and provenance['fit_rows']==int(fit.sum())
        saved=joblib.load(root/f'models/neural_reference_v1/07_inner_fold{fold}_preprocessor.joblib')
        replay=clone(joblib.load(root/'models/preprocessing/03/03_scaled_pipeline.joblib'))
        replay.fit(s['raw'].loc[fit]) # Verification only; no model fit or search.
        assert saved.get_feature_names_out().tolist()==replay.get_feature_names_out().tolist()==provenance['encoded_names']
        a=canonical_csr(saved.transform(s['raw'].loc[~fit]));b=canonical_csr(replay.transform(s['raw'].loc[~fit]))
        assert (a!=b).nnz==0
        rows.append(dict(fold=fold,group_overlap=0,fit_rows=int(fit.sum()),fold_local_preprocessing_reproduced=True))
        for index in (0,1):
            cache=read_json(root/f'.cache/neural_reference_v1/inner_{index}_{fold}.json')
            r=scores[scores.setting.eq(index)&scores.fold.eq(fold)].iloc[0]
            assert cache['plan_sha256']==s['frozen']['plan_sha256'] and cache['macro_f1']==r.macro_f1
    display(scores[['setting','fold','macro_f1','epochs','fit_seconds']])
    display(pd.read_csv(root/'results/tables/neural_reference_v1/07_search_summary.csv'))
    display(review_table(s,pd.DataFrame(rows),'search_provenance'))
    display(make_model(s['frozen']['configuration'],42))
    print('The original selection is reproduced. No MLP was fitted, retuned or early-stopped during this review.')


def verify_models(s):
    root=s['root'];y=labels(s['links']['model_selection']);metrics=pd.read_csv(root/'results/tables/neural_reference_v1/07_model_selection_metrics.csv',float_precision='round_trip')
    per=pd.read_csv(root/'results/tables/neural_reference_v1/07_per_class_metrics.csv',float_precision='round_trip');checks=[]
    assert len(metrics)==6
    losses=pd.read_csv(root/'results/tables/neural_reference_v1/07_final_training_losses.csv',float_precision='round_trip')
    for candidate in SETS:
        x=s['matrices'][candidate]['model_selection']
        for seed in SEEDS:
            name=f'07_mlp_{candidate}_seed{seed}';m=joblib.load(root/f'models/neural_reference_v1/{name}.joblib')
            settings=read_json(root/f'models/neural_reference_v1/{name}_settings.json')
            assert m.get_params()==make_model(s['frozen']['configuration'],seed).get_params()
            assert m.n_iter_==30 and m.t_==30*len(s['links']['train'])
            assert settings['classes']==list(CLASSES) and settings['encoded_names']==s['maps'][candidate]['encoded_names']
            assert settings['source_names']==s['maps'][candidate]['retained_original_names']
            assert settings['input_sha256']==sha256(root/f'data/processed/05/05_{candidate}_scaled_train.npz')
            pred,prob=predict(m,x);saved=pd.read_csv(root/f'data/processed/neural_reference_v1/{name}_predictions.csv',float_precision='round_trip')
            for col in s['links']['model_selection']:assert np.array_equal(saved[col],s['links']['model_selection'][col]),col
            assert np.array_equal(prob,saved[['p_'+c for c in CLASSES]])
            assert np.array_equal(saved.predicted_attack_cat,np.array(CLASSES)[pred])
            rec,pc,cm,ncm,pr,attack=evaluate(y,pred,prob)
            assert np.array_equal(saved.attack_score,attack) and np.array_equal(saved.predicted_binary_attack,pred!=6)
            row=metrics[metrics.feature_set.eq(candidate)&metrics.seed.eq(seed)].iloc[0]
            for key,val in rec.items():assert np.isclose(val,row[key],atol=1e-12,rtol=0),key
            selected=per[per.feature_set.eq(candidate)&per.seed.eq(seed)]
            assert selected.attack_cat.tolist()==list(CLASSES)
            for col in ('precision','recall','f1','support','correct','predicted_support'):
                assert np.allclose(selected[col],pc[col],atol=1e-12,rtol=0),col
            for suffix,actual in [('raw',cm),('normalized',ncm)]:
                f=pd.read_csv(root/f'results/tables/neural_reference_v1/{name}_confusion_{suffix}.csv',float_precision='round_trip')
                assert f.actual_class.tolist()==list(CLASSES) and f.columns[1:].tolist()==list(CLASSES)
                assert np.allclose(f.iloc[:,1:],actual,atol=1e-12,rtol=0)
            f=pd.read_csv(root/f'results/tables/neural_reference_v1/{name}_precision_recall.csv',float_precision='round_trip')
            assert np.allclose(f,pr,atol=1e-12,rtol=0,equal_nan=True)
            curve=losses[losses.feature_set.eq(candidate)&losses.seed.eq(seed)]
            assert curve.epoch.tolist()==list(range(1,31)) and np.array_equal(curve.loss,m.loss_curve_)
            checks.append(dict(feature_set=candidate,seed=seed,model_reloaded=True,predictions_equal=True,
                class_and_column_order=True,rows_equal=True,metrics_and_matrices_equal=True,epochs=30))
    s['metrics']=metrics;s['per_class']=per
    display(review_table(s,pd.DataFrame(checks),'model_reload_checks'))
    display(metrics[['feature_set','seed','macro_f1','accuracy','weighted_f1','worms_correct','worms_support']])


def verify_controls(s):
    root=s['root'];y=labels(s['links']['model_selection']);record=pd.read_csv(root/'results/tables/neural_reference_v1/07_verified_lightgbm_controls.csv',float_precision='round_trip')
    original=pd.read_csv(root/'results/tables/revision_v2/06_model_selection_metrics.csv',float_precision='round_trip');checks=[]
    for candidate in SETS:
        mapping=read_json(root/f'models/preprocessing/05/05_{candidate}_tree_columns.json')
        assert mapping['retained_original_names']==s['maps'][candidate]['retained_original_names']
        x=canonical_csr(sparse.load_npz(root/f'data/processed/05/05_{candidate}_tree_model_selection.npz'))
        parent=canonical_csr(sparse.load_npz(root/'data/processed/04/04_tree_augmented_model_selection.npz'))
        assert x.shape==(len(y),len(mapping['encoded_names'])) and (x!=parent[:,mapping['indices_in_04']]).nnz==0
        m=joblib.load(root/f'models/revision_v2/06_lightgbm_{candidate}.joblib')
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',message='X does not have valid feature names');pred,prob=predict(m,x)
        saved=pd.read_csv(root/f'data/processed/revision_v2/06_lightgbm_{candidate}_predictions.csv',float_precision='round_trip')
        for col in s['links']['model_selection']:assert np.array_equal(saved[col],s['links']['model_selection'][col]),col
        assert np.array_equal(prob,saved[['p_'+c for c in CLASSES]]) and np.array_equal(np.array(CLASSES)[pred],saved.predicted_attack_cat)
        rec,pc,cm,ncm,*_=evaluate(y,pred,prob)
        row=record[record.feature_set.eq(candidate)].iloc[0]
        old=original[original.model.eq('lightgbm')&original.feature_set.eq(candidate)].iloc[0]
        for key,val in rec.items():assert np.isclose(val,row[key],atol=1e-12,rtol=0) and np.isclose(val,old[key],atol=1e-12,rtol=0),key
        for suffix,actual in [('raw',cm),('normalized',ncm)]:
            f=pd.read_csv(root/f'results/tables/revision_v2/06_lightgbm_{candidate}_confusion_{suffix}.csv',float_precision='round_trip')
            assert np.allclose(f.iloc[:,1:],actual,atol=1e-12,rtol=0)
        checks.append(dict(feature_set=candidate,saved_06_model_verified=True,probabilities_equal=True,rows_and_classes_equal=True,macro_f1=rec['macro_f1']))
    s['baseline']=record;display(review_table(s,pd.DataFrame(checks),'lightgbm_checks'))


def show_results(s):
    root=s['root'];tables=root/'results/tables/neural_reference_v1'
    for name in ('seed_summary','paired_feature_differences'):
        display(pd.read_csv(tables/f'07_{name}.csv'))
    display(s['per_class'][['feature_set','seed','attack_cat','precision','recall','f1','correct','support']])
    combined=pd.read_csv(tables/'07_all_class_results.csv')
    display(combined[combined.model.eq('LightGBM')][['feature_set','attack_cat','precision','recall','f1','correct','support']])
    display(pd.read_csv(tables/'07_important_confusions.csv'))
    binary=['model','feature_set','seed','attack_recall','attack_precision','false_positive_rate','false_positive','false_negative','attack_average_precision']
    display(pd.concat([s['metrics'][binary],s['baseline'][binary]],ignore_index=True))
    costs=['model','feature_set','seed','fit_seconds','inference_seconds','inference_ms_per_row']
    display(pd.concat([s['metrics'][costs],s['baseline'][costs]],ignore_index=True))
    display(s['metrics'][['feature_set','seed','parameters','model_bytes']])
    for filename in ('07_matched_reference_scores.png','07_loss_and_class_recall.png','07_reference_confusion_matrices.png'):
        display(Image(filename=str(root/'results/figures/neural_reference_v1'/filename)))


def finish(s):
    verify_hashes(s['root'],s['protected'])
    runtime=read_json(s['root']/f'docs/{REVIEW}/07_runtime_check.json')
    summary=s['metrics'].groupby('feature_set').macro_f1.mean()
    assert np.isclose(summary.original42,.51975285584554953) and np.isclose(summary.total_and_byte44,.52414922903845396)
    assert s['metrics'].worms_correct.eq(0).all() and s['metrics'].worms_support.eq(13).all()
    text='''# Notebook 07: the result in simple words

| Inputs | MLP mean macro F1, three seeds | Saved LightGBM macro F1 |
|---|---:|---:|
| original42 | 0.5198 | 0.6307 |
| total_and_byte44 | 0.5241 | 0.6320 |

The MLP is a useful **bounded neural reference**, not our best-performing model. All six MLP runs correctly classify **0 of 13 Worms** rows. The matching LightGBM models find 8/13 and 9/13.

Adding total bytes and byte balance raises mean MLP macro F1 by 0.44 percentage points. Paired seed changes are +1.73, -1.20 and +0.78 points: some runs improve and another worsens. This is not a stable improvement across seeds or proof of performance on new traffic.

For 44 features and fixed seed 42, the MLP finds **271/1,226 DoS** rows, versus **86/1,226** for LightGBM. But it finds fewer Analysis (21 vs 30), Backdoor (15 vs 26), Exploits (2,942 vs 3,192), Fuzzers (1,304 vs 1,391) and Worms (0 vs 9). The full notebook keeps all per-class counts, false alarms, costs and seed variation visible.

We tried just **two configurations for 30 epochs**, using three grouped folds inside fitting. Falling training loss means fitting is progressing; it proves neither generalization nor convergence. No settings, epochs, seeds or weighting were changed in this review. Stronger MLP optimization would need a separately versioned experiment with its search and stopping rules declared first and selected only in fitting-only grouped folds.

The existing rule keeps total_and_byte44 as the provisional MLP input and seed 42 as the reference checkpoint. Keep both input sets and the exact LightGBM controls for Notebook 08. Model-selection has prior exposure; these remain optimistic development results. Calibration labels and official-test records are reserved. This is flow intrusion classification, with no DNS/temporal/botnet-specific or final-test claim.
'''
    for name in (f'docs/{REVIEW}/07_findings.md','docs/neural_reference_v1/07_results_in_simple_words.md','docs/neural_reference_v1/07_findings.md'):
        p=s['root']/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text,encoding='utf-8');s['outputs'].append(p)
    display(Markdown(text))
    print(f'Current runtime check: {runtime["status"]}. This is separate from the scikit-learn MLP results.')
    path=s['root']/f'data/processed/{REVIEW}/07_artifact_manifest.json'
    write_json(path,dict(version=REVIEW,mode='Frozen-model replay; no predictive-model fitting or retuning',
        dependencies_sha256=s['protected'],artifact_sha256={p.relative_to(s['root']).as_posix():sha256(p) for p in s['outputs']},
        calibration_targets_loaded=False,official_test_records_loaded=False,protected_unchanged=True))
    print(f'{len(s["protected"])} protected files unchanged. Byte-level final seal is pending the independent verifier.')
