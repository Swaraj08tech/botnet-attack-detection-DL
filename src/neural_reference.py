"""Notebook 07: a bounded, group-safe, unweighted neural reference.

This module never opens official test records or calibration targets. Existing
preprocessors are cloned for the inner search; final inputs stay exactly as saved.
"""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.special import softmax
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits, threadpool_info
from IPython.display import display, Markdown

from src.baseline_models import CLASSES, NORMAL, evaluate
from src.data.load_data import sha256
from src.data.feature_engineering import canonical_csr, sparse_row_keys, overlap_counts
from src.research_revision.common import read_json, write_json, protected_check

VERSION = 'neural_reference_v1'
SETS = ('original42', 'total_and_byte44')
SEEDS = (42, 43, 44)
EPOCHS = 30
CONFIGS = ({'hidden_layer_sizes': (64, 32), 'alpha': .0001},
           {'hidden_layer_sizes': (128, 64), 'alpha': .01})
FORBIDDEN = {'id', 'label', 'attack_cat', 'input_row', 'row_position',
             'feature_group_id', 'role', 'source_id', 'csv_line', 'partition'}


def labels(link):
    y = pd.Categorical(link.attack_cat, categories=CLASSES).codes
    if (y < 0).any() or not np.array_equal(link.label, y != NORMAL):
        raise ValueError('Unknown categories or inconsistent binary labels')
    return y


def assert_groups(folds, links):
    for name in ('input_row', 'row_position', 'feature_group_id'):
        if not np.array_equal(folds[name], links[name]):
            raise ValueError(f'Inner fold row mismatch: {name}')
    if set(folds.internal_holdout_fold) != {0, 1, 2}:
        raise ValueError('Three frozen fitting folds required')
    if folds.groupby('feature_group_id').internal_holdout_fold.nunique().max() != 1:
        raise ValueError('Identical original patterns cross inner folds')


def make_model(config, seed):
    # n_iter_no_change > max_iter prevents training-loss-based early termination.
    return MLPClassifier(hidden_layer_sizes=tuple(config['hidden_layer_sizes']),
        activation='relu', solver='adam', alpha=config['alpha'], batch_size=512,
        learning_rate='constant', learning_rate_init=.001, max_iter=EPOCHS,
        shuffle=True, random_state=seed, tol=0., early_stopping=False,
        n_iter_no_change=EPOCHS + 1, beta_1=.9, beta_2=.999, epsilon=1e-8)


def fit_fixed(model, x, y):
    begin = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=2):
        warnings.simplefilter('always'); model.fit(x, y)
    seconds = time.perf_counter() - begin
    if model.n_iter_ != EPOCHS or len(model.loss_curve_) != EPOCHS:
        raise RuntimeError('The declared fixed-epoch budget was not completed')
    if not np.isfinite(model.loss_curve_).all() or not all(
        np.isfinite(a).all() for a in model.coefs_ + model.intercepts_):
        raise RuntimeError('Nonfinite loss or parameters')
    unexpected = [str(w.message) for w in caught if not (
        issubclass(w.category, ConvergenceWarning) and 'Maximum iterations' in str(w.message))]
    if unexpected:
        raise RuntimeError(f'Unexpected fitting warnings: {unexpected}')
    return seconds, [str(w.message) for w in caught]


def predict(model, x):
    with threadpool_limits(limits=2):
        probability = model.predict_proba(x)
    assert np.array_equal(model.classes_, np.arange(10))
    assert model.n_features_in_ == x.shape[1]
    return np.argmax(probability, axis=1), probability


def measured_inference(model, x):
    predict(model, x[:min(100, x.shape[0])])  # Warm up, without learning.
    times = []
    for _ in range(3):
        begin = time.perf_counter(); pred, prob = predict(model, x)
        times.append(time.perf_counter() - begin)
    return pred, prob, float(np.median(times))


def progress(s, message):
    text = time.strftime('%H:%M:%S') + ' ' + message
    print(text, flush=True)
    with (s['root']/f'.cache/{VERSION}/progress.log').open('a', encoding='utf-8') as f:
        f.write(text + '\n')


def save_table(s, frame, name):
    path = s['out']['tables']/f'07_{name}.csv'
    frame.to_csv(path, index=False, float_format='%.17g')
    s['outputs'].append(path); return frame


def save_figure(s, fig, name):
    fig.tight_layout()
    path = s['out']['figures']/f'07_{name}.png'
    fig.savefig(path, dpi=160, bbox_inches='tight'); s['outputs'].append(path)
    plt.show(); plt.close(fig)


def note(s, result, decision, uncertainty):
    text = f'**Result:** {result}\n\n**Decision:** {decision}\n\n**Still uncertain:** {uncertainty}'
    s['notes'].append(text); display(Markdown(text))


def current_protected(root):
    hashes = protected_check(root)
    prior = read_json(root/'docs/revision_v2/deliverable_manifest.json')
    notebook_changes = {}
    for name, expected in prior['file_sha256'].items():
        actual = sha256(root/name)
        if actual != expected:
            # Existing notebook saves are recorded, never repaired or rewritten here.
            if not name.startswith('notebooks/0'):
                raise RuntimeError(f'Earlier sealed artifact changed: {name}')
            notebook_changes[name] = dict(sealed_sha256=expected,starting_sha256=actual)
        hashes[name] = actual
    for i in ('02', '03', '04', '05', '06'):
        path = root/f'data/processed/revision_v2/{i}_artifact_manifest.json'
        manifest = read_json(path)
        for section in ('dependencies_sha256', 'artifact_sha256'):
            for name, expected in manifest[section].items():
                assert sha256(root/name) == expected, name
                hashes[name] = expected
        hashes[path.relative_to(root).as_posix()] = sha256(path)
    for name in ('docs/revision_v2/deliverable_manifest.json', 'README.md', 'docs/research_notes.md'):
        hashes[name] = sha256(root/name)
    return hashes, notebook_changes


def start(root):
    root = Path(root); checked, old_notebook_saves = current_protected(root)
    out = {k: root/f'{v}/{VERSION}' for k,v in {
        'tables':'results/tables', 'figures':'results/figures', 'data':'data/processed',
        'models':'models', 'docs':'docs'}.items()}
    for p in out.values(): p.mkdir(parents=True, exist_ok=True)
    cache = root/f'.cache/{VERSION}'; cache.mkdir(parents=True, exist_ok=True)
    assignment = pd.read_csv(root/'results/splits/02_train_validation_assignments.csv',
        usecols=['row_position','source_id','feature_group_id','role'])
    split = read_json(root/'results/splits/02_split_manifest.json')
    assert sha256(root/'results/splits/02_train_validation_assignments.csv') == split['assignment_sha256']
    assert assignment.groupby('feature_group_id').role.nunique().max() == 1
    assert np.array_equal(assignment.row_position, np.arange(len(assignment)))
    assert set(assignment.role) == {'train','model_selection','calibration'}
    target_map = read_json(root/'models/preprocessing/03/03_target_mapping.json')
    assert target_map == {c:i for i,c in enumerate(CLASSES)}
    links = {}; matrices = {}; maps = {}; support = []; audits = []; overlaps = []
    definitions = read_json(root/'models/preprocessing/05/05_comparison_sets.json')['sets']
    for role in ('train', 'model_selection', 'calibration'):
        cols = None if role != 'calibration' else ['input_row','row_position','source_id','feature_group_id','role']
        link = pd.read_csv(root/f'data/processed/05/05_row_links_{role}.csv', usecols=cols)
        expected = assignment[assignment.role.eq(role)].reset_index(drop=True)
        assert np.array_equal(link.input_row, np.arange(len(link)))
        for col in expected:
            assert np.array_equal(link[col], expected[col]), (role,col)
        links[role] = link
        if role != 'calibration':
            y = labels(link)
            for i,c in enumerate(CLASSES):
                mask = y == i
                support.append(dict(role=role, attack_cat=c, rows=int(mask.sum()),
                    original_patterns=link.loc[mask,'feature_group_id'].nunique()))
    for candidate in SETS:
        mapping = read_json(root/f'models/preprocessing/05/05_{candidate}_scaled_columns.json')
        names = mapping['encoded_names']; sources = mapping['original_source_for_each_encoded_column']
        assert mapping['retained_original_names'] == definitions[candidate]['retained']
        assert len(names) == len(set(names)) == len(sources)
        assert not (set(sources) | set(names)) & FORBIDDEN
        assert set(sources) == set(mapping['retained_original_names'])
        full_names = read_json(root/'models/preprocessing/04/04_scaled_augmented_feature_names.json')
        assert names == [full_names[i] for i in mapping['indices_in_04']]
        maps[candidate] = mapping; matrices[candidate] = {}; keys = {}
        for role in links:
            x = canonical_csr(sparse.load_npz(root/f'data/processed/05/05_{candidate}_scaled_{role}.npz'))
            parent = canonical_csr(sparse.load_npz(root/f'data/processed/04/04_scaled_augmented_{role}.npz'))
            assert x.shape == (len(links[role]), len(names))
            assert (x != parent[:,mapping['indices_in_04']]).nnz == 0
            keys[role] = sparse_row_keys(x)
            if role != 'calibration': matrices[candidate][role] = x
            audits.append(dict(feature_set=candidate,role=role,rows=x.shape[0],
                encoded_columns=x.shape[1],dtype=str(x.dtype),finite=True,verified_parent=True))
        for a,b in (('train','model_selection'),('train','calibration'),('model_selection','calibration')):
            overlaps.append(dict(feature_set=candidate,role_a=a,role_b=b,**overlap_counts(keys[a],keys[b])))
    folds = pd.read_csv(root/'data/processed/05/05_internal_folds.csv')
    assert_groups(folds, links['train'])
    raw = pd.read_csv(root/'data/processed/05/05_original42_raw_train.csv',float_precision='round_trip')
    assert list(raw) == split['predictors'] and len(raw) == len(links['train'])
    # Reproduce groups from predictors alone; targets never enter the group key.
    actual_group = raw.groupby(list(raw),sort=False,dropna=False,observed=True).ngroup()
    group_check = pd.DataFrame({'actual':actual_group,'frozen':links['train'].feature_group_id})
    assert group_check.groupby('actual').frozen.nunique().max() == 1
    assert group_check.groupby('frozen').actual.nunique().max() == 1
    saved_y = pd.read_csv(root/'data/processed/05/05_row_links_train.csv')
    assert np.array_equal(folds.attack_cat, saved_y.attack_cat)
    s = dict(root=root,out=out,checked=checked,notes=[],outputs=[],links=links,
        matrices=matrices,maps=maps,folds=folds,raw=raw,prior_notebook_saves=old_notebook_saves)
    save_table(s,pd.DataFrame(audits),'input_checks')
    overlap = save_table(s,pd.DataFrame(overlaps),'exact_role_overlap')
    assert overlap.shared_vectors.eq(0).all()
    display(save_table(s,pd.DataFrame(support),'class_support').query('role == "train"'))
    conflicts = links['train'].groupby('feature_group_id').attack_cat.nunique().gt(1)
    print(f'Fitting has {len(conflicts):,} original patterns; {int(conflicts.sum()):,} have conflicting categories. All versions stay together.')
    path = out['tables']/'07_starting_integrity.json'
    write_json(path,dict(protected_sha256=checked,earlier_notebook_save_differences=old_notebook_saves,
        calibration_labels_loaded=False,calibration_inputs='Shape, parent equality and overlap audit only',
        official_test='Bytes hashed only; no record access'))
    s['outputs'].append(path)
    pd.set_option('display.max_columns',14); plt.rcParams.update({'font.size':10,'figure.dpi':110})
    print(f'Checked {len(checked)} earlier files. Fitting: {len(links["train"]):,}; model-selection: {len(links["model_selection"]):,}.')
    print('193 / 195 encoded columns. Calibration targets and official-test records were not loaded.')
    if old_notebook_saves:
        print('Earlier notebook saves differ from the old seal; their current bytes are protected for this run. All earlier model/data/table manifests match.')
    return s


def worked_example(s):
    # Deliberately invented two-input weights; this is an explanation, not an experiment.
    x = np.array([.5, -.25]); w = np.array([[1., -.5],[.5, 1.]])
    hidden = np.maximum(0., x @ w + np.array([.1,.2]))
    scores = hidden @ np.array([[.1]*6+[1.,.1,.1,.1], [.1]*10])
    probability = softmax(scores)
    display(pd.DataFrame({'step':['Two toy inputs','Weighted sum + ReLU','Ten scores -> softmax','Prediction','Loss if actual class is Normal'],
        'value':[str(x),str(hidden),str(np.round(probability,3)),CLASSES[int(probability.argmax())],f'{-np.log(probability[NORMAL]):.4f}']}))
    print('Invented example only. Real models use all 193 or 195 inputs and two hidden layers.')


def declare(s):
    environment = dict(python=platform.python_version(),executable=__import__('sys').executable,
        platform=platform.platform(),threads=2,precision='float64; saved CSR inputs retained',
        versions={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','joblib','lightgbm','nbclient','nbformat']},
        threadpools=threadpool_info(),backend='scikit-learn MLPClassifier, CPU',
        torch_limitation='Installed PyTorch import failed: WinError 4551, Application Control blocked torch.dll or a dependency. No security-policy change attempted.')
    path = s['out']['docs']/'07_environment.json'; write_json(path,environment); s['outputs'].append(path)
    plan = dict(version=VERSION,sets=list(SETS),classes=list(CLASSES),
        configurations=[dict(hidden_layer_sizes=list(c['hidden_layer_sizes']),alpha=c['alpha']) for c in CONFIGS],
        settings=make_model(CONFIGS[0],42).get_params(),epochs=EPOCHS,
        search='Two jointly declared capacity/L2 settings x three existing grouped fitting folds, original42 only; clone/refit inner preprocessing.',
        search_fits=6,search_seed=42,final_seeds=list(SEEDS),final_fits=6,
        rule='Highest mean fixed-ten-class macro F1; within .001 prefer first/smaller configuration. Same chosen settings for both feature sets.',
        feature_choice='Mean of three final MLP seeds; within .001 prefer original42. Do not choose a best seed.',
        reference_seed=42,stopping='Exactly 30 full epochs; no early stopping, convergence-based stop or later epoch change.',
        weights='Unweighted; no sample weights/resampling; weighting deferred',
        optimizer='Adam; learning rate .001, batch 512, beta1 .9, beta2 .999, epsilon 1e-8',
        regularization='L2 alpha is sklearn penalty coefficient divided by batch sample count; no dropout',
        fairness='Same fitting rows/classes/scaled inputs for MLP sets; existing tree representation for exact LightGBM controls. MLP has six inner fits, matching Notebook06 LightGBM search count; families have different fixed training algorithms/budgets.',
        reserved='Calibration targets never loaded. Official-test records never opened.',
        exposure='Model-selection already used in Notebook06; comparisons are optimistic development evidence.',
        deviations=['Use installed scikit-learn MLP instead of blocked PyTorch; the requested MLP experiment is fully implemented.'],
        assignment_sha256=sha256(s['root']/'results/splits/02_train_validation_assignments.csv'),
        inner_folds_sha256=sha256(s['root']/'data/processed/05/05_internal_folds.csv'),
        implementation_sha256=sha256(Path(__file__)))
    path = s['root']/f'configs/07_{VERSION}_plan.json'
    if path.exists():
        assert read_json(path) == json.loads(json.dumps(plan)), 'Version the plan before changing an existing search'
    else: write_json(path,plan)
    s['outputs'].append(path); s['plan']=plan; s['plan_hash']=sha256(path)
    display(pd.DataFrame([dict(setting=i,hidden_layers=str(c['hidden_layer_sizes']),L2_alpha=c['alpha'],epochs=EPOCHS,folds=3) for i,c in enumerate(CONFIGS)]))
    print('Plan saved before fitting. Six search fits + six final fits. No extra search after model-selection results.')


def tune(s):
    y = labels(s['links']['train']); scores=[]; losses=[]; support=[]
    for fold in range(3):
        hold = s['folds'].internal_holdout_fold.eq(fold).to_numpy(); fit=~hold
        assert not set(s['folds'].loc[fit,'feature_group_id']) & set(s['folds'].loc[hold,'feature_group_id'])
        pre = clone(joblib.load(s['root']/'models/preprocessing/03/03_scaled_pipeline.joblib'))
        pre.fit(s['raw'].loc[fit])
        a=canonical_csr(pre.transform(s['raw'].loc[fit])); b=canonical_csr(pre.transform(s['raw'].loc[hold]))
        path=s['out']['models']/f'07_inner_fold{fold}_preprocessor.joblib'
        joblib.dump(pre,path); s['outputs'].append(path)
        provenance=dict(fold=fold,fit_rows=int(fit.sum()),holdout_rows=int(hold.sum()),
            fit_positions_sha256=hashlib.sha256(s['folds'].loc[fit,'row_position'].to_numpy(dtype=np.int64).tobytes()).hexdigest(),
            encoded_names=pre.get_feature_names_out().tolist(),original_pattern_overlap=0,plan_sha256=s['plan_hash'])
        path=s['out']['models']/f'07_inner_fold{fold}_provenance.json'; write_json(path,provenance); s['outputs'].append(path)
        for i,c in enumerate(CLASSES):
            mask=hold&(y==i); support.append(dict(fold=fold,attack_cat=c,rows=int(mask.sum()),patterns=s['folds'].loc[mask,'feature_group_id'].nunique()))
        for index,config in enumerate(CONFIGS):
            checkpoint=s['root']/f'.cache/{VERSION}/inner_{index}_{fold}.json'
            if checkpoint.exists():
                r=read_json(checkpoint); assert r['plan_sha256']==s['plan_hash']
                progress(s,f'Verified cached fitting-only result: setting {index}, fold {fold}')
            else:
                progress(s,f'Fitting setting {index}, inner fold {fold}: {int(fit.sum()):,} rows, {EPOCHS} epochs')
                m=make_model(config,42); seconds,caught=fit_fixed(m,a,y[fit]); pred,_=predict(m,b)
                r=dict(setting=index,fold=fold,macro_f1=float(f1_score(y[hold],pred,labels=np.arange(10),average='macro',zero_division=0)),
                    fit_seconds=seconds,epochs=m.n_iter_,loss_curve=m.loss_curve_,warnings=caught,plan_sha256=s['plan_hash'])
                write_json(checkpoint,r)
            scores.append({k:v for k,v in r.items() if k not in ('loss_curve','warnings')})
            losses.extend(dict(stage='inner',setting=index,fold=fold,epoch=e+1,loss=v) for e,v in enumerate(r['loss_curve']))
            progress(s,f'Setting {index}, fold {fold}: macro F1 {r["macro_f1"]:.4f}; {r["fit_seconds"]:.1f} seconds')
    scores=save_table(s,pd.DataFrame(scores),'inner_fold_scores')
    summary=scores.groupby('setting').agg(mean_macro_f1=('macro_f1','mean'),min_macro_f1=('macro_f1','min'),max_macro_f1=('macro_f1','max'),fit_seconds=('fit_seconds','sum')).reset_index()
    index=int(summary[summary.mean_macro_f1.ge(summary.mean_macro_f1.max()-.001)].setting.min())
    s['config']=CONFIGS[index]; s['setting']=index
    display(save_table(s,summary,'search_summary'))
    save_table(s,pd.DataFrame(losses),'inner_losses'); save_table(s,pd.DataFrame(support),'inner_class_support')
    path=s['out']['models']/'07_frozen_settings.json'
    write_json(path,dict(plan_sha256=s['plan_hash'],chosen_setting=index,configuration=s['config'],epochs=EPOCHS,
        final_seeds=SEEDS,full_fitting_refit=True,chosen_without_model_selection=True))
    s['outputs'].append(path)
    note(s,f'Setting {index} was selected using only the three grouped folds inside fitting.',
        'Freeze these settings and train for 30 epochs for both feature sets and all three seeds.',
        'This small search does not establish the best possible neural network. Folds hold recorded patterns together, not verified hosts or capture periods.')


def final_fits(s):
    y=labels(s['links']['train']); v_y=labels(s['links']['model_selection']); rows=[]; per=[]; losses=[]
    for candidate in SETS:
        x=s['matrices'][candidate]['train']; v=s['matrices'][candidate]['model_selection']
        for seed in SEEDS:
            name=f'mlp_{candidate}_seed{seed}'; path=s['out']['models']/f'07_{name}.joblib'
            settings_path=s['out']['models']/f'07_{name}_settings.json'
            m=make_model(s['config'],seed)
            if path.exists():
                settings=read_json(settings_path)
                assert settings['plan_sha256']==s['plan_hash'] and settings['setting']==s['setting']
                assert settings['input_sha256']==sha256(s['root']/f'data/processed/05/05_{candidate}_scaled_train.npz')
                m=joblib.load(path); seconds=settings['fit_seconds']
                assert m.get_params()==make_model(s['config'],seed).get_params()
                progress(s,f'Reloaded identical saved final fit: {name}')
            else:
                progress(s,f'Final fitting-only training: {name}, {EPOCHS} epochs')
                seconds,caught=fit_fixed(m,x,y); joblib.dump(m,path)
                settings=dict(plan_sha256=s['plan_hash'],setting=s['setting'],settings=m.get_params(),fit_seconds=seconds,
                    seed=seed,epochs=EPOCHS,classes=list(CLASSES),encoded_names=s['maps'][candidate]['encoded_names'],
                    source_names=s['maps'][candidate]['retained_original_names'],
                    input_sha256=sha256(s['root']/f'data/processed/05/05_{candidate}_scaled_train.npz'),warnings=caught)
                write_json(settings_path,settings)
            assert m.n_iter_==EPOCHS
            pred,prob,infer=measured_inference(m,v)
            reload_pred,reload_prob=predict(joblib.load(path),v)
            assert np.array_equal(pred,reload_pred) and np.array_equal(prob,reload_prob)
            metrics,pc,cm,ncm,pr,attack=evaluate(v_y,pred,prob)
            rows.append(dict(model='MLP',feature_set=candidate,seed=seed,fit_seconds=seconds,inference_seconds=infer,
                inference_ms_per_row=1000*infer/len(v_y),model_bytes=path.stat().st_size,
                parameters=sum(a.size for a in m.coefs_+m.intercepts_),**metrics))
            pc.insert(0,'seed',seed); pc.insert(0,'feature_set',candidate); pc.insert(0,'model','MLP'); per.append(pc)
            link=s['links']['model_selection'].copy(); link['predicted_attack_cat']=np.array(CLASSES)[pred]
            link['predicted_binary_attack']=pred!=NORMAL; link['attack_score']=attack
            for i,c in enumerate(CLASSES): link['p_'+c]=prob[:,i]
            save_path=s['out']['data']/f'07_{name}_predictions.csv'; link.to_csv(save_path,index=False,float_format='%.17g'); s['outputs'].append(save_path)
            for suffix,value in [('confusion_raw',cm),('confusion_normalized',ncm)]:
                frame=pd.DataFrame(value,columns=CLASSES); frame.insert(0,'actual_class',CLASSES)
                save_table(s,frame,name+'_'+suffix)
            save_table(s,pr,name+'_precision_recall')
            losses.extend(dict(feature_set=candidate,seed=seed,epoch=e+1,loss=loss) for e,loss in enumerate(m.loss_curve_))
            s['outputs'].extend([path,settings_path])
            progress(s,f'{name}: macro F1 {metrics["macro_f1"]:.4f}; Worms {metrics["worms_correct"]}/13; fit {seconds:.1f}s')
    s['metrics']=save_table(s,pd.DataFrame(rows),'model_selection_metrics')
    s['per_class']=save_table(s,pd.concat(per,ignore_index=True),'per_class_metrics')
    s['losses']=save_table(s,pd.DataFrame(losses),'final_training_losses')
    display(s['metrics'][['feature_set','seed','macro_f1','accuracy','weighted_f1','worms_correct','worms_support','fit_seconds']])
    note(s,'Every final run used the full frozen fitting role and its fixed 30-epoch budget.',
        'Report all six runs. Seed 42 is the predeclared reference checkpoint, even if another seed scores higher.',
        'Training loss shows optimization progress, not generalization. Completing 30 epochs does not mean optimization converged.')


def compare(s):
    baseline=pd.read_csv(s['root']/'results/tables/revision_v2/06_model_selection_metrics.csv')
    baseline=baseline[baseline.model.eq('lightgbm') & baseline.feature_set.isin(SETS)].copy()
    baseline_per=pd.read_csv(s['root']/'results/tables/revision_v2/06_per_class_metrics.csv')
    baseline_per=baseline_per[baseline_per.model.eq('lightgbm') & baseline_per.feature_set.isin(SETS)].copy()
    assert len(baseline)==2
    base=[]; all_per=[s['per_class']]; y=labels(s['links']['model_selection'])
    for candidate in SETS:
        path=s['root']/f'data/processed/revision_v2/06_lightgbm_{candidate}_predictions.csv'
        saved=pd.read_csv(path,float_precision='round_trip')
        for col in s['links']['model_selection']:
            assert np.array_equal(saved[col],s['links']['model_selection'][col]), col
        pred=pd.Categorical(saved.predicted_attack_cat,categories=CLASSES).codes
        probabilities=saved[['p_'+c for c in CLASSES]].to_numpy()
        metrics,pc,*_=evaluate(y,pred,probabilities)
        original=baseline[baseline.feature_set.eq(candidate)].iloc[0]
        for key,val in metrics.items(): assert np.isclose(val,original[key],rtol=0,atol=1e-12),key
        tree=canonical_csr(sparse.load_npz(s['root']/f'data/processed/05/05_{candidate}_tree_model_selection.npz'))
        m=joblib.load(s['root']/f'models/revision_v2/06_lightgbm_{candidate}.joblib')
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',message='X does not have valid feature names')
            bpred,bprob,infer=measured_inference(m,tree)
        assert np.array_equal(pred,bpred) and np.array_equal(probabilities,bprob)
        base.append(dict(model='LightGBM (saved 06)',feature_set=candidate,seed=42,
            fit_seconds=float(original.fit_seconds),inference_seconds=infer,inference_ms_per_row=1000*infer/len(y),**metrics))
        pc.insert(0,'seed',42); pc.insert(0,'feature_set',candidate); pc.insert(0,'model','LightGBM'); all_per.append(pc)
    s['baseline']=save_table(s,pd.DataFrame(base),'verified_lightgbm_controls')
    combined=save_table(s,pd.concat(all_per,ignore_index=True),'all_class_results')
    summary=s['metrics'].groupby('feature_set').agg(mean_macro_f1=('macro_f1','mean'),sd_macro_f1=('macro_f1','std'),min_macro_f1=('macro_f1','min'),max_macro_f1=('macro_f1','max'),mean_accuracy=('accuracy','mean'),mean_fit_seconds=('fit_seconds','mean')).reset_index()
    summary=summary.merge(s['baseline'][['feature_set','macro_f1']],on='feature_set').rename(columns={'macro_f1':'saved_lightgbm_macro_f1'})
    summary['mean_MLP_minus_LightGBM_pp']=100*(summary.mean_macro_f1-summary.saved_lightgbm_macro_f1)
    display(save_table(s,summary,'seed_summary')); s['summary']=summary
    delta=s['metrics'].pivot(index='seed',columns='feature_set',values='macro_f1')
    delta['engineered_minus_original_pp']=100*(delta.total_and_byte44-delta.original42)
    display(save_table(s,delta.reset_index(),'paired_feature_differences'))
    display(save_table(s,combined[combined.attack_cat.isin(['Analysis','Backdoor','DoS','Worms'])],'weak_class_results')[['model','feature_set','seed','attack_cat','recall','f1','correct','support']])
    # Absolute confusions for every seed and both fixed tree controls.
    confusions=[]
    for candidate in SETS:
        for family,seeds in [('mlp',SEEDS),('lightgbm',(42,))]:
            for seed in seeds:
                path=(s['out']['data']/f'07_mlp_{candidate}_seed{seed}_predictions.csv' if family=='mlp' else s['root']/f'data/processed/revision_v2/06_lightgbm_{candidate}_predictions.csv')
                f=pd.read_csv(path,usecols=['attack_cat','predicted_attack_cat'])
                for c in ('Analysis','Backdoor','DoS','Worms'):
                    actual=f.attack_cat.eq(c); count=int((actual & f.predicted_attack_cat.eq('Exploits')).sum())
                    confusions.append(dict(model=family,feature_set=candidate,seed=seed,actual_class=c,actual_rows=int(actual.sum()),
                        predicted_as_Exploits=count,percent=100*count/int(actual.sum())))
    display(save_table(s,pd.DataFrame(confusions),'important_confusions'))
    # Seed 42 figures are predeclared examples; all seeds' full matrices are saved.
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for i,candidate in enumerate(SETS):
        r=s['metrics'][s['metrics'].feature_set.eq(candidate)]
        axes[i].scatter(r.seed,r.macro_f1,label='MLP seeds',color='#3b73b9',s=55)
        axes[i].axhline(s['baseline'].set_index('feature_set').loc[candidate,'macro_f1'],color='#cf683c',label='Saved LightGBM')
        axes[i].set(title=candidate,xlabel='Training seed',ylabel='Ten-class macro F1',xticks=SEEDS)
        axes[i].set_ylim(0,max(.75,float(s['metrics'].macro_f1.max())+.05)); axes[i].legend()
    save_figure(s,fig,'matched_reference_scores')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for candidate in SETS:
        f=s['losses'][(s['losses'].feature_set.eq(candidate)) & s['losses'].seed.eq(42)]
        axes[0].plot(f.epoch,f.loss,label=candidate)
    axes[0].set(xlabel='Epoch',ylabel='Unweighted training loss',title='Training progress: fixed seed 42'); axes[0].legend()
    pcs=combined[combined.seed.eq(42)&combined.feature_set.eq('total_and_byte44')]
    for family,color in [('MLP','#3b73b9'),('LightGBM','#cf683c')]:
        f=pcs[pcs.model.eq(family)].set_index('attack_cat').loc[list(CLASSES)]
        offset=-.18 if family=='MLP' else .18
        axes[1].bar(np.arange(10)+offset,f.recall,width=.36,label=family,color=color)
    axes[1].set(xticks=np.arange(10),xticklabels=CLASSES,ylabel='Recall',title='Each class: 44 features, seed 42'); axes[1].tick_params(axis='x',rotation=70); axes[1].legend()
    save_figure(s,fig,'loss_and_class_recall')
    fig,axes=plt.subplots(1,2,figsize=(15,6))
    for ax,candidate in zip(axes,SETS):
        cm=pd.read_csv(s['out']['tables']/f'07_mlp_{candidate}_seed42_confusion_raw.csv').iloc[:,1:].to_numpy()
        norm=cm/cm.sum(axis=1,keepdims=True)
        ax.imshow(norm,vmin=0,vmax=1,cmap='Blues')
        for i in range(10):
            for j in range(10):
                ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=8,color='white' if norm[i,j]>.55 else 'black')
        ax.set(xticks=np.arange(10),yticks=np.arange(10),xticklabels=CLASSES,yticklabels=CLASSES,
               xlabel='Predicted class',ylabel='Actual class',title=f'{candidate}: counts; colour = row recall')
        ax.tick_params(axis='x',rotation=75)
    save_figure(s,fig,'reference_confusion_matrices')
    mean=summary.set_index('feature_set').mean_macro_f1
    s['chosen_set']='original42' if mean.original42>=mean.max()-.001 else 'total_and_byte44'
    change=float(mean.total_and_byte44-mean.original42)
    note(s,f'Mean MLP macro F1 changes by {100*change:+.3f} percentage points with total bytes and byte balance. The table gives each seed and the matched LightGBM differences.',
        f'Carry {s["chosen_set"]} as the provisional MLP feature choice under the declared .001 tie rule; retain both sets for Notebook 08.',
        'Three training seeds vary initialization/order only. Their range is not a confidence interval. The controls have one saved seed; this is not a statistical superiority test. Worms has only 13 model-selection rows.')


def real_example(s):
    candidate=s['chosen_set']; m=joblib.load(s['out']['models']/f'07_mlp_{candidate}_seed42.joblib')
    x=s['matrices'][candidate]['model_selection'][:1]; pred,prob=predict(m,x)
    link=s['links']['model_selection'].iloc[0]
    display(pd.DataFrame({'class':CLASSES,'model_probability':prob[0]}).sort_values('model_probability',ascending=False))
    print(f'Example row position {int(link.row_position)}: actual {link.attack_cat}; predicted {CLASSES[pred[0]]}.')
    print(f'Input shape: 1 x {x.shape[1]}; hidden layers {m.hidden_layer_sizes}; output: ten probabilities.')
    print('The largest score gives the class. These scores have not been confidence-calibrated.')


def verify_saved(s):
    checks=[]; y=labels(s['links']['model_selection'])
    for candidate in SETS:
        for seed in SEEDS:
            name=f'mlp_{candidate}_seed{seed}'
            m=joblib.load(s['out']['models']/f'07_{name}.joblib')
            settings=read_json(s['out']['models']/f'07_{name}_settings.json')
            assert settings['classes']==list(CLASSES)
            assert settings['encoded_names']==s['maps'][candidate]['encoded_names']
            assert settings['source_names']==s['maps'][candidate]['retained_original_names']
            assert set(settings['source_names']).isdisjoint(FORBIDDEN)
            pred,prob=predict(m,s['matrices'][candidate]['model_selection'])
            saved=pd.read_csv(s['out']['data']/f'07_{name}_predictions.csv',float_precision='round_trip')
            for col in s['links']['model_selection']:
                assert np.array_equal(saved[col],s['links']['model_selection'][col]),col
            assert np.array_equal(saved[['p_'+c for c in CLASSES]].to_numpy(),prob)
            assert np.array_equal(saved.predicted_attack_cat,np.array(CLASSES)[pred])
            assert np.array_equal(saved.predicted_binary_attack,pred!=NORMAL)
            assert np.array_equal(saved.attack_score,1-prob[:,NORMAL])
            recomputed,pc,cm,ncm,*_=evaluate(y,pred,prob)
            row=s['metrics'][s['metrics'].feature_set.eq(candidate)&s['metrics'].seed.eq(seed)].iloc[0]
            for key,val in recomputed.items(): assert np.isclose(row[key],val,atol=1e-12,rtol=0),key
            assert np.array_equal(pd.read_csv(s['out']['tables']/f'07_{name}_confusion_raw.csv').iloc[:,1:],cm)
            saved_pc=s['per_class'][s['per_class'].feature_set.eq(candidate)&s['per_class'].seed.eq(seed)]
            for key in ('precision','recall','f1','support','correct'):
                assert np.allclose(saved_pc[key],pc[key],atol=1e-12,rtol=0),key
            checks.append(dict(feature_set=candidate,seed=seed,dimensions=True,column_order=True,class_order=True,
                row_links=True,reloaded_predictions_equal=True,recomputed_metrics_equal=True))
    for fold in range(3):
        fit=~s['folds'].internal_holdout_fold.eq(fold).to_numpy()
        pre=joblib.load(s['out']['models']/f'07_inner_fold{fold}_preprocessor.joblib')
        provenance=read_json(s['out']['models']/f'07_inner_fold{fold}_provenance.json')
        assert provenance['fit_positions_sha256']==hashlib.sha256(s['folds'].loc[fit,'row_position'].to_numpy(dtype=np.int64).tobytes()).hexdigest()
        assert provenance['encoded_names']==pre.get_feature_names_out().tolist()
    display(save_table(s,pd.DataFrame(checks),'reload_verification'))
    for name,expected in s['checked'].items(): assert sha256(s['root']/name)==expected,name
    print(f'Six model reloads pass; {len(s["checked"])} earlier files and both raw CSV hashes are unchanged.')


def finish(s):
    rows=s['summary'].set_index('feature_set'); chosen=s['chosen_set']
    findings='\n'.join(f'- {c}: MLP mean macro F1 {rows.loc[c,"mean_macro_f1"]:.4f}, seed range {rows.loc[c,"min_macro_f1"]:.4f}-{rows.loc[c,"max_macro_f1"]:.4f}; saved LightGBM {rows.loc[c,"saved_lightgbm_macro_f1"]:.4f}; mean difference {rows.loc[c,"mean_MLP_minus_LightGBM_pp"]:+.3f} percentage points.' for c in SETS)
    if (rows.mean_MLP_minus_LightGBM_pp < 0).all():
        verdict='LightGBM remains the stronger development benchmark on both sets. Keep the MLP as the non-convolutional neural reference, not as a replacement.'
    else:
        verdict='The tables show where the MLP exceeds the saved tree score. Keep both references and the class-level tradeoffs; this limited experiment does not prove a robust or final win.'
    parts=[f'# Notebook 07: a simple neural reference\n\n{findings}',
        f'## Decision\n\n{verdict} Provisional MLP set: {chosen}. The seed-42 model is the predeclared reference; no best-seed choice or seed ensemble was used.',
        '## What remains uncertain\n\nThirty epochs and two configurations are a bounded reference, not exhaustive tuning or evidence of convergence. Training-seed variation is not sampling uncertainty. Model-selection has prior exposure and these scores are optimistic development evidence. Worms has only 13 rows; one changed correct prediction moves recall by 7.69 percentage points. Repeated and conflicting patterns limit discrimination; source-pattern grouping does not prove host/time independence. Raw scores are not calibrated confidence. No Botnet-specific, DNS-stream or temporal performance claim is supported.',
        '## Notebook 08 contract\n\nUse the two feature sets original42 and total_and_byte44, the exact saved float64 scaled matrices and complete encoded column maps from Notebook05, the same ten classes and frozen rows, and the existing three grouped folds inside fitting. Refit any inner preprocessing only on inner training. Keep calibration targets and official-test outcomes reserved. Compare an explicitly named tabular Conv1D adaptation with the saved seed-42 MLP for each matching set and exact Notebook06 LightGBM controls. Also use seeds 42, 43, 44 for matched neural repeats. Use unweighted loss, a predeclared six-fit grouped search budget and fixed epoch/refit rules; differences from the 30-epoch MLP budget must be declared and cost-reported before fitting. Do not reuse model-selection for early stopping or further informal tuning. Preserve full categorical blocks; save a deterministic source-feature/token order and predeclare at least two order permutations inside fitting. Column position is not network time. Report macro F1, all ten classes and counts, the specific Exploits confusions, binary/AP metrics, costs and seed ranges. Any attention ablation needs a matched backbone/budget; no weighting without a separate weighted tabular control. An authentic NET/DNS/temporal reproduction requires measured DNS transactions, endpoints/timestamps/window provenance and verified botnet ground truth, which these CSVs lack.',
        '## Execution and environment\n\nCPU scikit-learn MLPClassifier; PyTorch DLL loading is blocked by Windows Application Control (WinError 4551). The MLP experiment ran without PyTorch. Saved CSR float64 inputs are unchanged; no densification of the complete dataset or new scaling was added. All six final checkpoints, row-linked probabilities, metrics and confusion matrices were reloaded and verified. Earlier artifact manifests match; pre-existing notebook-save hash differences are recorded, and their starting bytes are preserved. See execution.json for fresh-kernel status and independent tests.',
        '## Section results\n\n'+'\n\n'.join(s['notes'])]
    text='\n\n'.join(parts)
    path=s['out']['docs']/'07_findings.md'; path.write_text(text,encoding='utf-8'); s['outputs'].append(path)
    decision=dict(feature_set=chosen,reference_seed=42,reference_model=f'models/{VERSION}/07_mlp_{chosen}_seed42.joblib',
        settings_sha256=sha256(s['out']['models']/'07_frozen_settings.json'),status='Development reference; no final test claim',
        baseline_verdict=verdict)
    path=s['out']['models']/'07_development_decision.json'; write_json(path,decision); s['outputs'].append(path)
    for name,expected in s['checked'].items(): assert sha256(s['root']/name)==expected,name
    manifest=dict(version=VERSION,notebook='07',plan_sha256=s['plan_hash'],
        dependencies_sha256=s['checked'],artifact_sha256={p.relative_to(s['root']).as_posix():sha256(p) for p in dict.fromkeys(s['outputs'])},
        implementation_sha256=sha256(Path(__file__)),calibration_targets_loaded=False,calibration_predicted=False,
        official_test_records_loaded=False,protected_unchanged=True)
    write_json(s['out']['data']/'07_artifact_manifest.json',manifest)
    display(Markdown('\n\n'.join(parts[:3])))
