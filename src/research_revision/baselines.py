"""Bounded, fitting-only group tuning and matched representation experiments."""
import time
import warnings
import json
import importlib.metadata
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier
from threadpoolctl import threadpool_limits
from IPython.display import display
from src.baseline_models import CLASSES,SETS,make_model,evaluate,exact_settings
from src.data.load_data import sha256
from src.data.feature_engineering import canonical_csr
from .common import table,figure,explain,read_json,write_json,verify_candidates

FAMILIES=('logistic','decision_tree','lightgbm')
CONFIGS={'logistic':[{'C':.1},{'C':1.}],
         'decision_tree':[{'max_depth':16,'min_samples_leaf':5},{'max_depth':24,'min_samples_leaf':2}],
         'lightgbm':[{'num_leaves':31},{'num_leaves':63}]}
TOL=.001

def model(family, configuration):
    if family=='logistic':
        m=make_model('logistic'); m.set_params(estimator__C=configuration['C']); return m
    if family=='decision_tree':
        return make_model('decision_tree').set_params(**configuration)
    if family=='lightgbm':
        return LGBMClassifier(objective='multiclass',n_estimators=120,learning_rate=.1,
           num_leaves=configuration['num_leaves'],max_depth=-1,min_child_samples=20,
           reg_lambda=1.,reg_alpha=0.,colsample_bytree=1.,subsample=1.,subsample_freq=0,
           class_weight=None,n_jobs=2,random_state=42,deterministic=True,force_col_wise=True,
           zero_as_missing=False,verbosity=-1)
    raise ValueError(family)

def declare(s):
    shapes,overlaps=verify_candidates(s); assert overlaps.shared_vectors.eq(0).all()
    folds=pd.read_csv(s['root']/'data/processed/05/05_internal_folds.csv')
    assert np.array_equal(folds.row_position,s['links']['train'].row_position)
    assert folds.groupby('feature_group_id').internal_holdout_fold.nunique().max()==1
    plan=dict(version='revision_v2',question='Do deterministic engineered summaries help matched models?',
      feature_sets=list(SETS),classes=list(CLASSES),fit_role='train',selection_role='model_selection',
      calibration='Integrity only; never predicted',official_test='Hash only in this notebook',seed=42,
      tuning='Two configurations per family, same three existing grouped fitting folds, original42 only. Refit preprocessing on each inner-training fold.',
      configuration_candidates=CONFIGS,exact_candidate_settings={f:[exact_settings(model(f,c)) for c in CONFIGS[f]] for f in FAMILIES},
      inner_budget_fits=18,final_budget='One final train-only fit per family/set; reuse a protected existing fit only when exact settings match.',
      rationale='Tune a family on the common original representation, then freeze its settings across five variants. Feature-specific search budget is zero for every set.',
      weights='Unweighted; no sample weights, resampling or SMOTE',primary_metric='Fixed-ten-class macro F1',
      inner_tie_rule='Within .001 of best mean macro F1: first/lower-capacity declared configuration. Every fold must converge.',
      development_tie_rule='Within .001 of best: fewer source features, then logistic, decision_tree, lightgbm, then higher score, then declared set order.',
      stopping_rule='120 fixed boosting rounds; no early stopping on model-selection rows. No expanded search after seeing scores.',
      prior_exposure='Ten earlier experiments used the same model-selection role; all further selection is optimistic development evidence.',
      fold_assignment_sha256=sha256(s['root']/'data/processed/05/05_internal_folds.csv'),
      original_assignment_sha256=sha256(s['root']/'results/splits/02_train_validation_assignments.csv'),
      source_train_sha256=sha256(s['root']/'data/raw/UNSW_NB15_training-set.csv'),
      implementation_sha256=sha256(s['root']/'src/research_revision/baselines.py'),
      versions={p:importlib.metadata.version(p) for p in ['scikit-learn','lightgbm','numpy','scipy']})
    path=s['root']/'configs/revision_v2_baseline_plan.json'
    if path.exists(): assert read_json(path)==plan,'Plan changed; review/version before another search'
    else: write_json(path,plan)
    s['outputs'].append(path); s['plan']=plan; s['folds']=folds
    print('Plan written before new fitting:',path); display(pd.DataFrame([dict(family=f,representation='scaled' if f=='logistic' else 'tree',configs=json.dumps(CONFIGS[f]),inner_fits=6,sets=5) for f in FAMILIES]))

def tuning(s):
    folds=s['folds']; raw=s['fit'][s['original']]
    y=pd.Categorical(s['fit'].attack_cat,categories=CLASSES).codes
    plan_hash=sha256(s['root']/'configs/revision_v2_baseline_plan.json')
    cache=s['root']/'.cache/revision_v2_tuning'; cache.mkdir(parents=True,exist_ok=True)
    rows=[]; configs=[]; support=[]
    for fold in range(3):
        hold=folds.internal_holdout_fold.eq(fold).to_numpy(); fit=~hold
        assert not set(folds.loc[hold,'feature_group_id'])&set(folds.loc[fit,'feature_group_id'])
        for cat in CLASSES:
            mask=hold&(y==CLASSES.index(cat)); support.append(dict(fold=fold,attack_cat=cat,rows=int(mask.sum()),patterns=folds.loc[mask,'feature_group_id'].nunique()))
        transformed={}
        for rep in ['scaled','tree']:
            # This is deliberately not a slice of the full-fitting fitted matrix.
            p=clone(joblib.load(s['root']/f'models/preprocessing/03/03_{rep}_pipeline.joblib'))
            p.fit(raw.loc[fit]); a=canonical_csr(p.transform(raw.loc[fit])); b=canonical_csr(p.transform(raw.loc[hold]))
            transformed[rep]=(a,b)
            provenance=dict(plan_sha256=plan_hash,fold=fold,representation=rep,preprocessor_fit_rows=int(fit.sum()),
                holdout_rows=int(hold.sum()),preprocessor_fit_row_position_sha256=__import__('hashlib').sha256(folds.loc[fit,'row_position'].to_numpy(dtype=np.int64).tobytes()).hexdigest(),
                encoded_names=p.get_feature_names_out().tolist(),zero_original_group_overlap=True)
            path=s['out']['models']/f'06_inner_{rep}_fold{fold}_pipeline.joblib'; joblib.dump(p,path); s['outputs'].append(path)
            path=s['out']['models']/f'06_inner_{rep}_fold{fold}_provenance.json'; write_json(path,provenance); s['outputs'].append(path)
        for family in FAMILIES:
            a,b=transformed['scaled' if family=='logistic' else 'tree']
            for index,configuration in enumerate(CONFIGS[family]):
                cp=cache/f'{family}_{index}_{fold}.json'
                if cp.exists():
                    result=read_json(cp); assert result['plan_sha256']==plan_hash
                    print('Verified tuning checkpoint:',family,index,fold,flush=True)
                else:
                    m=model(family,configuration); begin=time.perf_counter()
                    print('Fitting inner fold:',family,'config',index,'fold',fold,flush=True)
                    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=2):
                        warnings.simplefilter('always'); m.fit(a,y[fit]); pred=m.predict(b)
                    result=dict(family=family,config_index=index,fold=fold,macro_f1=float(f1_score(y[hold],pred,labels=np.arange(10),average='macro',zero_division=0)),
                        fit_seconds=time.perf_counter()-begin,converged=not any(issubclass(w.category,ConvergenceWarning) for w in caught),
                        warning_messages=[str(w.message) for w in caught],fit_rows=int(fit.sum()),holdout_rows=int(hold.sum()),plan_sha256=plan_hash)
                    write_json(cp,result)
                rows.append(result)
                print(f'Inner macro F1 {result["macro_f1"]:.4f}; {result["fit_seconds"]:.1f}s',flush=True)
    scores=pd.DataFrame([{k:v for k,v in r.items() if k!='warning_messages'} for r in rows])
    display(table(s,scores,'grouped_tuning_folds'))
    summary=scores.groupby(['family','config_index']).agg(mean_macro_f1=('macro_f1','mean'),min_macro_f1=('macro_f1','min'),max_macro_f1=('macro_f1','max'),total_seconds=('fit_seconds','sum'),all_converged=('converged','all')).reset_index()
    selected={}
    for family in FAMILIES:
        v=summary[summary.family.eq(family)&summary.all_converged]
        if v.empty: raise RuntimeError(f'No converged configuration for {family}; bounded search did not produce a valid candidate')
        best=v.mean_macro_f1.max(); chosen=v[v.mean_macro_f1.ge(best-TOL)].sort_values('config_index').iloc[0]
        selected[family]=CONFIGS[family][int(chosen.config_index)]
    display(table(s,summary,'grouped_tuning_summary')); display(table(s,pd.DataFrame(support),'inner_class_support').query('attack_cat == "Worms"'))
    path=s['out']['models']/'06_frozen_family_settings.json'; write_json(path,dict(plan_sha256=plan_hash,chosen=selected,exact_settings={f:exact_settings(model(f,c)) for f,c in selected.items()})); s['outputs'].append(path)
    s['selected_configs']=selected
    explain(s,'The bounded search uses only fitting rows. Fold-local category vocabularies and scaling are refitted before each group-held-out prediction.',
        'Freeze one setting per family and use it unchanged for all five representations. Do not use model-selection results to expand this search.',
        'Three folds and two configurations are a modest benchmark budget, not exhaustive tuning. Pattern groups are not verified independent hosts; fold ranges are not confidence intervals.')

def complexity(m,family):
    if family=='logistic': return int(sum(e.coef_.size+e.intercept_.size for e in m.estimators_))
    if family=='decision_tree': return int(m.tree_.node_count)
    return int(sum(t['num_leaves'] for t in m.booster_.dump_model()['tree_info']))

def experiments(s):
    y=pd.Categorical(s['links']['train'].attack_cat,categories=CLASSES).codes
    eval_y=pd.Categorical(s['links']['model_selection'].attack_cat,categories=CLASSES).codes
    plan_hash=sha256(s['root']/'configs/revision_v2_baseline_plan.json'); rows=[]; classes=[]
    for family in FAMILIES:
        configuration=s['selected_configs'][family]; rep='scaled' if family=='logistic' else 'tree'
        for candidate in SETS:
            name=family+'_'+candidate; experiment='revised_'+name
            x=canonical_csr(sparse.load_npz(s['root']/f'data/processed/05/05_{candidate}_{rep}_train.npz'))
            v=canonical_csr(sparse.load_npz(s['root']/f'data/processed/05/05_{candidate}_{rep}_model_selection.npz'))
            m=model(family,configuration); settings=exact_settings(m)
            model_path=s['out']['models']/f'06_{name}.joblib'; settings_path=s['out']['models']/f'06_{name}_settings.json'
            reused=False; caught=[]
            if model_path.exists():
                saved=read_json(settings_path); assert saved['plan_sha256']==plan_hash and saved['settings']==settings
                m=joblib.load(model_path); seconds=saved['fit_seconds']; reused=saved['reused_prior_exact_fit']; converged=saved['converged']
            else:
                old=s['root']/f'models/baselines/06/06_{name}.joblib'
                if old.exists() and exact_settings(joblib.load(old))==settings:
                    m=joblib.load(old); oldmetrics=pd.read_csv(s['root']/'results/tables/06_model_selection_metrics.csv')
                    previous=oldmetrics[oldmetrics.model.eq(family)&oldmetrics.feature_set.eq(candidate)].iloc[0]
                    seconds=float(previous.fit_seconds); converged=bool(previous.eligible); reused=True
                    print('Reusing exact protected fit:',name,flush=True)
                else:
                    print('Final train-only fit:',name,flush=True); begin=time.perf_counter()
                    with warnings.catch_warnings(record=True) as caught,threadpool_limits(limits=2):
                        warnings.simplefilter('always'); m.fit(x,y)
                    seconds=time.perf_counter()-begin; converged=not any(issubclass(w.category,ConvergenceWarning) for w in caught)
                joblib.dump(m,model_path)
                write_json(settings_path,dict(settings=settings,plan_sha256=plan_hash,fit_seconds=seconds,
                    converged=converged,reused_prior_exact_fit=reused,input_sha256=sha256(s['root']/f'data/processed/05/05_{candidate}_{rep}_train.npz'),
                    warnings=[str(w.message) for w in caught]))
            assert exact_settings(m)==settings
            assert np.array_equal(m.classes_,np.arange(10)) and m.n_features_in_==x.shape[1]
            pred=m.predict(v); probability=m.predict_proba(v)
            reload=joblib.load(model_path)
            assert np.array_equal(pred,reload.predict(v)) and np.allclose(probability,reload.predict_proba(v),rtol=0,atol=0)
            metrics,per,cm,ncm,pr,score=evaluate(eval_y,pred,probability)
            rows.append(dict(experiment=experiment,model=family,feature_set=candidate,representation=rep,fit_seconds=seconds,
                source_features=len(read_json(s['root']/'models/preprocessing/05/05_comparison_sets.json')['sets'][candidate]['retained']),
                encoded_columns=x.shape[1],complexity_units=complexity(m,family),complexity_type='coefficients including intercepts' if family=='logistic' else 'tree nodes' if family=='decision_tree' else 'total boosting leaves',
                model_bytes=model_path.stat().st_size,eligible=converged,reused_prior_exact_fit=reused,**metrics))
            per.insert(0,'feature_set',candidate); per.insert(0,'model',family); classes.append(per)
            predictions=s['links']['model_selection'].copy(); predictions['predicted_attack_cat']=np.asarray(CLASSES)[pred]
            predictions['predicted_binary_attack']=pred!=CLASSES.index('Normal'); predictions['attack_score']=score
            for i,c in enumerate(CLASSES): predictions['p_'+c]=probability[:,i]
            path=s['out']['data']/f'06_{name}_predictions.csv'; predictions.to_csv(path,index=False,float_format='%.17g'); s['outputs'].append(path)
            reloaded=pd.read_csv(path,float_precision='round_trip')
            assert np.array_equal(reloaded.row_position,s['links']['model_selection'].row_position)
            assert np.array_equal(reloaded[['p_'+c for c in CLASSES]].to_numpy(),probability)
            for normalized,mat in [(False,cm),(True,ncm)]:
                frame=pd.DataFrame(mat,columns=CLASSES); frame.insert(0,'actual_class',CLASSES)
                table(s,frame,name+'_confusion_'+('normalized' if normalized else 'raw'))
            table(s,pr,name+'_precision_recall'); s['outputs'].extend([model_path,settings_path])
            print(f'{name}: macro F1 {metrics["macro_f1"]:.4f}; Worms {metrics["worms_correct"]}/13; {seconds:.1f}s',flush=True)
    results=table(s,pd.DataFrame(rows),'model_selection_metrics'); perclass=table(s,pd.concat(classes,ignore_index=True),'per_class_metrics')
    s['results']=results; s['perclass']=perclass
    display(results[['model','feature_set','macro_f1','accuracy','weighted_f1','worms_correct','worms_support','fit_seconds','complexity_units']])

def compare(s):
    result=s['results']; per=s['perclass']; pairs=[('original42',v) for v in SETS[1:]]+[
        ('total_only43','total_and_byte44'),('total_only43','total_and_packet44'),('total_and_byte44','all45'),('total_and_packet44','all45')]
    rows=[]; changes=[]
    for f in FAMILIES:
        v=result[result.model.eq(f)].set_index('feature_set')
        for a,b in pairs:
            row=dict(model=f,reference=a,candidate=b)
            for metric in ['macro_f1','accuracy','weighted_f1','attack_recall','false_positive_rate','attack_average_precision','worms_correct']:
                row['delta_'+metric]=float(v.loc[b,metric]-v.loc[a,metric])
            rows.append(row)
            for c in CLASSES:
                p=per[per.model.eq(f)&per.attack_cat.eq(c)].set_index('feature_set')
                changes.append(dict(model=f,reference=a,candidate=b,attack_cat=c,support=int(p.loc[a,'support']),delta_f1=float(p.loc[b,'f1']-p.loc[a,'f1']),delta_correct=int(p.loc[b,'correct']-p.loc[a,'correct'])))
    comparisons=table(s,pd.DataFrame(rows),'feature_comparisons'); table(s,pd.DataFrame(changes),'class_tradeoffs'); s['comparisons']=comparisons
    display(comparisons)
    fig,axes=plt.subplots(1,2,figsize=(13,5)); x=np.arange(5)
    for j,f in enumerate(FAMILIES):
        v=result[result.model.eq(f)].set_index('feature_set').loc[list(SETS)]
        axes[0].plot(x,v.macro_f1,'o-',label=f)
        axes[1].plot(x,v.worms_correct,'o-',label=f)
    for ax in axes: ax.set_xticks(x,SETS,rotation=40); ax.legend()
    axes[0].set(ylabel='Fixed-ten-class macro F1',title='Matched settings within each family')
    axes[1].set(ylabel='Correct Worms rows out of 13',ylim=(-.5,13.5),title='One Worms row changes recall by 7.69 points')
    figure(s,fig,'matched_models')
    for f in FAMILIES:
        rare=per[per.model.eq(f)&per.attack_cat.isin(['Analysis','Backdoor','Shellcode','Worms'])]
        print(f+' — rare-class results for every set')
        display(rare[['model','feature_set','attack_cat','precision','recall','f1','support','correct']])
        best_set=result[result.model.eq(f)].sort_values('macro_f1',ascending=False).iloc[0].feature_set
        print(f+' — every class for its highest-scoring set (full 150-row table is saved)')
        display(per[per.model.eq(f)&per.feature_set.eq(best_set)])
    display(result[['model','feature_set','attack_recall','attack_precision','false_positive_rate','attack_average_precision']])
    # Existing ten unweighted experiments remain a separate, explicit historical control.
    old=pd.read_csv(s['root']/'results/tables/06_model_selection_metrics.csv')
    historical=old[['model','feature_set','macro_f1','accuracy','worms_correct','fit_seconds']].copy(); historical['status']='Earlier fixed-settings control; same selection rows'
    display(table(s,historical,'historical_controls'))
    eligible=result[result.eligible].copy(); best=eligible.macro_f1.max(); tied=eligible[eligible.macro_f1.ge(best-TOL)].copy()
    tied['priority']=tied.model.map({f:i for i,f in enumerate(FAMILIES)}); tied['set_order']=tied.feature_set.map({f:i for i,f in enumerate(SETS)})
    chosen=tied.sort_values(['source_features','priority','macro_f1','set_order'],ascending=[True,True,False,True]).iloc[0]
    s['choice']=dict(model=chosen.model,feature_set=chosen.feature_set,macro_f1=float(chosen.macro_f1),best_observed=float(best),gap=float(best-chosen.macro_f1),tie_tolerance=TOL,final_performance='Unknown',status='Provisional development candidate')
    path=s['out']['models']/'06_provisional_choice.json'; write_json(path,s['choice']); s['outputs'].append(path)
    explain(s,f'The declared rule selects {chosen.model} with {chosen.feature_set} (macro F1 {chosen.macro_f1:.4f}). This selection uses previously exposed model-selection rows.',
        'Carry this exact model/input contract as the performance bar. Retain class-level tradeoffs and negative engineered-feature findings.',
        'Scores are development point estimates, not final performance or statistical significance. Thirteen Worms rows cannot establish a stable rare-class advantage; raw scores are not calibrated confidence.')

def detailed_plots(s):
    result=s['results']; per=s['perclass']
    # Save both matrices for every experiment; display the best set in each family.
    chosen_by_family={f:result[result.model.eq(f)].sort_values('macro_f1',ascending=False).iloc[0].feature_set for f in FAMILIES}
    for f in FAMILIES:
        for candidate in SETS:
            fig,axes=plt.subplots(1,2,figsize=(13,6))
            for ax,suffix in zip(axes,['raw','normalized']):
                frame=pd.read_csv(s['out']['tables']/f'06_{f}_{candidate}_confusion_{suffix}.csv'); m=frame[list(CLASSES)].to_numpy()
                im=ax.imshow(m,cmap='Blues'); ax.set_xticks(range(10),CLASSES,rotation=80); ax.set_yticks(range(10),CLASSES)
                ax.set(title=f+' '+candidate+' '+suffix,xlabel='Predicted class',ylabel='Actual class')
                for i,j in __import__('itertools').product(range(10),repeat=2):
                    text=str(int(m[i,j])) if suffix=='raw' else f'{m[i,j]:.2f}'
                    if m[i,j]>0: ax.text(j,i,text,ha='center',va='center',fontsize=6,color='white' if m[i,j]>.5*m.max() else 'black')
                fig.colorbar(im,ax=ax,shrink=.6)
            if candidate==chosen_by_family[f]: figure(s,fig,f+'_'+candidate+'_confusions')
            else:
                fig.tight_layout(); path=s['out']['figures']/f'06_{f}_{candidate}_confusions.png'; fig.savefig(path,dpi=150,bbox_inches='tight'); plt.close(fig); s['outputs'].append(path)
    fig,ax=plt.subplots(figsize=(7,5))
    for f,candidate in chosen_by_family.items():
        pr=pd.read_csv(s['out']['tables']/f'06_{f}_{candidate}_precision_recall.csv')
        ax.plot(pr.recall,pr.precision,label=f+' '+candidate)
    prevalence=s['links']['model_selection'].label.mean(); ax.axhline(prevalence,color='grey',linestyle='--',label='Attack prevalence')
    ax.set(xlabel='Attack recall',ylabel='Attack precision',xlim=(0,1),ylim=(0,1.02),title='Raw attack scores: 1 − P(Normal)'); ax.legend(); figure(s,fig,'precision_recall')
    chosen=s['choice']; p=per[per.model.eq(chosen['model'])&per.feature_set.eq(chosen['feature_set'])]
    fig,ax=plt.subplots(figsize=(10,4)); ax.bar(p.attack_cat,p.recall); ax.set(ylim=(0,1),ylabel='Class recall',title='Provisional candidate: which classes are missed?'); ax.tick_params(axis='x',rotation=55); figure(s,fig,'chosen_class_recall')

def result_text(s):
    c=s['choice']; rows=[]
    for f in FAMILIES:
        v=s['results'][s['results'].model.eq(f)].set_index('feature_set'); best=v.macro_f1.idxmax()
        delta=100*(v.loc[best,'macro_f1']-v.loc['original42','macro_f1'])
        rows.append(f'{f}: original42 macro F1 {v.loc["original42","macro_f1"]:.4f}; highest observed set {best} {v.loc[best,"macro_f1"]:.4f} (difference {delta:+.3f} percentage points).')
    return '\n\n'.join(rows)+f'\n\nProvisional choice: {c["model"]}, {c["feature_set"]}. Every result uses the same 17,534 model-selection rows; Worms support is 13.'
