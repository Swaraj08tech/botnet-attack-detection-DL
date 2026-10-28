"""Fixed sparse baselines, ten-class accounting, and development-only decisions."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_recall_curve, precision_recall_fscore_support)

SETS = ('original42', 'total_only43', 'total_and_byte44', 'total_and_packet44', 'all45')
MODELS = ('logistic', 'decision_tree')
CLASSES = ('Analysis', 'Backdoor', 'DoS', 'Exploits', 'Fuzzers', 'Generic',
           'Normal', 'Reconnaissance', 'Shellcode', 'Worms')
NORMAL = CLASSES.index('Normal')
SEED = 42
TIE_TOLERANCE = 0.001


def make_model(family):
    if family == 'logistic':
        # sklearn 1.9 expresses pure L2 with l1_ratio=0; no deprecated penalty option.
        return OneVsRestClassifier(LogisticRegression(solver='liblinear', C=1.0,
            l1_ratio=0.0, dual=False, max_iter=500, tol=1e-4, fit_intercept=True,
            intercept_scaling=1, class_weight=None, random_state=SEED), n_jobs=1)
    if family == 'decision_tree':
        return DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=16,
            min_samples_leaf=5, min_samples_split=2, max_features=None,
            class_weight=None, ccp_alpha=0.0, random_state=SEED)
    raise ValueError(f'Unknown baseline: {family}')


def exact_settings(model):
    return {k: v for k, v in model.get_params(deep=True).items() if k != 'estimator'}


def evaluate(y, predicted, probabilities):
    """Fixed ten-class metrics; binary decisions come from multiclass predictions."""
    y=np.asarray(y, dtype=int); predicted=np.asarray(predicted, dtype=int)
    probabilities=np.asarray(probabilities, dtype=float)
    if probabilities.shape != (len(y),len(CLASSES)) or not np.isfinite(probabilities).all():
        raise ValueError('Ten aligned finite probability columns required')
    if np.any(probabilities < 0) or np.any(probabilities > 1): raise ValueError('Invalid probabilities')
    if not np.allclose(probabilities.sum(axis=1),1,atol=1e-10): raise ValueError('Probability sums')
    if not set(y).issubset(range(10)) or not set(predicted).issubset(range(10)):
        raise ValueError('Unknown class')
    labels=np.arange(len(CLASSES))
    precision,recall,f1,support=precision_recall_fscore_support(y,predicted,labels=labels,zero_division=0)
    cm=confusion_matrix(y,predicted,labels=labels)
    normalized=np.divide(cm,cm.sum(axis=1,keepdims=True),out=np.zeros_like(cm,dtype=float),where=cm.sum(axis=1,keepdims=True)>0)
    per_class=pd.DataFrame(dict(attack_cat=CLASSES,precision=precision,recall=recall,f1=f1,
        support=support,predicted_support=cm.sum(axis=0),correct=np.diag(cm),zero_support=support==0))
    actual_attack=y!=NORMAL; detected_attack=predicted!=NORMAL
    tn,fp,fn,tp=confusion_matrix(actual_attack,detected_attack,labels=[False,True]).ravel()
    ratio=lambda a,b: float(a/b) if b else float('nan')
    score=1-probabilities[:,NORMAL]
    both_classes=np.unique(actual_attack).size==2
    pr_precision,pr_recall,pr_thresholds=precision_recall_curve(actual_attack,score) if both_classes else (np.array([]),np.array([]),np.array([]))
    pr=pd.DataFrame(dict(precision=pr_precision,recall=pr_recall,
        score_threshold=np.append(pr_thresholds,np.nan) if both_classes else []))
    metrics=dict(macro_f1=float(np.mean(f1)),accuracy=float(accuracy_score(y,predicted)),
        weighted_f1=float(f1_score(y,predicted,labels=labels,average='weighted',zero_division=0)),
        attack_recall=ratio(tp,tp+fn),attack_precision=ratio(tp,tp+fp),
        false_positive_rate=ratio(fp,fp+tn),false_negative_rate=ratio(fn,fn+tp),
        attack_average_precision=float(average_precision_score(actual_attack,score)) if both_classes else float('nan'),
        true_negative=int(tn),false_positive=int(fp),false_negative=int(fn),true_positive=int(tp),
        worms_f1=float(f1[-1]),worms_recall=float(recall[-1]),worms_correct=int(cm[-1,-1]),worms_support=int(support[-1]))
    return metrics,per_class,cm,normalized,pr,score


def choose_candidate(results, feature_counts):
    """Practical tie rule declared before fitting; never a significance test."""
    eligible=results[results.eligible].copy()
    if eligible.empty: raise RuntimeError('No finite, converged candidate; no development choice')
    best=float(eligible.macro_f1.max())
    tied=eligible[eligible.macro_f1>=best-TIE_TOLERANCE-1e-12].copy()
    tied['source_features']=tied.feature_set.map(feature_counts)
    tied['model_priority']=tied.model.map({m:i for i,m in enumerate(MODELS)})
    tied['set_priority']=tied.feature_set.map({s:i for i,s in enumerate(SETS)})
    chosen=tied.sort_values(['source_features','model_priority','macro_f1','set_priority'],
        ascending=[True,True,False,True]).iloc[0]
    return dict(model=chosen.model,feature_set=chosen.feature_set,experiment=chosen.experiment,
        macro_f1=float(chosen.macro_f1),best_observed_macro_f1=best,
        gap_to_best=best-float(chosen.macro_f1),tie_tolerance=TIE_TOLERANCE,
        tied_candidates=tied.experiment.tolist(),status='Provisional development choice; final performance unknown')


def comparisons(results,per_class):
    pairs=[('original42',s,'Engineered versus original') for s in SETS[1:]]
    pairs += [('total_only43','total_and_byte44','Contribution of byte balance'),
              ('total_only43','total_and_packet44','Contribution of packet balance'),
              ('total_and_byte44','all45','Contribution of packet balance after byte balance'),
              ('total_and_packet44','all45','Contribution of byte balance after packet balance')]
    rows=[]; class_rows=[]
    metrics=['macro_f1','accuracy','weighted_f1','attack_recall','attack_precision',
             'false_positive_rate','attack_average_precision','worms_f1','worms_recall','worms_correct']
    for family in MODELS:
        r=results[results.model.eq(family)].set_index('feature_set')
        c=per_class[per_class.model.eq(family)]
        for reference,candidate,question in pairs:
            row=dict(model=family,reference=reference,candidate=candidate,question=question,
                both_eligible=bool(r.loc[reference,'eligible'] and r.loc[candidate,'eligible']))
            for metric in metrics:
                row['delta_'+metric]=float(r.loc[candidate,metric]-r.loc[reference,metric])
            rows.append(row)
            a=c[c.feature_set.eq(reference)].set_index('attack_cat')
            b=c[c.feature_set.eq(candidate)].set_index('attack_cat')
            for target in CLASSES:
                assert a.loc[target,'support']==b.loc[target,'support']
                class_rows.append(dict(model=family,reference=reference,candidate=candidate,attack_cat=target,
                    support=int(a.loc[target,'support']),delta_precision=float(b.loc[target,'precision']-a.loc[target,'precision']),
                    delta_recall=float(b.loc[target,'recall']-a.loc[target,'recall']),
                    delta_f1=float(b.loc[target,'f1']-a.loc[target,'f1']),
                    delta_correct=int(b.loc[target,'correct']-a.loc[target,'correct'])))
    return pd.DataFrame(rows),pd.DataFrame(class_rows)
