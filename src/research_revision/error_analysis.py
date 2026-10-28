"""Post-choice error diagnostics; never a fitted selector or another search."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from src.baseline_models import CLASSES
from .common import table,figure,explain

def minimum_vector_errors(counts):
    """Best possible row accuracy under one deterministic class per exact vector.

    An empirical constraint on these recorded inputs/population, not relabeling.
    It is not a macro-F1 upper bound and not an estimated Bayes error.
    """
    x=np.asarray(counts,dtype=np.int64)
    if x.ndim!=2 or (x<0).any(): raise ValueError('Nonnegative two-dimensional class counts required')
    return int((x.sum(axis=1)-x.max(axis=1)).sum())

def diagnose(s):
    choice=s['choice']; family=choice['model']; candidate=choice['feature_set']
    pred=pd.read_csv(s['out']['data']/f'06_{family}_{candidate}_predictions.csv',float_precision='round_trip')
    link=s['links']['model_selection']
    assert np.array_equal(pred.row_position,link.row_position)
    counts=pd.crosstab(link.feature_group_id,link.attack_cat).reindex(columns=CLASSES,fill_value=0)
    ambiguous=counts.gt(0).sum(axis=1).gt(1)
    flags=link.feature_group_id.map(ambiguous).to_numpy(); correct=pred.predicted_attack_cat.eq(link.attack_cat).to_numpy()
    rows=[]
    for population,mask in [('all selection rows',np.ones(len(link),dtype=bool)),('clear recorded pattern',~flags),('conflicting recorded pattern',flags)]:
        subset=link.loc[mask]; ct=pd.crosstab(subset.feature_group_id,subset.attack_cat).reindex(columns=CLASSES,fill_value=0)
        minimum=minimum_vector_errors(ct)
        rows.append(dict(population=population,rows=int(mask.sum()),distinct_original_patterns=subset.feature_group_id.nunique(),
           actual_errors=int((~correct[mask]).sum()),minimum_deterministic_vector_errors=minimum,
           observed_accuracy=float(correct[mask].mean()),empirical_vector_accuracy_ceiling=1-minimum/int(mask.sum())))
    summary=table(s,pd.DataFrame(rows),'ambiguity_error_diagnostic'); display(summary)
    classes=[]
    for cat in CLASSES:
        for status,flag in [('clear',False),('conflicting',True)]:
            mask=link.attack_cat.eq(cat).to_numpy()&(flags==flag)
            classes.append(dict(attack_cat=cat,pattern_status=status,rows=int(mask.sum()),distinct_original_patterns=link.loc[mask,'feature_group_id'].nunique(),correct=int(correct[mask].sum()),recall=float(correct[mask].mean()) if mask.any() else np.nan))
    per=table(s,pd.DataFrame(classes),'class_ambiguity_errors'); display(per)
    actual=link.attack_cat; guessed=pred.predicted_attack_cat
    mistakes=pd.DataFrame(dict(actual=actual,predicted=guessed,ambiguous=flags)).loc[actual.ne(guessed)]
    confusions=mistakes.groupby(['actual','predicted'],observed=True).agg(rows=('ambiguous','size'),rows_in_conflicting_patterns=('ambiguous','sum')).reset_index().sort_values('rows',ascending=False)
    display(table(s,confusions,'important_confusions').head(15))
    source=s['perclass'][s['perclass'].model.eq(family)]
    original=source[source.feature_set.eq('original42')].set_index('attack_cat')
    chosen=source[source.feature_set.eq(candidate)].set_index('attack_cat')
    delta=pd.DataFrame(dict(attack_cat=CLASSES,delta_class_f1=chosen.loc[list(CLASSES),'f1'].to_numpy()-original.loc[list(CLASSES),'f1'].to_numpy()))
    delta['macro_f1_contribution']=delta.delta_class_f1/10
    delta['contribution_percentage_points']=100*delta.macro_f1_contribution
    expected=s['results'].set_index(['model','feature_set'])
    assert np.isclose(delta.macro_f1_contribution.sum(),expected.loc[(family,candidate),'macro_f1']-expected.loc[(family,'original42'),'macro_f1'])
    display(table(s,delta,'chosen_gain_contributions'))
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    p=per.pivot(index='attack_cat',columns='pattern_status',values='recall').reindex(CLASSES)
    x=np.arange(10)
    axes[0].bar(x-.18,p['clear'],width=.35,label='Clear pattern'); axes[0].bar(x+.18,p['conflicting'],width=.35,label='Conflicting pattern')
    axes[0].set_xticks(x,CLASSES,rotation=65); axes[0].set(ylabel='Recall (descriptive subgroup)',ylim=(0,1),title='Both model errors and ambiguous inputs matter'); axes[0].legend()
    axes[1].barh(delta.attack_cat,delta.contribution_percentage_points); axes[1].axvline(0,color='black',linewidth=.8)
    axes[1].set(xlabel='Contribution to macro-F1 change (percentage points)',title=candidate+' minus original42, same ensemble')
    figure(s,fig,'ambiguity_and_gain')
    worms=float(delta.loc[delta.attack_cat.eq('Worms'),'contribution_percentage_points'].iloc[0]); total=float(delta.contribution_percentage_points.sum())
    explain(s,f'The chosen set changes macro F1 by {total:+.3f} percentage points versus original42 within {family}; Worms alone contributes {worms:+.3f} points. The printed confusion table quantifies errors toward Exploits, and the subgroup table locates conflicting recorded patterns.',
        'Keep the provisional rule unchanged. Carry original42 as a close reference and describe the engineered gain as uncertain, with class-level losses retained. Use the error diagnosis to define future matched ablations, not to start another search here.',
        'Clear and conflicting subgroups have different class mixtures; their recalls are descriptive, not causal attribution. The deterministic-vector ceiling concerns accuracy on this population, not macro F1, a theoretical Bayes limit or unseen data. Extra truly measured context could separate identical recorded flows; deterministic features cannot.')
