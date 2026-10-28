"""Fresh integrity/EDA/feature investigations, using preserved definitions."""
import itertools
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.base import clone
from IPython.display import display
from src.data.validate_data import quality_summary
from src.data.duplicate_investigation import (build_groups,repetition_summary,ambiguity_tables,
                                            cross_split_overlap,overlap_by_class,policy_class_counts)
from src.data.feature_engineering import retained_features,candidate_features,proposal_table,RETAINED,canonical_csr
from src.data.feature_selection import fit_bin_edges,apply_bins,association_score
from src.baseline_models import CLASSES,SETS
from .common import table,figure,explain,show_prior,read_json,write_json,ROLES,exact_roles,verify_candidates

def audit_original(s):
    # Historical audit only. Test records are never passed into learned analysis.
    test=pd.read_csv(s['root']/'data/raw/UNSW_NB15_testing-set.csv',low_memory=False)
    s['datasets']={'train':s['raw'],'test':test}
    s['groups']={r:build_groups(df) for r,df in s['datasets'].items()}
    prior=pd.read_csv(s['root']/'results/tables/dataset_summary.csv').set_index('Dataset')
    checks=[]
    for r,df in s['datasets'].items():
        now=quality_summary(df,r)
        for metric in ['Rows','Columns','Missing Values','Infinite Values','Duplicate Rows',
                       'Duplicate Features (excluding id and targets)','Duplicate Labeled Records (excluding id)']:
            checks.append(dict(dataset=r,metric=metric,previous=int(prior.loc[r,metric]),reproduced=int(now[metric])))
    checks=pd.DataFrame(checks); checks['matches']=checks.previous.eq(checks.reproduced)
    assert checks.matches.all(); display(table(s,checks,'audit_reproduction'))
    assert np.array_equal(s['groups']['train'].row_groups,s['assignment'].feature_group_id)
    explain(s,'All 14 core Notebook 01 audit counts reproduce; original-pattern IDs also reproduce the frozen assignment.',
            'Keep both source files and all frozen roles. Test access here only repeats the previously disclosed audit.',
            'Preservation hashes authenticate local bytes, not the dataset creator or capture process.')

def duplicates(s):
    summaries=[]; class_rows=[]
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,(r,df) in zip(axes,s['datasets'].items()):
        g=s['groups'][r]; overall,byclass=repetition_summary(df,g,r)
        summaries.append(overall); class_rows.append(byclass.rename(columns={'independent_feature_groups':'distinct_original_patterns'}))
        table(s,g.summary.join(g.category_counts.add_prefix('class_rows__')),'groups_'+r)
        bins=[1,2,3,5,10,20,50,100,200,500]
        n,_=np.histogram(g.summary.rows,bins=bins)
        ax.bar(['1','2','3–4','5–9','10–19','20–49','50–99','100–199','200–499'],n)
        ax.set(title=r.title(),ylabel='Recorded feature patterns (log scale)',xlabel='Rows per pattern',yscale='log')
        ax.tick_params(axis='x',rotation=50)
    figure(s,fig,'group_sizes')
    summary=table(s,pd.DataFrame(summaries),'repetition'); display(summary)
    cr=table(s,pd.concat(class_rows,ignore_index=True),'class_repetition')
    fig,ax=plt.subplots(figsize=(10,5))
    for offset,r in [(-.18,'train'),(.18,'test')]:
        v=cr[cr.dataset.eq(r)&cr.target.eq('attack_cat')].set_index('class').reindex(CLASSES)
        ax.barh(np.arange(10)+offset,v.repeated_group_row_pct,height=.34,label=r)
    ax.set_yticks(np.arange(10),CLASSES); ax.set(xlabel='Rows in repeated patterns (%)',xlim=(0,100)); ax.legend()
    figure(s,fig,'class_repetition')
    display(cr[cr.target.eq('attack_cat')])
    explain(s,'Training has 101,040 distinct patterns, 84,049 rows in repeated groups, and 74,301 excess rows; these are different quantities.',
        'Keep all rows but partition complete original patterns together. Report row support and pattern support.',
        'Two identical recorded vectors can describe different events. Distinct patterns are not verified independent events.')

def conflicts(s):
    outputs=[[],[],[],[]]
    for r,df in s['datasets'].items():
        for dest,out in zip(outputs,ambiguity_tables(df,s['groups'][r],r)): dest.append(out)
    merged=[table(s,pd.concat(out,ignore_index=True),name) for out,name in zip(outputs,['conflicts','conflict_combinations','conflict_pairs','conflict_examples'])]
    display(merged[0]); display(merged[1].sort_values('rows',ascending=False).head(15))
    constraints=[]
    for r,g in s['groups'].items():
        for target,counts in [('label',g.binary_counts),('attack_cat',g.category_counts)]:
            errors=int((counts.sum(axis=1)-counts.max(axis=1)).sum())
            constraints.append(dict(dataset=r,target=target,minimum_recorded_row_errors=errors,
                                    maximum_empirical_accuracy=1-errors/len(s['datasets'][r])))
    display(table(s,pd.DataFrame(constraints),'ambiguity_constraints'))
    display(merged[3][['dataset','conflict_target','group_id','row_position','label','attack_cat','dur','sbytes','dbytes']].head(12))
    prior=pd.read_csv(s['root']/'results/tables/02_target_ambiguity.csv')
    for row in merged[0].itertuples():
        old=prior[prior.dataset.eq(row.dataset)&prior.target.eq(row.target)].iloc[0]
        assert row.affected_groups==old.affected_groups and row.affected_rows==old.affected_rows
    pairs=merged[2]; fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for ax,r in zip(axes,['train','test']):
        m=np.zeros((10,10),dtype=int)
        for row in pairs[pairs.dataset.eq(r)&pairs.target.eq('attack_cat')].itertuples():
            a,b=CLASSES.index(row.class_a),CLASSES.index(row.class_b); m[a,b]=m[b,a]=row.shared_conflicting_groups
        im=ax.imshow(np.log1p(m),cmap='YlOrRd'); ax.set_xticks(range(10),CLASSES,rotation=80); ax.set_yticks(range(10),CLASSES)
        ax.set_title(r.title()+' — conflicting patterns')
        for i,j in itertools.product(range(10),repeat=2):
            if m[i,j]: ax.text(j,i,str(m[i,j]),ha='center',va='center',fontsize=7)
    fig.colorbar(im,ax=axes.ravel().tolist(),label='log(1 + number of conflicting patterns)',shrink=.65)
    figure(s,fig,'conflict_heatmap')
    explain(s,'Training contains 1,772 category-conflicting groups (30,528 rows) and 229 binary-conflicting groups (940 rows). Binary and category ambiguity differ.',
        'Retain original labels and keep each whole pattern in one role. Do not apply majority-label replacement.',
        'Missing context, quantization or annotation differences may explain conflicts; these data do not distinguish them. The empirical accuracy ceiling applies only to one deterministic prediction per recorded vector on the audited population; additional genuinely measured inputs or another population can change it. It is not a majority-relabeling instruction.')

def cross_split(s):
    matches,matchmap=cross_split_overlap(s['groups']['train'],s['groups']['test'])
    assert len(matches)==1302 and matches.train_rows.sum()==10279 and matches.test_rows.sum()==8541
    table(s,matches,'cross_split_exact_matches')
    rows=[]
    for target in ['binary','category']:
        for status,v in matches.groupby(target+'_status'):
            rows.append(dict(target=target,status=status,vectors=len(v),train_rows=int(v.train_rows.sum()),test_rows=int(v.test_rows.sum())))
    display(table(s,pd.DataFrame(rows),'cross_split_summary'))
    byclass=pd.concat([overlap_by_class(df,s['groups'][r],matches,r) for r,df in s['datasets'].items()],ignore_index=True)
    display(table(s,byclass[byclass.target.eq('attack_cat')],'cross_split_by_class'))
    mask=matchmap[s['groups']['test'].row_groups]>=0
    matched_training_group=matchmap[s['groups']['test'].row_groups]
    fit_mask=np.isin(matched_training_group,s['links']['train'].feature_group_id.unique())
    masks=pd.DataFrame(dict(row_position=np.arange(len(mask)),overlaps_supplied_training=mask,
                            nonoverlap_supplied_training=~mask,overlaps_actual_fitting=fit_mask,
                            nonoverlap_actual_fitting=~fit_mask))
    assert mask.sum()==8541 and (~mask).sum()==73791
    table(s,masks,'test_diagnostic_masks')
    diagnostic=[]
    for name,m in [('full_test',np.ones(len(mask),dtype=bool)),('overlaps_official_train',mask),
                   ('nonoverlap_official_train',~mask),('overlaps_fit_train',fit_mask)]:
        diagnostic.append(dict(population=name,rows=int(m.sum()),distinct_test_patterns=len(np.unique(s['groups']['test'].row_groups[m]))))
    display(table(s,pd.DataFrame(diagnostic),'test_overlap_populations'))
    explain(s,'Full-tuple equality confirms 1,302 shared patterns and 8,541 participating test rows. Category overlap is consistent for 754 patterns and conflicting for 548.',
        'Retain the full supplied test. Preserve masks relative both to supplied training and to actual fitting rows; name the reference population in every later diagnostic. Never resplit it.',
        'Dictionary hashing only locates candidates; full tuple equality determines a match. Disjoint exact vectors do not prove host/time independence.')

def semantics(s):
    fig,axes=plt.subplots(2,1,figsize=(11,5))
    for ax,(r,df) in zip(axes,s['datasets'].items()):
        # All rows, coloured by recorded class. ID is shown strictly as an identifier.
        y=pd.Categorical(df.attack_cat,categories=CLASSES).codes
        ax.scatter(df.id,y,c=y,cmap='tab10',s=.3,rasterized=True); ax.set_yticks(range(10),CLASSES,fontsize=7)
        ax.set(title=r+' class ordering',xlabel='ID (identifier; not time)')
    figure(s,fig,'id_ordering')
    rows=[]
    for r,df in s['datasets'].items():
        for name,value in [('service_dash',int(df.service.eq('-').sum())),('nonbinary_ftp_login',int((~df.is_ftp_login.isin([0,1])).sum())),
                           ('ftp_field_mismatch',int(df.is_ftp_login.ne(df.ct_ftp_cmd).sum()))]:
            rows.append(dict(dataset=r,check=name,rows=value))
    display(table(s,pd.DataFrame(rows),'semantics_checks'))
    prior_register=pd.read_csv(s['root']/'results/tables/02_feature_availability_register.csv').set_index('feature')
    availability=[]
    for col in s['original']:
        contextual=col in ['ct_srv_src','ct_srv_dst','ct_dst_ltm','ct_src_ltm','ct_src_dport_ltm','ct_dst_sport_ltm','ct_dst_src_ltm']
        availability.append(dict(feature=col,definition_status=prior_register.loc[col,'evidence_status'],
           prediction_scope='Completed flow plus connection history' if contextual else 'Completed-flow benchmark',
           unresolved='Past-only ordering/window and endpoint extraction unavailable' if contextual else 'Earliest availability and extraction details not verified'))
    display(table(s,pd.DataFrame(availability),'feature_availability').head(10))
    display(pd.DataFrame(availability).loc[pd.DataFrame(availability).feature.isin(['is_ftp_login','ct_ftp_cmd','ct_srv_src','ct_src_ltm'])])
    explain(s,'Class blocks follow ID ordering. FTP fields match throughout supplied training but differ on eight previously audited test rows; login values include 2 and 4.',
        'Exclude ID, preserve the literal service marker and both numeric FTP fields. Restrict claims to offline completed-flow use.',
        'Contextual ct_* counts require history. We cannot verify past-only extraction, timing or real-time availability. Predictive association alone is not proof of leakage.')

def policy_and_roles(s):
    policy=pd.concat([policy_class_counts(df,s['groups'][r],r) for r,df in s['datasets'].items()],ignore_index=True)
    table(s,policy,'policy_effects'); display(policy[policy.target.eq('attack_cat')&policy.dataset.eq('train')])
    links=s['links']; rows=[]
    for a,b in [('train','model_selection'),('train','calibration'),('model_selection','calibration')]:
        overlap=set(links[a].feature_group_id)&set(links[b].feature_group_id)
        assert not overlap; rows.append(dict(role_a=a,role_b=b,shared_original_patterns=len(overlap)))
    display(table(s,pd.DataFrame(rows),'frozen_role_overlap'))
    display(pd.read_csv(s['out']['tables']/'02_role_class_support.csv'))
    explain(s,'All original-pattern groups belong to one frozen role; every pair has zero exact overlap. Model selection has 13 Worms rows in 13 patterns.',
        'Adopt keep-all policy. Reuse frozen assignments rather than draw a more favourable split. Calibration remains reserved.',
        'Stratification cannot guarantee perfect balance with uneven/conflicting groups. Pattern support is only an upper bound on effective independent information.')

def fit_eda(s):
    link=s['links']['train']; fit=s['fit']; unique=~link.feature_group_id.duplicated()
    support=pd.DataFrame([dict(attack_cat=c,rows=int(fit.attack_cat.eq(c).sum()),distinct_original_patterns=link.loc[fit.attack_cat.eq(c),'feature_group_id'].nunique()) for c in CLASSES])
    display(table(s,support,'fitting_support'))
    fig,ax=plt.subplots(figsize=(10,4)); pos=np.arange(10)
    ax.bar(pos-.18,support.rows,width=.35,label='Rows'); ax.bar(pos+.18,support.distinct_original_patterns,width=.35,label='Distinct patterns')
    ax.set_xticks(pos,CLASSES,rotation=55); ax.set(ylabel='Support (log scale)',yscale='log',title='Only fitting rows: class imbalance and repeated patterns'); ax.legend()
    figure(s,fig,'class_support')
    numeric=fit[s['original']].select_dtypes('number'); rows=[]
    for col in numeric:
        x=numeric[col]; rows.append(dict(feature=col,min=x.min(),median=x.median(),p99=x.quantile(.99),max=x.max(),zero_pct=100*x.eq(0).mean(),skew=x.skew(),distinct_values=x.nunique()))
    summary=table(s,pd.DataFrame(rows),'numeric_summary'); display(summary.sort_values('skew',ascending=False))
    fig,axes=plt.subplots(1,2,figsize=(12,5)); v=summary.sort_values('zero_pct',ascending=False).head(12)
    axes[0].barh(v.feature,v.zero_pct); axes[0].set(xlabel='Zero rows (%)',title='Zeros may encode protocol behaviour')
    v=summary.sort_values('skew',ascending=False).head(12); axes[1].barh(v.feature,v['skew']); axes[1].set(xlabel='Skewness',title='Large positive tails')
    figure(s,fig,'zeros_and_skew')
    categorical=[]
    for col in ['proto','service','state']:
        for value,n in fit[col].value_counts().items():
            m=fit[col].eq(value); categorical.append(dict(feature=col,value=value,rows=n,distinct_original_patterns=link.loc[m,'feature_group_id'].nunique()))
    cat=table(s,pd.DataFrame(categorical),'categorical_support'); display(cat.groupby('feature',sort=False).head(12))
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,col in zip(axes,['proto','service','state']):
        v=cat[cat.feature.eq(col)].head(12); ax.bar(v.value.astype(str),v.rows); ax.set(title=col,ylabel='Fitting rows'); ax.tick_params(axis='x',rotation=70)
    figure(s,fig,'categorical_values')
    s['numeric_summary']=summary
    explain(s,f'Fitting has {len(fit):,} rows but {int(unique.sum()):,} distinct original patterns. Worms has 104 rows and 102 patterns; several numeric fields are mostly zero or highly skewed.',
        'Use fitting distributions to justify representation changes. Keep zeros, extremes and protocol markers unless a verified measurement error is established.',
        'A tail value can be valid network traffic. Histogram shape is not evidence that it should be clipped or deleted.')

def relationships(s):
    fit=s['fit']; link=s['links']['train']
    rows=[]
    for proto,g in fit.groupby('proto'):
        for col in ['dbytes','tcprtt','is_ftp_login','response_body_len']:
            rows.append(dict(proto=proto,feature=col,rows=len(g),zero_pct=100*g[col].eq(0).mean(),median=g[col].median()))
    display(table(s,pd.DataFrame(rows),'protocol_zero_context').query('proto in ["tcp","udp","icmp"]'))
    target=[]
    weights=1/link.groupby('feature_group_id').feature_group_id.transform('size')
    for c in CLASSES:
        m=fit.attack_cat.eq(c)
        for col in ['dur','sbytes','dbytes','sttl','ct_srv_src']:
            target.append(dict(attack_cat=c,feature=col,rows=int(m.sum()),distinct_original_patterns=link.loc[m,'feature_group_id'].nunique(),median=fit.loc[m,col].median(),zero_pct=100*fit.loc[m,col].eq(0).mean(),pattern_weighted_mean=np.average(fit.loc[m,col],weights=weights[m])))
    target=table(s,pd.DataFrame(target),'target_relationships'); display(target)
    fig,ax=plt.subplots(figsize=(10,4))
    columns=['dur','sbytes','dbytes','ct_srv_src']; vals=[]
    for col in columns: vals.append(np.log1p(fit.loc[fit.label.eq(0),col])); vals.append(np.log1p(fit.loc[fit.label.eq(1),col]))
    ax.boxplot(vals,showfliers=False,tick_labels=[col+' '+target for col in columns for target in ['Normal','Attack']]); ax.tick_params(axis='x',rotation=45)
    ax.set(ylabel='log(1 + original value); descriptive only',title='Completed-flow quantities differ by target'); figure(s,fig,'target_relationships')
    explain(s,'Protocol-specific zero frequencies differ, and class medians show associations between recorded measurements and outcomes.',
        'Preserve zero as a measured or extractor-dependent value. Use associations to describe the dataset, not to claim causal detection or discard features.',
        'Repeated patterns and capture-specific TTL/context relationships can inflate apparent relevance; prospective availability remains unverified.')

def preprocessing(s):
    fit=s['fit'][s['original']]; checks=[]; matrices={}; pipelines={}
    for rep in ['tree','scaled']:
        old=joblib.load(s['root']/f'models/preprocessing/03/03_{rep}_pipeline.joblib')
        new=clone(old).fit(fit); path=s['out']['models']/f'03_{rep}_pipeline.joblib'; joblib.dump(new,path); s['outputs'].append(path)
        reload=joblib.load(path); names=new.get_feature_names_out().tolist()
        assert names==read_json(s['root']/f'models/preprocessing/03/03_{rep}_feature_names.json')
        pipelines[rep]=new; matrices[rep]={}
        for role in ROLES:
            x=canonical_csr(new.transform(s['roles'][role][s['original']]))
            oldx=canonical_csr(sparse.load_npz(s['root']/f'data/processed/03/03_{rep}_{role}.npz'))
            assert x.shape==oldx.shape and (x!=oldx).nnz==0
            assert (x!=canonical_csr(reload.transform(s['roles'][role][s['original']]))).nnz==0
            out=s['out']['data']/f'03_{rep}_{role}.npz'; sparse.save_npz(out,x); s['outputs'].append(out)
            matrices[rep][role]=x
            checks.append(dict(representation=rep,role=role,rows=x.shape[0],columns=x.shape[1],finite=True,exactly_reproduces_prior=True,reload_matches=True))
        fp=s['out']['models']/f'03_{rep}_names.json'; write_json(fp,names); s['outputs'].append(fp)
    display(table(s,pd.DataFrame(checks),'transform_verification'))
    overlap=pd.concat([exact_roles(s,matrices[rep],rep) for rep in matrices],ignore_index=True)
    assert overlap.shared_vectors.eq(0).all(); display(table(s,overlap,'transformed_overlap'))
    s['pipelines']=pipelines; s['matrices']=matrices
    catpipe=pipelines['tree'].named_transformers_['cat']; encoder=catpipe if hasattr(catpipe,'categories_') else catpipe.named_steps[next(reversed(catpipe.named_steps))]
    unknown=[]
    for role in ROLES:
        for col,vocabulary in zip(['proto','service','state'],encoder.categories_):
            m=~s['roles'][role][col].isin(vocabulary)
            unknown.append(dict(role=role,feature=col,unknown_rows=int(m.sum()),unknown_values=' | '.join(map(str,sorted(s['roles'][role].loc[m,col].unique())))))
    display(table(s,pd.DataFrame(unknown),'unknown_categories'))
    example=s['roles']['train'][s['original']].iloc[:1].copy(); example.loc[:,'proto']='__unseen_protocol__'
    assert np.isfinite(pipelines['tree'].transform(example).data).all()
    fig,axes=plt.subplots(2,3,figsize=(12,6))
    scaled_names=pipelines['scaled'].get_feature_names_out().tolist()
    for j,col in enumerate(['sbytes','dur','sjit']):
        raw=fit[col].to_numpy(); ix=scaled_names.index('log__'+col); transformed=matrices['scaled']['train'][:,ix].toarray().ravel()
        # Central 99% displayed, but no training values are removed or clipped.
        axes[0,j].hist(raw,bins=50,range=(0,np.quantile(raw,.99))); axes[0,j].set(title=col+' raw (central 99%)',ylabel='Rows')
        axes[1,j].hist(transformed,bins=50); axes[1,j].set(title='log1p + train-fitted robust scaling',xlabel=col+' transformed',ylabel='Rows')
    figure(s,fig,'before_after')
    explain(s,'Fresh train-only fits exactly reproduce all six earlier matrices, feature order and saved/reloaded transformations. Transformed role pairs have zero exact overlap.',
        'Keep raw numeric values for trees and train-fitted log/robust scaling for linear or later neural models. Ignore unknown categories with documented all-zero one-hot blocks.',
        'Robust scaling centres by the fitting median and divides by its interquartile range; it does not eliminate outliers or create information. Some unknown categories have no learned category-specific effect.')

def engineering(s):
    proposals=proposal_table(); proposals['units']=['bytes','unitless','unitless','packets/second if dur is seconds','bytes/packet','bytes/packet']
    display(table(s,proposals,'proposals'))
    toy=pd.DataFrame(dict(sbytes=[100,100,0,0],dbytes=[100,0,100,0],spkts=[2,2,0,0],dpkts=[2,0,2,0],dur=[1,1,1,0]))
    display(table(s,pd.concat([toy,retained_features(toy)],axis=1),'worked_example'))
    cand=candidate_features(s['fit']); s['engineered']=cand
    rows=[]; link=s['links']['train']
    for col in cand:
        x=cand[col]; rows.append(dict(feature=col,rows=len(x),distinct_original_patterns=link.feature_group_id.nunique(),min=x.min(),median=x.median(),p99=x.quantile(.99),max=x.max(),zeros=int(x.eq(0).sum()),distinct_values=x.nunique()))
    display(table(s,pd.DataFrame(rows),'candidate_summary'))
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    axes[0].hist(np.log1p(cand.fe_total_bytes),bins=50); axes[0].set(xlabel='log(1 + total bytes)')
    for ax,col in zip(axes[1:],RETAINED[1:]): ax.hist(cand[col],bins=40); ax.set(xlabel=col+' (−1 to +1)')
    for ax in axes: ax.set_ylabel('Fitting rows')
    figure(s,fig,'engineered_distributions')
    unique=~link.feature_group_id.duplicated()
    corr=pd.DataFrame([dict(support='all fitting rows',rows=len(cand),spearman=cand.fe_byte_balance.corr(cand.fe_packet_balance,method='spearman')),
                       dict(support='one row per original pattern',rows=int(unique.sum()),spearman=cand.loc[unique,'fe_byte_balance'].corr(cand.loc[unique,'fe_packet_balance'],method='spearman'))])
    display(table(s,corr,'balance_redundancy'))
    fig,ax=plt.subplots(figsize=(6,5)); h=ax.hexbin(cand.fe_byte_balance,cand.fe_packet_balance,gridsize=35,bins='log',mincnt=1,cmap='viridis')
    ax.set(xlabel='Byte balance',ylabel='Packet balance',title='Related, but not identical, directional summaries'); fig.colorbar(h,ax=ax,label='Rows (log scale)'); figure(s,fig,'balance_relationship')
    relation=[]
    for c in CLASSES:
        mask=s['fit'].attack_cat.eq(c)
        for col in RETAINED:
            relation.append(dict(attack_cat=c,feature=col,rows=int(mask.sum()),distinct_original_patterns=link.loc[mask,'feature_group_id'].nunique(),median=cand.loc[mask,col].median(),zero_rows=int(cand.loc[mask,col].eq(0).sum())))
    display(table(s,pd.DataFrame(relation),'target_relationships'))
    checks=[]
    for col,original in [('fe_source_bytes_per_packet','smean'),('fe_destination_bytes_per_packet','dmean')]:
        diff=abs(cand[col]-s['fit'][original]); checks.append(dict(candidate=col,existing=original,exact_rows=int(diff.eq(0).sum()),within_half_unit_rows=int(diff.le(.5+1e-12).sum()),rows=len(diff)))
    rate_diff=abs(cand.fe_packets_per_second-s['fit'].rate)
    checks.append(dict(candidate='fe_packets_per_second',existing='rate',exact_rows=int(rate_diff.eq(0).sum()),within_half_unit_rows=int(rate_diff.le(.5).sum()),rows=len(rate_diff)))
    display(table(s,pd.DataFrame(checks),'existing_information'))
    decisions=proposals[['feature','formula','units']].copy(); decisions['decision']=decisions.feature.isin(RETAINED).map({True:'Retain for a model comparison',False:'Reject from this small candidate set'})
    decisions['reason']=['Volume as an explicit sum','Bounded direction ratio may help a linear representation','Packet direction may differ despite related byte balance','Existing rate field; extractor formula unresolved','Near smean up to rounding','Near dmean up to rounding']
    display(table(s,decisions,'decisions'))
    explain(s,f'Byte/packet balances correlate {corr.iloc[0].spearman:.4f} by row and {corr.iloc[1].spearman:.4f} by distinct original pattern. They are related, not exact copies.',
        'Retain total bytes and both balances as predeclared model candidates. Reject extra average-size/rate fields from this small set.',
        'These formulas add no new measured information. Only matched model experiments can show whether this representation helps; source/destination are not proven client/server.')

def engineering_verify(s):
    rows=[]; overlaps=[]
    for role in ROLES:
        expected=retained_features(s['roles'][role]); old=pd.read_csv(s['root']/f'data/processed/04/04_engineered_{role}.csv',float_precision='round_trip')
        # Saved engineered input contains only the three engineered feature columns.
        assert np.array_equal(expected.to_numpy(),old[RETAINED].to_numpy()),role
        path=s['out']['data']/f'04_engineered_{role}.csv'; expected.to_csv(path,index=False,float_format='%.17g'); s['outputs'].append(path)
        reload=pd.read_csv(path,float_precision='round_trip'); assert np.array_equal(expected.to_numpy(),reload.to_numpy())
        links=s['out']['data']/f'04_row_links_{role}.csv'; s['links'][role].to_csv(links,index=False); s['outputs'].append(links)
        rows.append(dict(role=role,rows=len(expected),columns=3,finite=bool(np.isfinite(expected.to_numpy()).all()),formulas_match_prior=True,reload_exact=True))
    display(table(s,pd.DataFrame(rows),'role_verification'))
    for rep in ['tree','scaled']:
        xs={r:canonical_csr(sparse.load_npz(s['root']/f'data/processed/04/04_{rep}_augmented_{r}.npz')) for r in ROLES}
        scaler=joblib.load(s['root']/'models/preprocessing/04/04_engineered_scaler.joblib')
        for role in ROLES:
            base=canonical_csr(sparse.load_npz(s['root']/f'data/processed/03/03_{rep}_{role}.npz'))
            values=retained_features(s['roles'][role]).to_numpy(copy=True)
            if rep=='scaled':
                # The saved named-column transformer includes log1p internally.
                values=scaler.transform(retained_features(s['roles'][role]))
            expected=canonical_csr(sparse.hstack([base,sparse.csr_matrix(values)],format='csr'))
            assert expected.shape==xs[role].shape and (expected!=xs[role]).nnz==0, (rep,role)
        overlaps.append(exact_roles(s,xs,rep+'_all45'))
    overlaps.append(exact_roles(s,{r:sparse.csr_matrix(retained_features(s['roles'][r]).to_numpy()) for r in ROLES},'engineered_three_alone_diagnostic'))
    display(table(s,pd.concat(overlaps,ignore_index=True),'representation_overlap'))
    explain(s,'The same three stateless formulas reproduce saved engineered values in every development role, with exact CSV round trips and frozen row links.',
        'Use all original features as the comparison baseline. Use previously train-fitted scaling of engineered columns, verified by the protected manifests.',
        'Calibration receives the fixed transformation for integrity only. No labels occur in formulas, and no validation outcome chooses a formula.')

def engineering_detail(s):
    cand=s['engineered']; fit=s['fit']; link=s['links']['train']
    denominator=[]
    for col,total in [('byte_balance',fit.sbytes+fit.dbytes),('packet_balance',fit.spkts+fit.dpkts)]:
        m=total.eq(0); denominator.append(dict(feature=col,zero_denominator_rows=int(m.sum()),zero_denominator_patterns=link.loc[m,'feature_group_id'].nunique()))
    display(table(s,pd.DataFrame(denominator),'zero_denominators'))
    extremes=[]
    for col in RETAINED:
        for ix in cand[col].nlargest(3).index:
            extremes.append(dict(feature=col,value=cand.loc[ix,col],row_position=int(link.loc[ix,'row_position']),attack_cat=fit.loc[ix,'attack_cat'],**fit.loc[ix,['sbytes','dbytes','spkts','dpkts']].to_dict()))
    display(table(s,pd.DataFrame(extremes),'extreme_examples'))
    show_prior(s,'04_redundancy_checks_fit')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for c in CLASSES:
        mask=fit.attack_cat.eq(c)
        axes[0].scatter(np.log1p(fit.loc[mask,'sbytes'].median()),np.log1p(fit.loc[mask,'dbytes'].median()),label=c,s=35)
        axes[1].scatter(cand.loc[mask,'fe_byte_balance'].median(),cand.loc[mask,'fe_packet_balance'].median(),label=c,s=35)
    axes[0].set(xlabel='log(1 + median source bytes)',ylabel='log(1 + median destination bytes)',title='Original counters by class')
    axes[1].set(xlabel='Median byte balance',ylabel='Median packet balance',title='Engineered directional summaries'); axes[1].legend(bbox_to_anchor=(1.02,1),fontsize=8)
    figure(s,fig,'original_and_engineered')
    explain(s,'No fitting row has a zero total-byte or total-packet denominator. The zero rule is still tested with toy flows. Extreme examples expose which source counts create the totals; rate alternatives closely track existing rate but differ in formula.',
        'Keep the transparent three-feature contract and the negative redundancy evidence. Do not clip high byte counts or call a rounded average a new measurement.',
        'Class medians summarize heterogeneous traffic. A high correlation, central trend or extreme example cannot predict a model benefit; the engineered-three-only overlap diagnostic discards original information and is not a model input set.')

def selection_evidence(s):
    raw=s['fit'][s['original']].copy(); engineered=retained_features(s['fit']); full=pd.concat([raw,engineered],axis=1); link=s['links']['train']
    copies=[]
    numeric=full.select_dtypes('number'); corr=numeric.corr(method='spearman')
    for a,b in itertools.combinations(numeric.columns,2):
        if np.array_equal(numeric[a],numeric[b]): copies.append(dict(feature_a=a,feature_b=b,rows=len(raw),distinct_original_patterns=link.feature_group_id.nunique()))
    display(table(s,pd.DataFrame(copies),'exact_copies'))
    pairs=pd.DataFrame([dict(feature_a=a,feature_b=b,spearman=corr.loc[a,b]) for a,b in itertools.combinations(numeric.columns,2) if abs(corr.loc[a,b])>=.95])
    display(table(s,pairs,'near_redundancy'))
    variation=[]
    for col in full:
        counts=full[col].value_counts(); dominant=counts.iloc[0]/len(full)
        variation.append(dict(feature=col,distinct_values=full[col].nunique(),dominant_pct=100*dominant,minority_rows=int(len(full)-counts.iloc[0]),minority_original_patterns=link.loc[~full[col].eq(counts.index[0]),'feature_group_id'].nunique(),flag_99_5_pct=bool(dominant>=.995)))
    variation=table(s,pd.DataFrame(variation),'variation'); display(variation[variation.flag_99_5_pct])
    fig,ax=plt.subplots(figsize=(7,5)); cols=['sbytes','dbytes','spkts','dpkts']+RETAINED
    m=corr.loc[cols,cols]; im=ax.imshow(m,cmap='coolwarm',vmin=-1,vmax=1); ax.set_xticks(range(len(cols)),cols,rotation=70); ax.set_yticks(range(len(cols)),cols); fig.colorbar(im,ax=ax,label='Spearman correlation'); figure(s,fig,'redundancy')
    explain(s,'FTP fields are exact copies in fitting rows. Several low-variation fields still have nonzero minority pattern support. Balance correlation repeats the engineering finding.',
        'Do not drop original predictors on these descriptive tests. Preserve both FTP fields because intended meanings differ and prior audit found supplied-data disagreement.',
        'High correlation need not mean interchangeable predictive use. Low variation can conceal meaningful rare protocol behaviour.')

def association_stability(s):
    folds=pd.read_csv(s['root']/'data/processed/05/05_internal_folds.csv'); link=s['links']['train']
    assert np.array_equal(folds.row_position,link.row_position) and np.array_equal(folds.attack_cat,link.attack_cat)
    assert folds.groupby('feature_group_id').internal_holdout_fold.nunique().max()==1
    full=pd.concat([s['fit'][s['original']],retained_features(s['fit'])],axis=1); outputs=[]
    support=[]
    for fold in range(3):
        hold=folds.internal_holdout_fold.eq(fold).to_numpy(); train=~hold
        assert not set(folds.loc[hold,'feature_group_id'])&set(folds.loc[train,'feature_group_id'])
        for c in CLASSES:
            mask=hold&folds.attack_cat.eq(c).to_numpy(); support.append(dict(fold=fold,attack_cat=c,rows=int(mask.sum()),original_patterns=folds.loc[mask,'feature_group_id'].nunique()))
        weights=1/folds.loc[hold].groupby('feature_group_id').feature_group_id.transform('size').to_numpy()
        for col in full:
            levels=full.loc[hold,col].to_numpy() if col in ['proto','service','state'] else apply_bins(full.loc[hold,col],fit_bin_edges(full.loc[train,col]))
            for target in ['attack_cat','label']:
                y=folds.loc[hold,target]
                outputs.append(dict(fold=fold,feature=col,target=target,holdout_rows=int(hold.sum()),holdout_original_patterns=folds.loc[hold,'feature_group_id'].nunique(),observed_feature_levels=len(np.unique(levels)),row_association=association_score(levels,y),pattern_weighted_association=association_score(levels,y,weights)))
    score=table(s,pd.DataFrame(outputs),'association_folds'); display(table(s,pd.DataFrame(support),'internal_fold_support').query('attack_cat == "Worms"'))
    prior=pd.read_csv(s['root']/'results/tables/05_association_by_internal_fold.csv')
    for col in ['row_association','pattern_weighted_association']:
        assert np.allclose(score[col],prior[col],equal_nan=True,atol=1e-12)
    summary=score.groupby(['target','feature'],sort=False).agg(mean_row=('row_association','mean'),mean_pattern_weighted=('pattern_weighted_association','mean'),min_pattern_weighted=('pattern_weighted_association','min'),max_pattern_weighted=('pattern_weighted_association','max'),defined_folds=('pattern_weighted_association','count')).reset_index()
    display(table(s,summary,'association_summary').query('target == "attack_cat"').sort_values('mean_pattern_weighted',ascending=False).head(15))
    fig,ax=plt.subplots(figsize=(9,5)); v=summary[summary.target.eq('attack_cat')].nlargest(12,'mean_pattern_weighted').sort_values('mean_pattern_weighted')
    ax.barh(v.feature,v.mean_pattern_weighted); ax.errorbar(v.mean_pattern_weighted,v.feature,xerr=[v.mean_pattern_weighted-v.min_pattern_weighted,v.max_pattern_weighted-v.mean_pattern_weighted],fmt='none',ecolor='black',capsize=3)
    ax.set(xlabel="Pattern-weighted Cramér's V (descriptive)",title='Group-held-out association; error bars are fold range'); figure(s,fig,'association')
    explain(s,'Three group-held-out association runs reproduce the earlier evidence. Numeric quantile edges are fitted on each inner-training fold; one pattern has total weight one in the weighted diagnostic.',
        'Use this as descriptive stability evidence only. Keep all five engineered ablations and all 42 original predictors for fair model comparisons.',
        'Cramér’s V measures association, not model improvement. Fold ranges are not confidence intervals. Rare classes, repeated patterns and conflicting targets limit rankings; constant FTP bins have undefined scores, not zero relevance.')

def shortlist(s):
    definitions=read_json(s['root']/'models/preprocessing/05/05_comparison_sets.json')['sets']
    rows=[dict(feature_set=k,source_count=len(definitions[k]['retained']),retained=' | '.join(definitions[k]['retained']),omitted=' | '.join(definitions[k]['omitted']),question=definitions[k]['reason']) for k in SETS]
    display(table(s,pd.DataFrame(rows),'shortlist'))
    path=s['out']['models']/'05_feature_sets.json'; write_json(path,definitions); s['outputs'].append(path)
    shapes,overlaps=verify_candidates(s); display(shapes); display(overlaps)
    assert overlaps.shared_vectors.eq(0).all()
    # Index the verified existing inputs rather than overwrite/duplicate hundreds of MB.
    index=[]
    for role in ROLES:
        for candidate in SETS:
            for rep,ext in [('raw','csv'),('tree','npz'),('scaled','npz')]:
                index.append(dict(role=role,feature_set=candidate,representation=rep,path=f'data/processed/05/05_{candidate}_{rep}_{role}.{ext}',row_links=f'data/processed/05/05_row_links_{role}.csv'))
    table(s,pd.DataFrame(index),'verified_input_index')
    explain(s,'All five source-name definitions keep whole categorical groups. Every saved role-linked input matches its protected predecessor and has zero cross-role exact matches.',
        'This notebook shortlists representation variants; it does not identify a winning subset. Each variant answers volume, byte direction, packet direction or joint contribution.',
        'No predictive selector is fitted. Removing columns in future may create new identical rows; audit and report any new matches without silently redrawing roles.')
