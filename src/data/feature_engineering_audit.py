"""Notebook 04 evidence and serialization helpers; no predictive model fitting."""

from itertools import combinations
import importlib.metadata
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler

from src.data.load_data import sha256
from src.data.duplicate_investigation import exact_feature_ids
from src.data.feature_engineering import (RETAINED, candidate_features, retained_features,
    proposal_table, safe_ratio, canonical_csr, sparse_row_keys, overlap_counts)

ROLES = ('train', 'model_selection', 'calibration')


def protected_files(root):
    """Earlier inputs, utilities and deliverables, excluding every 04 output."""
    files = []
    for directory in ('data/raw', 'data/processed/03', 'models/preprocessing/03',
                      'results/splits', 'results/tables', 'results/figures', 'docs', 'configs', 'src', 'notebooks'):
        for p in (root / directory).rglob('*'):
            if (p.is_file() and '__pycache__' not in p.parts and not p.name.startswith('04_')
                    and p.name not in ('feature_engineering.py', 'feature_engineering_audit.py')):
                files.append(p)
    return sorted(files)


def starting_point(root):
    root = Path(root)
    protected = {p.relative_to(root).as_posix(): sha256(p) for p in protected_files(root)}
    m2 = json.loads((root/'results/splits/02_split_manifest.json').read_text())
    m3 = json.loads((root/'data/processed/03/03_artifact_manifest.json').read_text())
    assert m2['source_sha256'] == m3['source_sha256']
    assert m2['assignment_sha256'] == m3['assignment_sha256']
    # Hash-only integrity check: never parse, transform or inspect official test records.
    for role, name in [('train', 'training'), ('test', 'testing')]:
        assert sha256(root/f'data/raw/UNSW_NB15_{name}-set.csv') == m2['source_sha256'][role]
    assignment_path = root/'results/splits/02_train_validation_assignments.csv'
    assert sha256(assignment_path) == m2['assignment_sha256']
    for relative, expected in m3['artifacts_sha256'].items():
        assert sha256(root / relative.replace('\\', '/')) == expected, relative
    source = pd.read_csv(root/'data/raw/UNSW_NB15_training-set.csv', low_memory=False)
    assign = pd.read_csv(assignment_path)
    assert len(source) == len(assign) == 175341
    assert np.array_equal(assign.row_position, np.arange(len(source)))
    for col, meta in [('id','source_id'), ('label','label'), ('attack_cat','attack_cat')]:
        assert source[col].equals(assign[meta]), col
    assert set(assign.role) == set(ROLES)
    assert assign.groupby('feature_group_id').role.nunique().max() == 1
    predictors = m2['predictors']
    assert predictors == m3['predictors'] and len(predictors) == 42
    assert set(source.columns) - set(predictors) == {'id','label','attack_cat'}
    ids, _ = exact_feature_ids(source, predictors)
    assert np.array_equal(ids, assign.feature_group_id), 'Frozen group definitions changed'
    positions = {r: np.flatnonzero(assign.role.eq(r)) for r in ROLES}
    assert {r:len(ix) for r,ix in positions.items()} == m3['roles']
    state = dict(root=root, m2=m2, m3=m3, protected=protected, source=source, assign=assign,
                 predictors=predictors, positions=positions)
    for folder in ('results/tables','results/figures','data/processed/04','models/preprocessing/04'):
        (root/folder).mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame([{'role':r, 'rows':len(ix),
        'distinct_feature_groups':assign.feature_group_id.iloc[ix].nunique()} for r,ix in positions.items()])
    save_table(state, summary, 'role_support')
    (root/'results/tables/04_starting_integrity.json').write_text(json.dumps({
        'source_sha256':m2['source_sha256'], 'assignment_sha256':m2['assignment_sha256'],
        'manifest03_sha256':sha256(root/'data/processed/03/03_artifact_manifest.json'),
        'verified_manifest03_artifacts':len(m3['artifacts_sha256']), 'protected_before':protected,
        'test_access':'SHA-256 bytes only; no official test CSV parsing or feature access'}, indent=2), encoding='utf-8')
    return state, summary


def save_table(state, frame, name):
    frame.to_csv(state['root']/f'results/tables/04_{name}.csv', index=False)
    return frame


def save_figure(state, fig, name):
    fig.tight_layout()
    fig.savefig(state['root']/f'results/figures/04_{name}.png', dpi=150, bbox_inches='tight')
    plt.show(); plt.close(fig)


def proposals(state):
    return save_table(state, proposal_table(), 'feature_proposals')


def fitting_investigation(state):
    ix = state['positions']['train']
    fit = state['source'].iloc[ix]
    groups = state['assign'].feature_group_id.iloc[ix]
    candidates = candidate_features(fit[state['predictors']])
    state.update(fit=fit, fit_groups=groups, candidates=candidates)
    denominators = {
        'fe_total_bytes':None, 'fe_byte_balance':fit.sbytes+fit.dbytes,
        'fe_packet_balance':fit.spkts+fit.dpkts, 'fe_packets_per_second':fit.dur,
        'fe_source_bytes_per_packet':fit.spkts, 'fe_destination_bytes_per_packet':fit.dpkts}
    rows, extremes = [], []
    for col in candidates:
        v = candidates[col]; zero = v.eq(0)
        d = denominators[col]; absent = pd.Series(False, index=fit.index) if d is None else d.eq(0)
        q = v.quantile([.01,.5,.95,.99,.999])
        rows.append(dict(feature=col, rows=len(v), distinct_feature_groups=groups.nunique(),
            unique_values=v.nunique(), zero_rows=int(zero.sum()), zero_groups=groups[zero].nunique(),
            denominator_zero_rows=int(absent.sum()), denominator_zero_groups=groups[absent].nunique(),
            min=v.min(), p01=q.loc[.01], median=q.loc[.5], p95=q.loc[.95], p99=q.loc[.99],
            p999=q.loc[.999], max=v.max(), skew=v.skew(),
            above_p99_rows=int(v.gt(q.loc[.99]).sum()), above_p99_groups=groups[v.gt(q.loc[.99])].nunique()))
        # Representative extremes from distinct original groups, not repeated copies.
        ordered = v.sort_values(ascending=False,kind='stable').index
        representative = groups.loc[ordered].drop_duplicates().head(3).index
        for pos in representative:
            extremes.append(dict(feature=col, value=v.loc[pos], row_position=int(pos),
                source_id=int(fit.loc[pos,'id']), feature_group_id=int(groups.loc[pos]),
                label=int(fit.loc[pos,'label']), attack_cat=fit.loc[pos,'attack_cat'],
                **{k:float(fit.loc[pos,k]) for k in ['sbytes','dbytes','spkts','dpkts','dur']}))
    stats = save_table(state, pd.DataFrame(rows), 'candidate_fit_summary')
    save_table(state, pd.DataFrame(extremes), 'candidate_extreme_examples')
    # Tolerances are stated in advance and diagnostic, never used to merge vectors.
    comparisons = [('fe_packets_per_second','rate',fit.dur.gt(0),1e-8),
                   ('fe_source_bytes_per_packet','smean',fit.spkts.gt(0),.5),
                   ('fe_destination_bytes_per_packet','dmean',fit.dpkts.gt(0),.5)]
    redundancy = []
    for new, old, valid, atol in comparisons:
        a, b = candidates.loc[valid,new], fit.loc[valid,old].astype(float)
        close = np.isclose(a,b,rtol=1e-6,atol=atol)
        redundancy.append(dict(candidate=new, comparator=old, valid_rows=int(valid.sum()),
            valid_groups=groups[valid].nunique(), exact_equal_rows=int(a.eq(b).sum()),
            close_rows=int(close.sum()), close_fraction=float(close.mean()), atol=atol, rtol=1e-6,
            median_absolute_difference=float((a-b).abs().median()), spearman=float(a.corr(b,method='spearman'))))
    valid = fit.dur.gt(0)
    alt = pd.Series(safe_ratio(fit.spkts+fit.dpkts-1,fit.dur),index=fit.index)
    close = np.isclose(alt[valid],fit.loc[valid,'rate'],rtol=1e-6,atol=1e-8)
    redundancy.append(dict(candidate='diagnostic: (spkts+dpkts-1)/dur',comparator='rate',
        valid_rows=int(valid.sum()), valid_groups=groups[valid].nunique(),
        exact_equal_rows=int(alt[valid].eq(fit.loc[valid,'rate']).sum()),close_rows=int(close.sum()),
        close_fraction=float(close.mean()),atol=1e-8,rtol=1e-6,
        median_absolute_difference=float((alt[valid]-fit.loc[valid,'rate']).abs().median()),
        spearman=float(alt[valid].corr(fit.loc[valid,'rate'],method='spearman'))))
    redundancy = save_table(state,pd.DataFrame(redundancy),'redundancy_checks_fit')
    # Check all candidates against all 39 existing numerical columns.
    equalities = []; correlations = []
    for new in candidates:
        for old in fit[state['predictors']].select_dtypes('number'):
            equalities.append(dict(candidate=new, original=old,
                exact_equal_rows=int(candidates[new].eq(fit[old]).sum()),
                exact_copy=bool(candidates[new].eq(fit[old]).all())))
            correlations.append(dict(candidate=new, original=old,
                spearman=float(candidates[new].corr(fit[old],method='spearman'))))
    save_table(state,pd.DataFrame(equalities),'exact_column_redundancy_fit')
    save_table(state,pd.DataFrame(correlations),'candidate_original_correlations_fit')
    candidate_corr = candidates.corr(method='spearman')
    save_table(state,candidate_corr.rename_axis('feature').reset_index(),'candidate_correlations_fit')
    support = []; rel = []
    weights = groups.map(groups.value_counts()).rdiv(1).to_numpy()
    for target in ['label','attack_cat']:
        for cls in sorted(fit[target].unique(),key=str):
            mask = fit[target].eq(cls)
            support.append(dict(target=target, target_class=str(cls), rows=int(mask.sum()),
                distinct_feature_groups=groups[mask].nunique()))
            for col in candidates:
                v = candidates.loc[mask,col]; zero=v.eq(0)
                rel.append(dict(target=target,target_class=str(cls),feature=col,rows=len(v),
                    distinct_feature_groups=groups[mask].nunique(), zero_rows=int(zero.sum()),
                    zero_feature_groups=groups.loc[v.index[zero]].nunique(), median=float(v.median()),
                    p25=float(v.quantile(.25)),p75=float(v.quantile(.75)),mean=float(v.mean()),
                    mean_inverse_group_size_weighted=float(np.average(v,weights=weights[mask.to_numpy()]))))
    save_table(state,pd.DataFrame(support),'fit_target_support')
    relationships=save_table(state,pd.DataFrame(rel),'candidate_target_relationships_fit')
    state.update(stats=stats,redundancy=redundancy,relationships=relationships,candidate_corr=candidate_corr)
    return stats,redundancy,pd.DataFrame(support)


def fitting_figures(state):
    c = state['candidates']; g = state['fit_groups']
    unique = ~g.duplicated()
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for ax,col in zip(axes.flat,c):
        values = c[col].to_numpy()
        display_values = values if 'balance' in col else np.log1p(values)
        bins=np.linspace(-1,1,41) if 'balance' in col else 40
        ax.hist(display_values,bins=bins,density=True,alpha=.5,label='All fitting rows')
        ax.hist(display_values[unique.to_numpy()],bins=bins,density=True,histtype='step',linewidth=1.5,label='One per exact group')
        ax.set(title=col.removeprefix('fe_').replace('_',' '),ylabel='Density',
               xlabel='Signed balance' if 'balance' in col else 'log1p(value), display only')
        ax.set_yscale('symlog',linthresh=.1)
    axes.flat[0].legend(fontsize=8)
    save_figure(state,fig,'candidate_distributions_fit')
    rel=state['relationships']; classes=state['m2']['class_vocabulary']
    fig,axes=plt.subplots(1,3,figsize=(14,5))
    for ax,col in zip(axes,RETAINED):
        d=rel[(rel.target=='attack_cat')&(rel.feature==col)].set_index('target_class').reindex(classes)
        ax.plot(d['median'],np.arange(len(classes)),'o',color='#527b94')
        ax.hlines(np.arange(len(classes)),d.p25,d.p75,color='#527b94')
        ax.set_yticks(np.arange(len(classes)),classes,fontsize=8)
        ax.set(title=col.removeprefix('fe_').replace('_',' '),xlabel='Median and interquartile range')
        if col=='fe_total_bytes':
            ax.set_xscale('log')
            ax.set_xlabel('Median and interquartile range (bytes, log scale)')
    save_figure(state,fig,'retained_class_distributions_fit')
    fig,ax=plt.subplots(figsize=(8,6))
    corr=state['candidate_corr']; im=ax.imshow(corr,vmin=-1,vmax=1,cmap='coolwarm')
    short=[x.removeprefix('fe_').replace('_',' ') for x in corr.columns]
    ax.set_xticks(range(len(short)),short,rotation=45,ha='right',fontsize=8)
    ax.set_yticks(range(len(short)),short,fontsize=8)
    for i in range(len(short)):
        for j in range(len(short)): ax.text(j,i,f'{corr.iloc[i,j]:.2f}',ha='center',va='center',fontsize=8)
    ax.set_title('Fitting-only candidate relationships')
    fig.colorbar(im,ax=ax,label='Spearman correlation')
    save_figure(state,fig,'candidate_correlations_fit')


def decisions(state):
    r=state['redundancy'].set_index('candidate')
    reasons={
        'fe_total_bytes':'Retain candidate: compact bidirectional volume; useful sum to test, fully derived from bytes.',
        'fe_byte_balance':'Retain candidate: bounded directional byte asymmetry, no duration division or learned constants.',
        'fe_packet_balance':'Retain candidate: bounded directional packet asymmetry; correlation with byte balance is not grounds for automatic removal.',
        'fe_packets_per_second':f"Reject from handoff: rate-like information already exists. Total/duration is not identical to rate; alternative (total-1)/duration is close on {r.loc['diagnostic: (spkts+dpkts-1)/dur','close_fraction']:.2%} of positive-duration fitting rows. Short/zero duration complicates meaning.",
        'fe_source_bytes_per_packet':f"Reject from handoff: within 0.5 bytes (rtol 1e-6) of smean on {r.loc['fe_source_bytes_per_packet','close_fraction']:.2%} of positive-source-packet fitting rows; likely existing rounded packet-size information.",
        'fe_destination_bytes_per_packet':f"Reject from handoff: within 0.5 bytes (rtol 1e-6) of dmean on {r.loc['fe_destination_bytes_per_packet','close_fraction']:.2%} of positive-destination-packet fitting rows; likely existing rounded packet-size information.",
    }
    frame=proposal_table()[['feature','formula']].copy()
    frame['decision']=frame.feature.map(lambda x:'retain candidate' if x in RETAINED else 'reject candidate')
    frame['reason']=frame.feature.map(reasons)
    frame['decision_basis']='Semantic simplicity and fitting-only redundancy/validity audit; not target ranking or model feature selection'
    state['decisions']=save_table(state,frame,'feature_decisions')
    config=dict(version='04-v1',retained=RETAINED,formulas=frame[frame.feature.isin(RETAINED)][['feature','formula']].to_dict('records'),
                zero_policy='0 sentinel on zero denominators; original counters preserved',
                base_predictors=state['predictors'],source_sha256=state['m2']['source_sha256']['train'],
                assignment_sha256=state['m2']['assignment_sha256'],fit_role='train',
                tree_append='raw retained features',scaled_append='log1p(total bytes) and raw balances, then train-only RobustScaler',
                target_inputs=False,test_csv_parsed=False)
    (state['root']/'models/preprocessing/04/04_feature_contract.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    return frame


def apply_development_roles(state):
    root=state['root']; out=root/'data/processed/04'; models=root/'models/preprocessing/04'
    scaler=ColumnTransformer([
        ('volume',Pipeline([('log',FunctionTransformer(np.log1p,feature_names_out='one-to-one')),
                            ('scale',RobustScaler())]),['fe_total_bytes']),
        ('balance',RobustScaler(),['fe_byte_balance','fe_packet_balance'])],verbose_feature_names_out=False)
    scaler.fit(state['candidates'][RETAINED])
    joblib.dump(scaler,models/'04_engineered_scaler.joblib')
    restored=joblib.load(models/'04_engineered_scaler.joblib')
    assert restored.get_feature_names_out().tolist()==RETAINED
    role_keys={k:{} for k in ('raw42','raw45','engineered_only_diagnostic','tree_augmented','scaled_augmented')}
    role_records=[]; check_rows=[]
    base_names={kind:json.loads((root/f'models/preprocessing/03/03_{kind}_feature_names.json').read_text())
                for kind in ('tree','scaled')}
    for kind in base_names:
        names=base_names[kind]+[f'engineered__{c}' for c in RETAINED]
        assert len(names)==len(set(names))
        (models/f'04_{kind}_augmented_feature_names.json').write_text(json.dumps(names,indent=2),encoding='utf-8')
    for role,ix in state['positions'].items():
        original=state['source'].iloc[ix][state['predictors']]
        e=retained_features(original)
        raw=original.copy()
        for col in RETAINED: raw[col]=e[col]
        assert raw.columns.tolist()==state['predictors']+RETAINED
        assert raw.index.equals(original.index) and raw[state['predictors']].equals(original)
        assert np.isfinite(raw.select_dtypes('number').to_numpy()).all()
        raw.to_csv(out/f'04_raw_augmented_{role}.csv',index=False,float_format='%.17g')
        e.to_csv(out/f'04_engineered_{role}.csv',index=False,float_format='%.17g')
        engineered_dtypes={col:np.float64 for col in RETAINED}
        raw_reload=pd.read_csv(out/f'04_raw_augmented_{role}.csv',float_precision='round_trip',dtype=engineered_dtypes)
        pd.testing.assert_frame_equal(raw.reset_index(drop=True),raw_reload,check_dtype=False,check_exact=True)
        reloaded=pd.read_csv(out/f'04_engineered_{role}.csv',float_precision='round_trip',dtype=engineered_dtypes)
        pd.testing.assert_frame_equal(e.reset_index(drop=True),reloaded,check_exact=True)
        linked=state['assign'].iloc[ix][['row_position','source_id','feature_group_id','role','label','attack_cat']].copy()
        linked.insert(0,'input_row',np.arange(len(ix)))
        y3=pd.read_csv(root/f'data/processed/03/03_targets_{role}.csv')
        assert np.array_equal(linked.row_position,y3.row_position)
        assert np.array_equal(linked.label,y3.label) and np.array_equal(linked.attack_cat,y3.attack_cat)
        linked.to_csv(out/f'04_row_links_{role}.csv',index=False)
        pd.testing.assert_frame_equal(linked.reset_index(drop=True),pd.read_csv(out/f'04_row_links_{role}.csv'),check_exact=True)
        role_keys['raw42'][role]=list(original.itertuples(index=False,name=None))
        role_keys['raw45'][role]=list(raw.itertuples(index=False,name=None))
        role_keys['engineered_only_diagnostic'][role]=list(e.itertuples(index=False,name=None))
        for kind in ('tree','scaled'):
            base=canonical_csr(sparse.load_npz(root/f'data/processed/03/03_{kind}_{role}.npz'))
            appended=e.to_numpy() if kind=='tree' else scaler.transform(e)
            if kind=='scaled': np.testing.assert_array_equal(appended,restored.transform(e))
            combined=canonical_csr(sparse.hstack([base,sparse.csr_matrix(appended)],format='csr'))
            assert combined.shape==(len(ix),len(base_names[kind])+3)
            assert (combined[:,:base.shape[1]] != base).nnz==0
            path=out/f'04_{kind}_augmented_{role}.npz'
            sparse.save_npz(path,combined,compressed=True)
            again=canonical_csr(sparse.load_npz(path))
            assert (again!=combined).nnz==0
            role_keys[f'{kind}_augmented'][role]=sparse_row_keys(again)
            role_records.append(dict(role=role,representation=f'{kind}_augmented',rows=combined.shape[0],
                                     columns=combined.shape[1],saved_reload_exact=True))
        check_rows.append(dict(role=role,rows=len(ix),original_columns=42,engineered_columns=3,
            original_values_preserved=True,row_order_labels_preserved=True,finite=True,csv_reload_exact=True))
    save_table(state,pd.DataFrame(check_rows),'role_verification')
    overlap=[]
    for representation,keys in role_keys.items():
        for a,b in combinations(ROLES,2):
            counts=overlap_counts(keys[a],keys[b])
            overlap.append(dict(representation=representation,role_a=a,role_b=b,**counts,
                new_shared_vs_raw42=counts['shared_vectors'],
                purpose='lossy diagnostic only, not a model input baseline' if representation=='engineered_only_diagnostic' else 'full comparison input'))
    overlap=pd.DataFrame(overlap)
    assert overlap[overlap.representation.eq('raw42')].shared_vectors.eq(0).all()
    # Do not assert engineered representations are disjoint: report matches without modifying roles.
    save_table(state,overlap,'representation_overlap')
    state.update(overlap=overlap,role_records=role_records)
    return save_table(state,pd.DataFrame(role_records),'representation_shapes'),overlap


def finish(state):
    root=state['root']
    assert {p:sha256(root/p) for p in state['protected']}==state['protected']
    # Detect removal/addition under protected collections, too.
    assert {p.relative_to(root).as_posix() for p in protected_files(root)}==set(state['protected'])
    newmatches=state['overlap'][(state['overlap'].representation!='engineered_only_diagnostic') &
                               (state['overlap'].representation!='raw42')].shared_vectors.sum()
    decision_lines='\n'.join(f"- `{r.feature}`: {r.reason}" for r in state['decisions'].itertuples())
    report=f'''# Notebook 04 — Feature Engineering

## What we created

We added three columns to the original 42 feature columns:

| New column | Formula | Meaning |
|---|---|---|
| fe_total_bytes | sbytes + dbytes | Total bytes in both directions |
| fe_byte_balance | (sbytes - dbytes) / (sbytes + dbytes) | Which direction carried more bytes |
| fe_packet_balance | (spkts - dpkts) / (spkts + dpkts) | Which direction carried more packets |

For either balance, +1 means entirely forward, -1 entirely back, and 0 equal amounts. We store 0 if the total is zero and keep the original counts to tell no traffic from equal traffic. For example, 900 bytes forward and 100 back give 1,000 total bytes and a byte balance of 0.8. Six packets forward and four back give a packet balance of 0.2.

The notebook shows these calculations directly in its code. The same formulas are also saved in src/data/feature_engineering.py and checked against the notebook calculations.

## What the training data shows

We studied only the 140,273 fitting rows, with 80,848 different original feature patterns. Repeated patterns are not proof of independent network events. The middle total-byte value is {state['stats'].set_index('feature').loc['fe_total_bytes','median']:,.0f} bytes; the largest is {state['stats'].set_index('feature').loc['fe_total_bytes','max']:,.0f}. Neither balance has a zero total in these fitting rows.

The two balances are closely related: Spearman correlation is {state['candidate_corr'].loc['fe_byte_balance','fe_packet_balance']:.4f}. We keep both as candidates, but the next notebook should check whether both are worth using. Label summaries describe differences; labels never enter a formula, and these differences do not prove better detection.

We also checked packet rate and average bytes per packet. We left them out: average sizes closely match smean and dmean, while rate already describes similar information and its exact calculation remains uncertain. Full numbers and reasons are in results/tables/04_feature_decisions.csv and 04_redundancy_checks_fit.csv. No original feature was removed.

## What we saved and checked

Each saved row set now has 45 raw feature columns: train (140,273 rows), model_selection (17,534) and calibration (17,534). Inputs are in data/processed/04/04_raw_augmented_*.csv; matching row positions and original labels are in 04_row_links_*.csv. Read the CSVs with float_precision='round_trip' and float64 for the three new columns to preserve their exact saved values.

We also appended the new features to the saved Notebook 03 inputs. The old columns remain exactly the same. For its scaled version, total bytes uses log1p, and a separate RobustScaler is fitted only on train. The saved original inputs and raw 45-column inputs remain comparison options.

Saved values, row order, labels, finite values and file hashes passed the checks. All earlier notebooks, raw CSVs, frozen assignments and earlier outputs remain unchanged. Exact matches between the full augmented row sets: {int(newmatches)} summed across representation/row-set pairs. Matches using only the three new columns are reported separately; those columns alone discard most original information. No split was changed. The official test CSV was not loaded; it was only hashed to check preservation.

## Limits and next step

These are summaries of existing measurements, not new information. They can be calculated after a flow is complete if its counters were recorded correctly. They do not prove early detection, botnet-specific labels or independent network context. Source/destination do not necessarily mean client/server. Feature meanings follow the dataset creators' paper, [Tables III and IV](https://doi.org/10.1109/MilCIS.2015.7348942), inspected through [the original paper mirror](https://studylib.net/doc/26070414/07348942). Exact extractor details remain unverified.

Notebook 05 should compare which original and new features are worth using, followed by baseline models. Keep the original 42-column and Notebook 03 inputs as comparisons. Use the fitting rows for learned changes, model_selection for later selection, calibration for later confidence adjustment and the official test only for final evaluation. Check overlap again if columns are removed. No models, feature selection, resampling or confidence calibration were performed here.

## File hashes

- Training CSV: {state['m2']['source_sha256']['train']}
- Frozen assignments: {state['m2']['assignment_sha256']}
- Notebook 03 output list: {sha256(root/'data/processed/03/03_artifact_manifest.json')}

All output hashes are saved in data/processed/04/04_artifact_manifest.json. Detailed checks are kept in results/tables/04_* so the notebook can focus on creating the features.
'''
    report_path=root/'docs/04_feature_engineering_findings.md'
    report_path.write_text(report,encoding='utf-8')
    verification=dict(protected_unchanged=True,protected_files=len(state['protected']),
        protected_before=state['protected'],protected_after=state['protected'],
        official_test_access='hash only',all_csv_and_sparse_reload_checks_pass=True,
        train_only_scaler=True,original_baselines_preserved=True,
        source_sha256=state['m2']['source_sha256'],assignment_sha256=state['m2']['assignment_sha256'])
    (root/'results/tables/04_integrity_verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    saved=[*root.glob('results/tables/04_*'),*root.glob('results/figures/04_*'),
           *root.glob('data/processed/04/04_*'),*root.glob('models/preprocessing/04/04_*'),report_path,
           root/'src/data/feature_engineering.py',root/'src/data/feature_engineering_audit.py',
           root/'scripts/build_feature_engineering_notebook.py',root/'scripts/run_feature_engineering.py',
           root/'tests/test_feature_engineering.py']
    saved=[p for p in saved if p.name not in ('04_artifact_manifest.json','04_notebook_execution.json','04_independent_verification.json')]
    manifest=dict(protocol='04-v1',source_train_sha256=state['m2']['source_sha256']['train'],
        assignment_sha256=state['m2']['assignment_sha256'],
        source_manifest03_sha256=sha256(root/'data/processed/03/03_artifact_manifest.json'),
        retained_features=RETAINED,formula_contract='models/preprocessing/04/04_feature_contract.json',
        roles={r:len(ix) for r,ix in state['positions'].items()},test_parsed=False,
        fitted_operations='engineered RobustScaler on train only; originals use frozen 03 inputs',
        versions={p:importlib.metadata.version(p) for p in ('numpy','pandas','scipy','scikit-learn','matplotlib','joblib','nbformat','nbclient')},
        protected_unchanged=True,artifact_sha256={p.relative_to(root).as_posix():sha256(p) for p in sorted(saved)})
    (root/'data/processed/04/04_artifact_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return dict(saved_artifacts=len(saved),protected_files_unchanged=len(state['protected']),
                new_full_representation_matches=int(newmatches),test_parsed=False)
