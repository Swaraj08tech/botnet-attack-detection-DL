"""Notebook 05 input checks, fitting-only evidence and candidate serialization."""

from itertools import combinations
from pathlib import Path
import ast
import importlib.metadata
import json

import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedGroupKFold

from src.data.load_data import sha256
from src.data.duplicate_investigation import exact_feature_ids
from src.data.feature_engineering import canonical_csr, retained_features, overlap_counts
from src.data.feature_selection import (ENGINEERED, CATEGORICAL, comparison_sets,
    encoded_sources, column_indices, fit_bin_edges, apply_bins, association_score)

ROLES = ('train', 'model_selection', 'calibration')
NEW_FILES = {'feature_selection.py', 'feature_selection_audit.py',
    'build_feature_selection_notebook.py', 'run_feature_selection.py',
    'verify_feature_selection.py', 'test_feature_selection.py'}


def protected_files(root):
    files = []
    for folder in ('data/raw', 'data/processed', 'models/preprocessing', 'results',
                   'docs', 'configs', 'src', 'notebooks', 'scripts', 'tests'):
        for p in (root / folder).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and not p.name.startswith('05_') and p.name not in NEW_FILES:
                files.append(p)
    return sorted(files)


def table(state, frame, name):
    frame.to_csv(state['root']/f'results/tables/05_{name}.csv', index=False)
    return frame


def figure(state, fig, name):
    fig.tight_layout()
    fig.savefig(state['root']/f'results/figures/05_{name}.png', dpi=150, bbox_inches='tight')
    plt.show(); plt.close(fig)


def setup(root):
    root = Path(root)
    before = {p.relative_to(root).as_posix():sha256(p) for p in protected_files(root)}
    m2 = json.loads((root/'results/splits/02_split_manifest.json').read_text())
    m3 = json.loads((root/'data/processed/03/03_artifact_manifest.json').read_text())
    m4 = json.loads((root/'data/processed/04/04_artifact_manifest.json').read_text())
    assert m4['assignment_sha256'] == m3['assignment_sha256'] == m2['assignment_sha256']
    assert m4['source_train_sha256'] == m2['source_sha256']['train']
    assert m4['source_manifest03_sha256'] == sha256(root/'data/processed/03/03_artifact_manifest.json')
    for key, manifest in [('artifacts_sha256',m3), ('artifact_sha256',m4)]:
        for relative,expected in manifest[key].items():
            assert sha256(root/relative.replace('\\','/')) == expected, relative
    # Only integrity hashing touches official test bytes; no test records are parsed.
    for role, token in [('train','training'),('test','testing')]:
        assert sha256(root/f'data/raw/UNSW_NB15_{token}-set.csv') == m2['source_sha256'][role]
    assert sha256(root/'results/splits/02_train_validation_assignments.csv') == m2['assignment_sha256']
    run4 = json.loads((root/'results/tables/04_notebook_execution.json').read_text())
    assert sha256(root/'data/processed/04/04_artifact_manifest.json') == run4['artifact_manifest_sha256']
    notebook04_path=root/'notebooks/04_feature_engineering.ipynb'
    notebook04=nbformat.read(notebook04_path,4)
    # Rebuild cell sources in memory from the manifest-verified builder. Never write NB04.
    builder=root/'scripts/build_feature_engineering_notebook.py'
    tree=ast.parse(builder.read_text(encoding='utf-8'))
    def omitted_statement(node):
        if not isinstance(node,ast.Expr) or not isinstance(node.value,ast.Call): return False
        function=node.value.func
        return ((isinstance(function,ast.Attribute) and isinstance(function.value,ast.Name)
                 and function.value.id=='nbf' and function.attr=='write')
                or (isinstance(function,ast.Name) and function.id=='print'))
    tree.body=[node for node in tree.body if not omitted_statement(node)]
    scope={'__file__':str(builder)}
    exec(compile(tree,str(builder),'exec'),scope)
    expected04=scope['nb']
    assert len(notebook04.cells)==len(expected04.cells)
    assert all(a.cell_type==b.cell_type and a.source==b.source for a,b in zip(notebook04.cells,expected04.cells)), 'Notebook 04 source differs from verified builder'
    code04=[c for c in notebook04.cells if c.cell_type=='code']
    assert all(c.execution_count is not None for c in code04)
    assert not any(o.output_type=='error' for c in code04 for o in c.outputs)
    notebook_provenance=dict(actual_sha256=sha256(notebook04_path),recorded_execution_sha256=run4['notebook_sha256'],
        file_hash_matches_old_execution_record=sha256(notebook04_path)==run4['notebook_sha256'],
        all_code_and_markdown_match_manifest_verified_builder=True,saved_error_outputs=0,
        explanation='Non-source notebook content differs from the old execution record; all code/Markdown and every manifest artifact verify. Earlier files were not modified.')
    source = pd.read_csv(root/'data/raw/UNSW_NB15_training-set.csv',low_memory=False)
    assign = pd.read_csv(root/'results/splits/02_train_validation_assignments.csv')
    original = m2['predictors']; universe = original + ENGINEERED
    assert original == m3['predictors'] and m4['retained_features'] == ENGINEERED
    assert len(source) == len(assign) == 175341
    assert np.array_equal(assign.row_position,np.arange(len(source)))
    for a,b in [('id','source_id'),('label','label'),('attack_cat','attack_cat')]:
        assert source[a].equals(assign[b])
    ids,_ = exact_feature_ids(source, original)
    assert np.array_equal(ids,assign.feature_group_id)
    assert assign.groupby('feature_group_id').role.nunique().max()==1
    assert set(assign.role)==set(ROLES)
    raw={}; links={}; positions={}; matrices={k:{} for k in ('tree','scaled')}
    names={k:json.loads((root/f'models/preprocessing/04/04_{k}_augmented_feature_names.json').read_text()) for k in matrices}
    for k in names:
        assert set(encoded_sources(names[k],original))==set(universe)
    for role in ROLES:
        ix=np.flatnonzero(assign.role.eq(role)); positions[role]=ix
        x=pd.read_csv(root/f'data/processed/04/04_raw_augmented_{role}.csv',float_precision='round_trip',dtype={c:'float64' for c in ENGINEERED})
        y=pd.read_csv(root/f'data/processed/04/04_row_links_{role}.csv')
        assert x.columns.tolist()==universe and x.shape==(len(ix),45)
        pd.testing.assert_frame_equal(x[original],source.iloc[ix][original].reset_index(drop=True),check_dtype=False,check_exact=True)
        pd.testing.assert_frame_equal(x[ENGINEERED],retained_features(source.iloc[ix][original]).reset_index(drop=True),check_exact=True)
        for col in ['row_position','source_id','feature_group_id','role','label','attack_cat']:
            assert np.array_equal(y[col],assign.iloc[ix][col])
        assert np.array_equal(y.input_row,np.arange(len(ix)))
        assert np.isfinite(x.select_dtypes('number').to_numpy()).all()
        raw[role]=x; links[role]=y
        for k in matrices:
            block=canonical_csr(sparse.load_npz(root/f'data/processed/04/04_{k}_augmented_{role}.npz'))
            assert block.shape==(len(ix),len(names[k]))
            matrices[k][role]=block
    state=dict(root=root,m2=m2,m3=m3,m4=m4,protected=before,original=original,universe=universe,
               raw=raw,links=links,positions=positions,matrices=matrices,names=names,notebook04_provenance=notebook_provenance)
    for p in ('results/tables','results/figures','data/processed/05','models/preprocessing/05'):
        (root/p).mkdir(parents=True,exist_ok=True)
    support=pd.DataFrame([dict(role=r,rows=len(raw[r]),original_patterns=links[r].feature_group_id.nunique()) for r in ROLES])
    table(state,support,'input_role_support')
    (root/'results/tables/05_notebook04_provenance.json').write_text(json.dumps(notebook_provenance,indent=2),encoding='utf-8')
    (root/'results/tables/05_input_integrity.json').write_text(json.dumps(dict(
        source_sha256=m2['source_sha256'],assignment_sha256=m2['assignment_sha256'],
        manifest04_sha256=sha256(root/'data/processed/04/04_artifact_manifest.json'),
        verified04_artifacts=len(m4['artifact_sha256']),protected_before=before,
        test_access='Hash only; no official test CSV parsing'),indent=2),encoding='utf-8')
    return state,support


def define_sets(state):
    sets=comparison_sets(state['original']); state['sets']=sets
    spec=dict(version='05-v1',sets=sets,definition='original source names, all one-hot columns kept as a group',
        original_features=state['original'],engineered_features=ENGINEERED,
        declared_before_analysis=True,inner_folds=3,inner_seed=52,quantile_bins=8,
        low_variation_mode_share=0.995,near_redundancy_abs_spearman=0.95,
        feature_selection_rule='Keep these five predefined candidates; no score cutoff or fitted predictive selector',
        assignment_sha256=state['m2']['assignment_sha256'])
    path=state['root']/'models/preprocessing/05/05_comparison_sets.json'
    path.write_text(json.dumps(spec,indent=2),encoding='utf-8')
    state['set_definition_sha256']=sha256(path)
    rows=[dict(candidate=k,features=len(v['retained']),omitted=', '.join(v['omitted']) or 'None',reason=v['reason']) for k,v in sets.items()]
    return table(state,pd.DataFrame(rows),'predefined_sets')


def redundancy_study(state):
    x=state['raw']['train']; g=state['links']['train'].feature_group_id
    unique=~g.duplicated(); numeric=x.select_dtypes('number').columns.tolist()
    pairs=[]
    for a,b in combinations(x.columns,2):
        if (a in numeric)!=(b in numeric): continue
        same=x[a].eq(x[b])
        if same.all():
            pairs.append(dict(feature_a=a,feature_b=b,equal_rows=int(same.sum()),equal_original_patterns=g[same].nunique(),
                              total_rows=len(x),total_original_patterns=g.nunique()))
    copies=table(state,pd.DataFrame(pairs,columns=['feature_a','feature_b','equal_rows','equal_original_patterns','total_rows','total_original_patterns']),'exact_copies_fit')
    corr=x[numeric].corr(method='spearman')
    pattern_corr=x.loc[unique,numeric].corr(method='spearman')
    records=[]
    for a,b in combinations(numeric,2):
        value=corr.loc[a,b]
        records.append(dict(feature_a=a,feature_b=b,row_spearman=value,
            one_per_original_pattern_spearman=pattern_corr.loc[a,b],rows=len(x),original_patterns=g.nunique(),
            near_redundant_flag=bool(pd.notna(value) and abs(value)>=.95),exact_copy=bool(x[a].equals(x[b]))))
    correlations=table(state,pd.DataFrame(records),'numeric_pair_correlations_fit')
    variations=[]
    for col in x:
        mode=x[col].value_counts().index[0]; mask=x[col].eq(mode)
        pattern_counts=x.loc[unique,col].value_counts()
        variations.append(dict(feature=col,unique_values=x[col].nunique(),most_common_value=str(mode),
            mode_rows=int(mask.sum()),mode_original_patterns=g[mask].nunique(),mode_row_share=float(mask.mean()),
            mode_pattern_share=float(pattern_counts.max()/unique.sum()),rows=len(x),original_patterns=g.nunique(),
            constant=bool(x[col].nunique()==1),low_variation_flag=bool(mask.mean()>=.995)))
    variation=table(state,pd.DataFrame(variations),'feature_variation_fit')
    # Existing audit evidence is cited as provenance, not re-analysed or used to choose sets.
    prior=pd.read_csv(state['root']/'results/tables/02_ftp_field_audit.csv')
    prior.to_csv(state['root']/'results/tables/05_prior_ftp_audit_reference.csv',index=False)
    ftp=table(state,x.groupby(['is_ftp_login','ct_ftp_cmd']).size().rename('rows').reset_index(),'ftp_joint_values_fit')
    state.update(corr=corr,pattern_corr=pattern_corr,variation=variation,copies=copies,correlations=correlations)
    return copies,variation.sort_values('mode_row_share',ascending=False),correlations[correlations.near_redundant_flag],ftp


def relevance_study(state):
    x=state['raw']['train']; y=state['links']['train']; groups=y.feature_group_id
    splitter=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=52)
    folds=np.full(len(x),-1,dtype=int); records=[]; support=[]; bins_record={}
    for fold,(inner_train,inner_holdout) in enumerate(splitter.split(x,y.attack_cat,groups)):
        assert not set(groups.iloc[inner_train]) & set(groups.iloc[inner_holdout])
        folds[inner_holdout]=fold
        held_groups=groups.iloc[inner_holdout]
        weights=1/held_groups.map(held_groups.value_counts()).to_numpy()
        for part,ix in [('inner_train',inner_train),('inner_holdout',inner_holdout)]:
            for cls in state['m2']['class_vocabulary']:
                chosen=y.iloc[ix].attack_cat.eq(cls)
                support.append(dict(fold=fold,part=part,attack_cat=cls,rows=int(chosen.sum()),
                    original_patterns=groups.iloc[ix][chosen].nunique()))
        bins_record[str(fold)]={}
        for col in x:
            if col in CATEGORICAL:
                held=x.iloc[inner_holdout][col].to_numpy()
                bins_record[str(fold)][col]={'method':'literal category, no vocabulary fit'}
            else:
                edges=fit_bin_edges(x.iloc[inner_train][col],bins=8)
                held=apply_bins(x.iloc[inner_holdout][col],edges)
                bins_record[str(fold)][col]={'method':'inner-train quantile edges', 'edges':edges.tolist()}
            for target in ['attack_cat','label']:
                labels=y.iloc[inner_holdout][target].to_numpy()
                records.append(dict(fold=fold,feature=col,target=target,holdout_rows=len(inner_holdout),
                    holdout_original_patterns=held_groups.nunique(),observed_feature_levels=len(np.unique(held)),
                    row_association=association_score(held,labels),
                    pattern_weighted_association=association_score(held,labels,weights)))
    assert (folds>=0).all()
    assigned=y[['input_row','row_position','feature_group_id','label','attack_cat']].copy()
    assigned['internal_holdout_fold']=folds
    assert assigned.groupby('feature_group_id').internal_holdout_fold.nunique().max()==1
    assigned.to_csv(state['root']/'data/processed/05/05_internal_folds.csv',index=False)
    (state['root']/'models/preprocessing/05/05_internal_binning.json').write_text(json.dumps(dict(
        folds=3,seed=52,bins=8,fit_scope='each inner training partition of frozen train role',
        use='diagnostic association only; never model inputs or a fitted selector',cuts=bins_record),indent=2),encoding='utf-8')
    scores=table(state,pd.DataFrame(records),'association_by_internal_fold')
    summary=scores.groupby(['target','feature'],sort=False).agg(
        mean_row_score=('row_association','mean'),min_row_score=('row_association','min'),max_row_score=('row_association','max'),
        mean_pattern_weighted_score=('pattern_weighted_association','mean'),
        min_pattern_weighted_score=('pattern_weighted_association','min'),max_pattern_weighted_score=('pattern_weighted_association','max'),
        defined_folds=('row_association','count')).reset_index()
    table(state,summary,'association_summary')
    table(state,pd.DataFrame(support),'internal_fold_class_support')
    classes=y.groupby('attack_cat').agg(rows=('attack_cat','size'),original_patterns=('feature_group_id','nunique')).reset_index()
    table(state,classes,'fit_class_support')
    conflicts=y.groupby('feature_group_id').agg(rows=('label','size'),binary_targets=('label','nunique'),categories=('attack_cat','nunique'))
    conflicts=table(state,pd.DataFrame([dict(target=t,conflicting_patterns=int(conflicts[c].gt(1).sum()),
        affected_rows=int(conflicts.loc[conflicts[c].gt(1),'rows'].sum())) for t,c in [('label','binary_targets'),('attack_cat','categories')]]),'fit_conflict_support')
    state.update(scores=scores,association_summary=summary,class_support=classes,conflicts=conflicts)
    return summary,classes,conflicts


def evidence_figures(state):
    selected=['sbytes','dbytes','spkts','dpkts',*ENGINEERED]
    fig,ax=plt.subplots(figsize=(7,6)); c=state['corr'].loc[selected,selected]
    im=ax.imshow(c,vmin=-1,vmax=1,cmap='coolwarm')
    labels=[n.removeprefix('fe_').replace('_',' ') for n in selected]
    ax.set_xticks(range(len(labels)),labels,rotation=45,ha='right',fontsize=8)
    ax.set_yticks(range(len(labels)),labels,fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)): ax.text(j,i,f'{c.iloc[i,j]:.2f}',ha='center',va='center',fontsize=7)
    ax.set_title('Which measurements move together?')
    fig.colorbar(im,ax=ax,label='Fitting-row Spearman correlation')
    figure(state,fig,'selected_correlations')


def relevance_figure(state):
    d=state['association_summary']; d=d[d.target.eq('attack_cat')].copy()
    view=['sbytes','dbytes','rate','service','is_ftp_login','ct_ftp_cmd',*ENGINEERED]
    d=d.set_index('feature').loc[view]
    fig,ax=plt.subplots(figsize=(9,4)); positions=np.arange(len(view))
    ax.errorbar(d.mean_row_score,positions-.12,
        xerr=np.vstack([d.mean_row_score-d.min_row_score,d.max_row_score-d.mean_row_score]),fmt='o',label='Rows')
    ax.errorbar(d.mean_pattern_weighted_score,positions+.12,
        xerr=np.vstack([d.mean_pattern_weighted_score-d.min_pattern_weighted_score,d.max_pattern_weighted_score-d.mean_pattern_weighted_score]),
        fmt='s',label='Each original pattern has total weight 1')
    for position, feature in enumerate(view):
        if not np.isfinite(d.loc[feature, 'mean_row_score']):
            ax.text(.03, position, 'No score: value bands collapse to one group',
                    va='center', fontsize=8, color='dimgray')
    ax.set_yticks(positions,[v.removeprefix('fe_').replace('_',' ') for v in view])
    ax.set(xlabel='Association score: mean and range across 3 internal folds',xlim=(-.02,1.02),title='Label associations, not model performance')
    ax.legend(fontsize=8)
    figure(state,fig,'association_stability')


def shortlist(state):
    rows=[]
    for name,spec in state['sets'].items():
        rows.append(dict(candidate=name,features=len(spec['retained']),retained_original_names=json.dumps(spec['retained']),
            omitted_original_names=json.dumps(spec['omitted']),reason=spec['reason'],
            status='Carry to Notebook 06; no winner established',
            evidence='Small semantic ablations; fitting correlation motivates testing both balances separately; preserve original features and unresolved FTP semantics'))
    result=table(state,pd.DataFrame(rows),'shortlist')
    for name,spec in state['sets'].items():
        (state['root']/f'models/preprocessing/05/05_{name}_features.json').write_text(json.dumps(spec,indent=2),encoding='utf-8')
    state['shortlist']=result
    return result


def sparse_keys(matrix):
    x=canonical_csr(matrix)
    # Full bytes, not digest-only equality. Canonical zero removal equates signed zeros.
    return [(x.indices[x.indptr[i]:x.indptr[i+1]].astype('<i8').tobytes(),
             x.data[x.indptr[i]:x.indptr[i+1]].astype('<f8').tobytes()) for i in range(x.shape[0])]


def save_candidates(state):
    root=state['root']; out=root/'data/processed/05'; model=root/'models/preprocessing/05'
    verification=[]; audit=[]; mapping=[]
    for role in ROLES:
        state['links'][role].to_csv(out/f'05_row_links_{role}.csv',index=False)
        pd.testing.assert_frame_equal(state['links'][role],pd.read_csv(out/f'05_row_links_{role}.csv'),check_exact=True)
    for candidate,spec in state['sets'].items():
        retained=spec['retained']; raw_keys={}
        for role in ROLES:
            block=state['raw'][role][retained]
            assert block.columns.tolist()==retained and not set(block.columns)&{'id','label','attack_cat'}
            path=out/f'05_{candidate}_raw_{role}.csv'
            block.to_csv(path,index=False,float_format='%.17g')
            restored=pd.read_csv(path,float_precision='round_trip',dtype={c:'float64' for c in retained if c in ENGINEERED})
            pd.testing.assert_frame_equal(block,restored,check_dtype=False,check_exact=True)
            raw_keys[role]=list(restored.itertuples(index=False,name=None))
            verification.append(dict(candidate=candidate,representation='raw',role=role,rows=len(block),columns=len(retained),finite=True,
                feature_order_preserved=True,row_links_labels_verified=True,saved_reload_exact=True))
        for a,b in combinations(ROLES,2):
            counts=overlap_counts(raw_keys[a],raw_keys[b])
            audit.append(dict(candidate=candidate,representation='raw',role_a=a,role_b=b,**counts,new_shared_vs_original42=counts['shared_vectors']))
        del raw_keys
        for kind in ('tree','scaled'):
            ix=column_indices(state['names'][kind],state['original'],retained)
            selected_names=[state['names'][kind][i] for i in ix]
            sources=encoded_sources(selected_names,state['original'])
            for source_name in retained:
                all_indices=[i for i,s in enumerate(encoded_sources(state['names'][kind],state['original'])) if s==source_name]
                assert set(all_indices).issubset(ix), 'Partial one-hot group'
            (model/f'05_{candidate}_{kind}_columns.json').write_text(json.dumps(dict(
                retained_original_names=retained,encoded_names=selected_names,indices_in_04=ix,
                original_source_for_each_encoded_column=sources),indent=2),encoding='utf-8')
            mapping.extend(dict(candidate=candidate,representation=kind,position=j,encoded_name=n,original_feature=s)
                           for j,(n,s) in enumerate(zip(selected_names,sources)))
            keys={}
            for role in ROLES:
                block=canonical_csr(state['matrices'][kind][role][:,ix])
                path=out/f'05_{candidate}_{kind}_{role}.npz'
                sparse.save_npz(path,block,compressed=True)
                restored=canonical_csr(sparse.load_npz(path))
                assert block.shape==(len(state['raw'][role]),len(selected_names))
                assert (block!=restored).nnz==0 and np.isfinite(block.data).all()
                keys[role]=sparse_keys(restored)
                verification.append(dict(candidate=candidate,representation=kind,role=role,rows=block.shape[0],columns=block.shape[1],finite=True,
                    feature_order_preserved=True,row_links_labels_verified=True,saved_reload_exact=True))
            for a,b in combinations(ROLES,2):
                counts=overlap_counts(keys[a],keys[b])
                audit.append(dict(candidate=candidate,representation=kind,role_a=a,role_b=b,**counts,new_shared_vs_original42=counts['shared_vectors']))
            del keys
    # Any shared reduced representation is reported; it never triggers split changes here.
    audit=table(state,pd.DataFrame(audit),'candidate_cross_role_overlap')
    verified=table(state,pd.DataFrame(verification),'candidate_input_verification')
    table(state,pd.DataFrame(mapping),'encoded_column_groups')
    state.update(overlap=audit,verification=verified)
    return verified,audit


def finish(state):
    root=state['root']
    after={p.relative_to(root).as_posix():sha256(p) for p in protected_files(root)}
    assert after==state['protected'], 'Earlier file changed'
    assert sha256(root/'models/preprocessing/05/05_comparison_sets.json')==state['set_definition_sha256']
    matched=state['overlap'][state['overlap'].shared_vectors.gt(0)]
    reasons='\n'.join(f"- **{k} ({len(v['retained'])} features):** {v['reason']} Omitted: {', '.join(v['omitted']) or 'none'}." for k,v in state['sets'].items())
    report=f'''# Notebook 05 — Feature Selection

## What we will compare

Question: Which original and engineered features should be carried forward as candidates for baseline models?

{reasons}

All five sets retain every original predictor, including both FTP fields. These are named candidates, not a winning set. Full retained/omitted lists are in `results/tables/05_shortlist.csv` and `models/preprocessing/05/05_*_features.json`. Each categorical feature includes its complete group of one-hot columns. Source-feature counts differ from encoded-matrix column counts. The definitions were saved before this notebook's analysis and did not change afterwards.

## Evidence and limits

Only fitting rows informed evidence: 140,273 rows with 80,848 different original patterns. Row/pattern support, exact copies, most-common-value shares, correlations and near-redundancy flags are saved in the evidence tables. A >=99.5% most-common-value share and absolute Spearman >=0.95 are diagnostic flags, not removal rules. Correlation between the balances is {state['corr'].loc['fe_byte_balance','fe_packet_balance']:.4f} for rows and {state['pattern_corr'].loc['fe_byte_balance','fe_packet_balance']:.4f} for one row per original pattern.

Both FTP columns match exactly in fitting rows. Notebook 02's saved audit already records differences in the supplied data. This is referenced provenance; no new official-test records were read or used to rank, prune or add candidates. Unresolved field meanings are a reason to preserve both. Low variation and correlation alone do not establish uselessness or label leakage.

Label associations were checked in three grouped internal folds (seed 52), entirely inside the frozen train role. Numeric features use up to eight quantile bins with cuts fitted on each inner training partition, then applied unchanged to that partition's holdout. Categories use literal values, not one-hot fragments. Constant observed features produce unavailable scores. Scores are Cramer's V summaries; the fractional-weight version gives each original pattern total weight one while retaining every conflicting label. No independence test or p-value is reported. High cardinality, binning and class frequencies can bias these descriptive scores; their fold ranges are not confidence intervals or model-performance estimates. They do not measure incremental information beyond the other columns.

Rare-class and conflict support are reported for fitting rows and internal folds. Identical original vectors stay together, including conflicting labels. Exact patterns are not verified independent hosts or events. No majority relabeling, deduplication or resampling occurs. No predictive selector, model comparison or confidence calibration was fitted. Saved internal binning is an audit tool only and is not used in the candidate model matrices.

Both FTP fields have unavailable association scores: their nearly all-zero values make the quantile bands collapse to one group. This diagnostic cannot describe those rare nonzero values properly. It is not evidence of zero relevance; the separate FTP value audit remains important.

## Saved inputs and checks

Each candidate has raw, tree and scaled inputs for train, model_selection and calibration. All are fixed column slices of verified Notebook 04 inputs; no preprocessing is refitted. Source-name lists define raw-column order; encoded-column order follows Notebook 04 and is separately saved. Shared row-link files contain original labels, source positions and pattern IDs; these are not predictors.

Saved/reloaded raw values and sparse matrices match exactly, with finite entries and preserved rows, labels and whole categorical groups. Every candidate/representation has an exact cross-role overlap check with counts for both roles. Checks compare full tuples or canonical sparse index/value bytes after dictionary lookup, not hash equality alone. Representation/role-pair checks with matches: {len(matched)}. Any new matches must be reported before modelling; no split changes occur here. All earlier notebooks, raw files, assignments, scripts, utilities and saved artifacts have unchanged hashes. Official-test bytes were only hashed for preservation.

Provenance note: Notebook 04's current file hash matches its old execution record: {state['notebook04_provenance']['file_hash_matches_old_execution_record']}. Its code and Markdown were independently rebuilt in memory from the manifest-verified builder and match exactly; all 47 listed artifacts match their hashes and the saved notebook has no error outputs. The discrepancy is in non-source notebook content. Both hashes and this limitation are recorded in `05_notebook04_provenance.json`; no earlier file was changed or execution record rewritten.

## Notebook 06

Use these five sets with both frozen tree and scaled representations. Prespecify a small baseline family, such as one regularized linear classifier and one tree-based classifier, and give sets the same tuning budget, fitting rows, seeds and validation rules. Keep unweighted baselines; any use of Notebook 03's candidate class weights must be separately named and applied equally across sets. Keep the all-original 42 and all-45 comparisons.

Train only on train, select models on model_selection, and keep calibration for later confidence adjustment. Do not refit on validation or run the official test during baseline selection. Compare fixed-ten-class macro F1, every class's precision/recall/F1 and support, confusion matrices, and Notebook 02's binary measures. For binary threshold-dependent results, choose thresholds only on model_selection. Use paired differences and, if estimating uncertainty, resample entire original groups with the same group draws across sets; this remains conditional on unknown context dependence.

Compare total_only43 against original42 for volume; total_and_byte44 and total_and_packet44 against total_only43 for each balance; all45 against both 44-feature sets for the second balance; all45 against original42 for the complete addition. Engineered features help only if model_selection results improve under a fair matched comparison, with rare-class tradeoffs and stability made explicit. No winner is declared here. Reserve full-test and overlap/non-overlap reporting for the later frozen final pipeline under Notebook 02.

## Hashes and sources

- Training source: `{state['m2']['source_sha256']['train']}`
- Frozen assignments: `{state['m2']['assignment_sha256']}`
- Notebook 04 artifact manifest: `{sha256(root/'data/processed/04/04_artifact_manifest.json')}`
- Predeclared feature sets: `{state['set_definition_sha256']}`

All deliverable hashes are in `data/processed/05/05_artifact_manifest.json`. Method references: [Cramer's V in SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.contingency.association.html) and [StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html). The weighted extension is our descriptive calculation, not an inferential SciPy test. Dataset semantics and deployment limits remain as documented in Notebooks 02–04.
'''
    (root/'docs/05_feature_selection_findings.md').write_text(report,encoding='utf-8')
    integrity=dict(protected_before=state['protected'],protected_after=after,protected_unchanged=True,
        predefined_sets_unchanged=True,official_test_parsed=False,learned_selector_fitted=False,
        internal_group_overlap=0,frozen_assignments_unchanged=True)
    (root/'results/tables/05_integrity_verification.json').write_text(json.dumps(integrity,indent=2),encoding='utf-8')
    saved=[*root.glob('results/tables/05_*'),*root.glob('results/figures/05_*'),
           *root.glob('data/processed/05/05_*'),*root.glob('models/preprocessing/05/05_*'),root/'docs/05_feature_selection_findings.md']
    saved += [root/'src/data/feature_selection.py',root/'src/data/feature_selection_audit.py',
              root/'scripts/build_feature_selection_notebook.py',root/'scripts/run_feature_selection.py',
              root/'scripts/verify_feature_selection.py',root/'tests/test_feature_selection.py']
    saved=[p for p in saved if p.name not in ('05_artifact_manifest.json','05_notebook_execution.json','05_independent_verification.json')]
    manifest=dict(version='05-v1',source_train_sha256=state['m2']['source_sha256']['train'],
        assignment_sha256=state['m2']['assignment_sha256'],manifest04_sha256=sha256(root/'data/processed/04/04_artifact_manifest.json'),
        set_definition_sha256=state['set_definition_sha256'],sets=state['sets'],roles={r:len(state['raw'][r]) for r in ROLES},
        predictive_selectors=[],learned_objects='Internal diagnostic bin edges only, fitted on inner training rows',
        winner=None,official_test_parsed=False,protected_unchanged=True,
        notebook04_provenance=state['notebook04_provenance'],
        versions={p:importlib.metadata.version(p) for p in ('pandas','numpy','scipy','scikit-learn','matplotlib','nbformat','nbclient')},
        artifact_sha256={p.relative_to(root).as_posix():sha256(p) for p in sorted(saved)})
    (root/'data/processed/05/05_artifact_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return dict(candidates=len(state['sets']),artifacts=len(saved),protected_files=len(after),overlap_checks_with_matches=len(matched),winner=None)
