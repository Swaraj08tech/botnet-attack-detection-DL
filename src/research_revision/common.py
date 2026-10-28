"""Provenance, row alignment and readable notebook reporting."""
from pathlib import Path
import hashlib
import json
import importlib.metadata
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from IPython.display import display, Markdown
from src.data.load_data import sha256
from src.data.feature_engineering import canonical_csr, sparse_row_keys, overlap_counts
from src.baseline_models import CLASSES, SETS

ROLES = ('train', 'model_selection', 'calibration')
MUTABLE = {'README.md', 'docs/research_notes.md'}

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False, default=str), encoding='utf-8')

def protected_check(root, snapshot_copies=False):
    root = Path(root)
    snapshot = root / (root/'.cache/workflow_snapshot_path.txt').read_text().strip()
    manifest = read_json(snapshot/'snapshot_manifest.json')
    checked = {}
    for name, spec in manifest['files'].items():
        mutable = name in MUTABLE or (name.startswith('notebooks/0') and Path(name).name[:2] in ('02','03','04','05','06'))
        if not mutable:
            assert sha256(root/name) == spec['sha256'], f'Protected source changed: {name}'
            checked[name] = spec['sha256']
        if snapshot_copies:
            assert sha256(snapshot/name) == spec['sha256'], f'Snapshot changed: {name}'
    for name, expected in manifest['raw_sha256'].items():
        assert sha256(root/name) == expected, f'Raw data changed: {name}'
        checked[name] = expected
    return checked

def start(root, number):
    root = Path(root); checked = protected_check(root)
    out = {k: root/f'{v}/revision_v2' for k,v in
           {'tables':'results/tables','figures':'results/figures','data':'data/processed','models':'models','docs':'docs'}.items()}
    for p in out.values(): p.mkdir(parents=True, exist_ok=True)
    m2 = read_json(root/'results/splits/02_split_manifest.json')
    assignment = pd.read_csv(root/'results/splits/02_train_validation_assignments.csv')
    raw = pd.read_csv(root/'data/raw/UNSW_NB15_training-set.csv', low_memory=False)
    assert sha256(root/'results/splits/02_train_validation_assignments.csv') == m2['assignment_sha256']
    assert np.array_equal(assignment.row_position, np.arange(len(raw)))
    for a,b in [('source_id','id'),('label','label'),('attack_cat','attack_cat')]:
        assert np.array_equal(assignment[a], raw[b]), a
    assert set(assignment.role) == set(ROLES)
    assert assignment.groupby('feature_group_id').role.nunique().max() == 1
    original = m2['predictors']
    assert len(original) == 42 and not set(original)&{'id','label','attack_cat','role','row_position','feature_group_id'}
    assert np.array_equal(raw.label, raw.attack_cat.ne('Normal').astype(int))
    links = {}; roles = {}; support=[]
    for role in ROLES:
        expected = assignment[assignment.role.eq(role)].reset_index(drop=True)
        link = pd.read_csv(root/f'data/processed/05/05_row_links_{role}.csv')
        assert np.array_equal(link.input_row, np.arange(len(link)))
        for col in ['row_position','source_id','label','attack_cat','feature_group_id','role']:
            assert np.array_equal(link[col], expected[col]), (role,col)
        links[role]=link; roles[role]=raw.iloc[link.row_position].reset_index(drop=True)
        for cat in CLASSES:
            mask=link.attack_cat.eq(cat)
            support.append(dict(role=role,attack_cat=cat,rows=int(mask.sum()),
                                distinct_original_patterns=link.loc[mask,'feature_group_id'].nunique()))
    s=dict(root=root,number=number,out=out,original=original,raw=raw,assignment=assignment,
           roles=roles,links=links,fit=roles['train'],checked=checked,notes=[],outputs=[])
    table(s,pd.DataFrame(support),'role_class_support')
    write_json(out['tables']/f'{number}_starting_integrity.json',dict(protected=checked,
       assignment_sha256=m2['assignment_sha256'],role_rows={r:len(links[r]) for r in ROLES},
       versions={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','lightgbm','joblib']},
       official_test_record_access=number=='02',calibration_use='Integrity/transformation only; no fit, prediction or outcome analysis'))
    pd.set_option('display.max_columns',16); pd.set_option('display.max_rows',55)
    plt.rcParams.update({'figure.dpi':110,'font.size':10})
    print(f'Verified {len(checked)} protected files; train={len(links["train"]):,}, model_selection={len(links["model_selection"]):,}, calibration={len(links["calibration"]):,}.')
    return s

def table(s, frame, name):
    path=s['out']['tables']/f'{s["number"]}_{name}.csv'
    frame.to_csv(path,index=False,float_format='%.17g'); s['outputs'].append(path)
    return frame

def figure(s, fig, name):
    if fig.get_layout_engine() is None: fig.tight_layout()
    path=s['out']['figures']/f'{s["number"]}_{name}.png'
    fig.savefig(path,dpi=150,bbox_inches='tight'); s['outputs'].append(path)
    plt.show(); plt.close(fig)

def explain(s, observation, decision, limitation):
    text=f'**Evidence:** {observation}\n\n**Decision:** {decision}\n\n**Still uncertain:** {limitation}'
    s['notes'].append(text); display(Markdown(text))

def show_prior(s, name, head=None):
    f=pd.read_csv(s['root']/f'results/tables/{name}.csv')
    table(s,f,'preserved_'+name)
    display(f if head is None else f.head(head))
    return f

def exact_roles(s, matrices, name):
    keys={r:sparse_row_keys(canonical_csr(matrices[r])) for r in ROLES}
    rows=[]
    for a,b in [('train','model_selection'),('train','calibration'),('model_selection','calibration')]:
        rows.append(dict(representation=name,role_a=a,role_b=b,**overlap_counts(keys[a],keys[b])))
    return pd.DataFrame(rows)

def verify_candidates(s):
    definitions=read_json(s['root']/'models/preprocessing/05/05_comparison_sets.json')['sets']
    rows=[]; overlaps=[]
    for candidate in SETS:
        for rep in ['tree','scaled']:
            mapping=read_json(s['root']/f'models/preprocessing/05/05_{candidate}_{rep}_columns.json')
            assert mapping['retained_original_names']==definitions[candidate]['retained']
            names=mapping['encoded_names']; sources=mapping['original_source_for_each_encoded_column']
            assert len(names)==len(set(names)) and set(sources)==set(definitions[candidate]['retained'])
            assert not set(sources)&{'id','label','attack_cat','role','row_position','feature_group_id','input_row'}
            full_names=read_json(s['root']/f'models/preprocessing/04/04_{rep}_augmented_feature_names.json')
            assert names==[full_names[i] for i in mapping['indices_in_04']]
            matrices={}
            for role in ROLES:
                x=canonical_csr(sparse.load_npz(s['root']/f'data/processed/05/05_{candidate}_{rep}_{role}.npz'))
                full=canonical_csr(sparse.load_npz(s['root']/f'data/processed/04/04_{rep}_augmented_{role}.npz'))
                assert x.shape==(len(s['links'][role]),len(names)) and (x!=full[:,mapping['indices_in_04']]).nnz==0
                raw=pd.read_csv(s['root']/f'data/processed/05/05_{candidate}_raw_{role}.csv',float_precision='round_trip')
                assert list(raw)==definitions[candidate]['retained'] and len(raw)==len(x.indptr)-1
                # Numeric CSV values are read with round-trip semantics; categorical values stay literal.
                for col in s['original']:
                    assert np.array_equal(raw[col],s['roles'][role][col]), (candidate,role,col)
                matrices[role]=x
                rows.append(dict(candidate=candidate,representation=rep,role=role,rows=x.shape[0],columns=x.shape[1],finite=True,aligned=True))
            overlaps.append(exact_roles(s,matrices,f'{candidate}_{rep}'))
    audit=pd.concat(overlaps,ignore_index=True)
    table(s,pd.DataFrame(rows),'candidate_verification'); table(s,audit,'candidate_overlap')
    # Report, rather than edit the assignments, if a changed representation ever introduces matches.
    return pd.DataFrame(rows),audit

def finish(s, findings, decisions, limitations, next_question):
    verified=protected_check(s['root'])
    assert verified==s['checked']
    parts=[f'# Notebook {s["number"]}: revised findings', '\n## Findings\n'+findings,
           '\n## Decisions\n'+decisions,'\n## Limitations\n'+limitations,
           '\n## Saved outputs\n'+ '\n'.join('- `'+p.relative_to(s['root']).as_posix()+'`' for p in dict.fromkeys(s['outputs'])),
           '\n## Next question\n'+next_question, '\n## Section evidence\n'+'\n\n'.join(s['notes'])]
    path=s['out']['docs']/f'{s["number"]}_findings.md'; path.write_text('\n\n'.join(parts),encoding='utf-8')
    s['outputs'].append(path)
    dependencies={k:v for k,v in verified.items() if k.startswith(('data/','models/','results/splits/','configs/','src/'))}
    for p in sorted((s['root']/'src/research_revision').glob('*.py')):
        dependencies[p.relative_to(s['root']).as_posix()]=sha256(p)
    for filename in ['build_research_revision.py','run_research_revision.py']:
        p=s['root']/'scripts'/filename; dependencies[p.relative_to(s['root']).as_posix()]=sha256(p)
    manifest=dict(version='revision_v2',notebook=s['number'],dependencies_sha256=dependencies,
       artifact_sha256={p.relative_to(s['root']).as_posix():sha256(p) for p in dict.fromkeys(s['outputs'])},
       protected_unchanged=True,calibration_predicted=False,official_test_evaluated=False)
    write_json(s['out']['data']/f'{s["number"]}_artifact_manifest.json',manifest)
    display(Markdown('\n\n'.join(parts[1:6])))
