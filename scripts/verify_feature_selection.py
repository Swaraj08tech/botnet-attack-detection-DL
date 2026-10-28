"""Independent verification of all candidate files and frozen group folds."""

from pathlib import Path
import json
import sys

import nbformat
import numpy as np
import pandas as pd
from scipy import sparse

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.data.load_data import sha256


def main():
    manifest=json.loads((ROOT/'data/processed/05/05_artifact_manifest.json').read_text())
    for path,expected in manifest['artifact_sha256'].items(): assert sha256(ROOT/path)==expected,path
    run=json.loads((ROOT/'results/tables/05_notebook_execution.json').read_text())
    for path,expected in run['protected_before'].items(): assert sha256(ROOT/path)==expected,path
    assert sha256(ROOT/'notebooks/05_feature_selection.ipynb')==run['notebook_sha256']
    assert sha256(ROOT/'data/processed/05/05_artifact_manifest.json')==run['artifact_manifest_sha256']
    nb=nbformat.read(ROOT/'notebooks/05_feature_selection.ipynb',4)
    code=[c for c in nb.cells if c.cell_type=='code']
    assert all(c.execution_count is not None for c in code)
    assert not any(o.output_type=='error' for c in code for o in c.outputs)
    assignments=pd.read_csv(ROOT/'results/splits/02_train_validation_assignments.csv')
    folds=pd.read_csv(ROOT/'data/processed/05/05_internal_folds.csv')
    fit=assignments[assignments.role.eq('train')].reset_index(drop=True)
    assert folds.row_position.equals(fit.row_position) and folds.label.equals(fit.label) and folds.attack_cat.equals(fit.attack_cat)
    assert folds.feature_group_id.equals(fit.feature_group_id)
    assert folds.groupby('feature_group_id').internal_holdout_fold.nunique().max()==1
    assert set(folds.internal_holdout_fold)=={0,1,2}
    audit=pd.read_csv(ROOT/'results/tables/05_candidate_cross_role_overlap.csv')
    original=json.loads((ROOT/'results/splits/02_split_manifest.json').read_text())['predictors']
    engineered=['fe_total_bytes','fe_byte_balance','fe_packet_balance']
    roles=('train','model_selection','calibration'); files=0
    for candidate,spec in manifest['sets'].items():
        keys={}; keep=spec['retained']
        assert set(original).issubset(keep) and {'is_ftp_login','ct_ftp_cmd'}.issubset(keep)
        for role in roles:
            dtype={c:'float64' for c in keep if c in engineered}
            raw=pd.read_csv(ROOT/f'data/processed/05/05_{candidate}_raw_{role}.csv',float_precision='round_trip',dtype=dtype)
            previous=pd.read_csv(ROOT/f'data/processed/04/04_raw_augmented_{role}.csv',float_precision='round_trip',dtype={c:'float64' for c in engineered})
            pd.testing.assert_frame_equal(raw,previous[keep],check_dtype=False,check_exact=True)
            links=pd.read_csv(ROOT/f'data/processed/05/05_row_links_{role}.csv')
            expected=assignments[assignments.role.eq(role)].reset_index(drop=True)
            for col in ['row_position','source_id','feature_group_id','label','attack_cat','role']:
                assert links[col].equals(expected[col])
            assert np.array_equal(links.input_row,np.arange(len(raw)))
            keys[role]=pd.MultiIndex.from_frame(raw)
            files+=1
            for kind in ('tree','scaled'):
                mapping=json.loads((ROOT/f'models/preprocessing/05/05_{candidate}_{kind}_columns.json').read_text())
                full_names=json.loads((ROOT/f'models/preprocessing/04/04_{kind}_augmented_feature_names.json').read_text())
                assert mapping['retained_original_names']==keep
                assert [full_names[i] for i in mapping['indices_in_04']]==mapping['encoded_names']
                for cat in ['proto','service','state']:
                    assert all(i in mapping['indices_in_04'] for i,n in enumerate(full_names) if n.startswith('cat__'+cat+'_'))
                full=sparse.load_npz(ROOT/f'data/processed/04/04_{kind}_augmented_{role}.npz')
                saved=sparse.load_npz(ROOT/f'data/processed/05/05_{candidate}_{kind}_{role}.npz')
                assert saved.shape==(len(raw),len(mapping['encoded_names']))
                assert (saved!=full[:,mapping['indices_in_04']]).nnz==0
                assert np.isfinite(saved.data).all()
                files+=1
        for i,a in enumerate(roles):
            for b in roles[i+1:]:
                common=keys[a].unique().intersection(keys[b].unique())
                record=audit[(audit.candidate==candidate)&(audit.representation=='raw')&(audit.role_a==a)&(audit.role_b==b)].iloc[0]
                assert len(common)==record.shared_vectors
                assert keys[a].isin(common).sum()==record.role_a_rows_with_match
                assert keys[b].isin(common).sum()==record.role_b_rows_with_match
        print(f'Verified candidate: {candidate}',flush=True)
    result=dict(code_cells=len(code),errors=0,candidates=len(manifest['sets']),verified_input_files=files,
        verified_artifact_hashes=len(manifest['artifact_sha256']),earlier_files_unchanged=len(run['protected_before']),
        exact_raw_overlap_independently_checked=True,categorical_groups_preserved=True,inner_original_groups_disjoint=True,
        winner=None,official_test_parsed=False)
    (ROOT/'results/tables/05_independent_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
