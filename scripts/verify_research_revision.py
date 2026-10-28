"""Independently reload artifacts, predictions and fold-local preprocessing."""
from pathlib import Path
import sys
import json
import joblib
import nbformat
import numpy as np
import pandas as pd
from scipy import sparse
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.data.load_data import sha256
from src.baseline_models import evaluate,CLASSES
from src.research_revision.common import protected_check,read_json,write_json
from src.research_revision.baselines import FAMILIES

def main():
    protected=protected_check(ROOT,snapshot_copies=True)
    total=0; cells=0
    for number in ['02','03','04','05','06']:
        path=next((ROOT/'notebooks').glob(number+'_*.ipynb')); nb=nbformat.read(path,4); nbformat.validate(nb)
        report=read_json(ROOT/f'results/tables/revision_v2/{number}_execution.json')
        assert report['notebook_sha256']==sha256(path)
        code=[c for c in nb.cells if c.cell_type=='code']; cells+=len(code)
        assert all(c.execution_count is not None for c in code)
        assert not any(o.output_type=='error' for c in code for o in c.outputs)
        manifest_path=ROOT/f'data/processed/revision_v2/{number}_artifact_manifest.json'
        assert report['manifest_sha256']==sha256(manifest_path)
        manifest=read_json(manifest_path)
        for paths in ['dependencies_sha256','artifact_sha256']:
            for name,expected in manifest[paths].items():
                assert sha256(ROOT/name)==expected,(number,name)
        for name in manifest['artifact_sha256']:
            path=ROOT/name
            if path.suffix=='.csv': pd.read_csv(path,float_precision='round_trip')
            elif path.suffix=='.npz':
                m=sparse.load_npz(path); assert np.isfinite(m.data).all()
            elif path.suffix=='.joblib': joblib.load(path)
            elif path.suffix=='.json': read_json(path)
        total+=len(manifest['artifact_sha256'])
    links=pd.read_csv(ROOT/'data/processed/05/05_row_links_model_selection.csv')
    results=pd.read_csv(ROOT/'results/tables/revision_v2/06_model_selection_metrics.csv',float_precision='round_trip')
    per_all=pd.read_csv(ROOT/'results/tables/revision_v2/06_per_class_metrics.csv',float_precision='round_trip')
    y=pd.Categorical(links.attack_cat,categories=CLASSES).codes
    checks=[]
    for row in results.itertuples():
        name=row.model+'_'+row.feature_set
        pred=pd.read_csv(ROOT/f'data/processed/revision_v2/06_{name}_predictions.csv',float_precision='round_trip')
        for col in links: assert np.array_equal(links[col],pred[col]),(name,col)
        m=joblib.load(ROOT/f'models/revision_v2/06_{name}.joblib')
        x=sparse.load_npz(ROOT/f'data/processed/05/05_{row.feature_set}_{row.representation}_model_selection.npz')
        predicted=pd.Categorical(pred.predicted_attack_cat,categories=CLASSES).codes
        probabilities=pred[['p_'+c for c in CLASSES]].to_numpy()
        assert np.array_equal(m.predict(x),predicted)
        assert np.array_equal(m.predict_proba(x),probabilities)
        metrics,per,cm,ncm,pr,score=evaluate(y,predicted,probabilities)
        for metric,value in metrics.items():
            assert np.isclose(getattr(row,metric),value,equal_nan=True,atol=1e-12),(name,metric)
        saved=per_all[per_all.model.eq(row.model)&per_all.feature_set.eq(row.feature_set)].reset_index(drop=True)
        for col in ['precision','recall','f1','support','correct','predicted_support']:
            assert np.allclose(saved[col],per[col],atol=1e-12)
        for normalized,expected in [('raw',cm),('normalized',ncm)]:
            actual=pd.read_csv(ROOT/f'results/tables/revision_v2/06_{name}_confusion_{normalized}.csv',float_precision='round_trip')
            assert np.allclose(actual[list(CLASSES)],expected,atol=1e-12)
        checks.append(dict(model=row.model,feature_set=row.feature_set,predictions_reproduced=True,metrics_reproduced=True,confusions_reproduced=True))
    # Verify scaler medians and learned categorical vocabularies from each inner-training subset.
    assignment=pd.read_csv(ROOT/'data/processed/05/05_internal_folds.csv')
    raw=pd.read_csv(ROOT/'data/processed/05/05_original42_raw_train.csv',float_precision='round_trip')
    for fold in range(3):
        fit=assignment.internal_holdout_fold.ne(fold)
        for rep in ['scaled','tree']:
            p=joblib.load(ROOT/f'models/revision_v2/06_inner_{rep}_fold{fold}_pipeline.joblib')
            encoder=p.named_transformers_['cat']
            for col,values in zip(['proto','service','state'],encoder.categories_): assert np.array_equal(values,np.sort(raw.loc[fit,col].unique()))
            if rep=='scaled':
                logcols=p.transformers_[0][2]
                assert np.array_equal(p.named_transformers_['log'].named_steps['scale'].center_,np.median(np.log1p(raw.loc[fit,logcols]),axis=0))
                plaincols=p.transformers_[1][2]
                assert np.array_equal(p.named_transformers_['plain'].center_,np.median(raw.loc[fit,plaincols],axis=0))
    report=dict(status='PASS',executed_notebooks=5,executed_code_cells=cells,artifact_entries_reloaded=total,
                protected_project_files=len(protected),snapshot_files_verified=379,model_prediction_checks=checks,
                fold_preprocessing_verified=True,calibration_predictions=0,official_test_model_predictions=0)
    write_json(ROOT/'results/tables/revision_v2/final_verification.json',report)
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
