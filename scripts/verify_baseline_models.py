"""Independent saved-model predictions, fixed metrics and preservation checks."""

from pathlib import Path
import json
import sys

import joblib
import nbformat
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import (accuracy_score,average_precision_score,confusion_matrix,
    f1_score,precision_recall_fscore_support)

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.data.load_data import sha256
from src.baseline_models import SETS,MODELS,CLASSES,NORMAL,choose_candidate,make_model,exact_settings
from src.baseline_audit import write_json


def main():
    read=lambda path:json.loads((ROOT/path).read_text(encoding='utf-8'))
    manifest=read('data/processed/06/06_artifact_manifest.json')
    for path,expected in manifest['artifact_sha256'].items(): assert sha256(ROOT/path)==expected,path
    run=read('results/tables/06_notebook_execution.json')
    for path,expected in run['protected_before'].items(): assert sha256(ROOT/path)==expected,path
    assert sha256(ROOT/'notebooks/06_baseline_models.ipynb')==run['notebook_sha256']
    assert sha256(ROOT/'data/processed/06/06_artifact_manifest.json')==run['artifact_manifest_sha256']
    nb=nbformat.read(ROOT/'notebooks/06_baseline_models.ipynb',4); nbformat.validate(nb)
    code=[c for c in nb.cells if c.cell_type=='code']
    assert all(c.execution_count is not None for c in code)
    assert not any(o.output_type=='error' for c in code for o in c.outputs)
    plan=read('configs/06_baseline_plan.json')
    assert sha256(ROOT/'configs/06_baseline_plan.json')==manifest['plan_sha256']
    assert plan['feature_sets']==list(SETS) and plan['fit_role']=='train' and not plan['threshold_selection']
    assignments=pd.read_csv(ROOT/'results/splits/02_train_validation_assignments.csv')
    links=pd.read_csv(ROOT/'data/processed/05/05_row_links_model_selection.csv')
    expected=assignments[assignments.role.eq('model_selection')].reset_index(drop=True)
    for column in ('row_position','source_id','feature_group_id','role','label','attack_cat'):
        assert links[column].equals(expected[column])
    truth=links.attack_cat.map({c:i for i,c in enumerate(CLASSES)}).to_numpy()
    summary=pd.read_csv(ROOT/'results/tables/06_model_selection_metrics.csv',float_precision='round_trip').set_index('experiment')
    per_class=pd.read_csv(ROOT/'results/tables/06_per_class_metrics.csv',float_precision='round_trip')
    assert len(summary)==10 and len(per_class)==100
    for family in MODELS:
        assert plan['models'][family]['settings']==exact_settings(make_model(family))
        kind=plan['models'][family]['representation']
        for candidate in SETS:
            experiment=f'{family}_{candidate}'
            model=joblib.load(ROOT/f'models/baselines/06/06_{experiment}.joblib')
            metadata=read(f'models/baselines/06/06_{experiment}_settings.json')
            assert exact_settings(model)==metadata['settings']==plan['models'][family]['settings']
            assert metadata['fit_role']=='train' and metadata['fit_rows']==140273 and not metadata['calibration_predicted']
            assert metadata['fit_row_links_sha256']==sha256(ROOT/'data/processed/05/05_row_links_train.csv')
            assert metadata['plan_sha256']==manifest['plan_sha256']
            mapping=read(f'models/preprocessing/05/05_{candidate}_{kind}_columns.json')
            assert metadata['encoded_columns']==mapping['encoded_names'] and metadata['source_features']==mapping['retained_original_names']
            matrix=sparse.load_npz(ROOT/f'data/processed/05/05_{candidate}_{kind}_model_selection.npz').tocsr()
            assert model.n_features_in_==matrix.shape[1] and np.array_equal(model.classes_,np.arange(10))
            if family=='decision_tree': assert model.tree_.n_node_samples[0]==140273 and model.get_depth()<=16
            saved=pd.read_csv(ROOT/f'data/processed/06/06_{experiment}_model_selection_predictions.csv',float_precision='round_trip')
            pd.testing.assert_frame_equal(saved[links.columns],links,check_dtype=False,check_exact=True)
            assert len(saved)==17534 and saved.role.eq('model_selection').all() and saved.experiment.eq(experiment).all()
            predicted=model.predict(matrix); proba=model.predict_proba(matrix)
            assert saved.predicted_attack_cat.tolist()==[CLASSES[i] for i in predicted]
            np.testing.assert_array_equal(saved[[f'p_{c}' for c in CLASSES]].to_numpy(),proba)
            np.testing.assert_array_equal(saved.attack_score,1-proba[:,NORMAL])
            np.testing.assert_array_equal(saved.predicted_binary_attack,(predicted!=NORMAL).astype(int))
            row=summary.loc[experiment]
            for name,value in [('macro_f1',f1_score(truth,predicted,labels=np.arange(10),average='macro',zero_division=0)),
                ('weighted_f1',f1_score(truth,predicted,labels=np.arange(10),average='weighted',zero_division=0)),
                ('accuracy',accuracy_score(truth,predicted)),('attack_average_precision',average_precision_score(links.label,saved.attack_score))]:
                assert np.isclose(row[name],value,rtol=0,atol=1e-14),name
            cm=confusion_matrix(truth,predicted,labels=np.arange(10)); norm=cm/cm.sum(axis=1,keepdims=True)
            for label,array in [('raw',cm),('normalized',norm)]:
                stored=pd.read_csv(ROOT/f'results/tables/06_{experiment}_confusion_{label}.csv',index_col=0,float_precision='round_trip')
                assert list(stored.index)==list(CLASSES) and list(stored.columns)==list(CLASSES)
                np.testing.assert_allclose(stored.to_numpy(),array,rtol=0,atol=1e-15)
            p,r,f,s=precision_recall_fscore_support(truth,predicted,labels=np.arange(10),zero_division=0)
            class_rows=per_class[per_class.experiment.eq(experiment)].set_index('attack_cat').loc[list(CLASSES)]
            for name,array in [('precision',p),('recall',r),('f1',f),('support',s),('correct',np.diag(cm))]:
                np.testing.assert_allclose(class_rows[name],array,rtol=0,atol=1e-14)
            tn,fp,fn,tp=confusion_matrix(links.label,saved.predicted_binary_attack,labels=[0,1]).ravel()
            for name,value in [('attack_precision',tp/(tp+fp)),('attack_recall',tp/(tp+fn)),
                ('false_positive_rate',fp/(fp+tn)),('false_negative_rate',fn/(fn+tp))]:
                assert np.isclose(row[name],value,rtol=0,atol=1e-14)
            print(f'Verified saved model and results: {experiment}',flush=True)
    declared=read('models/baselines/06/06_provisional_decision.json')
    counts={s:len(read(f'models/preprocessing/05/05_{s}_features.json')['retained']) for s in SETS}
    assert choose_candidate(summary.reset_index(),counts)==declared==manifest['decision']
    differences=pd.read_csv(ROOT/'results/tables/06_feature_comparisons.csv',float_precision='round_trip')
    assert len(differences)==16
    for contrast in differences.itertuples(index=False):
        reference=summary.loc[f'{contrast.model}_{contrast.reference}']
        candidate=summary.loc[f'{contrast.model}_{contrast.candidate}']
        for column in differences.columns:
            if column.startswith('delta_'):
                metric=column.removeprefix('delta_')
                assert np.isclose(getattr(contrast,column),candidate[metric]-reference[metric],rtol=0,atol=1e-14)
    tradeoffs=pd.read_csv(ROOT/'results/tables/06_class_tradeoffs.csv',float_precision='round_trip')
    assert len(tradeoffs)==160
    indexed=per_class.set_index(['model','feature_set','attack_cat'])
    for contrast in tradeoffs.itertuples(index=False):
        reference=indexed.loc[(contrast.model,contrast.reference,contrast.attack_cat)]
        candidate=indexed.loc[(contrast.model,contrast.candidate,contrast.attack_cat)]
        for metric in ('precision','recall','f1','correct'):
            assert np.isclose(getattr(contrast,'delta_'+metric),candidate[metric]-reference[metric],rtol=0,atol=1e-14)
    # Predictions must exist only for the permitted role, and there must be exactly ten.
    predictions=list((ROOT/'data/processed/06').glob('*predictions.csv'))
    assert len(predictions)==10 and all('model_selection' in p.name for p in predictions)
    result=dict(code_cells=len(code),errors=0,models_reloaded_and_verified=10,prediction_rows_per_model=17534,
        fixed_ten_class_metrics_recomputed=True,all_confusion_matrices_verified=True,
        probability_order_and_binary_derivation_verified=True,decision_recomputed=True,
        all_feature_differences_and_class_tradeoffs_recomputed=True,
        verified_artifacts=len(manifest['artifact_sha256']),earlier_files_unchanged=len(run['protected_before']),
        calibration_predictions=0,official_test_evaluations=0,raw_and_assignment_integrity=True)
    write_json(ROOT/'results/tables/06_independent_verification.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
