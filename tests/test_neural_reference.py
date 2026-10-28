"""Tests for meaningful protocol failures and saved-model prediction contracts."""
import joblib
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from src.neural_reference import assert_groups, labels, make_model, fit_fixed, predict, CLASSES, CONFIGS, EPOCHS

def test_conflicting_labels_do_not_change_group_membership():
    links=pd.DataFrame({'input_row':[0,1,2],'row_position':[10,11,12],'feature_group_id':[7,7,8]})
    folds=links.copy(); folds['internal_holdout_fold']=[0,0,1]
    extra=pd.DataFrame({'input_row':[3],'row_position':[13],'feature_group_id':[9],'internal_holdout_fold':[2]})
    folds=pd.concat([folds,extra],ignore_index=True); links=folds.iloc[:,:3].copy()
    assert_groups(folds,links)
    folds.loc[1,'internal_holdout_fold']=1
    with pytest.raises(ValueError,match='cross inner folds'): assert_groups(folds,links)

def test_reordered_fold_rows_are_rejected():
    link=pd.DataFrame({'input_row':[0,1,2],'row_position':[10,11,12],'feature_group_id':[7,8,9]})
    folds=link.copy(); folds['internal_holdout_fold']=[0,1,2]
    with pytest.raises(ValueError,match='row mismatch'): assert_groups(folds.iloc[::-1],link)

def test_unknown_or_inconsistent_targets_are_rejected():
    with pytest.raises(ValueError): labels(pd.DataFrame({'attack_cat':['Botnet'],'label':[1]}))
    with pytest.raises(ValueError): labels(pd.DataFrame({'attack_cat':['Normal'],'label':[1]}))
    assert labels(pd.DataFrame({'attack_cat':CLASSES,'label':[c!='Normal' for c in CLASSES]})).tolist()==list(range(10))

def test_fixed_budget_sparse_softmax_round_trip(tmp_path):
    rng=np.random.default_rng(91); x=sparse.csr_matrix(rng.normal(size=(100,7)))
    y=np.tile(np.arange(10),10); m=make_model(CONFIGS[0],42).set_params(batch_size=32)
    _,caught=fit_fixed(m,x,y)
    assert m.n_iter_==EPOCHS and m.t_==EPOCHS*len(y)
    assert m.early_stopping is False and m.n_iter_no_change>EPOCHS
    pred,p=predict(m,x); assert p.shape==(100,10) and np.allclose(p.sum(1),1)
    assert np.array_equal(pred,p.argmax(1))
    path=tmp_path/'m.joblib'; joblib.dump(m,path); a,b=predict(joblib.load(path),x)
    assert np.array_equal(a,pred) and np.array_equal(b,p)

def test_prediction_contract_rejects_missing_classes():
    m=make_model(CONFIGS[0],42).set_params(batch_size=32); x=sparse.csr_matrix(np.ones((40,3)))
    fit_fixed(m,x,np.tile(np.arange(8),5))
    with pytest.raises(AssertionError): predict(m,x)
