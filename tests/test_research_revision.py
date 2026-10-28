"""Research-critical regressions: grouping, fold fitting, metric and input contracts."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
import joblib
from src.data.duplicate_investigation import exact_feature_ids
from src.data.feature_engineering import retained_features
from src.baseline_models import evaluate,CLASSES,NORMAL
from src.research_revision.baselines import model,CONFIGS,FAMILIES
from src.research_revision.error_analysis import minimum_vector_errors

ROOT=Path(__file__).resolve().parents[1]

def test_conflicting_targets_do_not_define_groups():
    frame=pd.DataFrame({'x':[2.,2.,3.],'label':[0,1,1],'attack_cat':['Normal','DoS','DoS']})
    groups,_=exact_feature_ids(frame,['x'])
    assert groups.tolist()==[0,0,1]

def test_frozen_inner_folds_keep_patterns_and_role_links():
    links=pd.read_csv(ROOT/'data/processed/05/05_row_links_train.csv')
    folds=pd.read_csv(ROOT/'data/processed/05/05_internal_folds.csv')
    assert np.array_equal(links.row_position,folds.row_position)
    assert np.array_equal(links.attack_cat,folds.attack_cat)
    assert folds.groupby('feature_group_id').internal_holdout_fold.nunique().max()==1
    assert set(folds.internal_holdout_fold)=={0,1,2}

def test_clone_fits_scaler_on_inner_rows_and_handles_unseen_categories():
    original=joblib.load(ROOT/'models/preprocessing/03/03_scaled_pipeline.joblib')
    columns=__import__('json').loads((ROOT/'results/splits/02_split_manifest.json').read_text())['predictors']
    raw=pd.read_csv(ROOT/'data/raw/UNSW_NB15_training-set.csv',nrows=120)[columns]
    fitting=raw.iloc[:60].copy(); holdout=raw.iloc[60:].copy()
    holdout['sbytes']=1e12; holdout['proto']='__new_protocol__'
    pipeline=clone(original).fit(fitting)
    logcols=pipeline.transformers_[0][2]
    assert np.array_equal(pipeline.named_transformers_['log'].named_steps['scale'].center_,np.median(np.log1p(fitting[logcols]),axis=0))
    assert '__new_protocol__' not in pipeline.named_transformers_['cat'].categories_[0]
    assert np.isfinite(pipeline.transform(holdout).data).all()

def test_engineering_zero_sentinel_and_direction_are_target_free():
    raw=pd.DataFrame(dict(sbytes=[0,900,100],dbytes=[0,100,900],spkts=[0,6,4],dpkts=[0,4,6],dur=[0,1,1]))
    result=retained_features(raw)
    assert np.allclose(result.fe_byte_balance,[0,.8,-.8])
    assert np.allclose(result.fe_packet_balance,[0,.2,-.2])
    with_targets=raw.assign(label=[0,1,0],attack_cat=['x','y','z'],id=[999,0,1])
    assert result.equals(retained_features(with_targets))

def test_fixed_ten_macro_and_binary_confusions():
    y=np.arange(10); pred=np.full(10,NORMAL); p=np.zeros((10,10)); p[:,NORMAL]=1
    metrics,per,cm,normalized,pr,score=evaluate(y,pred,p)
    assert np.isclose(metrics['macro_f1'],(2/11)/10)
    assert metrics['false_positive_rate']==0 and metrics['attack_recall']==0
    assert metrics['false_negative']==9 and cm.sum()==10
    assert len(per)==10 and per.support.eq(1).all()
    assert np.allclose(normalized.sum(axis=1),1) and np.array_equal(score,np.zeros(10))

def test_bounded_models_are_unweighted_and_same_setting_for_variants():
    assert sum(len(CONFIGS[f])*3 for f in FAMILIES)==18
    for family in FAMILIES:
        for config in CONFIGS[family]:
            params=model(family,config).get_params(deep=True)
            assert params.get('estimator__class_weight',params.get('class_weight')) is None
    boosted=model('lightgbm',CONFIGS['lightgbm'][0])
    assert boosted.get_params()['zero_as_missing'] is False
    assert boosted.get_params()['n_estimators']==120

def test_vector_error_constraint_does_not_count_excess_duplicates_as_errors():
    assert minimum_vector_errors([[2,1,0],[0,0,1]])==1
    assert minimum_vector_errors([[50,0],[0,100]])==0
