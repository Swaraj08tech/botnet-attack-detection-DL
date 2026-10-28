import unittest
import numpy as np
import pandas as pd

from src.baseline_models import (CLASSES,NORMAL,make_model,exact_settings,evaluate,
    choose_candidate,comparisons,SETS,MODELS)


class BaselineTests(unittest.TestCase):
    def test_fixed_macro_and_binary_definition(self):
        # Only two observed classes: absent classes must still count in ten-class macro F1.
        y=np.array([NORMAL,NORMAL,0,0]); pred=np.array([NORMAL,0,0,NORMAL])
        p=np.zeros((4,10)); p[:,NORMAL]=[.6,.3,.2,.7]; p[:,0]=1-p[:,NORMAL]
        metrics,classes,cm,norm,pr,score=evaluate(y,pred,p)
        self.assertAlmostEqual(metrics['macro_f1'],.1)
        self.assertEqual(cm.shape,(10,10)); self.assertEqual(classes.support.sum(),4)
        self.assertEqual(classes.zero_support.sum(),8)
        self.assertEqual(metrics['attack_recall'],.5); self.assertEqual(metrics['false_positive_rate'],.5)
        self.assertEqual(metrics['attack_precision'],.5)
        np.testing.assert_array_equal(score,1-p[:,NORMAL])
        self.assertTrue(np.isfinite(norm).all()); self.assertEqual(len(pr),5)

    def test_multiclass_attack_not_half_probability(self):
        # Normal is the most likely single class even though summed attack probability exceeds .5.
        p=np.array([[.3,.3,0,0,0,0,.4,0,0,0],[.7,0,0,0,0,0,.3,0,0,0]])
        metrics,*_=evaluate(np.array([NORMAL,0]),np.argmax(p,axis=1),p)
        self.assertEqual(metrics['false_positive_rate'],0)
        self.assertEqual(metrics['attack_recall'],1)

    def test_missing_binary_denominators_unavailable(self):
        p=np.zeros((2,10)); p[:,NORMAL]=1
        metrics,*_=evaluate([NORMAL,NORMAL],[NORMAL,NORMAL],p)
        self.assertTrue(np.isnan(metrics['attack_recall']))
        self.assertTrue(np.isnan(metrics['attack_average_precision']))
        self.assertTrue(np.isnan(metrics['attack_precision']))

    def test_practical_tie_prefers_original_and_ignores_unconverged(self):
        rows=pd.DataFrame([
            dict(model='decision_tree',feature_set='all45',experiment='a',macro_f1=.701,eligible=True),
            dict(model='logistic',feature_set='original42',experiment='b',macro_f1=.7001,eligible=True),
            dict(model='decision_tree',feature_set='original42',experiment='c',macro_f1=.7005,eligible=True),
            dict(model='logistic',feature_set='all45',experiment='d',macro_f1=.99,eligible=False)])
        decision=choose_candidate(rows,{'original42':42,'all45':45})
        self.assertEqual(decision['experiment'],'b')
        self.assertAlmostEqual(decision['gap_to_best'],.0009)

    def test_settings_and_all_named_contrasts(self):
        for m in MODELS:
            settings=exact_settings(make_model(m))
            self.assertFalse(any(v is not None for k,v in settings.items() if k.endswith('class_weight')))
        rows=[];classes=[]
        for family in MODELS:
            for i,candidate in enumerate(SETS):
                row=dict(model=family,feature_set=candidate,eligible=True)
                for name in ('macro_f1','accuracy','weighted_f1','attack_recall','attack_precision',
                             'false_positive_rate','attack_average_precision','worms_f1','worms_recall','worms_correct'):
                    row[name]=i*.01
                rows.append(row)
                for target in CLASSES:
                    classes.append(dict(model=family,feature_set=candidate,attack_cat=target,
                        precision=i*.01,recall=i*.01,f1=i*.01,support=13,correct=i))
        d,c=comparisons(pd.DataFrame(rows),pd.DataFrame(classes))
        self.assertEqual(len(d),16); self.assertEqual(len(c),160)
        match=d[(d.model=='logistic')&(d.reference=='total_and_byte44')&(d.candidate=='all45')]
        self.assertAlmostEqual(match.iloc[0].delta_macro_f1,.02)


if __name__=='__main__': unittest.main()
