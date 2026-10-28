"""Explain the frozen experiment's actual results; no fitting or model choices."""
import pandas as pd
from IPython.display import display, Markdown
from src.neural_reference import SETS, save_table


def explain_results(s):
    summary=s['summary'].set_index('feature_set')
    lines=[]
    for candidate in SETS:
        r=summary.loc[candidate]
        lines.append(f'**{candidate}:** The MLP averages {r.mean_macro_f1:.4f} macro F1 across three seeds; '
            f'the matching saved LightGBM scores {r.saved_lightgbm_macro_f1:.4f}. '
            f'The MLP difference is {r.mean_MLP_minus_LightGBM_pp:+.2f} percentage points.')
    delta=s['metrics'].pivot(index='seed',columns='feature_set',values='macro_f1')
    dif=100*(delta.total_and_byte44-delta.original42)
    direction='increase' if dif.mean()>0 else 'decrease'
    lines.append(f'**The added features:** They {direction} MLP macro F1 by {abs(dif.mean()):.2f} percentage points on average. '
        f'The three paired changes range from {dif.min():+.2f} to {dif.max():+.2f} points. '
        'This is a development comparison; it does not establish a reliable gain on new traffic.')
    if dif.min() < 0 < dif.max():
        lines.append('The added features help some seeds and hurt another. The direction changes across these runs, so the improvement is not consistent.')
    confusion=pd.read_csv(s['out']['tables']/'07_important_confusions.csv')
    rows=[]
    for candidate in SETS:
        pc=s['per_class'][s['per_class'].seed.eq(42)&s['per_class'].feature_set.eq(candidate)].set_index('attack_cat')
        for cat in ('Analysis','Backdoor','DoS','Worms'):
            r=pc.loc[cat]
            c=confusion[confusion.model.eq('mlp')&confusion.feature_set.eq(candidate)&confusion.seed.eq(42)&confusion.actual_class.eq(cat)].iloc[0]
            rows.append(dict(feature_set=candidate,actual_class=cat,
                correct=int(r.correct),out_of=int(r.support),predicted_as_Exploits=int(c.predicted_as_Exploits)))
    frame=save_table(s,pd.DataFrame(rows),'simple_weak_class_counts')
    p42=s['per_class'][s['per_class'].seed.eq(42)&s['per_class'].feature_set.eq('original42')].set_index('attack_cat')
    p44=s['per_class'][s['per_class'].seed.eq(42)&s['per_class'].feature_set.eq('total_and_byte44')].set_index('attack_cat')
    tree=pd.read_csv(s['out']['tables']/'07_all_class_results.csv')
    tree=tree[tree.model.eq('LightGBM')&tree.feature_set.eq('total_and_byte44')].set_index('attack_cat')
    lines.append(f'**A concrete tradeoff:** For seed 42, adding the two features changes correctly found DoS flows from '
        f'{int(p42.loc["DoS","correct"])} to {int(p44.loc["DoS","correct"])} out of 1,226. '
        f'But correctly found Exploits flows change from {int(p42.loc["Exploits","correct"])} to {int(p44.loc["Exploits","correct"])} '
        f'out of 3,340, and Fuzzers from {int(p42.loc["Fuzzers","correct"])} to {int(p44.loc["Fuzzers","correct"])} out of 1,818. '
        'This is a class tradeoff, not an improvement everywhere.')
    lines.append(f'**Compared with the matching tree:** The seed-42 44-feature MLP finds '
        f'{int(p44.loc["DoS","correct"])} DoS rows, versus {int(tree.loc["DoS","correct"])} for LightGBM. '
        f'LightGBM finds {int(tree.loc["Worms","correct"])} of the 13 Worms rows; the MLP finds '
        f'{int(p44.loc["Worms","correct"])}. The tree has the higher overall macro F1, but neither model solves every class.')
    lines.append('**The weak classes:** The small table below gives the fixed seed-42 counts. '
        'Correct counts show how many flows of that class were found; the last column shows errors toward Exploits. '
        'A high overall accuracy must not hide these missed classes. One changed Worms prediction changes its recall by 7.69 percentage points.')
    lines.append('**Decision:** Keep this MLP as a checked neural reference. Keep the exact tree controls and both feature sets for the later Conv1D comparison. '
        'Do not increase epochs, change class weights or pick a lucky seed after inspecting these results.')
    text='\n\n'.join(lines)
    s['notes'].append(text); display(Markdown(text)); display(frame)
    path=s['out']['docs']/'07_results_in_simple_words.md'; path.write_text(text+'\n\n'+frame.to_string(index=False)+'\n',encoding='utf-8')
    s['outputs'].append(path)
