# %% [markdown]
# # 11 · Multi-Scale Conv1D
#
# **The question:** Our network's convolution looks at **3 neighbouring features** at a time. Does it help to look at **3, 5 and 7** neighbours at once
# (a *multi-scale* convolution)?
#
# | | |
# |---|---|
# | **Input** | Scaled `total_and_byte44` inputs; the network weighting chosen in Notebook 10 |
# | **What we do** | Train the same network with a single-scale and a multi-scale convolution, 3 seeds each, and compare |
# | **Output** | A decision: keep multi-scale or not. Notebook 12 builds on the winner |
#
# **Decision rule (fixed before training, the same in Notebooks 11-13):** a new block is kept only if it raises the **3-seed mean macro F1** and is
# better in **at least 2 of the 3 seeds**.

# %% [markdown]
# ## 1. Setup

# %%
import sys
import warnings
from pathlib import Path
sys.path.insert(0, str(next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'src').is_dir())))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.metrics import f1_score

from src.notebook_helpers import TRAIN_CSV, RAW_SHA256, CLASSES, DATA, MODELS, sha256, save_table, save_figure, save_json, read_json, setup_style, write_leaderboard
from src.flow_networks import FlowData, BASE_CONFIG, ConvBranch, run_config, paired_comparison, leaderboard_row, show_code, DEVICE
setup_style()
assert sha256(TRAIN_CSV) == RAW_SHA256['train']

WEIGHTING = read_json(MODELS / '10_rare_class_choice.json')['network_weighting']
print(f'network weighting from Notebook 10: {WEIGHTING} | device: {DEVICE}')
data = FlowData('total_and_byte44')

# %% [markdown]
# ## 2. What does "multi-scale" mean?
#
# A convolution with window size 3 combines each feature with its 2 direct neighbours in our topic order. A window of 5 or 7 reaches further.
# Here is what each window sees around `sttl` (time-to-live) in branch A:

# %%
position = data.branch_a.index('sttl')
for k in (3, 5, 7):
    half = k // 2
    window = data.branch_a[max(0, position - half): position + half + 1]
    print(f'window {k}: {window}')

# %% [markdown]
# A larger window can combine, for example, TTL with TCP timing *and* window sizes in one step. The multi-scale block runs all three windows in
# parallel and mixes the results:

# %%
show_code(ConvBranch)

# %% [markdown]
# To keep the comparison fair, the multi-scale branch has about the same total width (64 channels) as the single-scale one, so it does not simply
# win by being bigger. The parameter counts are shown in the results.

# %% [markdown]
# ## 3. Train both versions (3 seeds each)

# %%
SINGLE = dict(BASE_CONFIG, weighting=WEIGHTING, scales=[3])
MULTI = dict(BASE_CONFIG, weighting=WEIGHTING, scales=[3, 5, 7])
single_runs, single_proba, _, single_hist = run_config(data, SINGLE, label='single-scale (3)')
multi_runs, multi_proba, _, multi_hist = run_config(data, MULTI, label='multi-scale (3, 5, 7)')

# %% [markdown]
# ## 4. Results

# %%
runs = pd.concat([single_runs, multi_runs], ignore_index=True)
save_table(runs, '11_runs')
display(runs[['experiment', 'seed', 'macro F1', 'accuracy', 'false-positive rate', 'Worms correct (of 13)', 'epochs', 'train minutes', 'parameters']])
summary = runs.groupby('experiment', sort=False)['macro F1'].agg(['mean', 'std', 'min', 'max'])
display(summary)

y_eval = data.y['model_selection'].numpy()
comparison, overall = paired_comparison(y_eval, multi_proba, single_proba, 'multi-scale vs single-scale')
save_table(comparison, '11_comparison')
display(comparison)

fig, ax = plt.subplots(figsize=(7, 4))
for i, (name, r) in enumerate([('single-scale (3)', single_runs), ('multi-scale (3, 5, 7)', multi_runs)]):
    ax.scatter([i] * len(r), r['macro F1'], s=50, color=['#4C72B0', '#DD8452'][i], zorder=3)
    ax.hlines(r['macro F1'].mean(), i - 0.2, i + 0.2, color='black')
ax.set_xticks([0, 1], ['single-scale (3)', 'multi-scale (3, 5, 7)']); ax.set(ylabel='macro F1 on model_selection', title='Each dot is one seed; line = mean')
plt.tight_layout(); save_figure(fig, '11_scales'); plt.show()

# %% [markdown]
# ## 5. Class by class

# %%
per_class = pd.DataFrame({
    'single-scale F1 (mean of seeds)': np.mean([f1_score(y_eval, p.argmax(1), average=None, labels=range(10), zero_division=0) for p in single_proba.values()], axis=0),
    'multi-scale F1 (mean of seeds)': np.mean([f1_score(y_eval, p.argmax(1), average=None, labels=range(10), zero_division=0) for p in multi_proba.values()], axis=0),
}, index=CLASSES)
per_class['change'] = per_class.iloc[:, 1] - per_class.iloc[:, 0]
save_table(per_class, '11_per_class', index=True)
display(per_class.round(3))

# %% [markdown]
# **What we see:**
#
# - **Multi-scale is slightly better on average** (0.593 vs 0.588) and better for 2 of the 3 seeds, but the seed-by-seed tests **disagree**:
#   clearly worse for seed 42, chance for seed 43, clearly better for seed 44. The average gain is small and not reliable on its own.
# - **Its clearest effect is stability:** the spread between seeds drops from 0.015 to 0.004. The single-scale network has one lucky seed (0.604) and two
#   weaker ones; the multi-scale network is consistently near 0.59.
# - Per class, multi-scale helps the hard, conflicting classes a little (Analysis +0.02, Backdoor +0.03, DoS +0.02) and costs a little on Worms (-0.02,
#   about one row out of 13).
# - Both versions have about the same size (60k vs 69k parameters), so the difference is not simply more capacity.

# %% [markdown]
# ## 6. Decision

# %%
wins = int(sum(multi_runs.set_index('seed')['macro F1'] > single_runs.set_index('seed')['macro F1']))
gain = multi_runs['macro F1'].mean() - single_runs['macro F1'].mean()
keep = bool(gain > 0 and wins >= 2)
decision = {'block': 'multi-scale convolution', 'mean gain': round(float(gain), 4), 'seeds where better': f'{wins} of 3',
            'bootstrap verdict over seeds': overall, 'kept': keep, 'scales': [3, 5, 7] if keep else [3]}
save_json(decision, MODELS / '11_decision.json')
display(pd.Series(decision).to_frame('decision'))
write_leaderboard('11', [leaderboard_row(f'FlowNet single-scale ({WEIGHTING})', single_runs),
                         leaderboard_row(f'FlowNet multi-scale ({WEIGHTING})', multi_runs)])
assert sha256(TRAIN_CSV) == RAW_SHA256['train']

# %% [markdown]
# ## Summary
#
# ### What we learned
#
# | Question | Answer |
# |---|---|
# | Does multi-scale convolution help? | **A little, and not reliably:** +0.006 mean macro F1, better in 2 of 3 seeds, bootstrap verdict depends on the seed |
# | What does it clearly do? | Makes the network **more stable across seeds** (std 0.015 → 0.004) |
# | Decision (by the fixed rule) | **Kept**: scales 3, 5 and 7 |
#
# ### What is still uncertain
#
# - Whether the small gain holds on new data; the ablation study (Notebook 15) removes it again under the final design, and Notebook 19 reports test results.
#
# ### What comes next
#
# **Notebook 12** tests the second block: cross-attention fusion between flow measurements and context.
