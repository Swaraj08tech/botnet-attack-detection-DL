# %% [markdown]
# # 13 · TCN-BiLSTM Block
#
# **The question:** Does adding a **TCN** (dilated convolutions) and a **BiLSTM** after the fusion improve the network?
#
# | | |
# |---|---|
# | **Input** | Scaled `total_and_byte44` inputs; weighting (Notebook 10), convolution (Notebook 11) and fusion (Notebook 12) chosen so far |
# | **What we do** | Train the network with and without the TCN-BiLSTM block, 3 seeds each |
# | **Output** | A decision: keep the block or not. Notebook 14 assembles the proposed model |
#
# **An honest note before we start.** TCNs and BiLSTMs were invented for **sequences in time**. Our rows are single completed flows with no timestamps
# (Notebook 01), so here the block runs over the **sequence of feature tokens**, not over time. It can still learn relations between distant features,
# but it cannot learn anything about the order of events. Notebook 08 already found that a BiLSTM alone added nothing; this notebook tests the fuller
# TCN-BiLSTM block under the improved setup, so the paper can report a fair answer.
#
# **Decision rule (same as Notebooks 11-12):** keep the block only if it raises the 3-seed mean macro F1 and is better in at least 2 of the 3 seeds.

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

from src.notebook_helpers import TRAIN_CSV, RAW_SHA256, CLASSES, MODELS, sha256, save_table, save_figure, save_json, read_json, setup_style, write_leaderboard
from src.flow_networks import FlowData, BASE_CONFIG, TCNBiLSTM, run_config, paired_comparison, leaderboard_row, show_code, DEVICE
setup_style()
assert sha256(TRAIN_CSV) == RAW_SHA256['train']

WEIGHTING = read_json(MODELS / '10_rare_class_choice.json')['network_weighting']
previous = read_json(MODELS / '12_decision.json')
SCALES, FUSION = previous['scales'], previous['fusion']
print(f'weighting: {WEIGHTING} | scales: {SCALES} | fusion: {FUSION} | device: {DEVICE}')
data = FlowData('total_and_byte44')

# %% [markdown]
# ## 2. The block
#
# - **TCN:** three convolutions with dilation 1, 2 and 4. Dilation means the window skips positions, so the third layer looks 4 tokens apart.
#   Each layer adds to its input (a *residual* connection), so the original signal is never lost.
# - **BiLSTM:** reads the token sequence from left to right and from right to left, and combines both readings.

# %%
show_code(TCNBiLSTM)

# %% [markdown]
# ## 3. Train both versions (3 seeds each)

# %%
WITHOUT = dict(BASE_CONFIG, weighting=WEIGHTING, scales=SCALES, fusion=FUSION, tcn_bilstm=False)
WITH = dict(BASE_CONFIG, weighting=WEIGHTING, scales=SCALES, fusion=FUSION, tcn_bilstm=True)
without_runs, without_proba, _, _ = run_config(data, WITHOUT, label='without TCN-BiLSTM')
with_runs, with_proba, _, _ = run_config(data, WITH, label='with TCN-BiLSTM')

# %% [markdown]
# ## 4. Results

# %%
runs = pd.concat([without_runs, with_runs], ignore_index=True)
save_table(runs, '13_runs')
display(runs[['experiment', 'seed', 'macro F1', 'accuracy', 'false-positive rate', 'Worms correct (of 13)', 'epochs', 'train minutes', 'parameters']])
display(runs.groupby('experiment', sort=False)[['macro F1', 'train minutes']].agg(['mean', 'std']))

y_eval = data.y['model_selection'].numpy()
comparison, overall = paired_comparison(y_eval, with_proba, without_proba, 'with vs without TCN-BiLSTM')
save_table(comparison, '13_comparison')
display(comparison)

per_class = pd.DataFrame({
    'without F1': np.mean([f1_score(y_eval, p.argmax(1), average=None, labels=range(10), zero_division=0) for p in without_proba.values()], axis=0),
    'with F1': np.mean([f1_score(y_eval, p.argmax(1), average=None, labels=range(10), zero_division=0) for p in with_proba.values()], axis=0),
}, index=CLASSES)
per_class['change'] = per_class.iloc[:, 1] - per_class.iloc[:, 0]
save_table(per_class, '13_per_class', index=True)
display(per_class.round(3))

# %% [markdown]
# **What we see:**
#
# - **With TCN-BiLSTM the mean macro F1 rises from 0.596 to 0.603** and is higher for 2 of 3 seeds. The seed tests: chance for seeds 42 and 43,
#   clearly better for seed 44. Again a small gain that is not reliable on its own.
# - The improvement is broad but small: 8 of the 10 classes gain a little (largest: Backdoor +0.016, Worms +0.013, Shellcode +0.012).
# - False alarms fall slightly (about 17.4% → 15.6% on average).
# - The block nearly doubles the size (148k vs 86k parameters), yet training time stays similar because it stops earlier.
# - This differs from Notebook 08, where a plain BiLSTM added nothing. Here the block includes the TCN with residual connections and sits on top of the
#   weighted, cross-attention design. Its gain still comes from relating **feature tokens**, not from time order (there is none in these rows).

# %% [markdown]
# ## 5. Decision

# %%
wins = int(sum(with_runs.set_index('seed')['macro F1'] > without_runs.set_index('seed')['macro F1']))
gain = with_runs['macro F1'].mean() - without_runs['macro F1'].mean()
keep = bool(gain > 0 and wins >= 2)
decision = {'block': 'TCN-BiLSTM', 'mean gain': round(float(gain), 4), 'seeds where better': f'{wins} of 3',
            'bootstrap verdict over seeds': overall, 'kept': keep, 'tcn_bilstm': keep, 'fusion': FUSION, 'scales': SCALES}
save_json(decision, MODELS / '13_decision.json')
display(pd.Series(decision).to_frame('decision'))
label = f'scales {SCALES}, {FUSION}-attention ({WEIGHTING})'
write_leaderboard('13', [leaderboard_row(f'FlowNet {label}', without_runs), leaderboard_row(f'FlowNet {label} + TCN-BiLSTM', with_runs)])
assert sha256(TRAIN_CSV) == RAW_SHA256['train']

# %% [markdown]
# ## Summary
#
# ### What we learned
#
# | Question | Answer |
# |---|---|
# | Does TCN-BiLSTM help? | **Slightly**: +0.007 mean macro F1, better in 2 of 3 seeds, bootstrap clear only for one seed |
# | Where? | Small gains spread over most classes; slightly fewer false alarms |
# | What does it cost? | About 1.7× more parameters (148k vs 86k) |
# | Decision (by the fixed rule) | **Kept** |
#
# ### The honest overall picture of Notebooks 11-13
#
# All three blocks passed the rule, but each by a small margin that depends on the seed. Together they lift the network from 0.588 (Notebook 10, sqrt
# weights) to 0.603. LightGBM with sqrt weights is at 0.666 on the same rows, so the network is still clearly behind.
#
# ### What comes next
#
# **Notebook 14** assembles the proposed model from these decisions, gives it a small fair tuning, and compares it with all earlier models.
