"""Train one Notebook 08 network (arm + seed) outside the notebook, using the notebook's own code.

The setup and training cells of notebooks/08_conv1d_attention_model.ipynb are executed unchanged, then
train_arm(arm, seed) saves the checkpoint that the notebook later loads (RETRAIN = False). This lets several
arms train in parallel processes instead of one after another.

Usage:  .venv/Scripts/python.exe scripts/pretrain_08_arm.py B_no_attention 43
"""
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
arm, seed = sys.argv[1], int(sys.argv[2])
nb = nbformat.read(ROOT / 'notebooks/08_conv1d_attention_model.ipynb', 4)
namespace = {'display': print, '__name__': '__pretrain__'}
for cell in nb.cells:
    if cell.cell_type != 'code':
        continue
    source = cell.source
    exec(compile(source, f'08 cell', 'exec'), namespace)
    if 'def run(arm):' in source:          # everything needed for training is now defined
        break
model, history = namespace['train_arm'](arm, seed)
print(f'DONE {arm} seed {seed}: {len(history)} epochs, best epoch {int(history["stopping loss"].idxmin()) + 1}')
