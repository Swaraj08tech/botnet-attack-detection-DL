"""Small shared plumbing for the v3 notebooks: paths, saving and plot style.

All analysis lives in the notebooks themselves. This file only holds the
boring repeated parts so every notebook can start with one short setup cell.
"""

from pathlib import Path
import hashlib
import json

import matplotlib.pyplot as plt
import pandas as pd


def find_root(start=None):
    """Walk upwards until the project folder (with data/raw and src) is found."""
    here = Path(start or Path.cwd()).resolve()
    for folder in (here, *here.parents):
        if (folder / 'data/raw').is_dir() and (folder / 'src').is_dir():
            return folder
    raise FileNotFoundError('Run the notebook from inside the project folder.')


ROOT = find_root()
RAW = ROOT / 'data/raw'
TRAIN_CSV = RAW / 'UNSW_NB15_training-set.csv'
TEST_CSV = RAW / 'UNSW_NB15_testing-set.csv'

# Every v3 output goes into its own folder so older results stay untouched.
TABLES = ROOT / 'results/tables/v3'
FIGURES = ROOT / 'results/figures/v3'
DATA = ROOT / 'data/processed/v3'
MODELS = ROOT / 'models/v3'
DOCS = ROOT / 'docs/v3'
for _folder in (TABLES, FIGURES, DATA, MODELS, DOCS):
    _folder.mkdir(parents=True, exist_ok=True)

TARGETS = ['id', 'label', 'attack_cat']
CATEGORICAL = ['proto', 'service', 'state']
CLASSES = ['Analysis', 'Backdoor', 'DoS', 'Exploits', 'Fuzzers', 'Generic',
           'Normal', 'Reconnaissance', 'Shellcode', 'Worms']
ROLES = ['train', 'model_selection', 'calibration']

# Expected fingerprints of the two supplied CSV files (recorded when they were placed).
RAW_SHA256 = {
    'train': 'bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa',
    'test': '734fe6642edf758f7c94d7d9149426b49d202fe8e7bf0bef47392489c3c0a559',
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def save_table(frame, name, index=False):
    path = TABLES / f'{name}.csv'
    frame.to_csv(path, index=index)
    return path


def save_figure(fig, name):
    path = FIGURES / f'{name}.png'
    fig.savefig(path, dpi=130, bbox_inches='tight')
    return path


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, default=str), encoding='utf-8')
    return path


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def setup_style():
    pd.set_option('display.max_columns', 60)
    pd.set_option('display.width', 200)
    pd.set_option('display.float_format', lambda v: f'{v:,.4f}')
    plt.rcParams.update({'figure.dpi': 100, 'axes.grid': True, 'grid.alpha': 0.3,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titleweight': 'bold'})


# ---------------------------------------------------------------------------
# One shared results table ("leaderboard") for every modelling notebook.
# Each notebook writes its own file; read_leaderboard() joins them, so the
# final comparison table in a paper comes from one place with one format.
# ---------------------------------------------------------------------------
LEADERBOARD_COLUMNS = ['notebook', 'model', 'feature set', 'runs', 'macro F1 mean', 'macro F1 std',
                       'accuracy', 'attack recall', 'false-positive rate', 'attack avg precision',
                       'Worms correct (of 13)', 'parameters', 'train minutes per run']


def write_leaderboard(notebook, rows):
    frame = pd.DataFrame(rows).reindex(columns=LEADERBOARD_COLUMNS)
    frame['notebook'] = str(notebook)
    frame.to_csv(TABLES / f'leaderboard_{notebook}.csv', index=False)
    return frame


def read_leaderboard(upto=None):
    """All leaderboard rows; with upto='08' only rows from notebooks 01-08 (what existed at that point)."""
    files = sorted(TABLES.glob('leaderboard_[0-9][0-9].csv'))
    if upto is not None:
        files = [f for f in files if f.stem[-2:] <= upto]
    parts = [pd.read_csv(p, dtype={'notebook': str}) for p in files]
    board = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=LEADERBOARD_COLUMNS)
    return board.sort_values('macro F1 mean', ascending=False).reset_index(drop=True)
