"""Shared code for the neural-network notebooks 10-15.

Notebook 08 explains every step of this code in detail. It lives here so that notebooks 10-15 can stay focused on the
one new idea each of them tests. Each notebook prints the source of the building block it introduces (show_code).

Contents
- load_flow_data(): the 41 ordered number features (two branches) + 3 text codes, for the train / model_selection /
  calibration roles of the scaled total_and_byte44 (or original42) inputs.
- class_weights(): 'none', 'sqrt' or 'balanced' loss weights from the training labels.
- Building blocks: ConvBranch (single- or multi-scale), SelfAttentionFusion, CrossAttentionFusion, TCNBiLSTM.
- FlowNet: one configurable network that combines the blocks (every experiment in 10-15 is a FlowNet config).
- train_network(): GPU training with early stopping on a grouped 10% stopping slice of the train rows; cached.
- score(), bootstrap_difference(): the evaluation used in every modelling notebook.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')   # needed for deterministic GPU matrix maths

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from IPython.display import Markdown, display
from scipy import sparse
from sklearn.metrics import accuracy_score, average_precision_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from src.notebook_helpers import CLASSES, DATA, MODELS, read_json

NORMAL, WORMS = CLASSES.index('Normal'), CLASSES.index('Worms')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINTS = MODELS / 'nn_checkpoints'
CHECKPOINTS.mkdir(parents=True, exist_ok=True)

BRANCH_A = ['dur', 'sbytes', 'dbytes', 'spkts', 'dpkts', 'smean', 'dmean', 'fe_total_bytes', 'fe_byte_balance',
            'rate', 'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit', 'tcprtt', 'synack', 'ackdat',
            'sttl', 'dttl', 'swin', 'dwin', 'stcpb', 'dtcpb', 'is_sm_ips_ports']
BRANCH_B = ['trans_depth', 'response_body_len', 'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst',
            'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'ct_state_ttl']
TEXT = ['proto', 'service', 'state']
ROLES = ['train', 'model_selection', 'calibration']


def show_code(*objects):
    """Display the source code of a building block inside the notebook."""
    for obj in objects:
        display(Markdown(f'```python\n{inspect.getsource(obj)}\n```'))


def set_deterministic(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


# ---------------------------------------------------------------------------------------------------------------- data
class FlowData:
    """Tensors for one feature set. numbers: (rows, n) float32 in branch order; codes: (rows, 3) int64; y: (rows,)."""

    def __init__(self, feature_set='total_and_byte44'):
        self.feature_set = feature_set
        columns = read_json(MODELS / '05_feature_sets.json')['columns'][f'{feature_set}/scaled']
        self.branch_a = [f for f in BRANCH_A if any(c in columns for c in (f, f'log__{f}', f'plain__{f}'))]
        self.branch_b = list(BRANCH_B)
        self.len_a, self.len_b = len(self.branch_a), len(self.branch_b)

        def column_of(feature):
            for prefix in ('log__', 'plain__', ''):
                if prefix + feature in columns:
                    return columns.index(prefix + feature)
            raise KeyError(feature)

        number_index = [column_of(f) for f in self.branch_a + self.branch_b]
        blocks = {t: [i for i, c in enumerate(columns) if c.startswith(f'cat__{t}_')] for t in TEXT}
        self.vocab = {t: len(b) + 1 for t, b in blocks.items()}
        self.names = self.branch_a + self.branch_b
        self.numbers, self.codes, self.y, self.targets = {}, {}, {}, {}
        for role in ROLES:
            m = sparse.load_npz(DATA / f'05_{feature_set}_scaled_{role}.npz').tocsc()
            self.numbers[role] = torch.from_numpy(m[:, number_index].toarray().astype(np.float32))
            codes = []
            for t in TEXT:
                block = m[:, blocks[t]].toarray()
                codes.append(np.where(block.max(axis=1) > 0, block.argmax(axis=1) + 1, 0))
            self.codes[role] = torch.from_numpy(np.stack(codes, axis=1).astype(np.int64))
            self.targets[role] = pd.read_csv(DATA / f'03_targets_{role}.csv')
            self.y[role] = torch.from_numpy(pd.Categorical(self.targets[role].attack_cat, categories=CLASSES).codes.astype(np.int64))
        # Grouped 10% stopping slice of the train rows (identical patterns kept together), as in Notebook 08.
        roles = pd.read_csv(DATA / '02_roles.csv')
        patterns = roles.loc[roles.role == 'train', 'pattern'].to_numpy()
        splitter = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
        self.learn_rows, self.stop_rows = next(splitter.split(np.zeros(len(patterns)), self.y['train'].numpy(), patterns))


def class_weights(y, scheme):
    """Loss weight per class. 'balanced': n / (10 * n_class); 'sqrt': its square root; 'none': all 1."""
    counts = np.bincount(np.asarray(y), minlength=len(CLASSES)).astype(float)
    balanced = counts.sum() / (len(CLASSES) * counts)
    weights = {'none': np.ones(len(CLASSES)), 'balanced': balanced, 'sqrt': np.sqrt(balanced)}[scheme]
    return weights / (weights * counts).sum() * counts.sum()        # keep the average loss scale comparable


# ------------------------------------------------------------------------------------------------------ building blocks
class ConvBranch(nn.Module):
    """Two Conv1D layers over an ordered feature sequence. scales=(3,) is the Notebook 08 branch;
    scales=(3, 5, 7) runs parallel convolutions with several window sizes and joins their channels (multi-scale)."""

    def __init__(self, scales=(3,), channels=64):
        super().__init__()
        per_scale = channels // len(scales)
        self.paths = nn.ModuleList([nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=k, padding=k // 2), nn.ReLU(),
            nn.Conv1d(32, per_scale, kernel_size=k, padding=k // 2), nn.ReLU()) for k in scales])
        self.mix = nn.Conv1d(per_scale * len(scales), channels, kernel_size=1)   # combine the scales into `channels`

    def forward(self, x):                                   # x: (batch, length)
        x = x[:, None, :]
        return self.mix(torch.cat([path(x) for path in self.paths], dim=1)).transpose(1, 2)   # (batch, length, channels)


class SelfAttentionFusion(nn.Module):
    """Notebook 08 fusion: join both branches into one sequence and let every position attend to every other."""

    def __init__(self, channels=64, heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)

    def forward(self, a, b, text_tokens, return_attention=False):
        h = torch.cat([a, b, text_tokens], dim=1)
        mixed, weights = self.attention(h, h, h, need_weights=return_attention)
        return self.norm(h + mixed), weights


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion: the flow-measurement tokens (branch A) ask questions of the context tokens
    (branch B + protocol/service/state), and the context tokens ask questions of the flow tokens.
    Each side keeps its own sequence and is enriched with the information of the other side."""

    def __init__(self, channels=64, heads=4):
        super().__init__()
        self.flow_asks_context = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.context_asks_flow = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.norm_flow, self.norm_context = nn.LayerNorm(channels), nn.LayerNorm(channels)

    def forward(self, a, b, text_tokens, return_attention=False):
        context = torch.cat([b, text_tokens], dim=1)
        flow_update, w_fc = self.flow_asks_context(a, context, context, need_weights=return_attention)
        context_update, w_cf = self.context_asks_flow(context, a, a, need_weights=return_attention)
        h = torch.cat([self.norm_flow(a + flow_update), self.norm_context(context + context_update)], dim=1)
        return h, (w_fc, w_cf)


class ConcatFusion(nn.Module):
    """No attention: the tokens are only placed side by side (used in the ablation study to remove attention)."""

    def __init__(self, channels=64, heads=4):
        super().__init__()

    def forward(self, a, b, text_tokens, return_attention=False):
        return torch.cat([a, b, text_tokens], dim=1), None


class TCNBiLSTM(nn.Module):
    """A temporal-convolution (TCN) block with growing dilation (1, 2, 4), followed by a bidirectional LSTM.
    On our data it runs over the fused feature-token sequence, not over time."""

    def __init__(self, channels=64):
        super().__init__()
        self.tcn = nn.ModuleList([nn.Conv1d(channels, channels, kernel_size=3, padding=d, dilation=d) for d in (1, 2, 4)])
        self.lstm = nn.LSTM(channels, channels // 2, batch_first=True, bidirectional=True)

    def forward(self, h):                                   # h: (batch, length, channels)
        x = h.transpose(1, 2)
        for conv in self.tcn:
            x = x + torch.relu(conv(x))                     # residual connection keeps the original signal
        out, _ = self.lstm(x.transpose(1, 2))
        return out


class FlowNet(nn.Module):
    """One configurable network. Every experiment in Notebooks 10-15 is a FlowNet with different settings."""

    def __init__(self, len_a, len_b, vocab, scales=(3,), fusion='self', tcn_bilstm=False, channels=64, dropout=0.2):
        super().__init__()
        self.len_a = len_a
        self.branch_a, self.branch_b = ConvBranch(scales, channels), ConvBranch(scales, channels)
        self.position = nn.Parameter(torch.zeros(1, len_a + len_b + len(TEXT), channels))
        self.embeddings = nn.ModuleList([nn.Embedding(vocab[t], channels) for t in TEXT])   # text columns become tokens
        self.fusion = {'self': SelfAttentionFusion, 'cross': CrossAttentionFusion, 'none': ConcatFusion}[fusion](channels)
        self.sequence = TCNBiLSTM(channels) if tcn_bilstm else None
        self.head = nn.Sequential(nn.Linear(channels, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, len(CLASSES)))

    def forward(self, numbers, codes, return_attention=False):
        a = self.branch_a(numbers[:, :self.len_a])
        b = self.branch_b(numbers[:, self.len_a:])
        text_tokens = torch.stack([emb(codes[:, i]) for i, emb in enumerate(self.embeddings)], dim=1)
        n_a, n_b = a.shape[1], b.shape[1]
        a = a + self.position[:, :n_a]
        b = b + self.position[:, n_a:n_a + n_b]
        text_tokens = text_tokens + self.position[:, n_a + n_b:]
        h, weights = self.fusion(a, b, text_tokens, return_attention)
        if self.sequence is not None:
            h = self.sequence(h)
        scores = self.head(h.mean(dim=1))
        return (scores, weights) if return_attention else scores


# ------------------------------------------------------------------------------------------------------------ training
def config_name(config):
    text = json.dumps(config, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:10]


def build(data, config):
    return FlowNet(data.len_a, data.len_b, data.vocab, scales=tuple(config['scales']), fusion=config['fusion'],
                   tcn_bilstm=config['tcn_bilstm'], channels=config['channels'], dropout=config['dropout'])


@torch.no_grad()
def predict_proba(model, numbers, codes, batch=8192):
    model.eval()
    out = []
    for i in range(0, len(numbers), batch):
        out.append(torch.softmax(model(numbers[i:i + batch].to(DEVICE), codes[i:i + batch].to(DEVICE)), dim=1).cpu())
    return torch.cat(out).numpy()


def train_network(data, config, seed, retrain=False, verbose=True):
    """Train one FlowNet on the learning rows; early stopping on the stopping rows; cache the result.

    config keys: scales, fusion, tcn_bilstm, channels, dropout, weighting, lr, max_epochs, patience, batch_size.
    Returns (model, history DataFrame)."""
    config = dict(config, feature_set=data.feature_set)
    path = CHECKPOINTS / f'{config_name(config)}_seed{seed}.pt'
    set_deterministic(seed)
    model = build(data, config).to(DEVICE)
    if path.exists() and not retrain:
        saved = torch.load(path, weights_only=False, map_location=DEVICE)
        model.load_state_dict(saved['weights'])
        return model, saved['history']

    y_train = data.y['train']
    weights = torch.tensor(class_weights(y_train.numpy()[data.learn_rows], config['weighting']), dtype=torch.float32, device=DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    lx, lc, ly = (t[data.learn_rows].to(DEVICE) for t in (data.numbers['train'], data.codes['train'], y_train))
    sx, sc, sy = (t[data.stop_rows].to(DEVICE) for t in (data.numbers['train'], data.codes['train'], y_train))
    generator = torch.Generator().manual_seed(seed)
    history, best, best_weights, waited = [], np.inf, None, 0
    for epoch in range(1, config['max_epochs'] + 1):
        start = time.time()
        model.train()
        order = torch.randperm(len(ly), generator=generator).to(DEVICE)
        total = 0.0
        for i in range(0, len(order), config['batch_size']):
            rows = order[i:i + config['batch_size']]
            optimizer.zero_grad()
            loss = loss_fn(model(lx[rows], lc[rows]), ly[rows])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(rows)
        model.eval()
        with torch.no_grad():
            stop_scores = torch.cat([model(sx[i:i + 8192], sc[i:i + 8192]) for i in range(0, len(sy), 8192)])
            stop_loss = loss_fn(stop_scores, sy).item()
            stop_f1 = f1_score(sy.cpu().numpy(), stop_scores.argmax(1).cpu().numpy(), average='macro', labels=range(10), zero_division=0)
        history.append({'epoch': epoch, 'train loss': total / len(ly), 'stopping loss': stop_loss,
                        'stopping macro F1': stop_f1, 'seconds': time.time() - start})
        if stop_loss < best - 1e-4:
            best, waited = stop_loss, 0
            best_weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= config['patience']:
                break
    model.load_state_dict(best_weights)
    history = pd.DataFrame(history)
    torch.save({'weights': best_weights, 'history': history, 'config': config, 'seed': seed}, path)
    if verbose:
        print(f'  trained seed {seed}: {len(history)} epochs (best {int(history["stopping loss"].idxmin()) + 1}), '
              f'{history.seconds.sum() / 60:.1f} min')
    return model, history


BASE_CONFIG = dict(scales=[3], fusion='self', tcn_bilstm=False, channels=64, dropout=0.2, weighting='none',
                   lr=0.001, max_epochs=100, patience=5, batch_size=512)


def run_config(data, config, seeds=(42, 43, 44), label=None, retrain=False):
    """Train (or load) a config for several seeds; return per-seed results and model_selection probabilities."""
    rows, probabilities, models, histories = [], {}, {}, {}
    for seed in seeds:
        model, history = train_network(data, config, seed, retrain=retrain)
        proba = predict_proba(model, data.numbers['model_selection'], data.codes['model_selection'])
        stop = predict_proba(model, data.numbers['train'][data.stop_rows], data.codes['train'][data.stop_rows])
        y_stop = data.y['train'].numpy()[data.stop_rows]
        rows.append({'experiment': label, 'seed': seed, **score(data.y['model_selection'].numpy(), proba),
                     'stopping macro F1': f1_score(y_stop, stop.argmax(1), average='macro', labels=range(10), zero_division=0),
                     'epochs': len(history), 'train minutes': history.seconds.sum() / 60,
                     'parameters': sum(p.numel() for p in model.parameters())})
        probabilities[seed], models[seed], histories[seed] = proba, model, history
        print(f'{label} seed {seed}: model_selection macro F1 {rows[-1]["macro F1"]:.4f} | stopping macro F1 {rows[-1]["stopping macro F1"]:.4f}')
    return pd.DataFrame(rows), probabilities, models, histories


# ---------------------------------------------------------------------------------------------------------- evaluation
def score(y_true, proba):
    pred = proba.argmax(axis=1)
    attack_true, attack_pred = y_true != NORMAL, pred != NORMAL
    fp = int((~attack_true & attack_pred).sum()); tn = int((~attack_true & ~attack_pred).sum())
    tp = int((attack_true & attack_pred).sum()); fn = int((attack_true & ~attack_pred).sum())
    return {'macro F1': f1_score(y_true, pred, average='macro', labels=range(10), zero_division=0),
            'accuracy': accuracy_score(y_true, pred),
            'attack recall': tp / (tp + fn), 'false-positive rate': fp / (fp + tn),
            'attack avg precision': average_precision_score(attack_true, 1 - proba[:, NORMAL]),
            'Worms correct (of 13)': int(((y_true == WORMS) & (pred == WORMS)).sum())}


_BOOT = {}


def bootstrap_difference(y_true, pred_a, pred_b, repeats=1000, seed=0):
    """Macro F1 of A minus B, with a 95% bootstrap interval over rows."""
    key = (len(y_true), repeats, seed)
    if key not in _BOOT:
        _BOOT[key] = np.random.default_rng(seed).integers(0, len(y_true), size=(repeats, len(y_true)), dtype=np.int32)

    def macro(t, p):
        return f1_score(t, p, average='macro', labels=range(10), zero_division=0)

    diffs = [macro(y_true[i], pred_a[i]) - macro(y_true[i], pred_b[i]) for i in _BOOT[key]]
    low, high = np.percentile(diffs, [2.5, 97.5])
    verdict = 'first clearly better' if low > 0 else 'second clearly better' if high < 0 else 'could be chance'
    return macro(y_true, pred_a) - macro(y_true, pred_b), low, high, verdict


def paired_comparison(y_true, probs_a, probs_b, name):
    """Seed-paired bootstrap comparison of two 3-seed experiments. probs_*: {seed: probabilities}."""
    rows = []
    for seed in sorted(set(probs_a) & set(probs_b)):
        diff, low, high, verdict = bootstrap_difference(y_true, probs_a[seed].argmax(1), probs_b[seed].argmax(1))
        rows.append({'comparison': name, 'seed': seed, 'difference': diff, '95% low': low, '95% high': high, 'verdict': verdict})
    frame = pd.DataFrame(rows)
    overall = frame.verdict.iloc[0] if frame.verdict.nunique() == 1 else 'depends on the seed'
    return frame, overall


def leaderboard_row(model_name, runs_frame, feature_set='total_and_byte44'):
    r = runs_frame
    return {'model': model_name, 'feature set': feature_set, 'runs': len(r),
            'macro F1 mean': r['macro F1'].mean(), 'macro F1 std': r['macro F1'].std() if len(r) > 1 else np.nan,
            'accuracy': r['accuracy'].mean(), 'attack recall': r['attack recall'].mean(),
            'false-positive rate': r['false-positive rate'].mean(), 'attack avg precision': r['attack avg precision'].mean(),
            'Worms correct (of 13)': r['Worms correct (of 13)'].mean(),
            'parameters': r['parameters'].iloc[0] if 'parameters' in r else np.nan,
            'train minutes per run': r['train minutes'].mean() if 'train minutes' in r else np.nan}
