"""Turn the readable sources in scripts/notebook_sources_v3/ into .ipynb files.

Source format: plain Python files split into cells by marker lines.
    # %% [markdown]   -> a markdown cell (each line starts with "# ")
    # %%              -> a code cell

Usage:
    .venv/Scripts/python.exe scripts/build_v3_notebooks.py          # all notebooks
    .venv/Scripts/python.exe scripts/build_v3_notebooks.py 04       # one notebook
Rebuilding clears saved outputs, so execute the notebook afterwards
(scripts/run_v3_notebooks.py).
"""
from pathlib import Path
import sys

import nbformat

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / 'scripts/notebook_sources_v3'

NAMES = {
    '01': '01_data_understanding.ipynb',
    '02': '02_duplicate_target_ambiguity_and_leakage_investigation.ipynb',
    '03': '03_training_eda_and_leakage_safe_preprocessing.ipynb',
    '04': '04_feature_engineering.ipynb',
    '05': '05_feature_selection.ipynb',
    '06': '06_baseline_models.ipynb',
    '07': '07_neural_reference_model.ipynb',
    '08': '08_conv1d_attention_model.ipynb',
    '09': '09_leakage_study.ipynb',
    '10': '10_rare_class_handling.ipynb',
    '11': '11_multiscale_conv1d.ipynb',
    '12': '12_cross_attention.ipynb',
    '13': '13_tcn_bilstm.ipynb',
    '14': '14_proposed_model.ipynb',
    '15': '15_ablation_study.ipynb',
    '16': '16_shap_lime.ipynb',
    '17': '17_confidence_calibration.ipynb',
    '18': '18_rag_incident_reports.ipynb',
    '19': '19_final_results.ipynb',
    '20': '20_shift_aware_detector.ipynb',
    '21': '21_final_results_v2.ipynb',
    '22': '22_detector_v3_and_score_ceiling.ipynb',
    '23': '23_final_improvement_v4.ipynb',
}


def parse(text):
    cells, kind, lines = [], None, []

    def flush():
        if kind is None:
            return
        body = '\n'.join(lines).strip('\n')
        if not body.strip():
            return
        if kind == 'markdown':
            md = '\n'.join(l[2:] if l.startswith('# ') else l.lstrip('#') for l in body.splitlines())
            cells.append(nbformat.v4.new_markdown_cell(md))
        else:
            cells.append(nbformat.v4.new_code_cell(body))

    for line in text.splitlines():
        if line.startswith('# %% [markdown]'):
            flush(); kind, lines = 'markdown', []
        elif line.startswith('# %%'):
            flush(); kind, lines = 'code', []
        else:
            lines.append(line)
    flush()
    return cells


def build(number):
    source = SOURCES / f'{number}.py'
    nb = nbformat.v4.new_notebook()
    nb.cells = parse(source.read_text(encoding='utf-8'))
    nb.metadata['kernelspec'] = {'name': 'botnet-research', 'display_name': 'Botnet research (.venv)', 'language': 'python'}
    nb.metadata['language_info'] = {'name': 'python'}
    target = ROOT / 'notebooks' / NAMES[number]
    nbformat.validate(nb)
    nbformat.write(nb, target)
    print(f'built {target.relative_to(ROOT)} ({len(nb.cells)} cells)')


if __name__ == '__main__':
    for n in (sys.argv[1:] or sorted(NAMES)):
        build(n)
