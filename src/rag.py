"""Knowledge base, retrieval and local-LLM client for Notebook 18 (verified incident reports).

- build_chunks(): MITRE ATT&CK techniques (one chunk each) + NIST / CISA PDFs (page text split into ~1,200-character chunks).
- Retriever: TF-IDF (word 1-2 grams) with cosine similarity; deterministic, no external service needed.
- ollama_chat(): calls the free local LLM through Ollama's HTTP API with temperature 0 and a fixed seed.
"""
from __future__ import annotations

import json
import re
import time

import numpy as np
import pandas as pd
import requests
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer

from src.notebook_helpers import ROOT

KB = ROOT / 'data/knowledge_base'
PDFS = {'NIST SP 800-61r3': 'nist/NIST.SP.800-61r3.pdf', 'NIST SP 800-94': 'nist/NIST.SP.800-94.pdf',
        'CISA IR Playbooks': 'cisa/CISA_Incident_and_Vulnerability_Response_Playbooks.pdf'}
OLLAMA = 'http://localhost:11434'


def attack_techniques():
    """All current (not revoked or deprecated) ATT&CK Enterprise techniques as a table."""
    data = json.loads((KB / 'mitre/enterprise-attack.json').read_text(encoding='utf-8'))
    rows = []
    for o in data['objects']:
        if o['type'] != 'attack-pattern' or o.get('revoked') or o.get('x_mitre_deprecated'):
            continue
        ref = next((r for r in o.get('external_references', []) if r.get('source_name') == 'mitre-attack'), None)
        if ref is None:
            continue
        rows.append({'technique': ref['external_id'], 'name': o['name'], 'url': ref.get('url', ''),
                     'tactics': ', '.join(p['phase_name'] for p in o.get('kill_chain_phases', [])),
                     'description': re.sub(r'\(Citation:[^)]*\)', '', o.get('description', '')).strip(),
                     'detection': re.sub(r'\(Citation:[^)]*\)', '', o.get('x_mitre_detection', '') or '').strip()})
    return pd.DataFrame(rows).sort_values('technique').reset_index(drop=True)


def build_chunks(chunk_chars=1200, overlap=200):
    chunks = []
    for _, t in attack_techniques().iterrows():
        text = f"{t.technique} {t['name']}. Tactics: {t.tactics}. {t.description[:1500]} Detection: {t.detection[:600]}"
        chunks.append({'source': 'MITRE ATT&CK', 'location': t.technique, 'text': re.sub(r'\s+', ' ', text)})
    for title, rel in PDFS.items():
        reader = PdfReader(KB / rel)
        for page_no, page in enumerate(reader.pages, start=1):
            text = re.sub(r'\s+', ' ', page.extract_text() or '').strip()
            for start in range(0, max(len(text) - overlap, 1), chunk_chars - overlap):
                piece = text[start:start + chunk_chars]
                if len(piece) > 200:
                    chunks.append({'source': title, 'location': f'page {page_no}', 'text': piece})
    frame = pd.DataFrame(chunks)
    frame.insert(0, 'chunk_id', [f'K{i:05d}' for i in range(len(frame))])
    return frame


class Retriever:
    def __init__(self, chunks):
        self.chunks = chunks.reset_index(drop=True)
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.5, stop_words='english', sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(self.chunks.text)

    def search(self, query, k=6, sources=None):
        scores = (self.matrix @ self.vectorizer.transform([query]).T).toarray().ravel()
        frame = self.chunks.assign(score=scores)
        if sources:
            frame = frame[frame.source.isin(sources)]
        return frame.sort_values('score', ascending=False).head(k)


def ollama_available(model):
    try:
        tags = requests.get(f'{OLLAMA}/api/tags', timeout=5).json()
        return any(m['name'].startswith(model) for m in tags.get('models', []))
    except Exception:
        return False


def ollama_chat(model, system, user, seed=42, timeout=600):
    """One deterministic chat call that must return JSON. Returns (parsed JSON or None, raw text, seconds)."""
    start = time.time()
    response = requests.post(f'{OLLAMA}/api/chat', timeout=timeout, json={
        'model': model, 'stream': False, 'format': 'json',
        'options': {'temperature': 0, 'seed': seed, 'num_ctx': 8192},
        'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]})
    response.raise_for_status()
    text = response.json()['message']['content']
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    return parsed, text, time.time() - start
