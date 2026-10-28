"""The versioned security-report record (schema 2): one JSON document from which the web view, the PDF, and the JSON / Markdown exports
are all rendered, so they always show the same values, wording, citations and statuses.

Six parts, in reading order:
  A summary            what was uploaded, predicted Normal / attack counts, review counts, batch triage priority (a heuristic), next check
  B detections         class counts, calibrated probabilities, prediction-set membership: attack-STATUS vs attack-TYPE uncertainty
  C flows              illustrative flow cards: real values, class probabilities, exact TreeSHAP decomposition, LLM explanation (checked)
  D sources            retrieved passages (MITRE ATT&CK, NIST SP 800-61r3 / 800-94, CISA playbooks) with metadata, links and spans
  E actions            Verify / Contain if confirmed / Recover, each with reason, citations, applicability, owner, disruption, validation
  F limits             unknowns, component status and coverage, source register, provenance, human review
plus a separate model appendix (benchmark metrics, not measured on the upload) and, only when the file had labels, an evaluation section.

Every claim has an origin (observed input, batch statistic, model output, XAI attribution, retrieved guidance, LLM synthesis, analyst-proposed)
and a scope (batch, attack type, one flow). Citation ids are namespaced: F<flow>.E<n> for a flow's evidence facts, S<n> for the report-wide
source register (deduplicated by chunk id), so ids from different flows cannot collide. Numbers come from code, never from the LLM.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import select

from app.core import analysis as rules
from app.core.detector import CLASSES
from app.core.feature_names import definition, friendly, plain
from app.core.report_views import build_overview, views
from app.core.reports import prob_text
from app.db import FlowPrediction

SCHEMA_VERSION = '2.1'
TEMPLATE_VERSION = 'summary + detailed 2.1'
GROUP_CAP = 500                           # flows per class explained for the group findings (first by row number)
ILLUSTRATIVE_TYPES = 4                    # attack types that get an illustrative flow card (most frequent first)
KEY_FEATURES = ['proto', 'service', 'state', 'dur', 'sbytes', 'dbytes', 'spkts', 'dpkts', 'sttl', 'dttl', 'smean', 'dmean']
VERIFY_QUERY = 'detection analysis validate alerts indicators precursors determine whether an incident occurred'
RECOVER_QUERY = 'recovery restore systems eradication return to normal operations'
REPO = Path(__file__).resolve().parents[3]

ORIGINS = {
    'observed': 'observed in the uploaded file',
    'batch_statistic': 'computed by code over the uploaded batch',
    'model_output': 'output of the detection model',
    'xai_attribution': 'TreeSHAP attribution of the detection model',
    'retrieved_guidance': 'passage retrieved from the knowledge base',
    'llm_synthesis': 'written by the local LLM from the cited evidence and passed automated checks',
    'analyst_proposed': 'proposed by this application as a next step; not prescribed by a cited source',
    'heuristic': 'application rule (heuristic), not a model output',
}


# ------------------------------------------------------------------------------------------------------------ helpers
def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _git_commit() -> str | None:
    try:
        out = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO, capture_output=True, text=True, timeout=5)  # noqa: S603, S607
        return out.stdout.strip() or None
    except Exception:                                                    # noqa: BLE001  not a git checkout (e.g. Docker image)
        return None


def _pct(x: float, digits: int = 1) -> str:
    return f'{x * 100:.{digits}f}%'


def _fmt(v):
    if isinstance(v, float):
        if v.is_integer():
            return str(int(v))
        return f'{v:.{4 if abs(v) >= 0.001 else 6}f}'.rstrip('0').rstrip('.')
    return str(v)


def source_url(passage: dict, documents: dict) -> str | None:
    """Canonical link of a passage: the ATT&CK page of a technique / mitigation, or the PDF at its page."""
    doc = documents.get(passage['source'], {})
    kind = doc.get('link_kind')
    if kind == 'attack' and re.fullmatch(r'T\d{4}(\.\d{3})?', str(passage['location'])):
        return 'https://attack.mitre.org/techniques/' + str(passage['location']).replace('.', '/') + '/'
    if kind == 'mitigation':
        m = re.match(r'(M\d{4})', str(passage['location']))
        return f'https://attack.mitre.org/mitigations/{m.group(1)}/' if m else None
    if kind == 'pdf' and doc.get('download_url'):
        page = re.search(r'page (\d+)', str(passage['location']))
        return doc['download_url'] + (f'#page={page.group(1)}' if page else '')
    return None


def ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def excerpt(text: str, limit: int = 360) -> str:
    """A readable excerpt: starts at a sentence start (PDF chunks can begin mid-sentence) and ends at a sentence end where possible."""
    text = re.sub(r'^\d{1,3}\s+(?=[A-Z])', '', ' '.join(str(text).split()))       # a page number left at the start of a PDF chunk
    if text and not text[0].isupper() and not text[0].isdigit():
        m = re.search(r'[.!?]\s+(?:\d{1,3}\s+(?=[A-Z]))?([A-Z0-9])', text)          # skip a footnote number at the next sentence start
        if m and m.start(1) < len(text) // 2:
            text = text[m.start(1):]
    if len(text) <= limit:
        return text
    cut = text[:limit]
    end = max(cut.rfind('. '), cut.rfind('! '), cut.rfind('? '))
    return (cut[:end + 1] if end > limit // 3 else cut.rsplit(' ', 1)[0]) + ' …'


def percentile_note(feature: str, value, reference: dict) -> str | None:
    """Position of a value among the FITTING rows (approved reference artifact), e.g. 'higher than about 97% of training flows'."""
    if not reference:
        return None
    if feature in reference.get('categories', {}):
        share = reference['categories'][feature].get(str(value))
        return (f'{_pct(share)} of training flows have this value' if share is not None else 'this value never appears in the training flows')
    qs = reference.get('percentiles', {}).get(feature)
    if qs is None or value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    below = int(np.searchsorted(np.asarray(qs), v, side='left'))
    equal_hi = int(np.searchsorted(np.asarray(qs), v, side='right'))
    if equal_hi - below > 3:                                             # many training flows share exactly this value
        return f'between the {ordinal(below)} and {ordinal(min(equal_hi, 100))} percentile of training flows (a very common value)'
    return f'higher than about {min(below, 100)}% of training flows'


# ------------------------------------------------------------------------------------------------------------ action profiles
ACTION_KINDS = [
    ('credentials', r'password|credential|private key|secret|rotate|re-?key'),
    ('domain', r'\bDNS\b|domain name|domains?\b'),
    ('isolation', r'isolate (the |an |affected |compromised )?(host|system|device|machine)|quarantin|disconnect'),
    ('filtering', r'filter|block|rate[- ]limit|upstream|firewall|blackhol|scrub|content delivery|\bCDN'),
    ('hardening', r'sandbox|application isolation|disable|remove|uninstall|least privilege|minimi[sz]|restrict|pre-compromise|harden|'
                  r'unnecessary (ports|services)|\bclosed?\b'),
    ('patching', r'patch|update|upgrade|vulnerab|scan'),
]


def action_profile(text: str, cls: str) -> dict:
    """What a containment step needs before it is justified, who does it, what it may break, how to check and undo it. Specific to the kind
    of step: a predicted class alone never justifies blocking, isolation or credential changes."""
    kind = next((k for k, pattern in ACTION_KINDS if re.search(pattern, text, re.IGNORECASE)), 'other')
    base = f'A {cls} prediction alone is not enough.'
    profiles = {
        'filtering': ('Filter or block the confirmed traffic', f'Packet captures or sensor logs confirm the traffic is malicious AND its source, '
                      f'destination and service are identified (the uploaded file has none of these). {base}', 'Network engineering',
                      'Can also block legitimate users of the same service or address range.',
                      'Sensor counts for the matching traffic drop while service health checks stay normal.',
                      'Remove the rule or let it expire; record the rule and who approved it.'),
        'credentials': ('Reset exposed credentials or keys', f'Authentication logs or endpoint alerts show that a named account, password or key '
                        f'was exposed or misused. Network-flow predictions cannot show this. {base}', 'System administrator / identity team',
                        'Users and services must sign in again; jobs using the old secret fail until updated.',
                        'The old credentials are rejected and no new suspicious sign-ins appear.',
                        'Not reversible: update dependent services before rotating.'),
        'domain': ('Block a confirmed malicious domain', f'DNS or proxy logs name a specific malicious domain contacted from your network. The '
                   f'uploaded flows contain no domain names. {base}', 'DNS / network administrator',
                   'Blocking a shared or mistaken domain breaks legitimate services.', 'Lookups of the domain fail and no business service reports errors.',
                   'Remove the domain from the block list.'),
        'isolation': ('Isolate a confirmed compromised host', f'Endpoint or log evidence identifies a specific compromised host. {base}',
                      'System owner, approved by the incident lead', 'The host and its services are unavailable while isolated.',
                      'The host has no network access except management, and no new alerts come from it.', 'Reconnect after cleanup and a clean scan.'),
        'other': ('Apply the cited response step', f'Verification confirms {cls} activity on an identified system; adapt the step to that system. '
                  f'{base}', 'Incident response lead', 'Depends on the change; review it before applying.',
                  'The activity is no longer seen in sensor logs and normal services still work.', 'Keep a record of each change so it can be undone.'),
    }
    if kind in ('hardening', 'patching'):
        return {'kind': kind, 'prevention': True,
                'note': 'General prevention (hardening or patching) for this attack type; not an incident-response step for these flows.'}
    title, requires, owner, disruption, validation, rollback = profiles[kind]
    return {'kind': kind, 'prevention': False, 'title': title, 'requires': requires, 'owner': owner, 'disruption': disruption,
            'validation': validation, 'rollback': rollback}


# ------------------------------------------------------------------------------------------------------------ reasoning evidence
def _num(v) -> str:
    """A number as written in the facts (and expected back from the LLM): plain, no exponent, no trailing zeros."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return str(int(f)) if f.is_integer() else f'{f:.4f}'.rstrip('0').rstrip('.')


def typical_fact(feature: str, value, cls: str, other: str, reference: dict) -> dict | None:
    """How typical a measured value is for the predicted class and for one other class, from the training (fitting) rows only."""
    if not reference:
        return None
    discrete = reference.get('discrete', {}).get(feature)
    if discrete is not None:
        key = str(value)
        try:
            key = f'{float(value):g}'
        except (TypeError, ValueError):
            pass
        a, b = discrete.get(cls, {}).get(key, 0.0), discrete.get(other, {}).get(key, 0.0)
        return {'feature': feature, 'value': _num(value), 'kind': 'share', 'class': cls, 'other': other, 'share': a, 'other_share': b,
                'text': f'Value {feature} = {_num(value)} occurs in {a * 100:.1f}% of training {cls} flows and {b * 100:.1f}% of training {other} flows'}
    cont = reference.get('continuous', {}).get(feature)
    if cont is None or cls not in cont or other not in cont:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    lo, _, hi = cont[cls]
    olo, _, ohi = cont[other]
    where = 'inside' if lo <= v <= hi else ('below' if v < lo else 'above')
    theirs = (f'the middle half of training {other} flows is {_num(round(olo, 4))} to {_num(round(ohi, 4))}' if olo != ohi
              else f'at least half of training {other} flows have exactly {_num(round(olo, 4))}')
    if lo == hi:                                                     # at least half of the class shares one exact value
        ours = (f'at least half of training {cls} flows have exactly this value' if v == lo
                else f'at least half of training {cls} flows have exactly {_num(round(lo, 4))}')
        where = 'equal' if v == lo else where
        text = f'Value {feature} = {_num(value)}: {ours}; {theirs}'
    else:
        text = f'Value {feature} = {_num(value)} is {where} the middle half of training {cls} flows ({_num(round(lo, 4))} to {_num(round(hi, 4))}); {theirs}'
    return {'feature': feature, 'value': _num(value), 'kind': 'range', 'class': cls, 'other': other, 'where': where, 'text': text}


SLOT_TITLES = {'certainty': 'How sure the model is', 'reasons': 'Main reasons', 'typical': 'How typical the values are',
               'contrast': 'Why not {runner_up}', 'batch': 'In this file'}


def slot_of(cites: list[str], facts: dict, text: str = '') -> str:
    """The reasoning slot of a sentence: what it says first (comparison wording, training-flow wording, the analysed dataset, contributions),
    then the kind of evidence most of its citations point to."""
    low = text.lower()
    for slot, words in (('contrast', ('favour', 'compared with')), ('typical', ('of training',)), ('batch', ('analysed dataset',)),
                        ('reasons', ('contribution',))):
        if any(w in low for w in words):
            return slot
    texts = [facts.get(c, '') for c in cites]
    counts = {slot: sum(x.startswith(prefix) for x in texts)
              for slot, prefix in (('contrast', 'Compared with'), ('typical', 'Value '), ('batch', 'In the analysed dataset'), ('reasons', 'Column '))}
    best = max(counts, key=lambda s: counts[s])                     # dict order breaks ties
    return best if counts[best] else 'certainty'


def pct_text(p: float) -> str:
    return 'below 0.1%' if p < 0.001 else f'{p * 100:.1f}%'


# ------------------------------------------------------------------------------------------------------------ the builder
class RecordBuilder:
    def __init__(self, app, db, analysis, progress=None):
        self.app, self.db, self.item = app, db, analysis
        self.progress = progress or (lambda stage, detail='': None)
        self.a = analysis.summary
        self.timings: dict[str, float] = {}
        self.sources: dict[str, dict] = {}          # global register: S<n> -> passage
        self._by_chunk: dict[str, str] = {}
        self.components: dict[str, dict] = {}

    # ------------------------------------------------------------------------------------------ sources
    def register(self, passage: dict, used_by: str) -> str:
        key = passage.get('chunk_id') or passage.get('content_sha256')
        if key in self._by_chunk:
            sid = self._by_chunk[key]
            if used_by not in self.sources[sid]['used_by']:
                self.sources[sid]['used_by'].append(used_by)
            return sid
        sid = f'S{len(self.sources) + 1}'
        docs = self.app.state.knowledge_sources.get('documents', {})
        doc = docs.get(passage['source'], {})
        self._by_chunk[key] = sid
        self.sources[sid] = {
            'id': sid, 'chunk_id': passage.get('chunk_id'), 'source': passage['source'], 'publisher': doc.get('publisher'),
            'title': doc.get('title'), 'version': doc.get('version'), 'location': str(passage['location']),
            'url': source_url(passage, docs), 'document_url': doc.get('download_url'), 'retrieved': doc.get('retrieved'),
            'pdf_created': doc.get('pdf_created'), 'document_sha256': doc.get('document_sha256'),
            'content_sha256': passage.get('content_sha256') or hashlib.sha256(' '.join(str(passage['text']).split()).encode()).hexdigest(),
            'text': str(passage['text']), 'excerpt': excerpt(passage['text']),
            'retrieval': {'method': passage.get('method'), 'score': passage.get('score'), 'query': passage.get('query')},
            'used_by': [used_by],
            'trust': 'untrusted reference text: shown as data, never followed as instructions',
        }
        return sid

    def documents(self) -> list[dict]:
        """The source documents behind the register (one row per document family actually cited)."""
        docs = self.app.state.knowledge_sources.get('documents', {})
        used = sorted({s['source'] for s in self.sources.values()})
        return [{'source': name, **{k: docs.get(name, {}).get(k) for k in ('publisher', 'title', 'version', 'pdf_created', 'retrieved',
                                                                            'download_url', 'document_sha256')},
                 'passages': sum(1 for s in self.sources.values() if s['source'] == name)} for name in used]

    def phase_guidance(self, query: str, used_by: str) -> tuple[str | None, dict]:
        engine = self.app.state.reports
        found = engine._search(query, 3, ['NIST SP 800-61r3', 'NIST SP 800-94', 'CISA IR Playbooks'])
        found = found[found.score >= 0.02]
        if found.empty:
            return None, {'status': 'empty', 'query': query}
        r = found.iloc[0]
        sid = self.register({'chunk_id': str(r.chunk_id), 'source': r.source, 'location': r.location, 'text': r.text,
                             'method': 'TF-IDF similarity to a response-phase query', 'score': round(float(r.score), 4), 'query': query}, used_by)
        return sid, {'status': 'ok', 'query': query, 'score': round(float(r.score), 4)}

    # ------------------------------------------------------------------------------------------ flows
    def flow_rows(self) -> pd.DataFrame:
        rows = self.db.execute(select(FlowPrediction.id, FlowPrediction.flow_index, FlowPrediction.predicted_class, FlowPrediction.confidence,
                                      FlowPrediction.prediction_set, FlowPrediction.severity)
                               .where(FlowPrediction.analysis_id == self.item.id)).all()
        frame = pd.DataFrame(rows, columns=['id', 'flow_index', 'predicted_class', 'confidence', 'prediction_set', 'severity'])
        sets = frame.prediction_set
        frame['set_size'] = sets.map(len)
        frame['has_normal'] = sets.map(lambda s: 'Normal' in s)
        frame['attack_members'] = sets.map(lambda s: sum(c != 'Normal' for c in s))
        frame['malicious'] = frame.predicted_class != 'Normal'
        frame['status_uncertain'] = frame.has_normal & (frame.attack_members > 0)
        frame['type_uncertain'] = frame.malicious & ~frame.has_normal & (frame.attack_members > 1)
        return frame

    SELECTION_RULE = ('Illustrative flows: for each of the four most frequent predicted attack types, the flow at the MEDIAN confidence of that '
                      'type (a typical case, not the best one); the first flow of the review order (highest heuristic severity, then confidence); '
                      'and the lowest-confidence flow whose shortlist contains Normal and an attack type (or, if there is none, the lowest-confidence '
                      'attack-type-uncertain flow). Each card explains one flow, not its whole group.')

    def selected_flows(self, frame: pd.DataFrame) -> list[tuple[int, str]]:
        picks = []
        for d in self.a['attack_distribution'][:ILLUSTRATIVE_TYPES]:
            rows = frame[frame.predicted_class == d['attack_type']].sort_values(['confidence', 'flow_index'])
            if len(rows):
                k = (len(rows) - 1) // 2
                picks.append((int(rows.iloc[k].id), f'typical {d["attack_type"]} flow: median confidence of the {d["count"]} flows predicted as '
                                                    f'{d["attack_type"]} (rank {k + 1} of {len(rows)} by confidence; one example, not the whole group)'))
        order = {s: k for k, s in enumerate(rules.SEVERITY_ORDER)}
        attacks = frame[frame.malicious].assign(rank=lambda f: f.severity.map(order).fillna(9))
        if len(attacks):
            top = attacks.sort_values(['rank', 'confidence', 'flow_index'], ascending=[True, False, True]).iloc[0]
            picks.append((int(top.id), f'review priority: first flow of the review order (heuristic severity {top.severity}, then confidence)'))
        uncertain = frame[frame.status_uncertain].sort_values(['confidence', 'flow_index'])
        if len(uncertain):
            picks.append((int(uncertain.iloc[0].id), 'uncertain flow: the lowest-confidence flow whose 90% shortlist contains both Normal and an '
                                                     'attack type (attack status uncertain)'))
        else:
            typed = frame[frame.type_uncertain].sort_values(['confidence', 'flow_index'])
            if len(typed):
                picks.append((int(typed.iloc[0].id), 'uncertain flow: the lowest-confidence predicted attack with several attack types on its '
                                                     'shortlist (attack type uncertain)'))
        if not picks and len(frame):                                    # Normal-only batch: one illustrative Normal flow, no attack sources
            k = (len(frame) - 1) // 2
            mid = frame.sort_values(['confidence', 'flow_index']).iloc[k]
            picks.append((int(mid.id), 'typical flow: median confidence in a batch with no predicted attacks'))
        seen, out = set(), []
        for fid, why in picks:
            if fid not in seen:
                seen.add(fid)
                out.append((fid, why))
        return out

    # ------------------------------------------------------------------------------------------ group findings (TreeSHAP over a whole class)
    def group_findings(self, frame: pd.DataFrame) -> list[dict]:
        """For each main predicted attack type: TreeSHAP of the predicted class for its flows, and how often each measurement is one of the
        three strongest SUPPORTING contributions of a flow. Counted per flow, never summed across flows (signed sums can hide patterns)."""
        detector, refs, out = self.app.state.detector, self.app.state.feature_reference, []
        t0 = time.time()
        for d in self.a['attack_distribution'][:ILLUSTRATIVE_TYPES]:
            cls = d['attack_type']
            ids = frame[frame.predicted_class == cls].sort_values('flow_index').id.tolist()
            used = ids[:GROUP_CAP]
            flows = pd.DataFrame(self.db.scalars(select(FlowPrediction.flow).where(FlowPrediction.id.in_(used))).all())
            try:
                full = detector.explain_full(flows, np.full(len(flows), CLASSES.index(cls)))
            except Exception as error:                                 # noqa: BLE001  reported, never hidden
                out.append({'class': cls, 'status': 'failed', 'reason': f'{type(error).__name__}: {error}', 'population': len(ids)})
                continue
            supporting, opposing = Counter(), Counter()
            for r in full:
                for c in [c for c in r['contributions'] if c['contribution'] > 0][:3]:
                    supporting[c['feature']] += 1
                strongest_against = next((c for c in r['contributions'] if c['contribution'] < 0), None)
                if strongest_against:
                    opposing[strongest_against['feature']] += 1
            n = len(full)

            shown = {}                                                     # value of each measurement per flow, as the explanation shows it
            for r in full:
                for c in r['contributions']:
                    shown.setdefault(c['feature'], []).append(c['value'])

            def describe(feature, shown=shown):
                values = pd.Series(shown[feature])
                numeric = pd.to_numeric(values, errors='coerce')
                if numeric.notna().all():
                    values = numeric
                if pd.api.types.is_numeric_dtype(values):
                    median = float(values.median())
                    return {'value_summary': f'median {_fmt(median)}', 'reference': percentile_note(feature, median, refs)}
                mode = values.mode().iloc[0]
                return {'value_summary': f'most common value {mode} ({(values == mode).mean() * 100:.0f}% of these flows)',
                        'reference': percentile_note(feature, mode, refs)}

            out.append({'class': cls, 'status': 'available', 'population': len(ids), 'analysed': n, 'origin': 'xai_attribution',
                        'top_supporting': [{'feature': f, 'name': friendly(f), 'plain': plain(f), 'flows': k, 'share': k / n, **describe(f)}
                                           for f, k in supporting.most_common(3)],
                        'most_frequent_opposing': [{'feature': f, 'name': friendly(f), 'flows': k, 'share': k / n}
                                                   for f, k in opposing.most_common(1)],
                        'method': (f'TreeSHAP of the {cls} score for {"all" if n == len(ids) else f"the first {n} (by row number) of the"} '
                                   f'{len(ids)} flows predicted as {cls}. For each measurement: in how many flows it is one of the three strongest '
                                   'supporting contributions. Contributions are not added across flows. Values are summarised by median or most '
                                   'common value and compared with the training flows.')})
        self.timings['group_xai'] = time.time() - t0
        return out

    def flow_card(self, flow: FlowPrediction, why: str, index: int, total: int) -> dict:
        state, engine, detector = self.app.state, self.app.state.reports, self.app.state.detector
        fid = f'F{flow.flow_index}'
        frame = pd.DataFrame([flow.flow])
        t0 = time.time()
        self.progress('explanations', f'flow {index} of {total}: #{flow.flow_index}')
        pred = detector.predict(frame)
        try:
            full = detector.explain_full(frame, pred.predicted)[0]
            xai_status = {'status': 'available', 'method': 'TreeSHAP (LightGBM pred_contrib), grouped by original column'}
        except Exception as error:                                     # noqa: BLE001  reported, never hidden
            full, xai_status = None, {'status': 'failed', 'reason': f'{type(error).__name__}: {error}'}
        self.timings['xai'] = self.timings.get('xai', 0) + time.time() - t0
        probabilities = dict(zip(CLASSES, [float(p) for p in pred.proba[0]], strict=True))
        refs = state.feature_reference
        values = [{'feature': f, 'name': friendly(f), 'value': _fmt(flow.flow.get(f)), 'definition': definition(f, state.feature_descriptions),
                   'reference': percentile_note(f, flow.flow.get(f), refs)} for f in KEY_FEATURES]
        xai = None
        if full:
            contribs = full['contributions']
            top = contribs[:5]
            signs = {np.sign(c['contribution']) for c in top if c['contribution'] != 0}
            rest = sum(c['contribution'] for c in contribs[5:])
            xai = {**xai_status, 'class': full['class'], 'units': 'log-odds score units of the base LightGBM model',
                   'base_value': full['base_value'], 'margin': full['margin'], 'temperature': full['temperature'], 'class_offset': full['class_offset'],
                   'deployed_class_score': full['deployed_class_score'], 'remainder_of_other_features': rest, 'other_features': len(contribs) - 5,
                   'top': [{**c, 'value': _fmt(c['value']), 'name': friendly(c['feature']), 'direction': 'supports' if c['contribution'] > 0 else 'opposes',
                            'definition': definition(c['feature'], state.feature_descriptions), 'reference': percentile_note(c['feature'], c['value'], refs)}
                           for c in top],
                   'all': [{'feature': c['feature'], 'value': _fmt(c['value']), 'contribution': c['contribution']} for c in contribs],
                   'one_sided': len(signs) == 1,
                   'how_to_read': ('Each contribution moves the model score of the predicted class up (supports) or down (opposes). Base value + all '
                                   'contributions = the model score; the deployed detector divides it by the temperature and adds a fixed class offset '
                                   'before turning scores into probabilities. Contributions are not percentages of the probability, and they '
                                   'describe the model, not the cause of the traffic.')}
        else:
            xai = xai_status
        lime = {'status': 'not generated', 'reason': 'LIME was only run in research Notebook 16 for an older detector; it is not computed per flow '
                                                    'for the deployed model.'}

        # LLM explanation of this flow (retrieval + generation + checks), with the flow's evidence as facts
        # reasoning evidence, all computed by code: attack-or-normal probability, how typical the strongest values are (training rows only),
        # and what separates the prediction from the runner-up class (TreeSHAP of both classes)
        cls = flow.predicted_class
        runner_up, ru_prob = max(((c, pr) for c, pr in probabilities.items() if c != cls), key=lambda kv: kv[1])
        status = 1.0 - probabilities['Normal'] if flow.malicious else probabilities['Normal']
        contrast, typical = None, []
        if full:
            try:
                ru_full = detector.explain_full(frame, np.array([CLASSES.index(runner_up)]))[0]
                theirs = {c['feature']: c['contribution'] for c in ru_full['contributions']}
                diffs = sorted(({'feature': c['feature'], 'name': friendly(c['feature']), 'value': _fmt(c['value']),
                                 'difference': c['contribution'] - theirs.get(c['feature'], 0.0)} for c in full['contributions']),
                               key=lambda d: -d['difference'])
                contrast = {'runner_up': runner_up, 'probability': ru_prob, 'favouring': [d for d in diffs[:2] if d['difference'] > 0],
                            'favouring_runner_up': [d for d in diffs[::-1][:1] if d['difference'] < 0],
                            'method': f'TreeSHAP of {cls} minus TreeSHAP of {runner_up} for each measurement (difference of model scores, exact)'}
            except Exception:                                          # noqa: BLE001  the contrast is optional evidence
                contrast = None
            other = 'Normal' if flow.malicious else runner_up
            for c in [c for c in full['contributions'] if c['contribution'] > 0][:2]:
                fact = typical_fact(c['feature'], c['value'], cls, other, state.class_reference)
                if fact:
                    typical.append(fact)
        extra = []
        llm = None
        if flow.malicious:
            dist = next((d for d in self.a['attack_distribution'] if d['attack_type'] == flow.predicted_class), None)
            if dist:
                extra.append(f'In the analysed dataset, {dist["count"]} flows ({dist["percentage_of_malicious"] * 100:.1f}% of the predicted attacks) '
                             f'were classified as {flow.predicted_class}, with average confidence {dist["average_confidence"]:.2f}')
            extra.append(f'Probability that this flow is an attack of any type: {prob_text(status)} (the nine attack classes together)')
            extra += [f['text'] for f in typical]
            if contrast:
                extra += [f"Compared with {runner_up}, the second most likely class ({pct_text(ru_prob)}), column {d['feature']} = {d['value']} "
                          f"favours {cls} by {d['difference']:+.3f} in model score" for d in contrast['favouring']]
            explanation = [{'feature': c['feature'], 'value': c['value'], 'contribution': c['contribution']} for c in (full['contributions'] if full else [])]

            def step(stage, detail=f'flow {index} of {total}: #{flow.flow_index} ({flow.predicted_class})'):
                self.progress({'evidence': 'retrieval', 'writing': 'generation', 'verifying': 'checks', 'repairing': 'checks'}.get(stage, 'checks'), detail)

            result = engine.generate(flow.predicted_class, flow.confidence, flow.prediction_set, explanation, progress=step, extra_facts=extra)
            for k, v in result.timings.items():
                self.timings[k] = self.timings.get(k, 0) + v
            llm = self._namespace(result, fid)
        facts = llm['facts'] if llm else {}
        interpretation, alternatives = None, []
        if full:
            sup = [c for c in xai['top'] if c['contribution'] > 0][:2]
            opp = [c for c in xai['top'] if c['contribution'] < 0][:1]
            interpretation = ((f"For this flow, the strongest reasons towards {flow.predicted_class} were "
                               + ' and '.join(f"{c['name'][:1].lower() + c['name'][1:]} = {c['value']}" for c in sup) + '. ' if sup else '')
                              + (f"{opp[0]['name']} = {opp[0]['value']} pointed away from {flow.predicted_class}." if opp
                                 else 'None of the five strongest measurements pointed away from it.'))
        for c, pr in sorted(probabilities.items(), key=lambda kv: -kv[1]):
            if c != flow.predicted_class and (pr >= 0.05 or c in flow.prediction_set):
                alternatives.append({'class': c, 'probability': pr, 'in_set': c in flow.prediction_set})

        # the reasoning: up to five points, one per slot. A checked LLM sentence fills its slot; otherwise code writes it from the evidence.
        written = {}
        status_ids = {k for k, v in facts.items() if v.startswith('Probability that this flow is an attack')}
        for e in (llm or {}).get('explanation', []):
            slot = slot_of(e['cites'], facts, e['text'])
            if slot == 'certainty' and status_ids and not (status_ids & set(e['cites']) and 'any type' in e['text'].lower()):
                continue                                               # incomplete: the code version states both probabilities
            if slot in ('typical', 'contrast') and f'training {cls}' not in e['text'] and f'favours {cls}' not in e['text']:
                continue                                               # must be about the predicted class itself; else the code version
            written.setdefault(slot, e)
        shortlist = ', '.join(flow.prediction_set)
        code = {
            'certainty': ((f"The model gives {cls} a probability of {prob_text(flow.confidence)} and is {prob_text(status)} sure this flow is an "
                           f"attack of any type; its 90% shortlist is {shortlist}.") if flow.malicious else
                          (f"The model gives Normal a probability of {prob_text(flow.confidence)}; the remaining {1 - flow.confidence:.2f} is spread over "
                           f"the attack types, and its 90% shortlist is {shortlist}."), 'model_output'),
            'reasons': (interpretation, 'xai_attribution') if interpretation else None,
            'typical': ((typical[0]['text'] + '.').replace('Value ', 'The value ', 1), 'xai_attribution') if typical else None,
            'contrast': ((f"Compared with {runner_up} ({pct_text(ru_prob)}), "
                          + ' and '.join(f"{d['name'][:1].lower() + d['name'][1:]} = {d['value']}" for d in contrast['favouring'])
                          + f" favoured {cls} most.") if contrast and contrast['favouring'] else None, 'xai_attribution'),
            'batch': (next((x for x in extra if x.startswith('In the analysed dataset')), '') + '.', 'batch_statistic')
            if flow.malicious and extra and extra[0].startswith('In the analysed dataset') else None,
        }
        reasoning = []
        for slot in ('certainty', 'reasons', 'typical', 'contrast', 'batch'):
            title = SLOT_TITLES[slot].format(runner_up=runner_up)
            if slot in written:
                e = written[slot]
                reasoning.append({'slot': slot, 'title': title, 'text': e['text'], 'cites': e['cites'], 'origin': 'llm_synthesis'})
            elif code.get(slot) and code[slot][0]:
                reasoning.append({'slot': slot, 'title': title, 'text': code[slot][0], 'cites': [], 'origin': code[slot][1]})
        return {
            'id': fid, 'flow_id': flow.id, 'flow_index': flow.flow_index, 'selection': why,
            'prediction': {'class': flow.predicted_class, 'malicious': flow.malicious, 'confidence': flow.confidence,
                           'confidence_level': flow.confidence_level, 'severity_heuristic': flow.severity,
                           'prediction_set': flow.prediction_set, 'normal_in_set': 'Normal' in flow.prediction_set,
                           'probabilities': probabilities, 'origin': 'model_output'},
            'values': values, 'xai': xai, 'lime': lime, 'facts': facts, 'llm': llm,
            'interpretation': interpretation, 'alternatives': alternatives, 'reasoning': reasoning, 'contrast': contrast, 'typical': typical,
            'status_probability': status,
            'missing_confirmation': ('Packet captures, sensor or host logs for this flow. The uploaded file has no addresses, ports or timestamps, so the '
                                     'flow has to be located by protocol, service, state and byte counts.') if flow.malicious else None,
            'ground_truth_note': 'Labels in the uploaded file, if any, are not used here; see the separate evaluation section.',
        }

    def _namespace(self, result, fid: str) -> dict:
        """Map the generator's local ids (E3, S2) to report-wide ids (F222.E3, S7) and keep every check result."""
        r = result.to_dict()
        local_to_global = {}
        for s in r['sources_detail']:
            local_to_global[s['id']] = self.register(s, used_by=fid)
        for k in r['facts']:
            local_to_global[k] = f'{fid}.{k}'

        def cites(ids):
            return [local_to_global.get(c, f'{fid}.?{c}') for c in ids]

        PREFIX = {'evidence_explanation': 'X', 'techniques': 'T', 'claims': 'C', 'actions': 'A'}    # E is reserved for evidence facts

        def items(kind, origin):
            out = []
            for n, i in enumerate(r[kind], start=1):
                support = i.get('support')
                if support:
                    support = {**support, 'source': local_to_global.get(support['source'], support['source'])}
                out.append({'id': f'{fid}.{PREFIX[kind]}{n}', 'text': i.get('text') or f"{i.get('id')} {i.get('name') or ''}".strip(),
                            'technique': i.get('id') if kind == 'techniques' else None, 'cites': cites(i.get('cites', [])), 'origin': origin,
                            'status': 'passed automated checks', 'check': i.get('reason', 'supported'), 'round': i.get('round', 1), 'support': support})
            return out

        return {
            'status': r['status'], 'note': r['note'], 'model': r['llm_model'], 'seconds': round(r['seconds'], 2), 'retrieval': r['retrieval'],
            'facts': {f'{fid}.{k}': v for k, v in r['facts'].items()},
            'findings': [{'text': f['text'], 'cites': cites(f['cites']), 'origin': 'model_output' if f['cites'][0] in ('E1', 'E2') else 'xai_attribution',
                          'status': 'computed by code'} for f in r['findings']],
            'explanation': items('evidence_explanation', 'llm_synthesis'),
            'techniques': items('techniques', 'llm_synthesis'), 'context': items('claims', 'llm_synthesis'),
            'actions': items('actions', 'llm_synthesis'),
            'mapped_techniques': [{**p, 'origin': 'heuristic', 'note': 'fixed class-to-technique mapping: a hypothesis for context, not proof'}
                                  for p in r['possible_attacks']],
            'checked': r['checked_items'], 'removed': [{**x, 'cites': cites(x.get('cites', []))} for x in r['removed']],
        }

    # ------------------------------------------------------------------------------------------ actions
    def actions(self, cards: list[dict]) -> dict:
        if any(card['prediction']['malicious'] for card in cards):     # Normal-only batches get no attack guidance
            verify_sid, verify_info = self.phase_guidance(VERIFY_QUERY, 'actions')
            recover_sid, recover_info = self.phase_guidance(RECOVER_QUERY, 'actions')
        else:
            verify_sid = recover_sid = None
            verify_info = recover_info = {'status': 'not applicable', 'reason': 'no predicted attacks'}
        rows, prevention, by_text = [], [], {}

        def add(phase, **kw):
            rows.append({'id': '', 'phase': phase, **kw})

        review = int(self.a.get('needs_review_count', 0))
        if review:
            add('Verify', kind='review', title='Review the low-confidence predictions', scope='batch',
                text=f'Open the flow list filtered to confidence Low and review the {review} flows below 70% before acting on any of them.',
                reason='Low-confidence predictions are the most likely to be wrong.', cites=[], origin='analyst_proposed',
                applicability='Always; read-only.', owner='Security analyst', disruption='None (read-only).',
                validation='Each reviewed flow is marked confirmed, false positive or unclear in the checklist.')
        for card in cards:
            p = card['prediction']
            if not p['malicious']:
                continue
            cls, fid = p['class'], card['id']
            v = {x['feature']: x['value'] for x in card['values']}
            technique = next((s['id'] for s in self.sources.values() if s['source'] == 'MITRE ATT&CK' and fid in s['used_by']), None)
            weaken = 'the traffic belongs to a known service or job of your own team, or other sensors show nothing for that period'
            if p['normal_in_set']:
                weaken += '; Normal is also on this flow\'s shortlist'
            add('Verify', kind='verify', title=f'Check flow #{card["flow_index"]} in logs or captures', scope=f'flow #{card["flow_index"]} ({cls})',
                text=f'Find flow #{card["flow_index"]} ({v.get("proto")}/{v.get("service")}, state {v.get("state")}, {v.get("sbytes")} bytes sent, '
                     f'{v.get("dbytes")} received) in packet captures or sensor logs from the same period and decide whether it is {cls} activity.',
                reason=f'The model predicts {cls} with calibrated probability {prob_text(p["confidence"])}; its 90% shortlist is '
                       f'{", ".join(p["prediction_set"])}. A prediction is not a confirmed incident.',
                cites=[c for c in (f'{fid}.E1', f'{fid}.E2', technique, verify_sid) if c and (c.startswith('S') or c in card['facts'])],
                origin='analyst_proposed', applicability='Always first; read-only. Nothing should be blocked on the prediction alone.',
                owner='Security analyst', disruption='None (read-only).',
                validation=f'Confirmed if logs or captures show activity matching the cited description of {cls}; weakened if {weaken}.')
            candidates = list((card['llm'] or {}).get('actions', []))
            if not candidates:
                mitigation = next((s for s in self.sources.values() if s['source'] == 'MITRE ATT&CK mitigation' and fid in s['used_by']), None)
                if mitigation:
                    candidates = [{'text': mitigation['excerpt'], 'cites': [mitigation['id']], 'check': None, 'support': None, 'retrieved': True}]
            for a in candidates:
                profile = action_profile(a['text'], cls)
                key = a['text'].strip().lower()
                if profile['prevention']:
                    if key not in by_text:
                        by_text[key] = None
                        prevention.append({'scope': f'attack type {cls}', 'kind': profile['kind'], 'text': a['text'], 'cites': a['cites'],
                                           'origin': 'retrieved_guidance', 'note': profile['note']})
                    continue
                if key in by_text and by_text[key] is not None:          # same guidance for another class: one row, wider scope
                    by_text[key]['scope'] += f', {cls}'
                    continue
                add('Contain if confirmed', kind=profile['kind'], title=profile['title'], scope=f'attack type {cls}', text=a['text'],
                    reason=f'Guidance retrieved for the {cls} prediction (see the cited passage). It applies only after verification.',
                    cites=a['cites'], origin='retrieved_guidance' if a.get('retrieved') else 'llm_synthesis', check=a.get('check'),
                    support=a.get('support'), applicability=profile['requires'], owner=profile['owner'], disruption=profile['disruption'],
                    validation=profile['validation'], rollback=profile['rollback'])
                by_text[key] = rows[-1]
        if any(r['phase'] == 'Contain if confirmed' for r in rows) and recover_sid:
            add('Recover', kind='recover', title='Restore and monitor after a confirmed incident', scope='batch',
                text=self.sources[recover_sid]['excerpt'], reason='Recovery guidance retrieved for a confirmed incident.', cites=[recover_sid],
                origin='retrieved_guidance', applicability='Only after a confirmed incident has been contained.',
                owner='Incident response lead with the system owners', disruption='Restoring systems may need planned downtime.',
                validation='Restored systems behave normally and monitoring shows no recurrence for an agreed period.', rollback=None)
        order = {ph: k for k, ph in enumerate(('Verify', 'Contain if confirmed', 'Recover'))}
        rows.sort(key=lambda r: order[r['phase']])                       # stable: verify steps first, in card order
        for k, r in enumerate(rows, start=1):
            r['id'], r['priority'] = f'A{k}', k
        return {'rows': rows, 'prevention': prevention, 'guidance_retrieval': {'verify': verify_info, 'recover': recover_info},
                'note': 'The application never executes these actions; the checklist only records analyst decisions. Containment steps need the '
                        'evidence named under "Requires"; a predicted class alone never justifies blocking, isolation or credential changes. '
                        'General prevention is listed separately from incident response.'}

    # ------------------------------------------------------------------------------------------ parts A, B, F and provenance
    def detections(self, frame: pd.DataFrame) -> dict:
        n = len(frame)
        by_class = []
        for c in CLASSES:
            rows = frame[frame.predicted_class == c]
            if len(rows):
                by_class.append({'class': c, 'count': int(len(rows)), 'share': len(rows) / n, 'average_confidence': float(rows.confidence.mean()),
                                 'severity_heuristic': rules.SEVERITY[c], 'status_uncertain': int(rows.status_uncertain.sum()),
                                 'type_uncertain': int(rows.type_uncertain.sum())})
        by_class.sort(key=lambda r: -r['count'])
        sets = {'singleton': int((frame.set_size == 1).sum()), 'multiple': int((frame.set_size > 1).sum()),
                'contains_normal': int(frame.has_normal.sum()), 'attack_status_uncertain': int(frame.status_uncertain.sum()),
                'attack_type_uncertain': int(frame.type_uncertain.sum()),
                'predicted_attack_with_normal_in_set': int((frame.malicious & frame.has_normal).sum()),
                'predicted_normal_with_attack_in_set': int((~frame.malicious & (frame.attack_members > 0)).sum())}
        card = self.app.state.model_card
        return {'total': n, 'by_class': by_class, 'confidence_levels': self.a['confidence_distribution'], 'sets': sets,
                'notes': ['Probabilities are calibrated on held-out rows of the training file (temperature scaling). They describe the model, not '
                          'proof, and can be less reliable on traffic that differs from the training data.',
                          'The 90% prediction set (shortlist) targets 90% coverage on average over flows like the calibration rows; it is not a '
                          'per-flow guarantee, and a single-class set is not certainty.',
                          'Attack-STATUS uncertain: the shortlist contains Normal and at least one attack type (attack or not is unclear). '
                          'Attack-TYPE uncertain: predicted attack and the shortlist has several attack types but not Normal.',
                          'Confidence, heuristic severity and business impact are different quantities.'],
                'shift_evidence': [f"On the official test file (shifted from the training file) the detector flagged "
                                   f"{_pct(card['evaluation']['binary']['false_positive_rate'])} of Normal flows as attacks (model card)."
                                   if card.get('evaluation') else None,
                                   'For the previous detector v2 the 90% sets contained the true class for 79.1% of official test flows (Notebook 21); '
                                   'this was not re-measured for v3.']}

    def summary(self, frame: pd.DataFrame, detections: dict, actions: dict) -> dict:
        a, n = self.a, len(frame)
        sets = detections['sets']
        candidates = [(sets['attack_status_uncertain'], f"{sets['attack_status_uncertain']} flows have a shortlist with both Normal and an attack "
                                                         'type: whether they are attacks at all is uncertain.'),
                      (sets['attack_type_uncertain'], f"{sets['attack_type_uncertain']} predicted attacks have several attack types in their "
                                                       'shortlist: they are likely attacks, but the type is uncertain.'),
                      (a['needs_review_count'], f"{a['needs_review_count']} predictions have confidence below 70%.")]
        uncertainty = max(candidates, key=lambda c: c[0])[1] if n else 'No flows were analysed.'
        first = next((r for r in actions['rows'] if r.get('kind') == 'verify'), None)
        risk = a['risk']
        return {
            'scope': (f'Offline analysis of {n:,} completed network flows uploaded as "{self.item.dataset_name}". The file contains flow '
                      'measurements only: no live traffic, IP addresses, ports, timestamps or payloads.'),
            'predicted_normal': a['safe_count'], 'predicted_attack': a['malicious_count'],
            'predicted_normal_share': a['safe_percentage'], 'predicted_attack_share': a['malicious_percentage'],
            'average_confidence': a['average_confidence'], 'needs_review': a['needs_review_count'],
            'attack_status_uncertain': sets['attack_status_uncertain'], 'attack_type_uncertain': sets['attack_type_uncertain'],
            'status_confidence': ({'levels': a['status_confidence_distribution'], 'average': a['average_status_confidence'],
                                   'meaning': 'how sure the model is that a flow is an attack (any type) or Normal: the calibrated probabilities '
                                              'of the nine attack classes added together. The class confidence (above) is about the exact type.'}
                                  if 'status_confidence_distribution' in a else None),
            'triage_priority': {'level': risk['level'], 'rule': risk['rule'], 'origin': 'heuristic',
                                'meaning': 'Batch triage priority: an application heuristic that orders review work from the predicted classes and '
                                           'fixed severity rules. It is not a model output, not observed business impact and not a confirmed compromise.'},
            'most_important_uncertainty': uncertainty,
            'next_check': first['text'] if first else ('No predicted attacks: spot-check a sample of low-confidence flows.' if n else None),
            'origins': {'counts': 'batch_statistic', 'triage_priority': 'heuristic'},
        }

    def provenance(self) -> dict:
        state, bundle = self.app.state, self.app.state.detector.dir
        manifest = state.detector.manifest or {}
        files = manifest.get('files', {})
        engine = state.reports
        return {
            'analysis': {'id': self.item.id, 'dataset_name': self.item.dataset_name, 'created_at': self.item.created_at.isoformat(),
                         'flows': self.item.total_flows, 'input_sha256': (self.a.get('input') or {}).get('sha256'),
                         'input_note': None if (self.a.get('input') or {}).get('sha256') else 'not recorded for analyses created before report schema 2'},
            'detector': {'name': state.model_card.get('name'), 'version': state.model_card.get('version'),
                         'bundle_created': manifest.get('created'), 'bundle_source_commit': manifest.get('source_commit'),
                         'files_sha256': {k: files.get(k, {}).get('sha256') or _sha256_file(bundle / k)
                                          for k in ('model.txt', 'preprocessing.json', 'calibration.json', 'model_card.json')}},
            'preprocessing': {'inputs': '42 UNSW-NB15 flow features', 'engineered': ['fe_total_bytes', 'fe_byte_balance'],
                              'model_columns': len(state.detector.columns), 'feature_set': 'total_and_byte44'},
            'calibration': {'temperature': state.detector.temperature, 'alpha': state.detector.alpha,
                            'class_offsets': None if state.detector.class_bias is None else [float(b) for b in state.detector.class_bias]},
            'explanation': {'method': 'TreeSHAP (LightGBM pred_contrib) of the base booster, grouped by original column',
                            'exactness': 'exact for the deployed class score: (base value + contributions) / temperature + class offset; not a '
                                         'decomposition of the probability'},
            'knowledge': {'downloaded': state.knowledge_sources.get('downloaded'), 'chunks_sha256': files.get('knowledge_chunks.csv', {}).get('sha256'),
                          'documents': state.knowledge_sources.get('documents', {})},
            'llm': {'model': engine.model, 'runtime': 'local Ollama', 'settings': {'temperature': 0, 'seed': 42, 'num_ctx': 8192},
                    'note': 'uploaded flows are sent only to the local model'},
            'application': {'code_commit': _git_commit(), 'report_schema': SCHEMA_VERSION, 'template': TEMPLATE_VERSION},
        }

    # ------------------------------------------------------------------------------------------ build
    def build(self) -> dict:
        start = time.time()
        self.progress('detections', f'{self.item.total_flows:,} flows')
        t0 = time.time()
        frame = self.flow_rows()
        detections = self.detections(frame)
        self.timings['detections_summary'] = time.time() - t0
        groups = self.group_findings(frame)
        picks = self.selected_flows(frame)
        cards = []
        for i, (fid, why) in enumerate(picks, start=1):
            cards.append(self.flow_card(self.db.get(FlowPrediction, fid), why, i, len(picks)))
        self.progress('checks', 'actions and sources')
        actions = self.actions(cards)
        summary = self.summary(frame, detections, actions)
        llm_statuses = [c['llm']['status'] for c in cards if c['llm']]
        xai_statuses = [c['xai']['status'] for c in cards]
        retrieval = [c['llm']['retrieval'].get('status') for c in cards if c['llm']]
        self.components = {
            'detection': {'status': 'available', 'detail': f'{self.item.total_flows:,} flows analysed in {self.item.seconds:.1f} s'},
            'xai': {'status': 'available' if xai_statuses and all(s == 'available' for s in xai_statuses) else
                    ('partial' if 'available' in xai_statuses else ('not run' if not xai_statuses else 'failed')),
                    'detail': f'{xai_statuses.count("available")} of {len(xai_statuses)} selected flows explained (TreeSHAP); LIME not generated'},
            'retrieval': {'status': 'not run' if not retrieval else ('available' if all(s == 'ok' for s in retrieval) else
                                                                    ('partial' if 'ok' in retrieval else 'unavailable')),
                          'detail': f'{len(self.sources)} distinct passages from {len({s["source"] for s in self.sources.values()})} source families'},
            'llm': {'status': 'not run' if not llm_statuses else ('available' if all(s == 'generated' for s in llm_statuses) else
                                                                 ('partial' if 'generated' in llm_statuses else 'unavailable')),
                    'detail': ', '.join(f'{s}: {llm_statuses.count(s)}' for s in sorted(set(llm_statuses))) or 'no predicted attacks',
                    'reasons': sorted({c['llm']['note'] for c in cards if c['llm'] and c['llm']['note']})},
            'checks': {'status': 'available' if llm_statuses else 'not run',
                       'detail': f"{sum(c['llm']['checked'] for c in cards if c['llm'])} generated statements checked, "
                                 f"{sum(len(c['llm']['removed']) for c in cards if c['llm'])} rejected",
                       'method': 'rule-based: citation validity, typed fact binding (values, contributions, signs, probabilities, prediction set), '
                                 'class / column / number checks, no value judgements or causal claims, lexical overlap with cited passages, real '
                                 'ATT&CK IDs present in cited passages. Not a semantic proof.'},
            'pdf': {'status': 'pending'},
        }
        self.components['group_xai'] = {'status': 'not run' if not groups else ('available' if all(g['status'] == 'available' for g in groups)
                                                                           else 'partial'),
                                         'detail': f"{sum(g.get('analysed', 0) for g in groups)} flows of {len(groups)} attack types explained "
                                                   'for the group findings'}
        coverage = {'flows_explained': len(cards), 'flows_total': len(frame), 'selection_rule': self.SELECTION_RULE,
                    'group_flows_explained': sum(g.get('analysed', 0) for g in groups),
                    'attack_types_with_llm': sum(1 for c in cards if c['llm'] and c['llm']['status'] == 'generated'),
                    'attack_types_detected': len(self.a['attack_distribution'])}
        self.timings['total_record'] = time.time() - start
        record = {
            'schema': 'nad.security-report', 'schema_version': SCHEMA_VERSION, 'template_version': TEMPLATE_VERSION,
            'generated_at': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
            'question': 'What did the detector predict, what evidence supports it, what remains uncertain, and what should an analyst check next?',
            'provenance': self.provenance(), 'components': self.components, 'coverage': coverage,
            'summary': summary, 'detections': detections, 'group_findings': groups, 'flows': cards,
            'sources': {'register': list(self.sources.values()), 'documents': self.documents(),
                        'note': 'Scraping/ingestion collected these documents; retrieval selected passages; the LLM may only cite them. A retrieval '
                                'score measures word similarity, not the probability that a claim is true.'},
            'actions': actions,
            'limits': {
                'unknown': ['No IP addresses, ports, host names, timestamps or payloads are in the uploaded file, so hosts, timelines and botnet '
                            'or DNS attribution cannot be established.',
                            'Predictions are not confirmed incidents; class-to-ATT&CK mappings are hypotheses for context.',
                            'Heuristic severity and triage priority are application rules, not measured impact.',
                            f"Illustrative flows cover {len(cards)} of {len(frame):,} flows; other flows of the same type may differ."],
                'human_review': 'not reviewed yet (see the action checklist)',
                'origins': ORIGINS,
            },
            'appendix': {'model_card': {k: self.app.state.model_card.get(k) for k in ('name', 'version', 'evaluation', 'in_distribution', 'comparison')},
                         'grouped_evaluation': rules.grouped_evaluation(self.app.state.model_card.get('evaluation')),
                         'note': 'Benchmark results on the official UNSW-NB15 test file: context for the model, not performance measured on this upload.'},
            'evaluation': ({'scope': 'Labels found in the uploaded file were compared with the predictions after the analysis. They were not used '
                                     'for predictions, explanations, retrieval or actions.', **self.a['evaluation']} if self.a.get('evaluation') else None),
            'timings': {k: round(v, 3) for k, v in self.timings.items()},
        }
        record['overview'] = build_overview(record)
        record['views'] = views(record)
        json.dumps(record, default=str)                                  # fail early if anything is not serialisable
        return record
