"""The two presentations of one security-report record: Summary (students, general users) and Detailed (analysts, reviewers).

Both views are computed from the same stored record by the functions below. They never call the model, the retriever or the LLM, and they
add no numbers of their own: every count, probability, measurement and citation they show is read from the record. The Summary only
selects and rephrases; the Detailed view shows everything with its evidence. `views(record)` is stored in the record when the report is
built (schema 2.1) and recomputed on request for older schema-2.0 records, so switching views can never change a conclusion.
"""
from __future__ import annotations

import re

from app.core.feature_names import FRIENDLY, plain

# UNSW-NB15 category descriptions, paraphrased from the dataset paper (Moustafa & Slay, 2015). Dataset documentation, not incident evidence.
CLASS_MEANING = {
    'Normal': 'ordinary traffic',
    'Generic': 'traffic resembling attacks that work against block ciphers in general',
    'Exploits': 'traffic resembling attempts to use a known weakness in software',
    'Fuzzers': 'traffic resembling random input sent to make a program misbehave',
    'DoS': 'traffic resembling attempts to make a service unavailable',
    'Reconnaissance': 'traffic resembling scans that gather information about systems',
    'Analysis': 'traffic resembling port scans, spam or HTML-file probing',
    'Backdoor': 'traffic resembling attempts to bypass security to reach a system',
    'Shellcode': 'traffic resembling small code payloads used in exploits',
    'Worms': 'traffic resembling self-copying malicious code spreading between computers',
}
CLASS_SOURCE = 'UNSW-NB15 category descriptions (Moustafa & Slay, 2015)'


def prob_pct(p: float) -> str:
    """A calibrated probability for text: never "1.00" or "100%" (rounding is not certainty)."""
    return 'above 99.5%' if p >= 0.995 else f'{p * 100:.1f}%'


def _n(x: int) -> str:
    return f'{int(x):,}'


def _pct(x: float) -> str:
    return f'{x * 100:.1f}%'


def _top_classes(record: dict, k: int = 3) -> list[dict]:
    return [c for c in record['detections']['by_class'] if c['class'] != 'Normal'][:k]


def _group(record: dict, cls: str) -> dict | None:
    return next((g for g in record.get('group_findings') or [] if g['class'] == cls and g.get('status') == 'available'), None)


def _card(record: dict, cls: str) -> dict | None:
    return next((c for c in record['flows'] if c['prediction']['class'] == cls and c['xai'].get('status') == 'available'), None)


# ------------------------------------------------------------------------------------------------------------ overall explanation
def build_overview(record: dict) -> dict:
    """ONE overall explanation of the batch: 3-5 short bullets (about 100-150 words), each one fact, every fact read from the record.

    Group-level reasons are used only where a group analysis exists (TreeSHAP over the flows of a predicted class, counted per flow, never
    summed). Otherwise the bullet is scoped to the inspected example ("In the inspected example ...").
    """
    s, sets = record['summary'], record['detections']['sets']
    total = record['detections']['total']
    bullets = []
    tops = _top_classes(record)
    if s['predicted_attack']:
        parts = [f"{c['class']} ({_n(c['count'])})" for c in tops]
        groups = ', '.join(parts[:-1]) + (' and ' if len(parts) > 1 else '') + parts[-1]
        bullets.append({'text': f"{_n(s['predicted_attack'])} of {_n(total)} flows ({_pct(s['predicted_attack_share'])}) were predicted as attacks and "
                                f"{_n(s['predicted_normal'])} as Normal. The largest predicted groups are {groups}.",
                        'origin': 'model_output', 'cites': [], 'kind': 'finding'})
    else:
        bullets.append({'text': f"None of the {_n(total)} flows were predicted as attacks.", 'origin': 'model_output', 'cites': [], 'kind': 'finding'})
    for c in tops[:2]:
        g, card = _group(record, c['class']), _card(record, c['class'])
        if g and g['top_supporting']:
            f = g['top_supporting'][0]
            scope = 'the' if g['analysed'] == g['population'] else f"{_n(g['analysed'])} of the"
            bullets.append({'text': f"For {scope} {_n(g['population'])} {c['class']} predictions, the most frequent reason was the "
                                    f"{plain(f['feature'])} (a top-three reason in {_n(f['flows'])} of {_n(g['analysed'])} flows).",
                            'origin': 'xai_attribution', 'cites': [], 'kind': 'reason', 'scope': f"group: predicted {c['class']}"})
        elif card:
            sup = [x for x in card['xai']['top'] if x['contribution'] > 0][:2]
            if sup:
                names = ' and the '.join(plain(x['feature']) for x in sup)
                bullets.append({'text': f"In the inspected {c['class']} example (flow #{card['flow_index']}), the strongest reasons were the {names}.",
                                'origin': 'xai_attribution', 'cites': [f"{card['id']}.E3"] if card['facts'] else [], 'kind': 'reason',
                                'scope': f"one flow: #{card['flow_index']}"})
    if total:
        sc = s.get('status_confidence')
        if sc:
            unc = (f"For {_n(sc['levels']['high'])} of {_n(total)} flows the model is at least 90% sure whether it is an attack; the uncertainty "
                   f"is mostly about the exact type ({_n(s['needs_review'])} predictions have type confidence below 70%).")
        else:
            unc = f"{_n(s['needs_review'])} predictions have confidence below 70%"
            if sets['attack_type_uncertain']:
                unc += f", and {_n(sets['attack_type_uncertain'])} predicted attacks could be more than one attack type"
            unc += '.'
        if sets['attack_status_uncertain']:
            unc += f" For {_n(sets['attack_status_uncertain'])} flows, Normal traffic is still a possible answer."
        bullets.append({'text': unc, 'origin': 'batch_statistic', 'cites': [], 'kind': 'uncertainty'})
    bullets.append({'text': 'These are predictions about completed flows, not confirmed incidents or a forecast of future attacks. Check the '
                            'flagged flows against logs before blocking anything.',
                    'origin': 'analyst_proposed', 'cites': [], 'kind': 'meaning'})
    words = sum(len(b['text'].split()) for b in bullets)
    return {'bullets': bullets[:5], 'words': words,
            'method': 'Written by code from the report record (counts, model outputs, TreeSHAP group findings); no LLM call. Each bullet states its '
                      'origin and scope.'}


# ------------------------------------------------------------------------------------------------------------ AI explanation examples
_CODES = re.compile(r'\b(' + '|'.join(sorted(map(re.escape, FRIENDLY), key=len, reverse=True)) + r')\b')


def plain_statement(text: str) -> str:
    """A checked LLM sentence in plain wording for the Summary: feature codes become their names, score values are dropped, and
    "0.66 confident" becomes "66% confident". Facts are not changed; the exact sentence stays in the Detailed report."""
    text = re.sub(r'\s*\(contributions? [^)]*\)', '', str(text))
    text = re.sub(r'\b0\.(\d\d) confident', lambda m: f'{int(m.group(1))}% confident', text)
    text = _CODES.sub(lambda m: (lambda n: n[:1].lower() + n[1:])(FRIENDLY[m.group(1)]), text)
    return re.sub(r'\s+([,.])', r'', text).strip()


def ai_examples(record: dict, k: int = 2) -> list[dict]:
    """The checked LLM explanation (2-3 points) of one example flow for each of the main predicted attack types."""
    out = []
    for c in _top_classes(record, k):
        card = next((x for x in record['flows'] if x['prediction']['class'] == c['class'] and x.get('reasoning')), None)
        if card:
            out.append({'class': c['class'], 'flow_id': card['id'], 'flow_index': card['flow_index'], 'selection': card['selection'],
                        'points': [{'title': r['title'], 'text': plain_statement(r['text']), 'exact': r['text'], 'cites': r['cites'], 'origin': r['origin']}
                                   for r in card['reasoning']],
                        'model': (card['llm'] or {}).get('model'), 'origin': 'reasoning',
                        'note': 'Five points per example flow. Points marked "AI" were written by the local language model and passed the checks; '
                                'the others are written by code from the same evidence. They explain this example, not every flow of the type.'})
            continue
        card = next((x for x in record['flows'] if x['prediction']['class'] == c['class'] and x['llm'] and x['llm']['explanation']), None)
        if card:
            points = [{'text': plain_statement(e['text']), 'exact': e['text'], 'cites': e['cites'], 'origin': 'llm_synthesis'}
                      for e in card['llm']['explanation']]
            column_facts = {k for k, v in card['facts'].items() if v.startswith('Column ')}
            if card.get('interpretation') and not any(set(p['cites']) & column_facts for p in points):
                # the AI sentence about the measurements did not pass the checks: say it from the TreeSHAP evidence instead
                points.insert(1, {'text': card['interpretation'] + ' (from the model evidence)', 'exact': card['interpretation'], 'cites': [],
                                  'origin': 'xai_attribution'})
            out.append({'class': c['class'], 'flow_id': card['id'], 'flow_index': card['flow_index'], 'selection': card['selection'],
                        'points': points,
                        'model': card['llm']['model'], 'origin': 'llm_synthesis',
                        'note': 'Written by the local language model from the evidence of this one flow; every number and direction was checked '
                                'against that evidence. It explains this example, not every flow of the type.'})
    return out


# ------------------------------------------------------------------------------------------------------------ summary pieces
def detected(record: dict) -> dict:
    """Section 2 of the Summary: does the analysed traffic show signs of an attack, and what the main groups mean."""
    s, sets = record['summary'], record['detections']['sets']
    if s['predicted_attack']:
        answer = (f"Yes. The model found signs of attack-like activity in {_n(s['predicted_attack'])} of {_n(record['detections']['total'])} "
                  'flows. These are predictions from flow measurements, not confirmed incidents, and they do not show whether an attack will '
                  'happen in the future.')
    else:
        answer = 'No. The model predicted every flow as Normal.' + (
            f" For {_n(sets['predicted_normal_with_attack_in_set'])} of them an attack type is still on the shortlist, so they are worth a look."
            if sets.get('predicted_normal_with_attack_in_set') else '')
    groups = []
    for c in _top_classes(record, 3):
        context = None
        card = next((x for x in record['flows'] if x['prediction']['class'] == c['class'] and x['llm']), None)
        if card and card['llm']['context']:
            item = card['llm']['context'][0]
            context = {'text': first_sentence(item['text'], 40), 'cites': item['cites'], 'origin': 'retrieved_guidance',
                       'note': 'what public guidance says about this kind of activity: a possible explanation, not proof'}
        groups.append({'class': c['class'], 'count': c['count'], 'share': c['share'], 'average_confidence': c['average_confidence'],
                       'meaning': CLASS_MEANING.get(c['class'], ''), 'meaning_source': CLASS_SOURCE, 'context': context})
    return {'question': 'Does the analysed traffic show signs of an attack?', 'answer': answer, 'groups': groups}


PHASES = ('Verify', 'Contain if confirmed', 'Recover')


def summary_actions(record: dict, per_phase: int = 2) -> list[dict]:
    """A few prioritised actions for the Summary, taken from the same action rows as the Detailed plan (same ids, text and conditions)."""
    out = []
    for phase in PHASES:
        seen_kinds = set()
        for a in record['actions']['rows']:
            if a['phase'] != phase or len([o for o in out if o['phase'] == phase]) >= per_phase:
                continue
            kind = a.get('kind', a['id'])
            if kind in seen_kinds:
                continue
            seen_kinds.add(kind)
            out.append({'id': a['id'], 'phase': phase, 'what': a.get('title') or a['text'], 'how': first_sentence(a['text']), 'when': a['applicability'],
                        'check': a['validation'], 'owner': a['owner'], 'cites': a['cites']})
    return out


def first_sentence(text: str, words: int = 32) -> str:
    """The first sentence of a step (source passages can be long), cut at a word boundary; the full text stays in the Detailed report."""
    text = ' '.join(str(text).split())
    for stop in ('. ', '; '):
        if stop in text[:400]:
            text = text[:text.index(stop) + 1]
            break
    parts = text.split()
    return text if len(parts) <= words else ' '.join(parts[:words]) + ' ...'


def outcome_checks(record: dict) -> dict:
    s, sets = record['summary'], record['detections']['sets']
    weaken = ['the flagged traffic belongs to a known service, backup, monitoring or test run by your own team',
              'other sensors or logs from the same period show nothing unusual']
    if sets['attack_status_uncertain']:
        weaken.append(f"Normal is still on the shortlist ({_n(sets['attack_status_uncertain'])} flows)")
    return {
        'evidence_needed': 'Packet captures, firewall or sensor logs and host logs from the period when the flows were recorded. The uploaded file '
                           'has no addresses, ports or timestamps, so flows must be matched by protocol, service, state and byte counts.',
        'confirms': 'Another source (logs or captures) shows the same activity, for example the same service, volumes and connection pattern '
                    'as the flagged flows.',
        'weakens': weaken,
        'mitigation_success': 'After a confirmed problem is fixed, the activity no longer appears in new captures or logs, and normal services '
                              'still work.' if s['predicted_attack'] else 'Not applicable: no attacks were predicted.',
        'origin': 'analyst_proposed',
    }


def key_limits(record: dict) -> list[str]:
    out = [record['limits']['unknown'][1], record['limits']['unknown'][0]]
    for name, label in (('xai', 'Model explanations'), ('retrieval', 'Source retrieval'), ('llm', 'The local language model'), ('checks', 'Automated checks')):
        c = record['components'].get(name, {})
        if c.get('status') not in ('available', 'not run', None):
            out.append(f"{label}: {c['status']} ({c.get('detail', '')}).")
    out.append(f"Detailed explanations cover {record['coverage']['flows_explained']} example flows of {_n(record['coverage']['flows_total'])}; "
               'other flows of the same type may differ.')
    return out


def key_sources(record: dict, k: int = 4) -> list[dict]:
    """A few clickable references for the Summary: the passages its sections cite, then the most used ones."""
    cited = {c for g in detected(record)['groups'] if g['context'] for c in g['context']['cites']}
    cited |= {c for a in summary_actions(record) for c in a['cites'] if c.startswith('S')}
    reg = record['sources']['register']
    order = [s for s in reg if s['id'] in cited] + sorted((s for s in reg if s['id'] not in cited), key=lambda s: -len(s['used_by']))
    return [{'id': s['id'], 'source': s['source'], 'location': s['location'], 'url': s.get('url')} for s in order[:k]]


def views(record: dict) -> dict:
    overview = record.get('overview') or build_overview(record)
    return {'summary': {'overview': overview, 'detected': detected(record), 'examples': ai_examples(record), 'actions': summary_actions(record),
                        'outcome': outcome_checks(record), 'limits': key_limits(record), 'sources': key_sources(record)}}
