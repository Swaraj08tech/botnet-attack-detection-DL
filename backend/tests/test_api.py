import io
import time

import numpy as np
import pandas as pd
import pytest

from tests.conftest import synthetic_flows


def csv_bytes(frame):
    return frame.to_csv(index=False).encode()


def poll(client, url, headers, timeout=60):
    """Poll a job like the browser does until it is done or failed."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(url, headers=headers).json()
        if job['status'] in ('done', 'failed'):
            return job
        time.sleep(0.05)
    raise AssertionError(f'job did not finish: {job}')


def analyse(client, headers, frame, name='flows.csv'):
    upload = client.post('/api/uploads', headers=headers, files={'file': (name, csv_bytes(frame), 'text/csv')}).json()
    assert upload['valid'], upload
    job = client.post('/api/analyses', headers=headers, json={'upload_id': upload['upload_id']})
    assert job.status_code == 202 and job.json()['stages'] == ['features', 'detection', 'confidence', 'analytics']
    done = poll(client, f"/api/analyses/jobs/{job.json()['job_id']}", headers)
    assert done['status'] == 'done', done
    return client.get(f"/api/analyses/{done['result']['analysis_id']}", headers=headers).json()


def test_health_is_public(client):
    r = client.get('/api/health')
    assert r.status_code == 200 and r.json()['model_loaded'] is True and r.json()['llm_available'] is False
    assert r.json()['report_generator'] is True


def test_login_and_protected_routes(client, admin_headers):
    assert client.get('/api/analyses').status_code == 401
    assert client.get('/api/auth/me', headers=admin_headers).json()['role'] == 'admin'
    assert client.post('/api/auth/login', data={'username': 'admin@test.local', 'password': 'wrong-password'}).status_code == 401


def test_validation_reports_clear_errors(client, admin_headers):
    def check(name, content):
        return client.post('/api/uploads', headers=admin_headers, files={'file': (name, content, 'text/csv')}).json()

    assert check('flows.txt', b'a,b\n1,2\n')['errors'][0]['title'] == 'Unsupported file type'
    assert check('flows.csv', b'\x00\x01\x02')['valid'] is False
    assert check('flows.csv', csv_bytes(synthetic_flows(5).iloc[:0]))['errors'][0]['title'] == 'Empty dataset'
    missing = check('flows.csv', csv_bytes(synthetic_flows(5).drop(columns=['sttl', 'sbytes'])))
    assert missing['errors'][0]['title'] == 'Missing required features'
    assert {i['name'] for i in missing['errors'][0]['items']} == {'sttl', 'sbytes'} and missing['upload_id'] is None
    bad = synthetic_flows(5).astype({'dur': object})
    bad.loc[2, 'dur'] = 'abc'
    bad.loc[1, 'rate'] = np.inf
    bad.loc[3, 'sload'] = -1
    titles = {e['title'] for e in check('flows.csv', csv_bytes(bad))['errors']}
    assert titles == {'Invalid values', 'Infinite values', 'Negative values'}
    ok = check('flows.csv', csv_bytes(synthetic_flows(20)))
    assert ok['valid'] and ok['rows'] == 20 and ok['upload_id'] and ok['has_labels'] is False


def test_analysis_numbers_are_consistent(client, admin_headers):
    result = analyse(client, admin_headers, synthetic_flows(300))
    a = result['analysis']
    assert result['total_flows'] == a['total_flows'] == 300
    assert a['safe_count'] + a['malicious_count'] == 300
    assert abs(a['safe_percentage'] + a['malicious_percentage'] - 1) < 1e-9
    assert sum(d['count'] for d in a['attack_distribution']) == a['malicious_count']
    assert all(d['attack_type'] != 'Normal' for d in a['attack_distribution'])            # Normal is never an attack
    assert sum(a['confidence_distribution'].values()) == 300
    assert a['needs_review_count'] == a['confidence_distribution']['low']
    assert sum(a['severity_distribution'].values()) == a['malicious_count']
    assert a['risk']['level'] in ('LOW', 'MODERATE', 'HIGH', 'CRITICAL') and a['risk']['explanation']
    assert 0 < a['average_confidence'] <= 1 and 'evaluation' not in a                        # no labels -> no accuracy


def test_flows_are_filterable_and_paginated(client, admin_headers):
    result = analyse(client, admin_headers, synthetic_flows(120))
    base = f"/api/analyses/{result['id']}/flows"
    page = client.get(base, headers=admin_headers, params={'page_size': 25}).json()
    assert page['total'] == 120 and len(page['items']) == 25 and page['items'][0]['flow_index'] == 1
    malicious = client.get(base, headers=admin_headers, params={'traffic': 'malicious', 'page_size': 100}).json()
    assert malicious['total'] == result['malicious_count'] and all(f['prediction'] == 'Malicious' for f in malicious['items'])
    low = client.get(base, headers=admin_headers, params={'confidence': 'low', 'page_size': 100}).json()
    assert low['total'] == result['analysis']['confidence_distribution']['low']
    assert client.get(base, headers=admin_headers, params={'search': '#7'}).json()['items'][0]['flow_index'] == 7
    flow = page['items'][0]
    detail = client.get(f"{base}/{flow['id']}", headers=admin_headers).json()
    assert len(detail['explanation']) > 0 and len(detail['probabilities']) == 10 and 'sttl' in detail['flow']
    assert client.get(f"/api/analyses/{result['id'] + 999}", headers=admin_headers).status_code == 404


def test_labelled_upload_is_evaluated(client, admin_headers):
    frame = synthetic_flows(40)
    frame['attack_cat'] = ['Normal', 'DoS'] * 20
    result = analyse(client, admin_headers, frame, 'official.csv')
    ev = result['analysis']['evaluation']
    assert ev['labelled_flows'] == 40 and ev['normal'] == 20 and ev['attacks'] == 20


def make_report(client, headers, frame):
    result = analyse(client, headers, frame)
    job = client.post(f"/api/analyses/{result['id']}/report", headers=headers)
    assert job.status_code == 202 and job.json()['stages'] == ['detections', 'explanations', 'retrieval', 'generation', 'checks', 'pdf']
    done = poll(client, f"/api/analyses/{result['id']}/report/job", headers)
    assert done['status'] == 'done' and done['stage'] == 'pdf', done
    return result, done['result']['report_id']


def test_pdf_report_can_be_generated_repeatedly(client, admin_headers):
    result, first = make_report(client, admin_headers, synthetic_flows(200))
    job = client.post(f"/api/analyses/{result['id']}/report", headers=admin_headers)          # works again for the same analysis
    second = poll(client, f"/api/analyses/{result['id']}/report/job", admin_headers)['result']['report_id']
    assert second != first
    latest = client.get(f"/api/analyses/{result['id']}", headers=admin_headers).json()['report']
    assert latest['id'] == second and latest['schema_version'] == '2.1' and latest['has_record']
    pdf = client.get(f"/api/reports/{second}/pdf", headers=admin_headers)
    assert pdf.status_code == 200 and pdf.content[:4] == b'%PDF' and pdf.headers['content-disposition'].startswith('inline')
    assert b'/Outlines' in pdf.content and b'/Helvetica' not in pdf.content and b'/Times-Roman' not in pdf.content   # bookmarks, only embedded fonts
    download = client.get(f"/api/reports/{second}/pdf", headers=admin_headers, params={'download': True})
    assert download.headers['content-disposition'].startswith('attachment')
    assert job.status_code == 202 and any(r['id'] == result['id'] and r['report'] for r in client.get('/api/reports', headers=admin_headers).json())


def test_report_record_schema_provenance_and_consistency(client, admin_headers):
    result, rid = make_report(client, admin_headers, synthetic_flows(200))
    r = client.get(f'/api/reports/{rid}/record', headers=admin_headers).json()
    a = result['analysis']
    assert r['schema_version'] == '2.1' and r['report_id'] == rid
    prov = r['provenance']
    assert prov['analysis']['input_sha256'] and len(prov['analysis']['input_sha256']) == 64
    assert prov['detector']['files_sha256']['model.txt'] and prov['explanation']['exactness'].startswith('exact for the deployed class score')
    s = r['summary']                                                    # summary counts are the analysis counts (computed by code)
    assert (s['predicted_normal'], s['predicted_attack'], s['needs_review']) == (a['safe_count'], a['malicious_count'], a['needs_review_count'])
    assert sum(c['count'] for c in r['detections']['by_class']) == a['total_flows']
    assert s['triage_priority']['origin'] == 'heuristic' and 'not a model output' in s['triage_priority']['meaning']
    comps = r['components']                                             # no LLM in tests: truthful statuses, no fake success
    assert comps['detection']['status'] == 'available' and comps['xai']['status'] == 'available' and comps['pdf']['status'] == 'available'
    assert comps['llm']['status'] == 'unavailable' and 'evidence_only_llm_unavailable' in comps['llm']['detail']
    ids = []
    for card in r['flows']:
        x = card['xai']
        assert abs(x['base_value'] + sum(c['contribution'] for c in x['all']) - x['margin']) < 1e-6      # exact decomposition
        assert abs(x['margin'] / x['temperature'] + x['class_offset'] - x['deployed_class_score']) < 1e-9
        assert card['lime']['status'] == 'not generated'
        ids += list(card['facts'])
        if card['llm']:
            statement_ids = [i['id'] for part in ('explanation', 'techniques', 'context', 'actions') for i in card['llm'][part]]
            assert not set(statement_ids) & set(card['facts'])              # statement ids never reuse evidence ids
        if card['llm']:
            assert all(k.startswith(card['id'] + '.E') for k in card['facts'])                       # namespaced evidence ids
    assert len(ids) == len(set(ids))
    register = [src['id'] for src in r['sources']['register']]
    assert len(register) == len(set(register)) and all(src['content_sha256'] for src in r['sources']['register'])
    cited = {c for row in r['actions']['rows'] for c in row['cites']}
    assert cited <= set(register) | set(ids)                                                           # every citation resolves
    assert all(row['phase'] in ('Verify', 'Contain if confirmed', 'Recover') for row in r['actions']['rows'])
    js = client.get(f'/api/reports/{rid}/record', headers=admin_headers, params={'download': True})
    assert js.headers['content-disposition'].startswith('attachment') and js.json()['report_id'] == rid
    flows = client.get(f"/api/analyses/{result['id']}/flows", headers=admin_headers, params={'page_size': 100})
    sets = r['detections']['sets']
    for kind, key in (('status', 'attack_status_uncertain'), ('type', 'attack_type_uncertain'), ('none', 'singleton')):
        got = client.get(f"/api/analyses/{result['id']}/flows", headers=admin_headers, params={'uncertainty': kind}).json()['total']
        assert got == sets[key], (kind, got, sets[key])
    assert flows.status_code == 200


def test_normal_only_batch_has_no_attack_sources_or_actions(client, admin_headers):
    frame = synthetic_flows(40)
    frame['sttl'] = 320.0                                               # the synthetic model maps this TTL band to Normal
    result, rid = make_report(client, admin_headers, frame)
    r = client.get(f'/api/reports/{rid}/record', headers=admin_headers).json()
    assert result['malicious_count'] == 0 and r['actions']['rows'] == [] and r['sources']['register'] == []
    assert r['components']['llm']['status'] == 'not run' and r['flows'] and r['flows'][0]['llm'] is None
    assert r['evaluation'] is None                                      # no labels in the file


def test_action_checklist_history_and_access(client, admin_headers):
    _, rid = make_report(client, admin_headers, synthetic_flows(120))
    r = client.get(f'/api/reports/{rid}/record', headers=admin_headers).json()
    action = r['actions']['rows'][0]['id']
    assert client.post(f'/api/reports/{rid}/actions/{action}', json={'status': 'reviewed', 'note': 'checked'}).status_code == 401
    assert client.post(f'/api/reports/{rid}/actions/ZZ', headers=admin_headers, json={'status': 'reviewed'}).status_code == 404
    assert client.post(f'/api/reports/{rid}/actions/{action}', headers=admin_headers, json={'status': 'executed'}).status_code == 422
    client.post(f'/api/reports/{rid}/actions/{action}', headers=admin_headers, json={'status': 'reviewed', 'note': 'looked at pcap'})
    state = client.post(f'/api/reports/{rid}/actions/{action}', headers=admin_headers, json={'status': 'planned'}).json()
    assert state['status'] == 'planned' and [h['status'] for h in state['history']] == ['reviewed', 'planned'] and state['by'] == 'admin@test.local'
    assert client.get(f'/api/reports/{rid}/actions', headers=admin_headers).json()[action]['status'] == 'planned'


def test_legacy_report_without_record_stays_readable(client, admin_headers):
    from app.db import AnalysisReport
    result = analyse(client, admin_headers, synthetic_flows(30))
    with client.app.state.sessions() as db:
        legacy = AnalysisReport(analysis_id=result['id'], filename='missing.pdf', pages=7, size_bytes=1)
        db.add(legacy)
        db.commit()
        lid = legacy.id
    r = client.get(f'/api/reports/{lid}/record', headers=admin_headers)
    assert r.status_code == 404 and 'schema-1' in r.json()['detail']
    listed = client.get(f"/api/analyses/{result['id']}", headers=admin_headers).json()['report']
    assert listed['schema_version'] == '1.0' and listed['has_record'] is False


def test_sample_upload_and_download(client, admin_headers):
    staged = client.post('/api/uploads/sample', headers=admin_headers, params={'per_class': 5}).json()
    assert staged['valid'] and staged['has_labels'] and staged['filename'].startswith('unsw_nb15_sample_')
    r = client.get('/api/samples/flows.csv', headers=admin_headers, params={'per_class': 5})
    sample = pd.read_csv(io.StringIO(r.text))
    assert 'attachment' in r.headers['content-disposition'] and sample.ground_truth.value_counts().max() <= 5


def test_incident_report_without_llm_is_evidence_only(client, admin_headers):
    result = analyse(client, admin_headers, synthetic_flows(60))
    flows = client.get(f"/api/analyses/{result['id']}/flows", headers=admin_headers, params={'traffic': 'malicious'}).json()['items']
    url = f"/api/analyses/{result['id']}/flows/{flows[0]['id']}/incident-report"
    report = client.post(url, headers=admin_headers, params={'wait': True}).json()
    assert report['llm_used'] is False and 'unavailable' in report['note'] and report['possible_attacks']
    assert client.get(url, headers=admin_headers).json() == report
    job = client.post(url, headers=admin_headers)
    assert job.status_code == 202
    assert poll(client, f'{url}/job', admin_headers)['status'] == 'done'


def test_overview_and_model(client, admin_headers):
    analyse(client, admin_headers, synthetic_flows(50))
    o = client.get('/api/overview', headers=admin_headers).json()
    assert o['files_analyzed'] >= 1 and o['total_flows'] >= 50 and o['total_threats'] + o['safe_flows'] == o['total_flows']
    assert all(t['attack_type'] != 'Normal' for t in o['attacks_by_type'])
    m = client.get('/api/model', headers=admin_headers).json()
    assert len(m['calibration']['class_thresholds']) == 10 and m['rules']['confidence_levels']['high'] == '>= 90%'
    assert len(m['features']) == 42


def test_predict_endpoint(client, admin_headers):
    r = client.post('/api/predict', headers=admin_headers, json={'flows': synthetic_flows(3).to_dict('records')})
    assert r.status_code == 200 and len(r.json()) == 3 and len(r.json()[0]['probabilities']) == 10


# ------------------------------------------------------------------------------------------------ two report views (schema 2.1)
def pdf_text(content: bytes) -> str:
    pypdf = pytest.importorskip('pypdf')
    return ' '.join(' '.join((p.extract_text() or '').split()) for p in pypdf.PdfReader(io.BytesIO(content)).pages)


def test_summary_and_detailed_views_share_the_same_facts(client, admin_headers):
    result, rid = make_report(client, admin_headers, synthetic_flows(200))
    r = client.get(f'/api/reports/{rid}/record', headers=admin_headers).json()
    v = r['views']['summary']
    bullets = v['overview']['bullets']
    assert 3 <= len(bullets) <= 5 and v['overview']['words'] <= 150
    assert bullets == r['overview']['bullets']                                   # stored once, used by both views
    a = result['analysis']
    assert bullets[0]['text'].startswith(f"{a['malicious_count']} of {a['total_flows']} flows")
    for b in bullets:                                                             # no generalisation from one flow to a group
        if b['kind'] == 'reason':
            assert b['scope'].startswith('group:') or b['text'].startswith('In the inspected')
    ids = {row['id'] for row in r['actions']['rows']}
    assert {x['id'] for x in v['actions']} <= ids                                # the Summary only selects existing actions
    summary = client.get(f'/api/reports/{rid}/pdf', headers=admin_headers, params={'view': 'summary'})
    detailed = client.get(f'/api/reports/{rid}/pdf', headers=admin_headers)
    assert summary.status_code == 200 and summary.content[:4] == b'%PDF' and 'summary.pdf' in summary.headers['content-disposition']
    assert 'detailed.pdf' in detailed.headers['content-disposition']
    s_text, d_text = pdf_text(summary.content), pdf_text(detailed.content)
    for b in bullets:                                                             # identical findings in both PDFs
        text = ' '.join(b['text'].split())
        assert text in s_text and text in d_text, text
    import pypdf
    assert len(pypdf.PdfReader(io.BytesIO(summary.content)).pages) <= 4
    assert 'Does the analysed traffic show signs of an attack?' in s_text and 'going to happen' not in s_text
    assert '1.00 confident' not in d_text and '100.0%' not in s_text


def test_summary_pdf_needs_login_and_a_record(client, admin_headers):
    _, rid = make_report(client, admin_headers, synthetic_flows(60))
    assert client.get(f'/api/reports/{rid}/pdf', params={'view': 'summary'}).status_code == 401
    assert client.get(f'/api/reports/{rid}/pdf', headers=admin_headers, params={'view': 'other'}).status_code == 422
    from app.db import AnalysisReport
    result = analyse(client, admin_headers, synthetic_flows(20))
    with client.app.state.sessions() as db:
        legacy = AnalysisReport(analysis_id=result['id'], filename='missing.pdf', pages=7, size_bytes=1)
        db.add(legacy)
        db.commit()
        lid = legacy.id
    r = client.get(f'/api/reports/{lid}/pdf', headers=admin_headers, params={'view': 'summary'})
    assert r.status_code == 404 and 'schema-1' in r.json()['detail']


def test_group_findings_and_selection_rule(client, admin_headers):
    _, rid = make_report(client, admin_headers, synthetic_flows(200))
    r = client.get(f'/api/reports/{rid}/record', headers=admin_headers).json()
    assert r['group_findings']
    for g in r['group_findings']:
        assert g['status'] == 'available' and g['analysed'] == g['population']
        assert all(f['flows'] <= g['analysed'] for f in g['top_supporting'])      # counts of flows, not summed contributions
        assert 'not added across flows' in g['method']
    picks = [c['selection'] for c in r['flows']]
    assert any(p.startswith('typical') and 'median confidence' in p for p in picks)
    assert 'MEDIAN' in r['coverage']['selection_rule'] and r['components']['group_xai']['status'] == 'available'
    for card in r['flows']:
        assert card['interpretation'] and isinstance(card['alternatives'], list)


def test_older_schema_2_0_record_gets_its_views_on_read(client, admin_headers):
    _, rid = make_report(client, admin_headers, synthetic_flows(80))
    from sqlalchemy.orm.attributes import flag_modified

    from app.db import AnalysisReport
    with client.app.state.sessions() as db:
        report = db.get(AnalysisReport, rid)
        report.record = {**{k: v for k, v in report.record.items() if k not in ('views', 'overview', 'group_findings')}, 'schema_version': '2.0'}
        flag_modified(report, 'record')
        db.commit()
    r = client.get(f'/api/reports/{rid}/record', headers=admin_headers).json()
    assert r['schema_version'] == '2.0' and 'group_findings' not in r and r['views']['summary']['overview']['bullets']
    assert client.get(f'/api/reports/{rid}/pdf', headers=admin_headers, params={'view': 'summary'}).status_code == 200


# ------------------------------------------------------------------------------------------------ two confidences and the grouped view
def test_attack_or_normal_confidence_is_reported_and_backfilled(client, admin_headers):
    result = analyse(client, admin_headers, synthetic_flows(150))
    a = result['analysis']
    levels = a['status_confidence_distribution']
    assert sum(levels.values()) == a['total_flows'] and 0 <= a['average_status_confidence'] <= 1
    assert a['average_status_confidence'] >= a['average_confidence'] - 1e-9          # a sum of probabilities is never below one of them
    from sqlalchemy.orm.attributes import flag_modified

    from app.db import Analysis
    with client.app.state.sessions() as db:                                         # an analysis created before the field existed
        item = db.get(Analysis, result['id'])
        item.summary = {k: v for k, v in item.summary.items() if not k.startswith(('status_', 'average_status'))}
        flag_modified(item, 'summary')
        db.commit()
    again = client.get(f"/api/analyses/{result['id']}", headers=admin_headers).json()['analysis']
    assert again['status_confidence_distribution'] == levels and again['average_status_confidence'] == a['average_status_confidence']


def test_grouped_view_is_derived_from_the_confusion_matrix(client, admin_headers):
    import json
    import pathlib

    from app.core.analysis import grouped_evaluation
    m = client.get('/api/model', headers=admin_headers).json()
    assert m['grouped_evaluation'] is None                                        # the synthetic test bundle has no benchmark: nothing invented
    card = pathlib.Path(__file__).resolve().parents[2] / 'serving' / 'bundle' / 'model_card.json'
    if not card.exists():
        pytest.skip('serving bundle not available')
    e = json.loads(card.read_text(encoding='utf-8'))['evaluation']
    g = grouped_evaluation(e)
    assert g['group'] == 'DoS / Analysis / Backdoor' and len(g['per_class']) == len(e['per_class']) - 2
    total = sum(sum(row) for row in e['confusion_matrix']['matrix'])
    assert sum(r['support'] for r in g['per_class']) == total
    assert g['accuracy'] >= e['accuracy'] and 'no new evaluation' in g['note']
    unchanged = {r['class']: r for r in e['per_class']}
    for r in g['per_class']:
        if r['class'] in unchanged and r['class'] not in ('Normal',):                # classes outside the group keep their recall
            assert abs(r['recall'] - unchanged[r['class']]['recall']) < 1e-9


def test_report_has_both_confidences_and_examples_section(client, admin_headers):
    _, rid = make_report(client, admin_headers, synthetic_flows(200))
    r = client.get(f'/api/reports/{rid}/record', headers=admin_headers).json()
    sc = r['summary']['status_confidence']
    assert sum(sc['levels'].values()) == r['detections']['total']
    for ex in r['views']['summary']['examples']:                                    # no LLM in tests: no AI-written point is invented
        assert ex['points'] and all(p['origin'] != 'llm_synthesis' for p in ex['points'])
    assert r['appendix']['grouped_evaluation'] is None or r['appendix']['grouped_evaluation']['group'] == 'DoS / Analysis / Backdoor'


def test_summary_pdf_marks_ai_written_reasoning_points(client, admin_headers):
    _, rid = make_report(client, admin_headers, synthetic_flows(200))
    from sqlalchemy.orm.attributes import flag_modified

    from app.db import AnalysisReport
    with client.app.state.sessions() as db:                                         # simulate one accepted LLM point in the reasoning
        report = db.get(AnalysisReport, rid)
        card = next(c for c in report.record['flows'] if c['prediction']['malicious'] and c['facts'])
        card['reasoning'][0] = {**card['reasoning'][0], 'origin': 'llm_synthesis', 'cites': [f"{card['id']}.E1"]}
        flag_modified(report, 'record')
        db.commit()
    r = client.get(f'/api/reports/{rid}/record', headers=admin_headers).json()
    points = next(ex for ex in r['views']['summary']['examples'] if ex['flow_id'] == card['id'])['points']
    assert points[0]['origin'] == 'llm_synthesis' and all(p['title'] for p in points)
    pdf = client.get(f'/api/reports/{rid}/pdf', headers=admin_headers, params={'view': 'summary'})
    assert pdf.status_code == 200 and '(AI, checked)' in pdf_text(pdf.content)

def test_flow_cards_carry_five_point_reasoning_without_the_llm(client, admin_headers):
    _, rid = make_report(client, admin_headers, synthetic_flows(200))
    r = client.get(f'/api/reports/{rid}/record', headers=admin_headers).json()
    for card in r['flows']:
        slots = [x['slot'] for x in card['reasoning']]
        assert slots[0] == 'certainty' and 'reasons' in slots and len(slots) >= 3
        assert all(x['origin'] != 'llm_synthesis' for x in card['reasoning'])         # no LLM in tests: every point written by code
        if card['contrast']:
            assert card['contrast']['runner_up'] != card['prediction']['class']
            assert all(d['difference'] > 0 for d in card['contrast']['favouring'])
        assert abs(card['status_probability'] - (1 - card['prediction']['probabilities']['Normal'] if card['prediction']['malicious']
                                              else card['prediction']['probabilities']['Normal'])) < 1e-9
