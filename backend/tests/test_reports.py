import pandas as pd
import pytest

from app.core.reports import ReportEngine
from tests.conftest import PREDICTORS


@pytest.fixture(scope='module')
def engine(bundle_dir):
    return ReportEngine(bundle_dir, 'http://127.0.0.1:9', 'none', PREDICTORS)


@pytest.fixture(scope='module')
def evidence(engine):
    return engine.evidence('DoS', 0.83, ['DoS', 'Exploits'], [{'feature': 'sbytes', 'value': 200, 'contribution': 1.2},
                                                              {'feature': 'sttl', 'value': 254, 'contribution': 0.7}])


def test_evidence_and_retrieval(evidence):
    facts, sources = evidence
    assert facts['E1'] == 'Predicted class: DoS with calibrated probability 0.83'
    assert 'DoS, Exploits' in facts['E2'] and 'sbytes = 200' in facts['E3']
    assert any('T1498' in s.text for s in sources.values())


def test_verifier_rules(engine, evidence):
    facts, sources = evidence
    s_attack = next(k for k, s in sources.items() if 'T1498' in s.text)
    assert engine.check_claim('The probability is 0.83.', ['E1'], facts, sources)[0]
    assert not engine.check_claim('The probability is 0.97.', ['E1'], facts, sources)[0]            # wrong number
    assert not engine.check_claim('Something happened.', [], facts, sources)[0]                       # no citation
    assert not engine.check_claim('Something happened.', ['S99'], facts, sources)[0]                  # unknown citation
    assert not engine.check_claim('The value of dbytes is high.', ['E3'], facts, sources)[0]          # column not in cited fact
    assert engine.check_claim('Network service discovery is involved.', ['E1'], facts, sources)[0]  # English "service" is fine
    assert engine.check_technique('T1498', [s_attack], sources)[0]
    assert not engine.check_technique('T9999', [s_attack], sources)[0]                                # not a real technique
    assert not engine.check_technique('T1046', ['E1'], sources)[0]                                    # not in a cited source


def test_report_for_normal_flow_is_skipped(engine):
    result = engine.generate('Normal', 0.99, ['Normal'], [{'feature': 'sbytes', 'value': 10, 'contribution': 0.5}])
    assert result.llm_used is False and 'Normal' in result.note


def test_uncertain_alert_covers_every_class_in_the_set(engine):
    facts, sources = engine.evidence('DoS', 0.4, ['Analysis', 'DoS', 'Normal'], [{'feature': 'sbytes', 'value': 200, 'contribution': 1.0}])
    locations = [s.location for s in sources.values()]
    assert 'T1498' in locations and 'T1046' in locations          # DoS technique and Analysis technique, nothing for Normal


def test_llm_failure_falls_back_to_evidence_only(bundle_dir):
    engine = ReportEngine(bundle_dir, 'http://127.0.0.1:9', 'none', PREDICTORS, timeout=2)
    engine.available = lambda: True                                   # pretend the LLM is up; the call itself then fails
    stages = []
    result = engine.generate('DoS', 0.8, ['DoS'], [{'feature': 'sbytes', 'value': 200, 'contribution': 1.0}], progress=stages.append)
    assert result.llm_used is False and 'did not answer' in result.note
    assert result.findings and result.facts['E1'].startswith('Predicted class')
    assert stages == ['evidence', 'writing']


def test_possible_attacks_cover_the_whole_shortlist(engine):
    result = engine.generate('DoS', 0.4, ['Analysis', 'DoS', 'Normal'], [{'feature': 'sbytes', 'value': 200, 'contribution': 1.0}])
    assert [p['class'] for p in result.possible_attacks] == ['DoS', 'Analysis']        # predicted first, Normal left out
    assert result.possible_attacks[0]['predicted'] and result.possible_attacks[0]['techniques'][0]['id'] == 'T1498'


def test_prediction_evidence_verifier(engine):
    facts = {'E1': 'Predicted class: DoS with calibrated probability 0.83',
             'E3': 'Column sbytes = 200 supports DoS (contribution +1.153)',
             'E4': 'Column sttl = 254 argues against DoS (contribution -0.372)'}
    ok = engine.check_evidence_claim
    assert ok('The value sbytes = 200 supports DoS with a contribution of +1.153.', ['E3'], facts)[0]
    assert ok('sttl = 254 argues against DoS (contribution −0.372).', ['E4'], facts)[0]          # unicode minus
    assert not ok('The value sbytes = 300 supports DoS.', ['E3'], facts)[0]                               # wrong number
    assert not ok('The large sbytes = 200 supports DoS.', ['E3'], facts)[0]                               # value judgement
    assert not ok('sbytes = 200 supports Exploits.', ['E3'], facts)[0]                                    # class not in the fact
    assert not ok('dbytes = 200 supports DoS.', ['E3'], facts)[0]                                         # column not in the fact
    assert not ok('sbytes = 200 supports DoS.', ['S1'], facts)[0]                                         # must cite E facts
    assert ok('The model predicts DoS with probability 0.83.', ['E1'], facts)[0]


def test_extra_facts_are_added_as_evidence(engine):
    facts, _ = engine.evidence('DoS', 0.5, ['DoS'], [{'feature': 'sbytes', 'value': 200, 'contribution': 1.0}],
                               extra_facts=['In the analysed dataset, 36 flows were classified as DoS'])
    assert facts['E4'] == 'In the analysed dataset, 36 flows were classified as DoS'


def test_generation_path_with_a_simulated_llm(bundle_dir):
    """The LLM answer is replaced by a fixed one: good items are kept, unsupported ones removed and counted."""
    engine = ReportEngine(bundle_dir, 'http://127.0.0.1:9', 'none', PREDICTORS)
    engine.available = lambda: True
    answer = {'attack_techniques': [{'id': 'T1498', 'cites': ['S1']}, {'id': 'T9999', 'cites': ['S1']}],
              'context': [{'text': 'Network Denial of Service flood traffic exhausts bandwidth.', 'cites': ['S1']}],
              'recommended_actions': [{'text': 'Isolate affected systems and networks during incident response containment.', 'cites': ['S2']}],
              'prediction_evidence': [{'text': 'The value sbytes = 200 supports DoS (contribution +1.200).', 'cites': ['E3']},
                                      {'text': 'An unusually large sbytes = 200 was seen.', 'cites': ['E3']}]}
    engine._chat = lambda system, user: answer
    explanation = [{'feature': 'sbytes', 'value': 200, 'contribution': 1.2}]
    r = engine.generate('DoS', 0.8, ['DoS'], explanation, extra_facts=['In the analysed dataset, 36 flows were classified as DoS'])
    assert r.llm_used and r.status == 'generated' and r.checked_items == 12          # 6 items in the first answer + 6 in the repair round
    assert [t['id'] for t in r.techniques] == ['T1498'] and len(r.claims) == 1 and r.claims[0]['round'] == 2
    assert r.claims[0]['support']['overlap'] >= 0.5 and 'not a check of meaning' in r.claims[0]['support']['method']
    assert [e['text'] for e in r.evidence_explanation] == ['The value sbytes = 200 supports DoS (contribution +1.200).']
    assert {x['round'] for x in r.removed} == {1, 2} and r.removed_items == 4    # fake technique ID + value judgement, in both rounds
    from types import SimpleNamespace

    from app.core.report_record import RecordBuilder
    builder = RecordBuilder.__new__(RecordBuilder)                                   # only the citation namespacing is exercised here
    builder.app, builder.sources, builder._by_chunk = SimpleNamespace(state=SimpleNamespace(knowledge_sources={})), {}, {}
    llm = builder._namespace(r, 'F7')
    statements = [i['id'] for part in ('explanation', 'techniques', 'context', 'actions') for i in llm[part]]
    assert statements == ['F7.X1', 'F7.T1', 'F7.C1', 'F7.A1'] and not set(statements) & set(llm['facts'])
    assert llm['explanation'][0]['cites'] == ['F7.E3'] and all(c.startswith('S') for i in llm['context'] + llm['actions'] for c in i['cites'])


def test_pasted_facts_are_not_an_explanation(engine):
    facts = {'E3': 'Column sbytes = 200 supports DoS (contribution +1.153)', 'E4': 'Column sttl = 254 supports DoS (contribution +0.500)'}
    pasted = 'Column sbytes = 200 supports DoS (contribution +1.153), Column sttl = 254 supports DoS (contribution +0.500)'
    assert not engine.check_evidence_claim(pasted, ['E3', 'E4'], facts)[0]
    assert engine.check_evidence_claim('The strongest evidence is sbytes = 200 (contribution +1.153) and sttl = 254 (contribution +0.500).',
                                       ['E3', 'E4'], facts)[0]


def test_incomplete_citations_are_completed_but_wrong_values_still_fail(engine):
    facts = {'E1': 'Predicted class: DoS with calibrated probability 0.63',
             'E3': 'Column sttl = 62 supports DoS (contribution +1.076)', 'E4': 'Column smean = 172 supports DoS (contribution +0.689)'}
    good = {'prediction_evidence': [{'text': 'The strongest evidence is sttl = 62 (contribution +1.076) and smean = 172 (contribution +0.689).', 'cites': ['E3']}]}
    item = engine.verify(good, facts, {})['evidence'][0]
    assert item['supported'] and item['cites'] == ['E3', 'E4'] and 'citations completed' in item['reason']
    wrong = {'prediction_evidence': [{'text': 'The strongest evidence is sttl = 62 (contribution +1.076) and smean = 999.', 'cites': ['E3']}]}
    assert not engine.verify(wrong, facts, {})['evidence'][0]['supported']


def test_prediction_set_claims_must_match_the_evidence(engine):
    facts = {'E1': 'Predicted class: DoS with calibrated probability 0.63', 'E2': 'Conformal prediction set (90% coverage): DoS, Exploits'}
    ok = engine.check_evidence_claim
    assert not ok('The model is 0.63 confident, and its 90% prediction set contains only DoS.', ['E1', 'E2'], facts)[0]
    assert ok('The model is 0.63 confident, and its 90% prediction set contains DoS and Exploits.', ['E1', 'E2'], facts)[0]
    single = {'E2': 'Conformal prediction set (90% coverage): Generic'}
    assert ok('Its 90% prediction set contains only Generic.', ['E2'], single)[0]


def test_direction_of_each_column_must_match_the_evidence(engine):
    facts = {'E3': 'Column ct_src_ltm = 11 supports Backdoor (contribution +3.021)',
             'E7': 'Column sload = 160000000 supports Backdoor (contribution +0.624)',
             'E4': 'Column sttl = 254 argues against Backdoor (contribution -0.300)'}
    ok = engine.check_evidence_claim
    assert not ok('The strongest evidence is ct_src_ltm = 11 (contribution +3.021), while sload = 160000000 (contribution +0.624) argues against it.',
                  ['E3', 'E7'], facts)[0]
    assert ok('The strongest evidence is ct_src_ltm = 11 (contribution +3.021), while sttl = 254 (contribution -0.300) argues against it.',
              ['E3', 'E4'], facts)[0]
    assert not ok('sttl = 254 (contribution -0.300) supports Backdoor.', ['E4'], facts)[0]


# ------------------------------------------------------------------------------------------- report schema 2: regression tests of the checks
FACTS = {'E1': 'Predicted class: DoS with calibrated probability 0.83',
         'E2': 'Conformal prediction set (90% coverage): DoS, Exploits',
         'E3': 'Column sbytes = 200 supports DoS (contribution +1.153)',
         'E4': 'Column sttl = 254 argues against DoS (contribution -0.372)'}


def test_check_catches_swapped_values_and_contributions(engine):
    ok = engine.check_evidence_claim
    assert not ok('The strongest evidence is sbytes = 254 (contribution +1.153).', ['E3', 'E4'], FACTS)[0]        # value of the other feature
    assert not ok('sbytes = 200 (contribution +0.372) supports DoS.', ['E3', 'E4'], FACTS)[0]                    # contribution of the other feature
    assert ok('sbytes = 200 (contribution +1.153) supports DoS, while sttl = 254 (contribution -0.372) argues against it.', ['E3', 'E4'], FACTS)[0]


def test_check_catches_wrong_probability_and_causal_claims(engine):
    ok = engine.check_evidence_claim
    assert not ok('The model is 0.95 confident that this flow is DoS.', ['E1'], FACTS)[0]
    assert ok('The model is 83% confident that this flow is DoS.', ['E1'], FACTS)[0]
    assert not ok('sbytes = 200 proves that the flow is DoS.', ['E3'], FACTS)[0]
    assert not ok('sbytes = 200 caused the DoS prediction.', ['E3'], FACTS)[0]


def test_check_rejects_foreign_citations_and_unrelated_passages(engine, evidence):
    facts, sources = evidence
    report = {'prediction_evidence': [{'text': 'sbytes = 200 supports DoS.', 'cites': ['E9']}],                     # id from another flow
              'context': [{'text': 'Shellcode payloads are injected into processes.', 'cites': ['S1']},             # passage says something else
                          {'text': 'This flow is a denial of service flood that exhausts bandwidth.', 'cites': ['S1']}],
              'recommended_actions': [{'text': 'Isolate affected systems.', 'cites': ['E1']}]}
    checked = engine.verify(report, facts, sources)
    assert not checked['evidence'][0]['supported']
    assert [c['supported'] for c in checked['claims']] == [False, False]
    assert 'analysed flows' in checked['claims'][1]['reason']
    assert not checked['actions'][0]['supported'] and 'source passage' in checked['actions'][0]['reason']


def test_supported_source_claim_records_span_and_method(engine, evidence):
    facts, sources = evidence
    s_attack = next(k for k, s in sources.items() if 'T1498' in s.text)
    checked = engine.verify({'context': [{'text': 'Network Denial of Service flood traffic exhausts bandwidth.', 'cites': [s_attack]}]}, facts, sources)
    item = checked['claims'][0]
    assert item['supported'] and item['support']['source'] == s_attack and item['support']['overlap'] >= 0.5
    assert 'not a check of meaning' in item['support']['method']
    span = item['support']
    assert str(sources[s_attack].text)[span['start']:span['end']] == span['span']          # offsets point at exactly the highlighted sentence


def test_retrieval_dedupes_rejects_low_scores_and_skips_normal(engine):
    sources, info = engine.retrieve('DoS', ['DoS', 'DoS'])
    keys = [s['chunk_id'] for s in sources.values()]
    assert len(keys) == len(set(keys)) and info['status'] == 'ok' and info['min_score'] > 0
    assert all(pd.isna(s['score']) or s['score'] >= info['min_score'] for s in sources.values())   # mapped passages have no score
    r = engine.generate('Normal', 0.9, ['Normal'], [{'feature': 'sbytes', 'value': 10, 'contribution': 0.5}])
    assert r.status == 'not_applicable' and r.sources == [] and r.retrieval['status'] == 'not_applicable'


def test_llm_failure_statuses(bundle_dir):
    from app.core.reports import LLMError
    explanation = [{'feature': 'sbytes', 'value': 200, 'contribution': 1.0}]

    def run(chat):
        e = ReportEngine(bundle_dir, 'http://127.0.0.1:9', 'none', PREDICTORS)
        e.available = lambda: True
        e._chat = chat
        return e.generate('DoS', 0.8, ['DoS'], explanation)

    def timeout(*_):
        raise LLMError('no answer within 240 s')

    assert run(timeout).status == 'llm_timeout'
    assert run(lambda *_: None).status == 'invalid_response'
    bad = {'attack_techniques': [{'id': 'T9999', 'cites': ['S1']}], 'prediction_evidence': [{'text': 'sbytes = 999 supports DoS.', 'cites': ['E3']}]}
    r = run(lambda *_: bad)
    assert r.status == 'all_rejected' and r.removed_items == 4 and r.facts['E1'].startswith('Predicted class')
    offline = ReportEngine(bundle_dir, 'http://127.0.0.1:9', 'none', PREDICTORS).generate('DoS', 0.8, ['DoS'], explanation)
    assert offline.status == 'evidence_only_llm_unavailable' and offline.findings


def test_batch_average_confidence_is_bound_to_the_dataset_fact(engine):
    """Regression from the end-to-end run: "average confidence" of the class is not the flow's probability (and vice versa)."""
    facts = {'E1': 'Predicted class: Exploits with calibrated probability 0.99', 'E2': 'Conformal prediction set (90% coverage): Exploits',
             'E3': 'Column ct_srv_dst = 1 supports Exploits (contribution +1.636)',
             'E8': 'In the analysed dataset, 45 flows (16.9% of the predicted attacks) were classified as Exploits, with average confidence 0.67'}
    head = 'The model is 0.99 confident that this flow is Exploits, and ct_srv_dst = 1 (contribution +1.636) supports it. '
    good = head + 'In the analysed dataset, 45 flows (16.9% of the predicted attacks) were classified as Exploits, with average confidence 0.67.'
    assert engine.check_evidence_claim(good, ['E1', 'E3', 'E8'], facts) == (True, 'supported')
    swapped = good.replace('average confidence 0.67', 'average confidence 0.99')          # the repair-round answer that used to pass
    ok, why = engine.check_evidence_claim(swapped, ['E1', 'E3', 'E8'], facts)
    assert not ok and why == 'wrong average confidence 0.99: the evidence says 0.67'
    assert engine.check_evidence_claim(good.replace('45 flows', '16 flows'), ['E1', 'E3', 'E8'], facts)[1].startswith('number 16')
    wrong_share = good.replace('16.9% of the predicted', '67% of the predicted')
    assert not engine.check_evidence_claim(wrong_share, ['E1', 'E3', 'E8'], facts)[0]
    assert engine.check_evidence_claim(good.replace('0.99 confident', '0.67 confident'), ['E1', 'E3', 'E8'], facts)[1].startswith('wrong probability 0.67')


def test_table_of_contents_lines_are_never_retrieved(engine):
    found = engine._search('6 Detection & Analysis 10 Containment table of contents', 20, ['NIST SP 800-61r3', 'NIST SP 800-94', 'CISA IR Playbooks'])
    assert len(found) and not found.text.str.contains(r'\.{6,}').any()


def test_pdf_text_has_no_missing_glyphs():
    from app.core.pdf_report import esc
    assert esc('inputs \u2192 195 columns <b> & \u4e2d') == 'inputs -&gt; 195 columns &lt;b&gt; &amp; ?'


def test_action_prerequisites_depend_on_the_kind_of_step():
    from app.core.report_record import action_profile
    filtering = action_profile('Filter Network Traffic: block the flood upstream.', 'DoS')
    creds = action_profile('Change system admin passwords and rotate private keys.', 'Generic')
    dns = action_profile('Prevent DNS resolution of known attacker domain names.', 'Generic')
    assert filtering['kind'] == 'filtering' and 'source, destination and service are identified' in filtering['requires']
    assert creds['kind'] == 'credentials' and 'Authentication logs' in creds['requires'] and 'Not reversible' in creds['rollback']
    assert dns['kind'] == 'domain' and 'no domain names' in dns['requires']
    for p in (filtering, creds, dns):
        assert 'prediction alone is not enough' in p['requires']
    assert len({p['owner'] for p in (filtering, creds, dns)}) == 3                 # not the same generic owner for every action
    assert action_profile('Regularly scan the internal network for vulnerabilities.', 'Worms')['prevention']
    assert action_profile('Ensure that unnecessary ports and services are closed.', 'DoS')['prevention']
    assert action_profile('Keep the remote desktop port closed to the internet.', 'DoS')['kind'] == 'hardening'


def test_probabilities_are_never_written_as_certainty(engine):
    from app.core.reports import prob_text
    assert prob_text(0.9978) == 'above 0.99' and prob_text(0.5751) == '0.58'
    facts = {'E1': 'Predicted class: Generic with calibrated probability above 0.99', 'E2': 'Conformal prediction set (90% coverage): Generic'}
    ok, why = engine.check_evidence_claim('The model is 1.00 confident that this flow is Generic.', ['E1'], facts)
    assert not ok and ('not certainty' in why or 'number 1 not in cited facts' in why)       # rejected either way
    assert engine.check_evidence_claim('The model is 0.99 confident that this flow is Generic.', ['E1'], facts)[0]


def test_direction_is_checked_for_every_column_of_a_list_and_signs_must_match(engine):
    """Regression from the end-to-end run: "while A and B argue against it" was accepted although A supports the class."""
    facts = {'E1': 'Predicted class: DoS with calibrated probability 0.58', 'E2': 'Conformal prediction set (90% coverage): DoS',
             'E3': 'Column smean = 100 supports DoS (contribution +1.395)', 'E4': 'Column proto = unas supports DoS (contribution +0.343)',
             'E5': 'Column ct_srv_src = 11 argues against DoS (contribution -0.336)'}
    wrong = ('The strongest evidence is smean = 100 (contribution +1.395), while proto = unas (contribution +0.343) and ct_srv_src = 11 '
             '(contribution -0.336) argue against it.')
    ok, why = engine.check_evidence_claim(wrong, ['E3', 'E4', 'E5'], facts)
    assert not ok and why == 'wrong direction: proto supports the class'
    right = ('The strongest evidence is smean = 100 (contribution +1.395) and proto = unas (contribution +0.343), while ct_srv_src = 11 '
             '(contribution -0.336) argues against it.')
    assert engine.check_evidence_claim(right, ['E3', 'E4', 'E5'], facts) == (True, 'supported')
    flipped = 'The strongest evidence is smean = 100 (contribution -1.395).'
    assert engine.check_evidence_claim(flipped, ['E3'], facts)[1] == 'wrong sign for smean: the evidence says +1.395'


def test_plain_statement_keeps_facts_and_drops_codes():
    from app.core.report_views import plain_statement
    text = 'The strongest evidence is sttl = 62 (contribution +0.498), while ct_srv_dst = 5 argues against it (contribution -0.285).'
    out = plain_statement(text)
    assert out == ('The strongest evidence is source time-to-live (TTL) = 62, while recent connections: same service and destination = 5 '
                   'argues against it.')
    assert plain_statement('The model is 0.66 confident that this flow is Exploits.') == 'The model is 66% confident that this flow is Exploits.'


def test_typical_value_facts_come_from_the_class_reference():
    from app.core.report_record import typical_fact
    ref = {'discrete': {'sttl': {'Exploits': {'254': 0.5955}, 'Normal': {'254': 0.1985}}},
           'continuous': {'smean': {'DoS': [68.0, 100.0, 146.0], 'Normal': [60.0, 80.0, 1200.0]}}}
    f = typical_fact('sttl', 254, 'Exploits', 'Normal', ref)
    assert f['text'] == 'Value sttl = 254 occurs in 59.6% of training Exploits flows and 19.9% of training Normal flows'
    g = typical_fact('smean', 100, 'DoS', 'Normal', ref)
    assert g['where'] == 'inside' and '(68 to 146)' in g['text'] and '60 to 1200' in g['text']
    assert typical_fact('dur', 1.0, 'DoS', 'Normal', ref) is None and typical_fact('sttl', 254, 'DoS', 'Normal', {}) is None


def test_reasoning_points_are_sorted_into_slots_by_their_evidence():
    from app.core.report_record import slot_of
    facts = {'F1.E1': 'Predicted class: DoS with calibrated probability 0.58', 'F1.E3': 'Column smean = 100 supports DoS (contribution +1.395)',
             'F1.E8': 'In the analysed dataset, 70 flows ...', 'F1.E10': 'Value smean = 100 is inside ...',
             'F1.E12': 'Compared with Exploits, the second most likely class (15.9%), column smean = 100 favours DoS by +0.900 in model score'}
    assert slot_of(['F1.E1'], facts) == 'certainty' and slot_of(['F1.E1', 'F1.E3'], facts) == 'reasons'
    assert slot_of(['F1.E10'], facts) == 'typical' and slot_of(['F1.E12'], facts) == 'contrast' and slot_of(['F1.E8'], facts) == 'batch'


def test_comparison_and_extra_probabilities_are_checked(engine):
    facts = {'E1': 'Predicted class: DoS with calibrated probability 0.58', 'E2': 'Conformal prediction set (90% coverage): DoS',
             'E9': 'Probability that this flow is an attack of any type: 0.98 (the nine attack classes together)',
             'E12': 'Compared with Exploits, the second most likely class (15.9%), column smean = 100 favours DoS by +0.900 in model score'}
    ok = 'Compared with Exploits, the second most likely class (15.9%), smean = 100 favours DoS by +0.900 in model score.'
    assert engine.check_evidence_claim(ok, ['E12'], facts) == (True, 'supported')
    flipped = ok.replace('favours DoS', 'favours Exploits')
    assert engine.check_evidence_claim(flipped, ['E12'], facts)[1] == 'wrong comparison: the measurement favours DoS, not Exploits'
    sure = 'The model is 0.58 confident that this flow is DoS and 0.98 confident it is an attack of any type.'
    assert engine.check_evidence_claim(sure, ['E1', 'E9'], facts) == (True, 'supported')
    swapped = 'The model is 0.98 confident that this flow is DoS.'
    assert not engine.check_evidence_claim(swapped, ['E1', 'E9'], facts)[0]


def test_reasoning_wording_rules():
    from app.core.report_record import pct_text, slot_of, typical_fact
    ref = {'continuous': {'smean': {'DoS': [100.0, 100.0, 100.0], 'Normal': [59.0, 80.0, 130.0]}}}
    assert typical_fact('smean', 100, 'DoS', 'Normal', ref)['text'] == ('Value smean = 100: at least half of training DoS flows have exactly this '
                                                                       'value; the middle half of training Normal flows is 59 to 130')
    assert pct_text(0.0004) == 'below 0.1%' and pct_text(0.159) == '15.9%'
    facts = {'a': 'Value sbytes = 774 is inside ...', 'b': 'Value sttl = 254 occurs in ...', 'c': 'Compared with Normal, ...'}
    assert slot_of(['a', 'b', 'c'], facts) == 'typical'
    assert slot_of(['c'], facts, 'The value sttl = 254 occurs in 99.6% of training Fuzzers flows.') == 'typical'
    assert slot_of(['a'], facts, 'Compared with Normal, sttl = 254 favours Fuzzers.') == 'contrast'                                # most of the citations decide the slot
