"""Verified, evidence-grounded incident reports (the method of Notebook 18, tightened for the application), one alert at a time.

Pipeline: evidence facts (E1..) from the detector -> retrieve ATT&CK + NIST/CISA passages (S1..) -> the local LLM selects ATT&CK techniques,
recommended actions and short technique-context sentences, each with source citations -> rule-based verifier checks every item ->
one repair round -> only verified items are kept.

Two kinds of LLM text, each verified by rules:
  * prediction evidence: why the detector predicted the class, from the evidence facts only (E#: probability, shortlist, TreeSHAP
    contributions, dataset statistics). Every number, column and class name must appear in the cited facts, and value judgements
    ("large", "unusual", ...) are rejected unless the facts contain them (Notebook 18 showed the LLM adds such interpretations).
  * threat intelligence: ATT&CK techniques, what the attack does and recommended actions, each citing a retrieved source passage (S#).
The plain findings list is still written directly from the evidence (always exact).
If the LLM is unavailable or fails (timeout, error, invalid answer), an evidence-only report is returned (marked as such): the app never
invents text and a report request never fails because of the LLM.

`generate(..., progress=callback)` reports the stage it is in (evidence, writing, verifying, repairing) so the app can show progress.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

SYSTEM = """You are an assistant for security analysts. The facts (E...) describe ONE network alert; the sources (S...) are passages from
MITRE ATT&CK, NIST and CISA. Use ONLY these sources. Do not repeat or interpret the facts; they are already in the report.
1. "attack_techniques": 1 to 3 ATT&CK techniques that fit the alert, whose IDs appear in the sources; cite the source.
   If the prediction set (E2) contains several attack classes, include techniques for each of them.
2. "context": 1 or 2 sentences that explain what such an attack does, taken from the cited ATT&CK source passages (reuse their wording).
3. "recommended_actions": 2 or 3 response actions taken from the cited mitigation, NIST or CISA passages (reuse their wording);
   prefer the ATT&CK mitigation passages, because they are specific to this attack.
4. "prediction_evidence": 4 or 5 SEPARATE items, one short sentence and one point each, that explain why the detector predicted this
   class, using ONLY the facts (E...). Write them in this order and skip a point only when its facts are missing:
   (a) how sure: the class probability, the probability that the flow is an attack of any type, and the 90% prediction set; cite those facts;
   (b) main reasons: the two or three columns with the largest contributions and their direction (supports / argues against), values and
       contributions copied exactly; cite those column facts;
   (c) how typical: what the "Value ..." facts say about how often these values occur in the training flows of each class; cite them;
   (d) why not the second class: what the "Compared with ..." facts say; cite them;
   (e) in this file: how many flows of the analysed dataset were classified as this class; cite that fact.
   Do not call any value high, low, large, small, unusual or suspicious. Write normal sentences; never paste the fact lines.
   Describe the 90% prediction set exactly as fact E2 lists it.
   Example of the style (for a different alert), separate items:
   {"text": "The model is 0.91 confident that this flow is Reconnaissance and 0.97 sure it is an attack of any type; its 90% prediction set
   contains Reconnaissance and Analysis.", "cites": ["E1", "E9", "E2"]}
   {"text": "The strongest evidence is proto = tcp (contribution +2.100) and sttl = 254 (contribution +1.300), while dur = 0.5 argues
   against it (contribution -0.400).", "cites": ["E3", "E4", "E5"]}
   {"text": "The value sttl = 254 occurs in 99.5% of training Reconnaissance flows and 19.9% of training Normal flows.", "cites": ["E10"]}
   {"text": "Compared with Analysis, the second most likely class (4.0%), ct_srv_dst = 2 favours Reconnaissance by +0.800 in model
   score.", "cites": ["E11"]}
   {"text": "In the analysed dataset, 25 flows were classified as Reconnaissance.", "cites": ["E8"]}
Items 1-3 must cite at least one source ID such as "S2"; item 4 cites only facts. Do not invent technique IDs, numbers or procedures.
Answer with JSON only:
{"attack_techniques": [{"id": "<ATT&CK ID>", "cites": ["S1"]}],
 "context": [{"text": "<sentence from the sources>", "cites": ["S1"]}],
 "recommended_actions": [{"text": "<action from the sources>", "cites": ["S4"]}],
 "prediction_evidence": [{"text": "<sentence from the facts>", "cites": ["E3"]}]}"""

REPAIR = """Some items of your previous report failed verification; each failed item comes with the reason.
Rewrite each failed item so that it cites a fact (E...) or source (S...) that really supports it, or remove the item if nothing supports it.
Keep all items that passed. Answer with the complete corrected report in the same JSON format."""

JUDGEMENT = {'large', 'larger', 'small', 'smaller', 'high', 'higher', 'low', 'lower', 'unusual', 'unusually', 'abnormal', 'suspicious',
             'anomalous', 'excessive', 'significant', 'significantly', 'typical', 'malicious-looking', 'huge', 'tiny', 'extreme', 'extremely'}
NUMBER = re.compile(r'(?<![A-Za-z0-9_.])[-+]?\d+(?:\.\d+)?')
CAUSAL = re.compile(r'\b(caus(e|es|ed|ing)|prov(e|es|ed|ing)|proof|confirm(s|ed)?|definitely|certainly|compromis\w*|infect\w*|the attacker)\b', re.I)
FLOW_REFERENCE = re.compile(r'\b(this|the|these) (flow|flows|alert|alerts|traffic|dataset|upload|network)\b', re.I)
CLAUSE_BREAK = re.compile(r',\s|;\s|\.\s|\s(?:while|whereas|but)\s')      # clause boundaries (a decimal point is not one)
FACT_LINE = re.compile(r'Column (\w+) = (.+?) (supports|argues against) (\w+) \(contribution ([+-][0-9.]+)\)')
RETRIEVAL_MIN_SCORE = 0.02          # TF-IDF cosine below this is treated as "no relevant passage"
TOC_LEADER = r'\.{6,}'                                                # dot leaders of a table of contents ("Containment ........ 10")


def prob_text(p: float) -> str:
    """A calibrated probability for text: two decimals, but never "1.00" (rounding is not certainty)."""
    return 'above 0.99' if p >= 0.995 else f'{p:.2f}'
SUPPORT_METHOD = ('lexical overlap: at least half of the content words of the claim appear in the cited passage; this is not a check of meaning '
                  '(semantic entailment)')
STOP = set('the a an and or of to in on for with this that is are was were be by as at from it its into via which these those their has have not'.split())
ENGLISH_WORDS = {'service', 'state', 'rate', 'proto'}


def _value(v) -> str:
    """Feature values as people write them: 160000000.0 -> 160000000, 0.21940463 -> 0.2194, 0.000003 -> 0.000003."""
    if isinstance(v, float):
        if v.is_integer():
            return str(int(v))
        return f'{v:.{4 if abs(v) >= 0.001 else 6}f}'.rstrip('0').rstrip('.')
    return str(v)


def _clean(v):
    """NaN (missing retrieval score) -> None, so records stay valid JSON."""
    return None if isinstance(v, float) and v != v else v


class LLMError(RuntimeError):
    """The language model could not be reached or did not answer in time."""


@dataclass
class ReportResult:
    alert_class: str
    summary: str
    facts: dict
    sources: list = field(default_factory=list)
    findings: list = field(default_factory=list)     # about the alert itself, written from the evidence
    claims: list = field(default_factory=list)       # technique context from ATT&CK passages (LLM, verified)
    techniques: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    possible_attacks: list = field(default_factory=list)   # every attack class on the shortlist with its mapped ATT&CK techniques (fixed mapping)
    evidence_explanation: list = field(default_factory=list)   # why the model predicted the class (LLM, verified against the E facts)
    checked_items: int = 0
    removed: list = field(default_factory=list)          # every generated item the verifier rejected, with the reason and round
    removed_items: int = 0
    # generated / evidence_only_llm_unavailable / llm_timeout / llm_error / invalid_response / all_rejected / not_applicable / not_run
    status: str = 'not_run'
    retrieval: dict = field(default_factory=dict)        # passages found per method, rejected as irrelevant, duplicates removed
    timings: dict = field(default_factory=dict)          # seconds per stage (retrieval, generation, checks)
    sources_detail: list = field(default_factory=list)   # full passages with retrieval method / score / query
    llm_used: bool = False
    llm_model: str | None = None
    seconds: float = 0.0
    note: str = ''

    def to_dict(self):
        return self.__dict__.copy()


class ReportEngine:
    def __init__(self, bundle_dir: str | Path, ollama_url: str, model: str, predictors: list[str], timeout: float = 240.0):
        bundle = Path(bundle_dir)
        self.chunks = pd.read_csv(bundle / 'knowledge_chunks.csv')
        self.chunks['text'] = self.chunks.text.astype(str).str.replace('\ufffd', '-', regex=False).str.replace('\uf0b7', '-', regex=False)
        mitigation_file = bundle / 'attack_mitigations.csv'
        self.mitigations = pd.read_csv(mitigation_file) if mitigation_file.exists() else pd.DataFrame(columns=['technique', 'mitigation', 'name', 'text'])
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.5, stop_words='english', sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(self.chunks.text)
        self.techniques = pd.read_csv(bundle / 'attack_techniques.csv').set_index('technique')
        mapping = pd.read_csv(bundle / 'attack_mapping.csv').fillna('')
        self.class_techniques = {r['class']: [t.strip() for t in str(r.techniques).split(',') if t.strip()] for _, r in mapping.iterrows()}
        self.mapping_confidence = {r['class']: str(r.confidence) for _, r in mapping.iterrows()}
        self.ollama_url, self.model, self.timeout = ollama_url.rstrip('/'), model, timeout
        self.http = httpx.Client(timeout=timeout)                 # one reusable client: creating one per call costs about a second
        self._available: tuple[float, bool] | None = None          # (checked at, result), cached for 30 s so /api/health stays fast
        self.predictors = predictors

    # -------------------------------------------------------------------------------------------- retrieval + evidence
    def _search(self, query, k, sources):
        scores = (self.matrix @ self.vectorizer.transform([query]).T).toarray().ravel()
        frame = self.chunks.assign(score=scores)
        frame = frame[frame.source.isin(sources) & ~frame.text.str.contains(TOC_LEADER)]      # table-of-contents lines are not guidance
        return frame.sort_values('score', ascending=False).head(k)

    def retrieve(self, alert_class: str, prediction_set: list[str], k_guidance: int = 2) -> tuple[dict, dict]:
        """Passages for one prediction: ATT&CK techniques mapped to every attack class on the shortlist (by ID), their ATT&CK mitigations,
        and the closest NIST / CISA guidance (TF-IDF). Duplicates are removed and guidance below RETRIEVAL_MIN_SCORE is rejected."""
        classes = [alert_class] + [c for c in prediction_set if c not in (alert_class, 'Normal')]
        techs = list(dict.fromkeys(t for c in classes for t in self.class_techniques.get(c, []) if t in self.techniques.index))[:4]
        rows = []
        attack = self.chunks[(self.chunks.source == 'MITRE ATT&CK') & self.chunks.location.isin(techs)]
        for tid in techs:
            for _, r in attack[attack.location == tid].iterrows():
                rows.append({'chunk_id': str(r.chunk_id), 'source': r.source, 'location': r.location, 'text': r.text,
                             'method': 'ATT&CK technique mapped to the class (fixed mapping, by ID)', 'score': None, 'query': tid})
        mitig = self.mitigations[self.mitigations.technique.isin(techs)].drop_duplicates('mitigation').head(4)
        for _, m in mitig.iterrows():
            rows.append({'chunk_id': f'MIT:{m.mitigation}:{m.technique}', 'source': 'MITRE ATT&CK mitigation',
                         'location': f'{m.mitigation} {m["name"]} (for {m.technique})', 'text': f'{m["name"]}: {m.text}',
                         'method': 'ATT&CK mitigation linked to a mapped technique (STIX "mitigates")', 'score': None, 'query': m.technique})
        names = ' '.join(self.techniques.loc[t, 'name'] for t in techs)
        query = f'{names} {" ".join(classes)} incident response containment'
        found = self._search(query, k_guidance + 3, ['NIST SP 800-61r3', 'NIST SP 800-94', 'CISA IR Playbooks'])
        rejected = int((found.score < RETRIEVAL_MIN_SCORE).sum())
        for _, r in found[found.score >= RETRIEVAL_MIN_SCORE].head(k_guidance).iterrows():
            rows.append({'chunk_id': str(r.chunk_id), 'source': r.source, 'location': r.location, 'text': r.text,
                         'method': 'TF-IDF similarity to the class and technique names', 'score': round(float(r.score), 4), 'query': query})
        seen, unique, duplicates = set(), [], 0
        for r in rows:
            key = (r['chunk_id'], hashlib.sha256(' '.join(r['text'].split()).encode()).hexdigest())
            if key[0] in seen or key[1] in seen:
                duplicates += 1
                continue
            seen.update(key)
            r['content_sha256'] = key[1]
            unique.append(r)
        sources = {f'S{n}': pd.Series(r) for n, r in enumerate(unique, start=1)}
        info = {'techniques': techs, 'passages': len(unique), 'duplicates_removed': duplicates, 'guidance_rejected_low_score': rejected,
                'min_score': RETRIEVAL_MIN_SCORE, 'status': 'ok' if unique else 'empty'}
        return sources, info

    def evidence(self, alert_class: str, probability: float, prediction_set: list[str], explanation: list[dict], extra_facts: list[str] | None = None):
        facts = {'E1': f'Predicted class: {alert_class} with calibrated probability {prob_text(probability)}',
                 'E2': 'Conformal prediction set (90% coverage): ' + ', '.join(prediction_set)}
        for n, item in enumerate(explanation[:5], start=3):
            direction = 'supports' if item['contribution'] > 0 else 'argues against'
            facts[f'E{n}'] = f"Column {item['feature']} = {_value(item['value'])} {direction} {alert_class} (contribution {item['contribution']:+.3f})"
        for text in extra_facts or []:                                  # e.g. statistics of this attack type in the analysed dataset
            facts[f'E{len(facts) + 1}'] = text
        sources, _ = self.retrieve(alert_class, prediction_set)
        return facts, sources

    # -------------------------------------------------------------------------------------------- verification
    @staticmethod
    def _words(text):
        return {w for w in re.findall(r'[a-z0-9_.]+', text.lower()) if len(w) > 3 and w not in STOP}

    def check_claim(self, text, cites, facts, sources):
        if not cites or any(c not in facts and c not in sources for c in cites):
            return False, 'invalid or missing citation'
        cited_facts = ' '.join(facts[c] for c in cites if c in facts)
        for column in set(re.findall(r'\b([a-z_]{3,})\b', text)):
            if column not in self.predictors:
                continue
            if column in ENGLISH_WORDS and not re.search(rf'\b{column}\s*(=|:)|column {column}', text):
                continue
            if column not in cited_facts and not any(column in sources[c].text for c in cites if c in sources):
                return False, f'column {column} not in cited evidence'
        for number in re.findall(r'\b0?\.\d{2}\b', text):
            if cited_facts and number.lstrip('0') not in cited_facts.replace('0.', '.'):
                return False, f'number {number} not in cited facts'
        source_text = ' '.join(sources[c].text for c in cites if c in sources)
        if source_text and not cited_facts:
            words = self._words(text)
            if words and len(words & self._words(source_text)) / len(words) < 0.5:
                return False, 'claim words not found in cited sources'
        return True, 'supported'

    def support_span(self, text, cites, sources) -> dict | None:
        """The sentence of a cited passage that shares the most content words with the claim (shown highlighted in the report)."""
        words, best = self._words(text), None
        for c in cites:
            if c not in sources:
                continue
            passage = str(sources[c].text)
            for m in re.finditer(r'[^.!?]+[.!?]?', passage):
                sentence = m.group(0).strip()
                if len(sentence) < 15:
                    continue
                start = m.start() + len(m.group(0)) - len(m.group(0).lstrip())        # offsets of the stripped sentence: passage[start:end] == span
                overlap = len(words & self._words(sentence)) / max(len(words), 1)
                if best is None or overlap > best['overlap']:
                    best = {'source': c, 'start': start, 'end': start + len(sentence), 'span': sentence, 'overlap': round(overlap, 3)}
        if best:
            best['method'] = SUPPORT_METHOD
        return best

    @staticmethod
    def _numbers(text: str) -> set[float]:
        """Absolute values of the numbers in a text ("-0.372" and "0.372" are the same contribution; unicode minus allowed)."""
        return {round(abs(float(n)), 6) for n in NUMBER.findall(text.replace('−', '-'))}

    def check_evidence_claim(self, text, cites, facts):
        """A sentence about WHY the model predicted the class: it may only use the cited evidence facts (E#)."""
        if not cites or any(c not in facts for c in cites):
            return False, 'must cite evidence facts (E) only'
        cited = ' '.join(facts[c] for c in cites)
        if 'Predicted class:' in text or 'Conformal prediction set (' in text or len(re.findall(r'\bColumn \w+ = ', text)) > 1:
            return False, 'pastes the fact lines instead of explaining them in sentences'
        allowed = self._numbers(cited)
        missing = set()
        for m in NUMBER.finditer(text.replace('\u2212', '-')):                 # "83%" may restate the fact 0.83
            value = round(abs(float(m.group(0))), 6)
            is_percent = text[m.end():m.end() + 1] == '%'
            if value not in allowed and not (is_percent and round(value / 100, 6) in allowed):
                missing.add(value)
        if missing:
            return False, f'number {sorted(missing)[0]:g} not in cited facts'
        for column in set(re.findall(r'\b([a-z_]{3,})\b', text)):
            if column in self.predictors and column not in ENGLISH_WORDS and column not in cited:
                return False, f'column {column} not in cited facts'
        for name in re.findall(r'\b(Analysis|Backdoor|DoS|Exploits|Fuzzers|Generic|Normal|Reconnaissance|Shellcode|Worms)\b', text):
            if name not in cited:
                return False, f'class {name} not in cited facts'
        problem = (self._prediction_set_problem(text, facts) or self._direction_problem(text, facts) or self._binding_problem(text, facts)
                   or self._probability_problem(text, facts) or self._dataset_problem(text, facts) or self._contrast_problem(text, facts))
        if problem:
            return False, problem
        if CAUSAL.search(text):
            return False, f'unsupported causal or certainty claim ("{CAUSAL.search(text).group(0)}"): attributions are not causes or proof'
        judged = {w for w in re.findall(r'[a-z-]+', text.lower()) if w in JUDGEMENT} - set(re.findall(r'[a-z-]+', cited.lower()))
        if judged:
            return False, f'value judgement "{sorted(judged)[0]}" not supported by the evidence'
        return True, 'supported'

    @staticmethod
    def _prediction_set_problem(text, facts) -> str:
        """Claims about the 90% prediction set must match fact E2 (e.g. "contains only DoS" when the set is DoS, Exploits is wrong)."""
        shortlist = next((v for v in facts.values() if v.startswith('Conformal prediction set')), None)
        if shortlist is None or 'prediction set' not in text:
            return ''
        members = [c.strip() for c in shortlist.split(':', 1)[1].split(',')]
        only = re.search(r'prediction set[^.]*?\bonly\b\s+(?:the\s+)?(?:class\s+)?([A-Za-z]+)', text)
        if only and (len(members) != 1 or only.group(1) != members[0]):
            return f'wrong prediction set: the set is {", ".join(members)}'
        return ''

    @staticmethod
    def _direction_problem(text, facts) -> str:
        """Each column a sentence mentions must have the direction of its fact ("supports" vs "argues against")."""
        direction = {}
        for fact in facts.values():
            m = re.match(r'Column (\w+) = .* (supports|argues against) ', fact)
            if m:
                direction[m.group(1)] = m.group(2)
        mentions = sorted((m.start(), m.end(), c) for c in direction for m in re.finditer(r'\b' + re.escape(c) + r'\b', text))
        cuts = [0] + [m.start() for m in CLAUSE_BREAK.finditer(text)] + [len(text)]
        clauses = list(zip(cuts[:-1], cuts[1:], strict=True))
        for k, (start, end, column) in enumerate(mentions):
            stop = min([s for s, _, _ in mentions[k + 1:]] + [len(text)])
            window = text[end:stop].split('. ')[0].lower()
            # the verb can also follow a list of columns ("while A and B argue against it"): read the whole clause the column is in
            clause = next(text[a:b] for a, b in clauses if a <= start < b).lower()
            if 'against' not in window and 'support' not in window and 'towards' not in window:
                window = clause
            says_against = 'against' in window
            says_supports = 'support' in window or 'towards' in window or 'in favour' in window or 'in favor' in window
            if direction[column] == 'supports' and says_against and not says_supports:
                return f'wrong direction: {column} supports the class'
            if direction[column] == 'argues against' and says_supports and not says_against:
                return f'wrong direction: {column} argues against the class'
        return ''

    @staticmethod
    def _binding_problem(text, facts) -> str:
        """Typed binding: every "feature = value" must match that feature's fact, and a contribution written right after a feature must be
        that feature's own contribution (catches swapped values and swapped contributions)."""
        typed = {}
        for fact in facts.values():
            m = FACT_LINE.match(fact)
            if m:
                typed[m.group(1)] = {'value': m.group(2).strip(), 'contribution': abs(float(m.group(5))), 'signed': float(m.group(5))}
        for feature, info in typed.items():
            for m in re.finditer(r'\b' + re.escape(feature) + r'\s*(=|is|of)\s*([^\s,;()]+)', text):
                if m.group(1) == '=' and m.group(2).rstrip('.') != info['value']:
                    return f'wrong value for {feature}: the evidence says {info["value"]}'
            for m in re.finditer(r'\b' + re.escape(feature) + r'\b[^()]{0,40}\((?:contribution|contributing|score)?\s*([+-−]?[0-9.]+)\)', text):
                written = m.group(1).replace('−', '-')
                value = abs(float(written))
                if round(value, 3) != round(info['contribution'], 3):
                    return f'wrong contribution for {feature}: the evidence says {info["contribution"]:.3f}'
                if written[:1] in '+-' and (written[0] == '-') != (info['signed'] < 0):
                    return f'wrong sign for {feature}: the evidence says {info["signed"]:+.3f}'
        return ''

    @staticmethod
    def _probability_problem(text, facts) -> str:
        """A probability or confidence stated in the text must be the one of fact E1 (as 0.63 or 63%)."""
        e1 = next((v for v in facts.values() if v.startswith('Predicted class:')), None)
        if e1 is None:
            return ''
        m = re.search(r'probability (above )?([0-9.]+)', e1)
        above, prob = bool(m.group(1)), float(m.group(2))
        # other probabilities a sentence may state, each only for its own subject
        status = next((float(x.group(2)) for v in facts.values()
                       if (x := re.match(r'Probability that this flow is an attack of any type: (above )?([0-9.]+)', v))), None)
        runner = next(((x.group(1), float(x.group(2)) / 100) for v in facts.values()
                       if (x := re.match(r'Compared with (\w+), the second most likely class \(([0-9.]+)%\)', v))), None)
        pattern = (r'([0-9]+(?:\.[0-9]+)?)(%?)\s*(?:confident|confidence|probability|likely)'
                   r'|(?:probability|confidence)(?: of)?\s*([0-9]+(?:\.[0-9]+)?)(%?)')
        for m in re.finditer(pattern, text):
            if m.group(3) and re.search(r'average\s+$', text[:m.start()]):
                continue                                                       # "average confidence" is the batch statistic, checked below
            number, percent = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
            value = float(number) / (100 if percent else 1)
            after = text[m.end():m.end() + 60].lower()
            if status is not None and abs(value - status) <= 0.006 and 'attack' in after and ('any type' in after or 'an attack' in after):
                continue                                                       # the attack-or-normal probability, stated as such
            if runner and abs(value - runner[1]) <= 0.0006 and runner[0] in text:
                continue                                                       # the runner-up class's probability, named with it
            if above and value >= 0.995:
                return f'probability written as {number}{percent}: a rounded probability is not certainty (the evidence says above {prob:.2f})'
            if abs(value - prob) > 0.006:
                return f'wrong probability {number}{percent}: the evidence says {prob_text(prob)}'
        return ''

    @staticmethod
    def _contrast_problem(text, facts) -> str:
        """A comparison with the second most likely class must keep its direction: the measurement favours the PREDICTED class."""
        e1 = next((v for v in facts.values() if v.startswith('Predicted class:')), '')
        predicted = e1.split(':', 1)[1].split()[0] if e1 else None
        for m in re.finditer(r'favou?rs?\s+(\w+)', text):
            if predicted and m.group(1) != predicted and m.group(1) not in ('the', 'it'):
                return f'wrong comparison: the measurement favours {predicted}, not {m.group(1)}'
        return ''

    @staticmethod
    def _dataset_problem(text, facts) -> str:
        """Typed binding for the batch statistic: flow count, share of the predicted attacks and average confidence must be the ones of the
        dataset fact (catches e.g. the flow's own probability written as the class's average confidence)."""
        fact = next((v for v in facts.values() if v.startswith('In the analysed dataset')), None)
        said_average = re.search(r'average\s+confidence(?:\s+of)?\s+([0-9]+(?:\.[0-9]+)?)(%?)', text)
        if fact is None:
            return 'average confidence not in the evidence' if said_average else ''
        stats = re.search(r'(\d+) flows \(([0-9.]+)% of the predicted attacks\).*average confidence ([0-9.]+)', fact)
        if stats is None:
            return ''
        count, share, average = int(stats.group(1)), float(stats.group(2)), float(stats.group(3))
        if said_average:
            value = float(said_average.group(1)) / (100 if said_average.group(2) else 1)
            if abs(value - average) > 0.006:
                return f'wrong average confidence {said_average.group(1)}{said_average.group(2)}: the evidence says {average:.2f}'
        for m in re.finditer(r'([0-9]+(?:\.[0-9]+)?)% of the predicted attacks', text):
            if abs(float(m.group(1)) - share) > 0.05:
                return f'wrong share of the predicted attacks {m.group(1)}%: the evidence says {share:.1f}%'
        if 'analysed dataset' in text or 'were classified as' in text:
            for m in re.finditer(r'\b(\d+) flows\b', text):
                if int(m.group(1)) != count:
                    return f'wrong number of flows {m.group(1)}: the evidence says {count}'
        return ''

    def _facts_used(self, text, facts) -> list[str]:
        """The evidence facts a sentence uses: column facts by their column name, the other facts by their numbers."""
        numbers, used = self._numbers(text), []
        for key, fact in facts.items():
            column = re.match(r'Column (\w+) = ', fact)
            if column:
                if re.search(r'\b' + re.escape(column.group(1)) + r'\b', text):
                    used.append(key)
            elif self._numbers(fact) & numbers:
                used.append(key)
        return sorted(used, key=lambda c: int(c[1:]))

    def check_technique(self, tid, cites, sources):
        if tid not in self.techniques.index:
            return False, 'not a real ATT&CK ID'
        if not any(c in sources and tid in sources[c].text for c in cites):
            return False, 'ID not in cited sources'
        return True, 'supported'

    @staticmethod
    def findings(alert_class, probability, prediction_set, explanation):
        """The findings about the alert itself, written directly from the evidence (always exact)."""
        lines = [{'text': f'The detector classifies this flow as {alert_class} with calibrated probability {prob_text(probability)}.', 'cites': ['E1']},
                 {'text': ('The 90% prediction set contains only ' + alert_class + ' (a single class on the shortlist is not a guarantee).')
                  if len(prediction_set) == 1
                  else 'The 90% prediction set contains ' + ', '.join(prediction_set) + ', so the exact attack type is uncertain.', 'cites': ['E2']}]
        for n, item in enumerate(explanation[:3], start=3):
            direction = 'towards' if item['contribution'] > 0 else 'away from'
            lines.append({'text': f"The value {item['feature']} = {item['value']} pushed the prediction {direction} {alert_class} "
                                  f"(contribution {item['contribution']:+.3f}).", 'cites': [f'E{n}']})
        return lines

    def verify(self, report, facts, sources):
        checked = {'claims': [], 'techniques': [], 'actions': [], 'evidence': []}
        for e in report.get('prediction_evidence', []) or []:
            text, cites = str(e.get('text', '')), [c for c in (e.get('cites', []) or []) if isinstance(c, str)]
            ok, why = self.check_evidence_claim(text, cites, facts)
            if not ok and why.split()[0] in ('number', 'column', 'class') and all(c in facts for c in cites):
                # Every value may still be in the evidence, only cited incompletely: check against all facts and complete the citations.
                needed = self._facts_used(text, facts)
                if needed and self.check_evidence_claim(text, needed, facts)[0]:
                    added = [c for c in needed if c not in cites]
                    cites = sorted(set(cites) | set(needed), key=lambda c: int(c[1:]))
                    ok, why = True, f'supported; citations completed by the verifier ({", ".join(added)})'
            checked['evidence'].append({**e, 'text': text, 'cites': cites, 'supported': ok, 'reason': why})
        for c in report.get('context', []) or []:
            cites = list(c.get('cites', []) or [])
            text = str(c.get('text', ''))
            if any(x in facts for x in cites):
                ok, why = False, 'context must cite a source passage'
            elif FLOW_REFERENCE.search(text):
                ok, why = False, 'a source passage cannot support a statement about the analysed flows'
            else:
                ok, why = self.check_claim(text, cites, facts, sources)
            checked['claims'].append({**c, 'supported': ok, 'reason': why, 'support': self.support_span(text, cites, sources) if ok else None})
        for t in report.get('attack_techniques', []) or []:
            tid = str(t.get('id', '')).strip()
            ok, why = self.check_technique(tid, list(t.get('cites', []) or []), sources)
            checked['techniques'].append({'id': tid, 'cites': t.get('cites', []), 'supported': ok, 'reason': why,
                                          'name': self.techniques.loc[tid, 'name'] if tid in self.techniques.index else None})
        for a in report.get('recommended_actions', []) or []:
            cites = list(a.get('cites', []) or [])
            if any(x in facts for x in cites) and not any(x in sources for x in cites):
                ok, why = False, 'an action must cite a source passage'
            else:
                ok, why = self.check_claim(str(a.get('text', '')), cites, facts, sources)
            checked['actions'].append({**a, 'supported': ok, 'reason': why,
                                       'support': self.support_span(str(a.get('text', '')), cites, sources) if ok else None})
        return checked

    # -------------------------------------------------------------------------------------------- LLM
    def available(self) -> bool:
        if self._available and time.monotonic() - self._available[0] < 30:
            return self._available[1]
        try:
            tags = self.http.get(f'{self.ollama_url}/api/tags', timeout=3).json()
            result = any(m['name'].startswith(self.model) for m in tags.get('models', []))
        except Exception:
            result = False
        self._available = (time.monotonic(), result)
        return result

    def _chat(self, system, user):
        try:
            r = self.http.post(f'{self.ollama_url}/api/chat', timeout=self.timeout, json={
                'model': self.model, 'stream': False, 'format': 'json', 'options': {'temperature': 0, 'seed': 42, 'num_ctx': 8192},
                'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]})
            r.raise_for_status()
        except httpx.TimeoutException as error:
            raise LLMError(f'no answer within {self.timeout:.0f} s') from error
        except httpx.HTTPError as error:
            self._available = None                                    # check again next time
            raise LLMError(str(error) or type(error).__name__) from error
        try:
            return json.loads(r.json()['message']['content'])
        except (json.JSONDecodeError, KeyError):
            return None

    @staticmethod
    def _prompt(facts, sources):
        return ('FACTS:\n' + '\n'.join(f'[{k}] {v}' for k, v in facts.items()) + '\n\nSOURCES:\n'
                + '\n'.join(f'[{k}] ({s.source}, {s.location}) {s.text[:900]}' for k, s in sources.items()))

    def possible_attacks(self, alert_class: str, prediction_set: list[str]) -> list[dict]:
        """Each attack class on the shortlist (predicted class first) with its ATT&CK techniques from the fixed class mapping."""
        classes = [alert_class] + [c for c in prediction_set if c != alert_class]
        out = []
        for c in classes:
            if c == 'Normal':
                continue
            techniques = [{'id': t, 'name': str(self.techniques.loc[t, 'name']) if t in self.techniques.index else None}
                          for t in self.class_techniques.get(c, [])]
            out.append({'class': c, 'predicted': c == alert_class, 'mapping_confidence': self.mapping_confidence.get(c, ''), 'techniques': techniques})
        return out

    # -------------------------------------------------------------------------------------------- public
    def generate(self, alert_class: str, probability: float, prediction_set: list[str], explanation: list[dict], progress=None,
                 extra_facts: list[str] | None = None) -> ReportResult:
        start = time.time()
        step = progress or (lambda stage: None)
        step('evidence')
        facts = {'E1': f'Predicted class: {alert_class} with calibrated probability {prob_text(probability)}',
                 'E2': 'Conformal prediction set (90% coverage): ' + ', '.join(prediction_set)}
        for n, item in enumerate(explanation[:5], start=3):
            direction = 'supports' if item['contribution'] > 0 else 'argues against'
            facts[f'E{n}'] = f"Column {item['feature']} = {_value(item['value'])} {direction} {alert_class} (contribution {item['contribution']:+.3f})"
        for text in extra_facts or []:
            facts[f'E{len(facts) + 1}'] = text
        t0 = time.time()
        sources, retrieval = self.retrieve(alert_class, prediction_set) if alert_class != 'Normal' else ({}, {'status': 'not_applicable', 'passages': 0})
        retrieval_seconds = time.time() - t0
        summary = (f'The detector flags this flow as {alert_class} (calibrated probability {prob_text(probability)}). '
                   + ('The prediction set contains only this class (not a guarantee).' if len(prediction_set) == 1
                      else f"The attack type is uncertain: the prediction set contains {', '.join(prediction_set)}."))
        result = ReportResult(alert_class=alert_class, summary=summary, facts=facts,
                              sources=[{'id': k, 'source': s.source, 'location': s.location} for k, s in sources.items()],
                              findings=self.findings(alert_class, probability, prediction_set, explanation),
                              possible_attacks=self.possible_attacks(alert_class, prediction_set), retrieval=retrieval,
                              sources_detail=[{'id': k, **{f: _clean(v) for f, v in s.items()}} for k, s in sources.items()])
        result.timings['retrieval'] = round(retrieval_seconds, 3)

        def finish(status, note=''):
            result.status, result.note, result.seconds = status, note, time.time() - start
            return result

        if alert_class == 'Normal':
            return finish('not_applicable', 'No incident explanation is generated for flows predicted as Normal.')
        if not self.available():
            return finish('evidence_only_llm_unavailable', 'Language model unavailable: evidence-only report (no generated text).')
        step('writing')
        t0 = time.time()
        try:
            report = self._chat(SYSTEM, self._prompt(facts, sources))
        except LLMError as error:
            result.timings['generation'] = round(time.time() - t0, 3)
            timeout = 'no answer within' in str(error)
            return finish('llm_timeout' if timeout else 'llm_error',
                          f'The language model did not answer ({error}): evidence-only report. Try again in a minute.')
        result.timings['generation'] = round(time.time() - t0, 3)
        if not isinstance(report, dict):
            return finish('invalid_response', 'The language model returned no valid JSON report: evidence-only report.')
        step('verifying')
        t0 = time.time()
        checked = self.verify(report, facts, sources)
        check_seconds = time.time() - t0
        rejected = [{'part': part, 'text': i.get('text') or i.get('id'), 'cites': i.get('cites', []), 'reason': i['reason'], 'round': 1}
                    for part, items in checked.items() for i in items if not i['supported']]
        checked_count = sum(len(items) for items in checked.values())
        round_used = 1
        if rejected:                                                     # one repair round, then verify again
            step('repairing')
            failed = [{'part': r['part'], 'text': r['text'], 'reason': r['reason']} for r in rejected]
            try:
                t1 = time.time()
                repaired = self._chat(SYSTEM + '\n' + REPAIR, self._prompt(facts, sources) + '\n\nYOUR PREVIOUS REPORT:\n' + json.dumps(report)
                                      + '\n\nFAILED ITEMS:\n' + json.dumps(failed))
                result.timings['generation'] = round(result.timings['generation'] + time.time() - t1, 3)
            except LLMError:
                repaired = None                                          # keep the verified items of the first answer
            if isinstance(repaired, dict):
                step('verifying')
                t1 = time.time()
                checked = self.verify(repaired, facts, sources)
                check_seconds += time.time() - t1
                round_used = 2
                checked_count += sum(len(items) for items in checked.values())
                rejected += [{'part': part, 'text': i.get('text') or i.get('id'), 'cites': i.get('cites', []), 'reason': i['reason'], 'round': 2}
                             for part, items in checked.items() for i in items if not i['supported']]
        result.timings['checks'] = round(check_seconds, 3)
        for items in checked.values():
            for i in items:
                i['round'] = round_used
        result.claims = [c for c in checked['claims'] if c['supported']]
        result.techniques = [t for t in checked['techniques'] if t['supported']]
        result.actions = [a for a in checked['actions'] if a['supported']]
        result.evidence_explanation = [e for e in checked['evidence'] if e['supported']]
        result.checked_items = checked_count
        result.removed = rejected
        result.removed_items = len(rejected)
        result.llm_used, result.llm_model = True, self.model
        kept = len(result.claims) + len(result.techniques) + len(result.actions) + len(result.evidence_explanation)
        return finish('generated' if kept else 'all_rejected',
                      '' if kept else 'Every generated statement failed the automated checks; only the evidence and sources are shown.')
