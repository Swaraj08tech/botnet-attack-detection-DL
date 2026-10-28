import { createContext } from 'react'
import type { ActionState, ActionStatus, FlowCard, Origin, ReportRecord, SourcePassage, Statement, Support } from './api'

/** Opens the citation drill-down; provided by the report page. */
export const CitationContext = createContext<(id: string, support?: Support | null) => void>(() => {})

export const ACTION_LABEL: Record<ActionStatus, string> = {
  open: 'Open', reviewed: 'Reviewed', planned: 'Planned', completed_by_analyst: 'Completed by analyst', not_applicable: 'Not applicable',
}

/** Short labels for where a statement comes from (the same words as in the PDF). */
export const ORIGIN_LABEL: Record<Origin, string> = {
  observed: 'observed in file',
  batch_statistic: 'batch statistic',
  model_output: 'model output',
  xai_attribution: 'XAI (TreeSHAP)',
  retrieved_guidance: 'retrieved source',
  llm_synthesis: 'LLM · passed checks',
  analyst_proposed: 'proposed next step',
  heuristic: 'heuristic',
}

export const STATUS_LABEL: Record<string, string> = {
  generated: 'generated',
  evidence_only_llm_unavailable: 'evidence only: local LLM unavailable',
  llm_timeout: 'evidence only: LLM timed out',
  llm_error: 'evidence only: LLM error',
  invalid_response: 'evidence only: LLM answer was not valid JSON',
  all_rejected: 'every generated statement failed the checks',
  not_applicable: 'not applicable (predicted Normal)',
}

export type Citation =
  | { kind: 'evidence'; id: string; text: string; flow: FlowCard }
  | { kind: 'source'; id: string; passage: SourcePassage }
  | { kind: 'missing'; id: string }

/** Resolve a citation id: F<flow>.E<n> is evidence of one flow, S<n> a passage of the source register. */
export function resolveCitation(record: ReportRecord, id: string): Citation {
  if (/^S\d+$/.test(id)) {
    const passage = record.sources.register.find(s => s.id === id)
    return passage ? { kind: 'source', id, passage } : { kind: 'missing', id }
  }
  for (const flow of record.flows) {
    if (id in flow.facts) return { kind: 'evidence', id, text: flow.facts[id], flow }
  }
  return { kind: 'missing', id }
}

/** Every statement a flow card contains that was written by the LLM and passed the checks. */
export function llmStatements(flow: FlowCard): Statement[] {
  if (!flow.llm) return []
  return [...flow.llm.explanation, ...flow.llm.techniques, ...flow.llm.context, ...flow.llm.actions]
}

/** Split a passage into [before, span, after] so the supporting sentence can be highlighted. Returns null when the span does not match. */
export function splitSpan(text: string, start: number, end: number, span: string): [string, string, string] | null {
  if (start < 0 || end > text.length || start >= end || text.slice(start, end) !== span) return null
  return [text.slice(0, start), text.slice(start, end), text.slice(end)]
}

const pct = (x: number) => `${(x * 100).toFixed(1)}%`
const line = (s: string) => s.replace(/\s+/g, ' ').trim()

/** Markdown export of the same record the web view and the PDF use. Untrusted passage text is quoted, never interpreted. */
export function recordToMarkdown(r: ReportRecord, states: Record<string, ActionState> = {}): string {
  const s = r.summary
  const out: string[] = [
    `# Security report — ${r.provenance.analysis.dataset_name}`,
    '',
    `Report #${r.report_id} · analysis #${r.provenance.analysis.id} · generated ${r.generated_at} · schema ${r.schema_version} · detector ${r.provenance.detector.version}`,
    '',
    `> ${line(r.question)}`,
    '',
    '## A. Summary',
    '',
    line(s.scope),
    '',
    '**Main findings** (same as the Summary view)',
    '',
    ...(r.views?.summary.overview.bullets ?? []).map(b => `- ${line(b.text)} _(${ORIGIN_LABEL[b.origin]}${b.scope ? `; scope: ${b.scope}` : ''})_`),
    '',
    `- Predicted Normal: ${s.predicted_normal} (${pct(s.predicted_normal_share)}) — model output`,
    `- Predicted attack: ${s.predicted_attack} (${pct(s.predicted_attack_share)}) — model output`,
    `- Needs review (confidence below 70%): ${s.needs_review}; attack status uncertain: ${s.attack_status_uncertain}; attack type uncertain: ${s.attack_type_uncertain}`,
    `- Batch triage priority: **${s.triage_priority.level}** (heuristic: ${s.triage_priority.rule}). ${line(s.triage_priority.meaning)}`,
    `- Most important uncertainty: ${line(s.most_important_uncertainty)}`,
    `- Next check: ${line(s.next_check)}`,
    '',
    '## B. Detections and uncertainty',
    '',
    '| Class | Flows | Share | Avg. confidence | Severity (heuristic) | Status uncertain | Type uncertain |',
    '|---|---:|---:|---:|---|---:|---:|',
    ...r.detections.by_class.map(c => `| ${c.class} | ${c.count} | ${pct(c.share)} | ${pct(c.average_confidence)} | ${c.severity_heuristic} | ${c.status_uncertain} | ${c.type_uncertain} |`),
    '',
    ...r.detections.notes.map(n => `- ${line(n)}`),
    ...r.detections.shift_evidence.map(n => `- ${line(n)}`),
    '',
    '## C. Why these predictions (illustrative flows)',
  ]
  for (const f of r.flows) {
    const x = f.xai
    out.push('', `### ${f.id} — flow #${f.flow_index}: predicted ${f.prediction.class} (${pct(f.prediction.confidence)})`, '', `_${line(f.selection)}_`, '',
      `90% shortlist: ${f.prediction.prediction_set.join(', ')}. Severity (heuristic): ${f.prediction.severity_heuristic}.`, '',
      `TreeSHAP (exact): base ${x.base_value.toFixed(3)} + contributions = margin ${x.margin.toFixed(3)}; margin / T ${x.temperature.toFixed(3)} + class offset ${x.class_offset} = deployed class score ${x.deployed_class_score.toFixed(3)}.`, '',
      '| Feature | Value | Contribution | Direction |', '|---|---|---:|---|',
      ...x.top.map(c => `| ${c.name ?? c.feature} (\`${c.feature}\`) | ${c.value} | ${c.contribution >= 0 ? '+' : ''}${c.contribution.toFixed(3)} | ${c.direction ?? ''} |`),
      `| ${x.other_features} other features | | ${x.remainder_of_other_features >= 0 ? '+' : ''}${x.remainder_of_other_features.toFixed(3)} | |`, '',
      `LIME: ${f.lime.status} — ${line(f.lime.reason)}`, '', '**Evidence**', '', ...Object.entries(f.facts).map(([k, v]) => `- [${k}] ${line(v)}`))
    if (f.llm) {
      const st = llmStatements(f)
      out.push('', `**LLM (${f.llm.model ?? '-'}, status: ${STATUS_LABEL[f.llm.status] ?? f.llm.status}; ${f.llm.checked} statements checked, ${f.llm.removed.length} rejected)**`, '')
      out.push(...(st.length ? st.map(i => `- ${line(i.text)} [${i.cites.join(', ')}] _(${ORIGIN_LABEL[i.origin]})_`) : ['- No generated statement passed the checks; the evidence above is the explanation.']))
      if (f.llm.removed.length) out.push('', 'Rejected by the checks:', ...f.llm.removed.map(x => `- (${x.part}, round ${x.round ?? 1}) ${x.reason}: "${line(String(x.text)).slice(0, 200)}"`))
    }
  }
  out.push('', '## D. Sources (retrieved passages; untrusted reference text)', '')
  for (const p of r.sources.register) {
    out.push(`- **[${p.id}]** ${p.source}, ${p.location}${p.url ? ` <${p.url}>` : ''} — retrieved ${p.retrieved ?? '-'}; ${p.retrieval.method ?? '-'}${p.retrieval.score !== null ? `, score ${p.retrieval.score}` : ''}; sha256 ${p.content_sha256.slice(0, 16)}…`,
      `  > ${line(p.excerpt || p.text).slice(0, 400)}`)
  }
  if (!r.sources.register.length) out.push('- No passages were retrieved for this report.')
  out.push('', '## E. Actions (nothing is executed by the application)', '',
    '| Id | Phase | Scope | Action | Reason | Cites | Applicability | Owner | Disruption | Validation | Analyst status |',
    '|---|---|---|---|---|---|---|---|---|---|---|',
    ...r.actions.rows.map(a => `| ${a.id} | ${a.phase} | ${a.scope} | ${line(a.text)} | ${line(a.reason)} | ${a.cites.join(', ')} | ${line(a.applicability)} | ${a.owner} | ${a.disruption} | ${line(a.validation)} | ${ACTION_LABEL[states[a.id]?.status ?? 'open']} |`),
    '', line(r.actions.note), '', '## F. Limits', '', ...r.limits.unknown.map(u => `- ${line(u)}`), `- Human review: ${r.limits.human_review}`)
  if (r.evaluation) {
    const e = r.evaluation
    out.push('', '## Appendix: labels in the uploaded file (kept separate)', '', line(e.scope), '',
      `- Attack vs Normal correct: ${e.attack_vs_normal_correct} of ${e.labelled_flows}; exact class correct: ${e.exact_class_correct}; true class in the 90% shortlist: ${e.truth_in_set}`)
  }
  const m = r.appendix.model_card.evaluation
  if (m) out.push('', '## Appendix: model benchmark (official UNSW-NB15 test file; not measured on this upload)', '',
    `Attack detection F1 ${pct(m.binary.f1)}, recall ${pct(m.binary.recall)}, false alarms ${pct(m.binary.false_positive_rate)}; 10-class accuracy ${pct(m.accuracy)}, macro F1 ${m.macro.f1.toFixed(3)}.`)
  out.push('', '## Provenance', '',
    `- Input sha256: ${r.provenance.analysis.input_sha256 ?? 'not recorded'}`,
    `- Detector: ${r.provenance.detector.name}; bundle commit ${r.provenance.detector.bundle_source_commit ?? '-'}`,
    `- Application commit: ${r.provenance.application.code_commit ?? '-'}; template ${r.template_version}`,
    `- LLM: ${r.provenance.llm.model} (${r.provenance.llm.runtime}); ${r.provenance.llm.note}`)
  return out.join('\n') + '\n'
}

/** Only https links to known reference hosts are rendered as links; anything else is shown as text. */
const LINK_HOSTS = ['attack.mitre.org', 'nvlpubs.nist.gov', 'csrc.nist.gov', 'www.cisa.gov', 'raw.githubusercontent.com']
export function safeUrl(url: string | null | undefined): string | null {
  if (!url) return null
  try {
    const u = new URL(url)
    return u.protocol === 'https:' && LINK_HOSTS.includes(u.hostname) ? u.toString() : null
  } catch { return null }
}
