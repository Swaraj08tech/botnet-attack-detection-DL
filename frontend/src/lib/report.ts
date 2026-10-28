import type { IncidentReport as Report, ReportItem } from './api'

export function reportToMarkdown(alertId: number, r: Report) {
  const lines = [`# Incident report – alert ${alertId} (${r.alert_class})`, '', r.summary, '', '## Evidence', ...Object.entries(r.facts).map(([k, v]) => `- [${k}] ${v}`)]
  const add = (title: string, items: ReportItem[], fmt: (i: ReportItem) => string) => { if (items.length) lines.push('', `## ${title}`, ...items.map(i => `- ${fmt(i)} [${i.cites.join(', ')}]`)) }
  add('Why the model predicted this (LLM, verified against the evidence)', r.evidence_explanation ?? [], i => i.text ?? '')
  add('Findings', r.findings, i => i.text ?? '')
  if (r.possible_attacks?.length) {
    lines.push('', '## Possible attack types (shortlist) and their ATT&CK techniques',
      ...r.possible_attacks.map(p => `- ${p.class}${p.predicted ? ' (predicted)' : ''}: ${p.techniques.map(t => `${t.id} ${t.name ?? ''}`.trim()).join('; ') || 'no ATT&CK match'}`))
  }
  add('ATT&CK techniques', r.techniques, i => `${i.id} ${i.name ?? ''}`)
  add('What this attack does', r.claims, i => i.text ?? '')
  add('Recommended actions', r.actions, i => i.text ?? '')
  lines.push('', '## Sources', ...r.sources.map(s => `- [${s.id}] ${s.source}, ${s.location}`))
  lines.push('', `_Every item was checked against the cited evidence or source. ${r.llm_used ? `Generated with ${r.llm_model}; ${r.removed_items} unsupported item(s) were removed.` : r.note}_`)
  return lines.join('\n')
}
