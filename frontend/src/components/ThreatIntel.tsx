import { useQuery } from '@tanstack/react-query'
import clsx from 'clsx'
import { BookOpen, ChevronDown, FileText, Sparkles } from 'lucide-react'
import { useState } from 'react'
import { Link } from 'react-router-dom'
import { ApiError, api } from '../lib/api'
import { num, pct, when } from '../lib/format'
import ReportView from './ReportView'
import { Button, Card, ClassBadge, ErrorBox, InfoTip, SeverityBadge, Spinner } from './ui'

/** Dataset-level threat intelligence (RAG + LLM, verified) created together with the PDF report: the same content as in the PDF. */
export default function ThreatIntel({ analysisId, onGenerate }: { analysisId: number; onGenerate: () => void }) {
  const intel = useQuery({
    queryKey: ['threat-intel', analysisId],
    queryFn: async () => { try { return await api.threatIntel(analysisId) } catch (e) { if (e instanceof ApiError && e.status === 404) return null; throw e } },
    retry: false,
  })
  const [open, setOpen] = useState<string | null>(null)
  const t = intel.data

  return (
    <Card title={<span className="flex items-center gap-1.5"><Sparkles className="h-4 w-4 text-violet-600" />AI Threat Intelligence (RAG + LLM, verified)
      <InfoTip text="Retrieval-augmented generation: for each main attack type, evidence from a representative flow and the dataset is combined with passages retrieved from MITRE ATT&CK, NIST and CISA. A local LLM writes the explanation; every statement must cite its evidence (E) or source (S) and is checked by rules." /></span>}
      subtitle={t ? `From the security report of ${when(t.created_at)} · ${t.llm_used ? `written by ${t.llm_model}` : 'evidence only (LLM unavailable)'}` : 'Created when the security report is generated'}>
      {intel.isLoading ? <Spinner /> : intel.error ? <ErrorBox error={intel.error} /> : !t ? (
        <div className="flex flex-wrap items-center justify-between gap-4 rounded-lg bg-violet-50/60 px-4 py-4 text-sm text-slate-700">
          <p className="max-w-2xl">Generate the security report to create the threat intelligence: for the four most frequent attack types, the local LLM explains why the
            model predicted them and which MITRE ATT&CK techniques and actions apply, using retrieved sources, and every statement is verified.</p>
          <Button onClick={onGenerate}><FileText className="h-4 w-4" />Generate Security Report</Button>
        </div>
      ) : (
        <div className="space-y-4">
          <div className="grid gap-3 text-sm sm:grid-cols-4">
            {[['Attack types covered', `${t.attack_types_covered} of ${t.attack_types_total}`], ['Generated statements checked', num(t.checked_items)],
              ['Removed as unsupported', num(t.removed_items)], ['Generation time', `${t.seconds.toFixed(0)} s`]].map(([k, v]) => (
              <div key={k} className="rounded-lg border border-slate-200 px-3 py-2"><p className="text-xs text-slate-500">{k}</p><p className="font-semibold tabular-nums text-slate-900">{v}</p></div>
            ))}
          </div>
          {t.note && <p className="rounded-lg bg-amber-50 px-3 py-2 text-xs text-amber-800">{t.note}</p>}
          <p className="flex items-start gap-2 text-xs text-slate-500"><BookOpen className="mt-0.5 h-3.5 w-3.5 shrink-0" />{t.method}</p>
          <div className="divide-y divide-slate-100 rounded-lg border border-slate-200">
            {t.sections.map(s => {
              const isOpen = open === s.attack_type
              return (
                <div key={s.attack_type}>
                  <button onClick={() => setOpen(isOpen ? null : s.attack_type)} aria-expanded={isOpen}
                    className="flex w-full flex-wrap items-center gap-3 px-4 py-3 text-left hover:bg-slate-50">
                    <ClassBadge name={s.attack_type} severity={s.severity} /><SeverityBadge severity={s.severity} />
                    <span className="text-sm text-slate-600">{num(s.count)} flows · {pct(s.percentage_of_malicious)} of malicious</span>
                    <span className="text-xs text-slate-500">{s.llm_used ? `${s.checked_items} checked · ${s.removed_items} removed` : 'evidence only'}</span>
                    <ChevronDown className={clsx('ml-auto h-4 w-4 text-slate-400 transition', isOpen && 'rotate-180')} />
                  </button>
                  {isOpen && (
                    <div className="space-y-3 border-t border-slate-100 bg-slate-50/40 px-4 py-4">
                      <p className="text-xs text-slate-500">Explained with representative flow{' '}
                        <Link className="font-medium text-brand-600 hover:underline" to={`/results/${analysisId}/flows/${s.representative_flow.id}`}>#{s.representative_flow.flow_index}</Link>
                        {' '}— the {s.attack_type} flow the model is most confident about ({pct(s.representative_flow.confidence)}).</p>
                      <ReportView alertId={s.representative_flow.flow_index} report={s} compact />
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        </div>
      )}
    </Card>
  )
}
