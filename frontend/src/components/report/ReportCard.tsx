import { CheckCircle2, FileText, LayoutList, MinusCircle, XCircle } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import type { AnalysisSummary } from '../../lib/api'
import { when } from '../../lib/format'
import ThreatIntel from '../ThreatIntel'
import { Button, Card } from '../ui'

const LABELS: Record<string, string> = { detection: 'Detection', xai: 'XAI', group_xai: 'Group XAI', retrieval: 'RAG', llm: 'LLM', checks: 'Checks', pdf: 'PDF' }

/** Results-page entry to the security report: what ran (detection, XAI, RAG, LLM, checks, PDF) and a link to the interactive report. */
export default function ReportCard({ analysis, onGenerate }: { analysis: AnalysisSummary; onGenerate: () => void }) {
  const navigate = useNavigate()
  const report = analysis.report
  if (report && !report.has_record) return <ThreatIntel analysisId={analysis.id} onGenerate={onGenerate} />    // schema-1 report: its old view still works
  return (
    <Card title={<span className="flex items-center gap-1.5"><FileText className="h-4 w-4 text-brand-600" />Security report: model evidence, sources and LLM explanations</span>}
      subtitle={report ? `Report #${report.id} · generated ${when(report.created_at)} · ${report.pages} PDF pages` : 'Not generated yet'}>
      {!report ? (
        <div className="flex flex-wrap items-center justify-between gap-4 rounded-lg bg-brand-50/50 px-4 py-4 text-sm text-slate-700">
          <p className="max-w-2xl">The report explains illustrative predictions with exact TreeSHAP evidence, retrieves passages from MITRE ATT&CK, NIST and CISA, lets the local LLM
            write explanations that must cite that evidence (every statement is checked), and builds an action checklist.</p>
          <Button onClick={onGenerate}><FileText className="h-4 w-4" />Generate Security Report</Button>
        </div>
      ) : (
        <div className="space-y-3">
          <ul className="flex flex-wrap gap-x-4 gap-y-1 text-xs" aria-label="Report components">
            {Object.entries(report.components ?? {}).map(([k, c]) => (
              <li key={k} className="flex items-center gap-1" title={c.detail}>
                {c.status === 'available' ? <CheckCircle2 className="h-3.5 w-3.5 text-emerald-600" /> : c.status === 'not run' ? <MinusCircle className="h-3.5 w-3.5 text-slate-400" /> : <XCircle className="h-3.5 w-3.5 text-amber-600" />}
                <b className="text-slate-700">{LABELS[k] ?? k}</b><span className="text-slate-500">{c.status}</span>
              </li>
            ))}
          </ul>
          {report.coverage && <p className="text-xs text-slate-500">{report.coverage.flows_explained} illustrative flows explained; LLM explanations for {report.coverage.attack_types_with_llm} of {report.coverage.attack_types_detected} predicted attack types.</p>}
          <div className="flex flex-wrap gap-2">
            <Button onClick={() => navigate(`/reports/${report.id}`)}><LayoutList className="h-4 w-4" />Open interactive report</Button>
            <Button variant="secondary" onClick={() => navigate(`/reports/${report.id}?tab=sources`)}>Sources</Button>
            <Button variant="secondary" onClick={() => navigate(`/reports/${report.id}?tab=actions`)}>Action checklist</Button>
          </div>
        </div>
      )}
    </Card>
  )
}
