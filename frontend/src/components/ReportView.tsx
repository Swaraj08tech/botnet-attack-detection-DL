import { CheckCircle2, Download, ExternalLink, Info, Printer, Sparkles, XCircle } from 'lucide-react'
import type { IncidentReport as Report, ReportItem } from '../lib/api'
import { reportToMarkdown } from '../lib/report'
import { Button, ClassBadge } from './ui'

const cite = (ids: string[]) => ids.map(id => <sup key={id} className="ml-0.5 rounded bg-slate-100 px-1 text-[10px] font-medium text-slate-600">{id}</sup>)

/** Marks text written by the LLM (and checked by the verifier) versus text written directly from the evidence. */
function Origin({ llm }: { llm: boolean }) {
  return llm
    ? <span className="ml-2 inline-flex items-center gap-1 rounded bg-violet-50 px-1.5 py-0.5 text-[10px] font-semibold normal-case tracking-normal text-violet-700 ring-1 ring-inset ring-violet-200"><Sparkles className="h-3 w-3" />LLM · verified</span>
    : <span className="ml-2 rounded bg-slate-100 px-1.5 py-0.5 text-[10px] font-semibold normal-case tracking-normal text-slate-600">exact, from the model</span>
}

function Section({ title, items, render, llm }: { title: string; items: ReportItem[]; render: (i: ReportItem) => React.ReactNode; llm: boolean }) {
  if (!items.length) return null
  return (
    <div>
      <h3 className="mb-2 flex items-center text-xs font-semibold uppercase tracking-wide text-slate-500">{title}<Origin llm={llm} /></h3>
      <ul className="space-y-1.5 text-sm">
        {items.map((i, k) => (
          <li key={k} className="flex gap-2">
            <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-emerald-500" aria-hidden />
            <span>{render(i)}{i.reason?.includes('citations completed') && <span className="ml-1 text-[11px] text-slate-400">({i.reason.replace('supported; ', '')})</span>}</span>
          </li>
        ))}
      </ul>
    </div>
  )
}

/** A verified incident explanation: evidence (E#, exact), LLM text checked against the evidence or retrieved sources (S#), and what the
 *  verifier removed. `compact` hides the summary and the export buttons (used inside the dataset threat intelligence). */
export default function ReportView({ alertId, report, compact = false }: { alertId: number; report: Report; compact?: boolean }) {
  function download() {
    const blob = new Blob([reportToMarkdown(alertId, report)], { type: 'text/markdown' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob); a.download = `incident-explanation-flow-${alertId}.md`; a.click()
    URL.revokeObjectURL(a.href)
  }
  const cls = report.alert_class
  return (
    <div className="print-area space-y-5">
      {!compact && (
        <div className="no-print flex flex-wrap gap-2">
          <Button variant="secondary" onClick={download}><Download className="h-4 w-4" />Markdown</Button>
          <Button variant="secondary" onClick={() => window.print()}><Printer className="h-4 w-4" />Print / PDF</Button>
        </div>
      )}
      {!compact && <p className="text-sm text-slate-800">{report.summary}</p>}
      {report.note && <p className="flex items-start gap-2 rounded-lg bg-amber-50 px-3 py-2 text-xs text-amber-800"><Info className="mt-0.5 h-4 w-4 shrink-0" />{report.note}</p>}

      <div>
        <h3 className="mb-2 flex items-center text-xs font-semibold uppercase tracking-wide text-slate-500">Prediction evidence<Origin llm={false} /></h3>
        <div className="overflow-x-auto rounded-lg border border-slate-200">
          <table className="min-w-full text-xs">
            <tbody className="divide-y divide-slate-100">
              {Object.entries(report.facts).map(([k, v]) => <tr key={k}><td className="w-10 bg-slate-50 px-2 py-1.5 font-semibold text-slate-700">{k}</td><td className="px-2 py-1.5 text-slate-700">{v}</td></tr>)}
            </tbody>
          </table>
        </div>
      </div>
      <Section llm title={`Why the model predicted ${cls}`} items={report.evidence_explanation ?? []} render={i => <>{i.text}{cite(i.cites)}</>} />
      {report.llm_used && !report.evidence_explanation?.length && (
        <p className="text-xs text-slate-500">No generated explanation passed verification for this prediction; the evidence above is exact.</p>
      )}
      {!compact && <Section llm={false} title="Findings" items={report.findings} render={i => <>{i.text}{cite(i.cites)}</>} />}
      {!compact && !!report.possible_attacks?.length && (
        <div>
          <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">Possible attack types and their ATT&CK techniques</h3>
          <ul className="space-y-1.5 text-sm">
            {report.possible_attacks.map(p => (
              <li key={p.class} className="flex flex-wrap items-center gap-2">
                <ClassBadge name={p.class} />{p.predicted && <span className="text-xs text-slate-500">predicted</span>}
                <span className="text-slate-400">→</span>
                {p.techniques.length ? p.techniques.map(t => (
                  <a key={t.id} className="inline-flex items-center gap-1 text-brand-600 hover:underline" href={`https://attack.mitre.org/techniques/${t.id.replace('.', '/')}/`} target="_blank" rel="noreferrer">
                    {t.id}{t.name && <span className="text-slate-600"> {t.name}</span>}<ExternalLink className="h-3 w-3" aria-hidden />
                  </a>
                )) : <span className="text-xs text-slate-500">no ATT&CK match</span>}
              </li>
            ))}
          </ul>
          <p className="mt-1 text-xs text-slate-500">From the fixed class-to-technique mapping, so every class on the shortlist is covered even when the type is uncertain.</p>
        </div>
      )}
      <Section llm title="MITRE ATT&CK techniques" items={report.techniques} render={i => (
        <>
          <a className="inline-flex items-center gap-1 font-medium text-brand-600 hover:underline" href={`https://attack.mitre.org/techniques/${(i.id ?? '').replace('.', '/')}/`} target="_blank" rel="noreferrer">
            {i.id}<ExternalLink className="h-3 w-3" aria-hidden />
          </a> {i.name}{cite(i.cites)}
        </>)} />
      <Section llm title="What this attack does" items={report.claims} render={i => <>{i.text}{cite(i.cites)}</>} />
      <Section llm title="Recommended actions" items={report.actions} render={i => <>{i.text}{cite(i.cites)}</>} />
      <div>
        <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">Retrieved sources (knowledge base)</h3>
        <ul className="grid gap-1 text-xs text-slate-600 sm:grid-cols-2">
          {report.sources.map(s => <li key={s.id}><span className="font-medium text-slate-800">[{s.id}]</span> {s.source}, {s.location}</li>)}
        </ul>
      </div>
      {!!report.removed?.length && (
        <div>
          <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-rose-600">Removed by the verifier</h3>
          <ul className="space-y-1 text-xs text-rose-700">
            {report.removed.map((r, k) => <li key={k} className="flex gap-2"><XCircle className="mt-0.5 h-3.5 w-3.5 shrink-0" /><span><b>{r.reason}</b> — “{String(r.text).slice(0, 180)}{String(r.text).length > 180 ? '…' : ''}”</span></li>)}
          </ul>
        </div>
      )}
      <p className="border-t border-slate-100 pt-3 text-xs text-slate-500">
        Every generated item was checked automatically against its cited evidence (E) or source (S).
        {report.llm_used
          ? ` Written with ${report.llm_model} in ${report.seconds.toFixed(0)} s; ${report.checked_items ?? '–'} statement(s) checked, ${report.removed_items} unsupported item(s) removed.`
          : ' No language model was used.'}
      </p>
    </div>
  )
}
