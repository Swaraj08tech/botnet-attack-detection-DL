import { useQuery } from '@tanstack/react-query'
import clsx from 'clsx'
import { ArrowLeft, CheckCircle2, Clipboard, Download, Eye, FileJson, FileText, HelpCircle, MinusCircle, RotateCcw, XCircle } from 'lucide-react'
import { useCallback, useRef, useState, type KeyboardEvent } from 'react'
import { Link, useParams, useSearchParams } from 'react-router-dom'
import ReportPanel, { PdfPreview } from '../components/ReportPanel'
import { CitationDialog } from '../components/report/Citations'
import SummaryView from '../components/report/SummaryView'
import { ActionsTab, DetectionsTab, FlowDrawer, LimitsTab, SourcesTab, SummaryTab, WhyTab } from '../components/report/Sections'
import { Button, ErrorBox, PageHeader, Spinner } from '../components/ui'
import { ApiError, api, downloadFile, type ReportRecord, type ReportView, type Support } from '../lib/api'
import { num, pct, when } from '../lib/format'
import { CitationContext, recordToMarkdown } from '../lib/record'
import { useToast } from '../lib/toast'

const TABS = [
  { key: 'assessment', label: 'Assessment' },
  { key: 'detections', label: 'Detections' },
  { key: 'why', label: 'Why this prediction' },
  { key: 'sources', label: 'Sources' },
  { key: 'actions', label: 'Actions' },
  { key: 'limits', label: 'Limits & model' },
] as const
type TabKey = (typeof TABS)[number]['key']

const VIEW_KEY = 'nad.reportView'
function storedView(): ReportView {
  try { return localStorage.getItem(VIEW_KEY) === 'detailed' ? 'detailed' : 'summary' } catch { return 'summary' }
}

/** Summary | Detailed switch (a radio group: arrow keys move between the two views). */
function ViewSwitch({ view, onChange }: { view: ReportView; onChange: (v: ReportView) => void }) {
  const options: [ReportView, string, string][] = [['summary', 'Summary', 'short, plain language'], ['detailed', 'Detailed', 'full evidence and audit']]
  return (
    <div role="radiogroup" aria-label="Report view" className="inline-flex rounded-xl border border-slate-200 bg-white p-1 shadow-sm"
      onKeyDown={e => { if (['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(e.key)) { e.preventDefault(); onChange(view === 'summary' ? 'detailed' : 'summary') } }}>
      {options.map(([key, label, hint]) => (
        <button key={key} type="button" role="radio" aria-checked={view === key} tabIndex={view === key ? 0 : -1} onClick={() => onChange(key)}
          className={clsx('rounded-lg px-4 py-1.5 text-left text-sm focus:outline-none focus-visible:ring-2 focus-visible:ring-brand-500',
            view === key ? 'bg-brand-600 text-white' : 'text-slate-600 hover:bg-slate-50')}>
          <span className="font-semibold">{label}</span><span className={clsx('ml-2 hidden text-xs sm:inline', view === key ? 'text-white/80' : 'text-slate-400')}>{hint}</span>
        </button>
      ))}
    </div>
  )
}

const COMPONENTS: [keyof ReportRecord['components'], string][] = [
  ['detection', 'Detection'], ['xai', 'XAI (TreeSHAP)'], ['retrieval', 'Retrieval (RAG)'], ['llm', 'Local LLM'], ['checks', 'Automated checks'], ['pdf', 'PDF'],
]

function StatusIcon({ status }: { status: string }) {
  if (status === 'available') return <CheckCircle2 className="h-4 w-4 text-emerald-600" aria-hidden />
  if (status === 'not run' || status === 'not applicable') return <MinusCircle className="h-4 w-4 text-slate-400" aria-hidden />
  return <XCircle className="h-4 w-4 text-amber-600" aria-hidden />
}

/** Which parts of the pipeline ran, with the honest reason when one did not. */
function ComponentStatuses({ r, onRetry }: { r: ReportRecord; onRetry: () => void }) {
  const degraded = COMPONENTS.some(([k]) => !['available', 'not run'].includes(r.components[k]?.status ?? 'available'))
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-3">
      <ul className="grid gap-2 text-xs sm:grid-cols-2 lg:grid-cols-3" aria-label="Report components">
        {COMPONENTS.map(([k, label]) => {
          const c = r.components[k]
          return (
            <li key={k} className="flex min-w-0 items-start gap-2">
              <StatusIcon status={c?.status ?? 'unknown'} />
              <span className="min-w-0"><b className="text-slate-800">{label}</b>: <span className={c?.status === 'available' ? 'text-emerald-700' : 'text-amber-700'}>{c?.status ?? 'unknown'}</span>
                {c?.detail && <span className="block truncate text-slate-500" title={c.detail}>{c.detail}</span>}</span>
            </li>
          )
        })}
      </ul>
      {degraded && (
        <div className="mt-2 flex flex-wrap items-center justify-between gap-2 border-t border-slate-100 pt-2 text-xs text-slate-600">
          <span>Some components did not run fully; the report shows evidence-only content for them instead of hiding the gap.</span>
          <Button variant="secondary" className="px-2.5 py-1 text-xs" onClick={onRetry}><RotateCcw className="h-3.5 w-3.5" />Retry (generate again)</Button>
        </div>
      )}
    </div>
  )
}

type Question = 'why' | 'normal' | 'first'

/** Answers built only from the report record (no new model or LLM call). */
function Answer({ q, r, go }: { q: Question; r: ReportRecord; go: (tab: TabKey, extra?: Record<string, string>) => void }) {
  const s = r.summary, sets = r.detections.sets
  if (q === 'normal') {
    return (
      <div className="space-y-1.5 text-sm text-slate-700">
        <p><b>{num(sets.attack_status_uncertain)}</b> of {num(r.detections.total)} flows have Normal on their 90% shortlist next to an attack type, so for them “Normal” cannot be ruled out
          {sets.predicted_normal_with_attack_in_set > 0 && <> ({num(sets.predicted_normal_with_attack_in_set)} of them were predicted Normal but an attack is still possible)</>}.</p>
        <p>For the illustrative flows, the model's probability of Normal is: {r.flows.map(f => `#${f.flow_index} ${pct(f.prediction.probabilities.Normal ?? 0)}`).join(', ')}.</p>
        <p>{r.detections.shift_evidence[0]}</p>
        <button type="button" className="text-xs font-medium text-brand-600 hover:underline" onClick={() => go('detections', { uncertainty: 'status' })}>List the attack-status-uncertain flows</button>
      </div>
    )
  }
  if (q === 'first') {
    const verify = r.actions.rows.find(a => a.phase === 'Verify')
    return (
      <div className="space-y-1.5 text-sm text-slate-700">
        <p>{s.next_check}</p>
        {verify && <p className="text-xs text-slate-600">First action in the checklist: <b>{verify.id}</b> — {verify.text}</p>}
        <p className="text-xs text-slate-500">{s.most_important_uncertainty}</p>
        <button type="button" className="text-xs font-medium text-brand-600 hover:underline" onClick={() => go('actions')}>Open the action checklist</button>
      </div>
    )
  }
  const f = r.flows[0]
  return (
    <div className="space-y-1.5 text-sm text-slate-700">
      {f ? <p>Each prediction is explained with exact TreeSHAP contributions of the model. For flow #{f.flow_index} ({f.prediction.class}, {pct(f.prediction.confidence)}), the strongest evidence is{' '}
        {f.xai.top.slice(0, 2).map(c => `${c.feature} = ${c.value} (${c.contribution >= 0 ? '+' : ''}${c.contribution.toFixed(3)})`).join(' and ')}.</p>
        : <p>No flow was explained in this report.</p>}
      <button type="button" className="text-xs font-medium text-brand-600 hover:underline" onClick={() => go('why')}>See the evidence for every illustrative flow</button>
    </div>
  )
}

export default function ReportPage() {
  const reportId = Number(useParams().reportId)
  const toast = useToast()
  const [params, setParams] = useSearchParams()
  const tab = (TABS.find(t => t.key === params.get('tab'))?.key ?? 'assessment') as TabKey
  const asked = params.get('view')
  // An explicit ?view= wins; a deep link to a detailed tab opens the Detailed view; otherwise the reader's last choice (default Summary).
  const view: ReportView = asked === 'summary' || asked === 'detailed' ? asked : params.get('tab') ? 'detailed' : storedView()
  const [citation, setCitation] = useState<{ id: string; support?: Support | null } | null>(null)
  const [drawer, setDrawer] = useState<number | null>(null)
  const [question, setQuestion] = useState<Question | null>(null)
  const [preview, setPreview] = useState(false)
  const [retry, setRetry] = useState(false)
  const tabRefs = useRef<(HTMLButtonElement | null)[]>([])

  const record = useQuery({
    queryKey: ['report-record', reportId],
    queryFn: async () => { try { return await api.reportRecord(reportId) } catch (e) { if (e instanceof ApiError && e.status === 404 && /schema-1/.test(e.message)) return null; throw e } },
    retry: false,
  })
  const r = record.data
  const analysisId = r?.provenance.analysis.id
  const analysis = useQuery({ queryKey: ['analysis', analysisId], queryFn: () => api.analysis(analysisId!), enabled: !!analysisId })
  const states = useQuery({ queryKey: ['report-actions', reportId], queryFn: () => api.actionStates(reportId), enabled: !!r })

  const openCitation = useCallback((id: string, support?: Support | null) => setCitation({ id, support }), [])
  const go = (t: TabKey, extra: Record<string, string> = {}) => setParams(p => { const n = new URLSearchParams(p); n.set('view', 'detailed'); n.set('tab', t); n.delete('uncertainty'); Object.entries(extra).forEach(([k, v]) => n.set(k, v)); return n }, { replace: true })
  const setView = (v: ReportView) => {
    try { localStorage.setItem(VIEW_KEY, v) } catch { /* storage unavailable */ }
    setParams(p => { const n = new URLSearchParams(p); n.set('view', v); return n }, { replace: true })     // keeps tab, flow and filters
  }
  const onTabKey = (e: KeyboardEvent, i: number) => {
    const next = e.key === 'ArrowRight' ? (i + 1) % TABS.length : e.key === 'ArrowLeft' ? (i - 1 + TABS.length) % TABS.length : e.key === 'Home' ? 0 : e.key === 'End' ? TABS.length - 1 : -1
    if (next < 0) return
    e.preventDefault(); go(TABS[next].key); tabRefs.current[next]?.focus()
  }

  if (record.isLoading) return <Spinner label="Loading the report…" />
  if (record.error) return <ErrorBox error={record.error} />
  if (!r) {
    return (
      <>
        <PageHeader kicker="Security report" title={`Report #${reportId}`} actions={<Link to="/reports"><Button variant="secondary"><ArrowLeft className="h-4 w-4" />All reports</Button></Link>} />
        <div className="rounded-xl border border-slate-200 bg-white p-6 text-sm text-slate-700">
          <p>This report was created with an older version of the application (schema 1). It exists only as a PDF; it has no report record, so the interactive view, citations and checklist are not available.</p>
          <div className="mt-4 flex flex-wrap gap-2"><Button onClick={() => setPreview(true)}><Eye className="h-4 w-4" />Open the PDF</Button></div>
        </div>
        {preview && <PdfPreview reportId={reportId} title={`Security report #${reportId}`} onClose={() => setPreview(false)} />}
      </>
    )
  }

  const markdown = () => recordToMarkdown(r, states.data ?? {})
  const saveMarkdown = () => {
    const a = document.createElement('a')
    a.href = URL.createObjectURL(new Blob([markdown()], { type: 'text/markdown' }))
    a.download = `security_report_analysis_${r.provenance.analysis.id}_report_${r.report_id}.md`; a.click()
    setTimeout(() => URL.revokeObjectURL(a.href), 1000)
  }
  const copy = async () => {
    try { await navigator.clipboard.writeText(markdown()); toast({ tone: 'success', title: 'Report copied as Markdown' }) }
    catch { toast({ tone: 'error', title: 'Copy failed', body: 'The browser blocked clipboard access; use Markdown download instead.' }) }
  }
  const fail = (e: unknown) => toast({ tone: 'error', title: 'Download failed', body: (e as Error).message })

  return (
    <CitationContext.Provider value={openCitation}>
      <PageHeader kicker={`Security report #${r.report_id} · schema ${r.schema_version}`} title={<span className="break-all">{r.provenance.analysis.dataset_name}</span>}
        description={<span className="flex flex-wrap gap-x-3 gap-y-1"><span>Generated {when(r.generated_at)}</span><span>analysis <Link className="text-brand-600 hover:underline" to={`/results/${r.provenance.analysis.id}`}>#{r.provenance.analysis.id}</Link></span>
          <span>{num(r.provenance.analysis.flows)} flows</span><span>detector {r.provenance.detector.version}</span></span>}
        actions={<div className="flex flex-wrap gap-2">
          <Button variant="secondary" onClick={() => setPreview(true)}><Eye className="h-4 w-4" />View {view === 'summary' ? 'Summary' : 'Detailed'} PDF</Button>
          <Button variant="secondary" onClick={() => downloadFile(api.reportPdfPath(r.report_id, true, 'summary')).catch(fail)}><Download className="h-4 w-4" />Download Summary PDF</Button>
          <Button variant="secondary" onClick={() => downloadFile(api.reportPdfPath(r.report_id, true, 'detailed')).catch(fail)}><Download className="h-4 w-4" />Download Detailed PDF</Button>
          <Button variant="secondary" onClick={() => downloadFile(api.reportRecordPath(r.report_id)).catch(fail)}><FileJson className="h-4 w-4" />JSON</Button>
          <Button variant="secondary" onClick={saveMarkdown}><FileText className="h-4 w-4" />Markdown</Button>
          <Button variant="ghost" onClick={copy} aria-label="Copy the report as Markdown"><Clipboard className="h-4 w-4" /></Button>
        </div>} />

      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <ViewSwitch view={view} onChange={setView} />
        <p className="text-xs text-slate-500">Both views show the same analysis, numbers and evidence; the Detailed view adds the full evidence and audit trail.</p>
      </div>

      {view === 'summary' ? (
        <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm sm:p-6">
          <SummaryView r={r} onEvidence={() => go('why')} onDetailed={() => setView('detailed')} />
        </div>
      ) : <>
      <ComponentStatuses r={r} onRetry={() => setRetry(true)} />

      <div className="mt-4 flex flex-wrap items-center gap-2" role="group" aria-label="Questions about this report">
        <HelpCircle className="h-4 w-4 text-slate-400" aria-hidden />
        {([['why', 'Why this prediction?'], ['normal', 'Could it be Normal?'], ['first', 'What should I check first?']] as [Question, string][]).map(([k, label]) => (
          <button key={k} type="button" aria-pressed={question === k} onClick={() => setQuestion(question === k ? null : k)}
            className={clsx('rounded-full px-3 py-1 text-xs font-medium ring-1 ring-inset focus:outline-none focus-visible:ring-2 focus-visible:ring-brand-500',
              question === k ? 'bg-brand-600 text-white ring-brand-600' : 'bg-white text-slate-700 ring-slate-300 hover:bg-slate-50')}>{label}</button>
        ))}
      </div>
      {question && (
        <div className="mt-2 rounded-xl border border-brand-200 bg-brand-50/40 p-4" aria-live="polite">
          <Answer q={question} r={r} go={(t, extra) => { setQuestion(null); go(t, extra) }} />
          <p className="mt-2 text-[11px] text-slate-500">Answered from the report record (model output, TreeSHAP and batch statistics); no new model or LLM call.</p>
        </div>
      )}

      <div className="mt-5 rounded-xl border border-slate-200 bg-white shadow-sm">
        <div role="tablist" aria-label="Report sections" className="flex gap-1 overflow-x-auto border-b border-slate-200 px-2">
          {TABS.map((t, i) => (
            <button key={t.key} ref={el => { tabRefs.current[i] = el }} role="tab" id={`tab-${t.key}`} aria-selected={tab === t.key} aria-controls={`panel-${t.key}`}
              tabIndex={tab === t.key ? 0 : -1} onClick={() => go(t.key)} onKeyDown={e => onTabKey(e, i)}
              className={clsx('whitespace-nowrap border-b-2 px-3 py-3 text-sm font-medium focus:outline-none focus-visible:bg-brand-50',
                tab === t.key ? 'border-brand-600 text-brand-700' : 'border-transparent text-slate-500 hover:text-slate-800')}>
              {t.label}{t.key === 'actions' && r.actions.rows.length > 0 && <span className="ml-1 text-xs text-slate-400">({r.actions.rows.length})</span>}
              {t.key === 'sources' && <span className="ml-1 text-xs text-slate-400">({r.sources.register.length})</span>}
            </button>
          ))}
        </div>
        <div role="tabpanel" id={`panel-${tab}`} aria-labelledby={`tab-${tab}`} className="p-4 sm:p-5">
          {tab === 'assessment' && <SummaryTab r={r} />}
          {tab === 'detections' && <DetectionsTab r={r} analysis={analysis.data?.analysis} onOpenFlow={setDrawer} preset={params.get('uncertainty')} />}
          {tab === 'why' && <WhyTab r={r} flowId={params.get('flow')} onFlow={id => go('why', { flow: id })} />}
          {tab === 'sources' && <SourcesTab r={r} onOpen={id => openCitation(id)} />}
          {tab === 'actions' && <ActionsTab r={r} />}
          {tab === 'limits' && <LimitsTab r={r} />}
        </div>
      </div>
      </>}

      {citation && <CitationDialog record={r} id={citation.id} support={citation.support} onClose={() => setCitation(null)} />}
      {drawer !== null && <FlowDrawer analysisId={r.provenance.analysis.id} flowId={drawer} onClose={() => setDrawer(null)} />}
      {preview && <PdfPreview reportId={r.report_id} view={view} title={`${view === 'summary' ? 'Summary' : 'Detailed'} report · ${r.provenance.analysis.dataset_name}`} onClose={() => setPreview(false)} />}
      {retry && analysis.data && <ReportPanel analysis={analysis.data} onClose={() => setRetry(false)} />}
    </CitationContext.Provider>
  )
}
