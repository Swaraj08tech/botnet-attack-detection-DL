import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { CheckCircle2, Download, Eye, FileText, LayoutList, RotateCcw, X } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { ApiError, api, downloadFile, fetchBlob, type AnalysisSummary, type Job, type ReportView } from '../lib/api'
import { when } from '../lib/format'
import { useToast } from '../lib/toast'
import { Button, ErrorBox, StepList } from './ui'

const REPORT_STEPS = [
  { key: 'detections', label: 'Summarising detections and uncertainty' },
  { key: 'explanations', label: 'Explaining illustrative flows (TreeSHAP)' },
  { key: 'retrieval', label: 'Retrieving sources (MITRE ATT&CK, NIST, CISA)' },
  { key: 'generation', label: 'Local LLM writing explanations from the evidence' },
  { key: 'checks', label: 'Checking every generated statement against its citation' },
  { key: 'pdf', label: 'Rendering the PDF from the same report record' },
]

type ReportJob = Job<{ report_id: number; pages: number; schema_version?: string; components?: Record<string, string>; llm_detail?: string; checks_detail?: string }>

/** Opens the PDF in an overlay (fetched with the token, shown from a local object URL). */
export function PdfPreview({ reportId, title, onClose, view = 'detailed' }: { reportId: number; title: string; onClose: () => void; view?: ReportView }) {
  const [url, setUrl] = useState<string | null>(null)
  const [error, setError] = useState<unknown>(null)
  useEffect(() => {
    let current: string | null = null
    fetchBlob(api.reportPdfPath(reportId, false, view)).then(r => { current = r.url; setUrl(r.url) }).catch(setError)
    return () => { if (current) URL.revokeObjectURL(current) }
  }, [reportId, view])
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])
  return (
    <div className="fixed inset-0 z-50 flex flex-col bg-ink-900/70 p-4 backdrop-blur-sm md:p-8" role="dialog" aria-modal="true" aria-label="Report preview">
      <div className="mx-auto flex h-full w-full max-w-5xl flex-col overflow-hidden rounded-xl bg-white shadow-2xl">
        <div className="flex items-center justify-between gap-3 border-b border-slate-200 px-4 py-3">
          <p className="truncate text-sm font-medium text-slate-800">{title}</p>
          <div className="flex shrink-0 gap-2">
            <Button variant="secondary" onClick={() => downloadFile(api.reportPdfPath(reportId, true, view))}><Download className="h-4 w-4" />Download {view === 'summary' ? 'Summary' : 'Detailed'} PDF</Button>
            <Button variant="ghost" onClick={onClose} aria-label="Close preview"><X className="h-4 w-4" /></Button>
          </div>
        </div>
        <div className="flex-1 bg-slate-100">
          {error ? <div className="p-6"><ErrorBox error={error} /></div>
            : url ? <iframe title="Security report PDF" src={url} className="h-full w-full" />
              : <p className="p-6 text-sm text-slate-500">Loading the report…</p>}
        </div>
      </div>
    </div>
  )
}

/** Generate the PDF security report of one analysis, with the real generation stages, then preview / download it. */
export default function ReportPanel({ analysis, onClose }: { analysis: AnalysisSummary; onClose: () => void }) {
  const id = analysis.id
  const queryClient = useQueryClient()
  const toast = useToast()
  const navigate = useNavigate()
  const [preview, setPreview] = useState<number | null>(null)
  const job = useQuery({
    queryKey: ['report-job', id],
    queryFn: async () => { try { return await api.reportJob(id) } catch (e) { if (e instanceof ApiError && e.status === 404) return null; throw e } },
    refetchInterval: q => (q.state.data && ['queued', 'running'].includes(q.state.data.status) ? 500 : false),
    retry: false,
  })
  const start = useMutation({ mutationFn: () => api.startReport(id), onSuccess: j => queryClient.setQueryData(['report-job', id], j) })
  const j = job.data as ReportJob | null | undefined
  const running = !!j && (j.status === 'queued' || j.status === 'running')
  const watched = useRef<string | null>(null)
  const [finished, setFinished] = useState<ReportJob | null>(null)

  useEffect(() => {
    if (!j) return
    if (j.status === 'queued' || j.status === 'running') { watched.current = j.job_id; return }
    if (watched.current !== j.job_id) return
    watched.current = null
    queryClient.invalidateQueries({ queryKey: ['analysis', id] }); queryClient.invalidateQueries({ queryKey: ['reports'] })
    queryClient.invalidateQueries({ queryKey: ['analyses'] }); queryClient.invalidateQueries({ queryKey: ['threat-intel', id] }); queryClient.invalidateQueries({ queryKey: ['report-record'] })
    if (j.status === 'done') {
      setFinished(j)
      toast({ tone: 'success', title: 'Report generated successfully', body: `${j.result.pages} pages · LLM: ${j.result.llm_detail ?? '–'} · ${j.result.checks_detail ?? ''}` })
    }
    else toast({ tone: 'error', title: 'Report generation failed', body: j.error })
  }, [j, id, queryClient, toast, analysis.dataset_name])

  const existing = finished ? { id: finished.result.report_id, pages: finished.result.pages, created_at: '', has_record: !!finished.result.schema_version } : analysis.report
  const llmDetail = finished ? finished.result.llm_detail : analysis.report?.components?.llm?.detail
  const checksDetail = finished ? finished.result.checks_detail : analysis.report?.components?.checks?.detail
  // The threat-analysis stages repeat for every attack type; the checklist only moves forward and the detail line names the type.
  const reached = useRef<{ job: string; index: number }>({ job: '', index: 0 })
  const stageIndex = j ? Math.max(0, REPORT_STEPS.findIndex(s => s.key === j.stage)) : 0
  if (j && reached.current.job !== j.job_id) reached.current = { job: j.job_id, index: 0 }
  if (j) reached.current.index = Math.max(reached.current.index, stageIndex)
  const current = j ? reached.current.index : 0
  const health = useQuery({ queryKey: ['health'], queryFn: api.health })
  const subStep = j && stageIndex < current ? ` · ${REPORT_STEPS[stageIndex].label.toLowerCase()}` : ''
  const detailText = j?.detail ? j.detail.charAt(0).toUpperCase() + j.detail.slice(1) + subStep : ''
  const steps = REPORT_STEPS.map((s, i) => (i === current && detailText ? { ...s, detail: <span>{detailText}</span> } : s))
  const failed = j?.status === 'failed' && !running

  return (
    <div className="fixed inset-0 z-40 flex items-start justify-center overflow-y-auto bg-ink-900/50 p-4 pt-[10vh] backdrop-blur-[2px]" role="dialog" aria-modal="true" aria-label="Security report">
      <div className="w-full max-w-lg rounded-xl bg-white shadow-2xl">
        <div className="flex items-start justify-between gap-3 border-b border-slate-100 px-5 py-4">
          <div className="min-w-0">
            <h2 className="flex items-center gap-2 font-semibold text-slate-900"><FileText className="h-4 w-4 text-brand-600" />Security Report</h2>
            <p className="mt-0.5 truncate text-xs text-slate-500">{analysis.dataset_name} · analysis #{id}</p>
          </div>
          <Button variant="ghost" className="px-2" onClick={onClose} aria-label="Close"><X className="h-4 w-4" /></Button>
        </div>
        <div className="space-y-4 px-5 py-5">
          {running && j && (
            <>
              <p className="text-sm font-medium text-slate-900">Generating Security Report…</p>
              {j.status === 'queued' && <p className="text-xs text-slate-500">Waiting for {j.queue_position} other report(s) to finish…</p>}
              <StepList steps={steps} current={current} />
              <p className="text-xs text-slate-500">{j.elapsed.toFixed(0)} s · usually 1–3 minutes with the local LLM (most of it is writing and checking the explanations). You can close this window; the report keeps being generated.</p>
            </>
          )}
          {failed && j && (
            <>
              <StepList steps={steps} current={current} failed />
              <ErrorBox error={`The report could not be generated: ${j.error}`} />
            </>
          )}
          {!running && !failed && finished && <StepList steps={REPORT_STEPS} current={REPORT_STEPS.length} done />}
          {!running && !failed && existing && (
            <div className="rounded-xl border border-emerald-200 bg-emerald-50 p-4">
              <p className="flex items-center gap-2 font-medium text-emerald-800"><CheckCircle2 className="h-5 w-5" />{finished ? 'Report Generated Successfully' : 'Report available'}</p>
              <p className="mt-1 text-sm text-emerald-800/80">
                {existing.pages} pages{existing.created_at ? ` · generated ${when(existing.created_at)}` : ''}. Summary, detections and uncertainty, why each illustrative
                prediction was made (TreeSHAP + checked LLM text), retrieved sources, an action checklist, limits and the model appendix.
                {existing.has_record ? <> LLM: {llmDetail ?? '–'}. Checks: {checksDetail ?? '–'}.</> : ' This is an older (schema 1) PDF-only report; generate again for the interactive view.'}
              </p>
              <div className="mt-3 flex flex-wrap gap-2">
                {existing.has_record && <Button onClick={() => navigate(`/reports/${existing.id}`)}><LayoutList className="h-4 w-4" />Open interactive report</Button>}
                <Button variant={existing.has_record ? 'secondary' : 'primary'} onClick={() => setPreview(existing.id)}><Eye className="h-4 w-4" />Preview PDF</Button>
                <Button variant="secondary" onClick={() => downloadFile(api.reportPdfPath(existing.id, true)).catch(e => toast({ tone: 'error', title: 'Download failed', body: (e as Error).message }))}>
                  <Download className="h-4 w-4" />Download PDF
                </Button>
              </div>
            </div>
          )}
          {!running && !existing && !failed && (
            <div className="space-y-2 text-sm text-slate-600">
              <p>Create the security report of this analysis: one versioned report record, shown in the app and rendered as PDF, with the model evidence
                (<b>TreeSHAP</b>), passages retrieved from MITRE ATT&CK, NIST and CISA (<b>RAG</b>), explanations written by the <b>local LLM</b> and checked against their
                citations, and an action checklist.</p>
              <p className={health.data?.llm_available ? 'text-xs text-emerald-700' : 'text-xs text-amber-700'}>
                {health.data?.llm_available ? 'Local LLM online: explanations will be written and checked (about 1–3 minutes).'
                  : 'Local LLM offline: the report will be evidence-only (TreeSHAP and retrieved sources) and will say so. Start Ollama for LLM explanations.'}</p>
            </div>
          )}
          <ErrorBox error={start.error} />
        </div>
        <div className="flex justify-end gap-2 border-t border-slate-100 px-5 py-3">
          <Button variant="ghost" onClick={onClose}>Close</Button>
          <Button variant={existing ? 'secondary' : 'primary'} loading={start.isPending || running} onClick={() => { setFinished(null); start.mutate() }}>
            {existing || failed ? <><RotateCcw className="h-4 w-4" />{failed ? 'Try again' : 'Generate again'}</> : <><FileText className="h-4 w-4" />Generate Report</>}
          </Button>
        </div>
      </div>
      {preview !== null && <PdfPreview reportId={preview} title={`Security report · ${analysis.dataset_name}`} onClose={() => setPreview(null)} />}
    </div>
  )
}
