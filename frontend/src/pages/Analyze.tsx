import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import clsx from 'clsx'
import { AlertTriangle, CheckCircle2, ChevronDown, Download, FileSpreadsheet, FlaskConical, Info, Play, ShieldCheck, UploadCloud, XCircle } from 'lucide-react'
import { useEffect, useRef, useState, type DragEvent } from 'react'
import { useNavigate } from 'react-router-dom'
import { Button, Card, ErrorBox, PageHeader, ProgressBar, StepList } from '../components/ui'
import { api, type UploadReport } from '../lib/api'
import { bytes, num } from '../lib/format'
import { useToast } from '../lib/toast'

const MAX_MB = 50
const ANALYSIS_STEPS = [
  { key: 'uploaded', label: 'Dataset uploaded' },
  { key: 'validated', label: 'Dataset validated' },
  { key: 'features', label: 'Preparing model features', detail: 'Checking values, one-hot encoding protocol / service / state, adding the engineered byte features' },
  { key: 'detection', label: 'Running attack detection', detail: 'LightGBM model classifies every flow into Normal or one of nine attack types' },
  { key: 'confidence', label: 'Calculating prediction confidence', detail: 'Calibrated probabilities and the 90% shortlist of possible classes' },
  { key: 'analytics', label: 'Preparing security analytics', detail: 'Attack distribution, severity, overall risk; saving every flow prediction' },
]
const WORKFLOW = ['Upload dataset', 'Validate', 'Analyze', 'Results']

type Phase = 'select' | 'uploading' | 'validated' | 'invalid' | 'analyzing' | 'complete' | 'failed'

function Workflow({ phase }: { phase: Phase }) {
  const index = { select: 0, uploading: 0, invalid: 1, validated: 1, analyzing: 2, failed: 2, complete: 3 }[phase]
  return (
    <ol className="mb-6 flex flex-wrap items-center gap-2 text-xs font-medium">
      {WORKFLOW.map((step, i) => (
        <li key={step} className="flex items-center gap-2">
          <span className={clsx('flex h-6 w-6 items-center justify-center rounded-full', i < index || phase === 'complete' ? 'bg-emerald-500 text-white'
            : i === index ? 'bg-brand-600 text-white' : 'bg-slate-200 text-slate-500')}>{i < index || phase === 'complete' ? '✓' : i + 1}</span>
          <span className={i <= index ? 'text-slate-800' : 'text-slate-400'}>{step}</span>
          {i < WORKFLOW.length - 1 && <span className="mx-1 h-px w-6 bg-slate-300 sm:w-10" aria-hidden />}
        </li>
      ))}
    </ol>
  )
}

function Requirements() {
  const [open, setOpen] = useState(false)
  const model = useQuery({ queryKey: ['model'], queryFn: api.model, enabled: open })
  return (
    <div className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 text-sm">
      <button type="button" onClick={() => setOpen(o => !o)} className="flex w-full items-center justify-between gap-2 text-left font-medium text-slate-700">
        <span className="flex items-center gap-2"><Info className="h-4 w-4 text-brand-600" />Accepted format</span>
        <ChevronDown className={clsx('h-4 w-4 transition', open && 'rotate-180')} />
      </button>
      <ul className="mt-2 space-y-1 text-xs text-slate-600">
        <li>• <b>CSV</b> file (comma-separated, one header row), up to {MAX_MB} MB and 200,000 flows.</li>
        <li>• One row per <b>completed network flow</b> with the <b>42 UNSW-NB15 flow features</b> (dur, proto, service, state, sbytes, dbytes, sttl, …). Extra columns are ignored.</li>
        <li>• Optional: an <code>attack_cat</code> or <code>ground_truth</code> column with the true class, to compare the predictions with it (never used to predict).</li>
      </ul>
      {open && (
        <div className="mt-3 max-h-64 overflow-y-auto rounded-md border border-slate-200 bg-white">
          <table className="min-w-full text-xs">
            <tbody className="divide-y divide-slate-100">
              {model.data?.features.map(f => <tr key={f.name}><td className="whitespace-nowrap px-3 py-1.5 font-mono text-slate-800">{f.name}</td><td className="px-3 py-1.5 text-slate-600">{f.description}</td></tr>)}
              {!model.data && <tr><td className="px-3 py-2 text-slate-500">Loading the feature list…</td></tr>}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

function ValidationResult({ report, onChange, onAnalyze, busy }: { report: UploadReport; onChange: () => void; onAnalyze: () => void; busy: boolean }) {
  return (
    <Card title="Selected dataset">
      <div className="flex flex-wrap items-start gap-4">
        <FileSpreadsheet className="h-10 w-10 shrink-0 text-brand-600" aria-hidden />
        <div className="min-w-0 flex-1">
          <p className="truncate font-medium text-slate-900">{report.filename}</p>
          <p className="text-sm text-slate-500">{bytes(report.size_bytes)}{report.rows !== null && <> · <b className="text-slate-700">{num(report.rows)}</b> network flows</>}{report.columns !== null && <> · {report.columns} columns</>}</p>
          {report.valid
            ? <p className="mt-2 flex items-center gap-1.5 text-sm font-medium text-emerald-700"><CheckCircle2 className="h-4 w-4" />Dataset validated successfully</p>
            : <p className="mt-2 flex items-center gap-1.5 text-sm font-medium text-rose-700"><XCircle className="h-4 w-4" />Dataset validation failed</p>}
        </div>
        <Button variant="ghost" onClick={onChange} disabled={busy}>Remove / change file</Button>
      </div>
      {report.errors.map(e => (
        <div key={e.title} className="mt-4 rounded-lg border border-rose-200 bg-rose-50 p-4 text-sm text-rose-800">
          <p className="font-semibold">{e.title}</p>
          <p className="mt-0.5">{e.message}</p>
          {e.items.length > 0 && (
            <ul className="mt-2 max-h-48 space-y-0.5 overflow-y-auto text-xs">
              {e.items.map(i => <li key={i.name}><code className="font-semibold">{i.name}</code>{i.description && <span className="text-rose-700/80"> – {i.description}</span>}</li>)}
            </ul>
          )}
        </div>
      ))}
      {report.warnings.length > 0 && (
        <ul className="mt-4 space-y-1.5">
          {report.warnings.map(w => <li key={w} className="flex gap-2 rounded-lg bg-amber-50 px-3 py-2 text-xs text-amber-800"><AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />{w}</li>)}
        </ul>
      )}
      <div className="mt-5 flex flex-wrap items-center gap-3 border-t border-slate-100 pt-4">
        <Button onClick={onAnalyze} disabled={!report.valid || busy} loading={busy} className="px-5 py-2.5">
          <Play className="h-4 w-4" />Analyze Network Traffic
        </Button>
        {!report.valid && <span className="text-xs text-slate-500">Fix the problems above and upload the file again.</span>}
      </div>
    </Card>
  )
}

export default function Analyze() {
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const toast = useToast()
  const input = useRef<HTMLInputElement>(null)
  const [phase, setPhase] = useState<Phase>('select')
  const [dragging, setDragging] = useState(false)
  const [uploaded, setUploaded] = useState(0)
  const [pending, setPending] = useState<{ name: string; size: number } | null>(null)
  const [report, setReport] = useState<UploadReport | null>(null)
  const [jobId, setJobId] = useState<string | null>(null)
  const [error, setError] = useState<string>('')

  const onValidated = (r: UploadReport) => { setReport(r); setPhase(r.valid ? 'validated' : 'invalid') }
  const onUploadError = (e: Error) => { setPhase('select'); setError(e.message) }
  const upload = useMutation({ mutationFn: (file: File) => api.upload(file, setUploaded), onSuccess: onValidated, onError: onUploadError })
  const sample = useMutation({ mutationFn: () => api.uploadSample(30), onSuccess: onValidated, onError: onUploadError })
  const download = useMutation({ mutationFn: () => api.downloadSample(30), onError: (e: Error) => toast({ tone: 'error', title: 'Download failed', body: e.message }) })
  const start = useMutation({
    mutationFn: (uploadId: string) => api.startAnalysis(uploadId),
    onSuccess: j => { setJobId(j.job_id); setPhase('analyzing') },
    onError: (e: Error) => { setError(e.message); setPhase('validated') },
  })
  const job = useQuery({
    queryKey: ['analysis-job', jobId], queryFn: () => api.analysisJob(jobId!), enabled: !!jobId,
    refetchInterval: q => (q.state.data && ['done', 'failed'].includes(q.state.data.status) ? false : 400),
  })

  useEffect(() => {
    const j = job.data
    if (!j || phase !== 'analyzing') return
    if (j.status === 'done') {
      setPhase('complete')
      queryClient.invalidateQueries({ queryKey: ['overview'] }); queryClient.invalidateQueries({ queryKey: ['analyses'] }); queryClient.invalidateQueries({ queryKey: ['reports'] })
    }
    if (j.status === 'failed') setPhase('failed')
  }, [job.data, phase, queryClient])

  // Show "Analysis Complete" briefly, then open the results.
  const completedId = phase === 'complete' ? job.data?.result.analysis_id : undefined
  useEffect(() => {
    if (!completedId) return
    const t = setTimeout(() => navigate(`/results/${completedId}`), 1500)
    return () => clearTimeout(t)
  }, [completedId, navigate])

  function choose(file: File | undefined) {
    if (!file || phase === 'uploading' || phase === 'analyzing') return
    setError('')
    if (!file.name.toLowerCase().endsWith('.csv')) { setError(`"${file.name}" is not a CSV file. Please choose a .csv file.`); return }
    if (file.size > MAX_MB * 1024 * 1024) { setError(`"${file.name}" is ${bytes(file.size)}; the limit is ${MAX_MB} MB.`); return }
    setPending({ name: file.name, size: file.size }); setUploaded(0); setReport(null); setPhase('uploading')
    upload.mutate(file)
  }
  function reset() {
    if (report?.upload_id && phase !== 'complete') api.discardUpload(report.upload_id).catch(() => {})
    setReport(null); setPending(null); setJobId(null); setError(''); setPhase('select')
  }
  function onDrop(e: DragEvent) { e.preventDefault(); setDragging(false); choose(e.dataTransfer.files[0]) }

  const j = job.data
  const stageIndex = j ? ANALYSIS_STEPS.findIndex(s => s.key === j.stage) : 2
  const flows = report?.rows ?? 0

  return (
    <>
      <PageHeader kicker="Analysis" title="Analyze Network Flows"
        description="Upload a dataset of completed network flows. It is validated against the trained model's requirements, then each flow is predicted as Normal or as one of nine attack types." />
      <Workflow phase={phase} />

      {(phase === 'select' || phase === 'uploading') && (
        <div className="grid gap-6 lg:grid-cols-3">
          <Card title="Upload network flow dataset" className="lg:col-span-2">
            <div onDragOver={e => { e.preventDefault(); if (phase === 'select') setDragging(true) }} onDragLeave={() => setDragging(false)} onDrop={onDrop}
              className={clsx('flex flex-col items-center justify-center rounded-xl border-2 border-dashed px-6 py-12 text-center transition',
                dragging ? 'border-brand-500 bg-brand-50' : 'border-slate-300 bg-slate-50/50')}>
              {phase === 'uploading' && pending ? (
                <div className="w-full max-w-sm">
                  <FileSpreadsheet className="mx-auto mb-3 h-9 w-9 text-brand-600" aria-hidden />
                  <p className="truncate font-medium text-slate-800">{pending.name}</p>
                  <p className="mb-3 text-xs text-slate-500">{bytes(pending.size)} · {uploaded < 1 ? `uploading ${Math.round(uploaded * 100)}%` : 'validating the dataset…'}</p>
                  <ProgressBar value={uploaded < 1 ? uploaded : null} />
                </div>
              ) : (
                <>
                  <UploadCloud className="mb-3 h-10 w-10 text-slate-400" aria-hidden />
                  <p className="font-medium text-slate-800">Drag and drop your CSV file here</p>
                  <p className="mt-1 text-xs text-slate-500">.csv · up to {MAX_MB} MB · UNSW-NB15 flow features</p>
                  <Button className="mt-4" onClick={() => input.current?.click()}>Browse File</Button>
                </>
              )}
              <input ref={input} type="file" accept=".csv,text/csv" className="hidden" aria-label="CSV file" onChange={e => { choose(e.target.files?.[0]); e.target.value = '' }} />
            </div>
            <div className="mt-4 space-y-3">
              {error && <ErrorBox error={error} />}
              <Requirements />
            </div>
          </Card>
          <Card title="No dataset at hand?" subtitle="Real flows from the official UNSW-NB15 test file">
            <div className="space-y-4 text-sm text-slate-600">
              <p>Use a ready-made sample: <b>300 flows</b>, 30 of each class, with the true class in a <code>ground_truth</code> column so you can also check the predictions.</p>
              <Button className="w-full" variant="secondary" onClick={() => { setError(''); setPending(null); setPhase('uploading'); sample.mutate() }} loading={sample.isPending}>
                <FlaskConical className="h-4 w-4" />Use sample dataset
              </Button>
              <Button className="w-full" variant="ghost" onClick={() => download.mutate()} loading={download.isPending}>
                <Download className="h-4 w-4" />Download sample CSV
              </Button>
            </div>
          </Card>
        </div>
      )}

      {(phase === 'validated' || phase === 'invalid') && report && (
        <div className="space-y-4">
          {error && <ErrorBox error={error} />}
          <ValidationResult report={report} onChange={reset} onAnalyze={() => report.upload_id && start.mutate(report.upload_id)} busy={start.isPending} />
        </div>
      )}

      {(phase === 'analyzing' || phase === 'complete' || phase === 'failed') && report && (
        <Card>
          <div className="mx-auto max-w-xl py-4">
            <div className="mb-5 flex items-center gap-3">
              {phase === 'complete' ? <CheckCircle2 className="h-9 w-9 text-emerald-500" /> : phase === 'failed' ? <XCircle className="h-9 w-9 text-rose-500" /> : <ShieldCheck className="h-9 w-9 text-brand-600" />}
              <div className="min-w-0">
                <h2 className="text-lg font-semibold text-slate-900">{phase === 'complete' ? 'Analysis Complete' : phase === 'failed' ? 'Analysis failed' : 'Analyzing Network Traffic'}</h2>
                <p className="truncate text-sm text-slate-500">{report.filename} · {num(flows)} flows</p>
              </div>
            </div>
            <div className="mb-5"><ProgressBar value={phase === 'complete' ? 1 : (Math.max(stageIndex, 2)) / ANALYSIS_STEPS.length} tone={phase === 'failed' ? 'bad' : phase === 'complete' ? 'good' : 'brand'} /></div>
            <StepList steps={ANALYSIS_STEPS} current={Math.max(stageIndex, 2)} failed={phase === 'failed'} done={phase === 'complete'} />
            {phase === 'analyzing' && j && <p className="mt-4 text-xs text-slate-500">{j.status === 'queued' ? `Waiting for ${j.queue_position} other analysis to finish… · ` : ''}{j.elapsed.toFixed(1)} s</p>}
            {phase === 'complete' && j && (
              <div className="mt-5 rounded-lg bg-emerald-50 px-4 py-3 text-sm text-emerald-800">
                {num(flows)} network flows successfully analyzed in {j.elapsed.toFixed(1)} s. Opening the results…
                <Button className="ml-3" variant="secondary" onClick={() => navigate(`/results/${j.result.analysis_id}`)}>View results now</Button>
              </div>
            )}
            {phase === 'failed' && (
              <div className="mt-5 space-y-3">
                <ErrorBox error={j?.error || 'The analysis failed.'} />
                <Button variant="secondary" onClick={reset}>Analyze another file</Button>
              </div>
            )}
          </div>
        </Card>
      )}
    </>
  )
}
