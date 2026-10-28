import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import clsx from 'clsx'
import { ArrowLeft, CheckCircle2, RotateCcw, Sparkles, XCircle } from 'lucide-react'
import { useEffect, useRef } from 'react'
import { Link, useParams } from 'react-router-dom'
import { Bar, BarChart, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import ReportView from '../components/ReportView'
import { Button, Card, ClassBadge, ErrorBox, InfoTip, PageHeader, PredictionBadge, SeverityBadge, Spinner, StepList } from '../components/ui'
import { ApiError, api, type FlowDetail as Flow } from '../lib/api'
import { CONFIDENCE_LABEL, CONFIDENCE_TIP, COLORS, SEVERITY_COLORS, pct, prob } from '../lib/format'
import { useToast } from '../lib/toast'

const KEY_FIELDS = ['proto', 'service', 'state', 'dur', 'sbytes', 'dbytes', 'spkts', 'dpkts', 'sttl', 'dttl', 'smean', 'dmean', 'rate', 'sload', 'dload', 'ct_srv_src', 'ct_dst_src_ltm', 'ct_state_ttl']
const INCIDENT_STEPS = [
  { key: 'evidence', label: 'Collecting evidence and matching sources (MITRE ATT&CK, NIST, CISA)' },
  { key: 'writing', label: 'Local LLM is writing the explanation' },
  { key: 'verifying', label: 'Checking every statement against its source' },
  { key: 'saving', label: 'Saving' },
]

function Verdict({ f }: { f: Flow }) {
  const malicious = f.prediction === 'Malicious'
  const others = f.prediction_set.filter(c => c !== f.predicted_class)
  return (
    <div className={clsx('rounded-xl border p-5', malicious ? 'border-rose-200 bg-rose-50/70' : 'border-emerald-200 bg-emerald-50/70')}>
      <div className="flex flex-wrap items-center gap-3">
        <PredictionBadge malicious={malicious} />
        {malicious ? <ClassBadge name={f.predicted_class} severity={f.severity} /> : <span className="text-sm font-medium text-emerald-800">Normal traffic</span>}
        {malicious && <span className="text-sm text-slate-600">Severity: <SeverityBadge severity={f.severity} /></span>}
      </div>
      <p className="mt-3 text-sm text-slate-700">
        The model gives this prediction a confidence of <b>{pct(f.confidence)}</b> ({CONFIDENCE_LABEL[f.confidence_level].toLowerCase()} confidence).{' '}
        {f.needs_review ? 'Because the confidence is below 70%, this flow needs review by an analyst. ' : ''}
        {others.length ? <>The 90% shortlist also contains {others.join(', ')}, so the model cannot rule {others.length === 1 ? 'it' : 'them'} out.</> : 'No other class is plausible.'}
      </p>
      {f.ground_truth && (
        <p className={clsx('mt-3 flex items-center gap-2 text-sm font-medium', f.ground_truth === f.predicted_class ? 'text-emerald-700' : 'text-rose-700')}>
          {f.ground_truth === f.predicted_class ? <CheckCircle2 className="h-4 w-4" /> : <XCircle className="h-4 w-4" />}
          {f.ground_truth === f.predicted_class ? <>Correct: the label in the file is {f.ground_truth}.</>
            : <>The label in the file is {f.ground_truth}{f.prediction_set.includes(f.ground_truth as never) ? ' (it is on the shortlist).' : ' (not on the shortlist).'}</>}
        </p>
      )}
    </div>
  )
}

function IncidentExplanation({ f }: { f: Flow }) {
  const queryClient = useQueryClient()
  const toast = useToast()
  const health = useQuery({ queryKey: ['health'], queryFn: api.health })
  const report = useQuery({ queryKey: ['incident', f.id], queryFn: () => api.incident(f.analysis_id, f.id), enabled: f.has_incident_report, retry: false })
  const job = useQuery({
    queryKey: ['incident-job', f.id],
    queryFn: async () => { try { return await api.incidentJob(f.analysis_id, f.id) } catch (e) { if (e instanceof ApiError && e.status === 404) return null; throw e } },
    refetchInterval: q => (q.state.data && ['queued', 'running'].includes(q.state.data.status) ? 1000 : false), retry: false,
  })
  const start = useMutation({ mutationFn: () => api.startIncident(f.analysis_id, f.id), onSuccess: j => queryClient.setQueryData(['incident-job', f.id], j) })
  const j = job.data
  const running = !!j && (j.status === 'queued' || j.status === 'running')
  const watched = useRef<string | null>(null)
  useEffect(() => {
    if (!j) return
    if (j.status === 'queued' || j.status === 'running') { watched.current = j.job_id; return }
    if (watched.current !== j.job_id) return
    watched.current = null
    queryClient.invalidateQueries({ queryKey: ['incident', f.id] }); queryClient.invalidateQueries({ queryKey: ['flow', f.analysis_id, f.id] })
    if (j.status === 'done') toast({ tone: 'success', title: `Explanation ready for flow #${f.flow_index}`, body: `Finished in ${j.elapsed.toFixed(0)} s.` })
    else toast({ tone: 'error', title: 'Explanation failed', body: j.error })
  }, [j, f.id, f.analysis_id, f.flow_index, queryClient, toast])
  const step = j ? (j.stage === 'repairing' ? 2 : Math.max(0, INCIDENT_STEPS.findIndex(s => s.key === j.stage))) : 0

  return (
    <Card title={<span className="flex items-center gap-1.5"><Sparkles className="h-4 w-4 text-brand-600" />Written explanation (optional, local language model)</span>}
      subtitle="Relates this flow to MITRE ATT&CK techniques and response guidance. Statements are checked automatically against their cited sources; failed ones are removed."
      actions={<Button className="no-print" variant={report.data ? 'secondary' : 'primary'} onClick={() => start.mutate()} loading={start.isPending || running}>
        {report.data ? <><RotateCcw className="h-4 w-4" />Regenerate</> : 'Explain this flow'}</Button>}>
      <div className="space-y-4">
        <ErrorBox error={start.error} />
        {running && j && (
          <div className="rounded-xl border border-brand-100 bg-brand-50/40 p-4">
            {j.status === 'queued' ? <p className="text-sm text-slate-700">Waiting for {j.queue_position} other explanation(s)…</p> : <StepList steps={INCIDENT_STEPS} current={step} />}
            <p className="mt-3 text-xs text-slate-500">{j.elapsed.toFixed(0)} s · usually 20–60 s with the local LLM. You can leave this page; it keeps running.</p>
          </div>
        )}
        {!running && j?.status === 'failed' && <ErrorBox error={`The explanation could not be written: ${j.error}`} />}
        {!running && report.data && <ReportView alertId={f.flow_index} report={report.data} />}
        {!running && !report.data && (
          <p className="text-sm text-slate-600">
            {f.prediction === 'Safe' ? 'This flow was classified as safe; an incident explanation is only useful for malicious flows.' : 'Generate a short, source-checked explanation of this attack.'}
            {health.data && !health.data.llm_available && ' The local LLM (Ollama) is offline, so only the evidence and sources will be included.'}
          </p>
        )}
      </div>
    </Card>
  )
}

export default function FlowDetail() {
  const id = Number(useParams().id)
  const flowId = Number(useParams().flowId)
  const flow = useQuery({ queryKey: ['flow', id, flowId], queryFn: () => api.flow(id, flowId) })
  if (flow.isLoading) return <Spinner label="Loading the flow and its explanation…" />
  if (flow.error) return <ErrorBox error={flow.error} />
  const f = flow.data!
  const probs = Object.entries(f.probabilities).map(([name, value]) => ({ name, value })).sort((x, y) => y.value - x.value)
  const shap = [...f.explanation].reverse().map(e => ({ name: `${e.feature} = ${typeof e.value === 'number' ? +e.value.toFixed(4) : e.value}`, value: e.contribution }))

  return (
    <>
      <Link to={`/results/${id}`} className="no-print mb-3 inline-flex items-center gap-1 text-sm text-slate-500 hover:text-slate-800"><ArrowLeft className="h-4 w-4" />Back to results</Link>
      <PageHeader kicker="Flow analysis" title={`Flow #${f.flow_index}`} description={<>{f.dataset_name} · analysis #{id}</>} />
      <div className="grid gap-6 lg:grid-cols-3">
        <div className="lg:col-span-3"><Verdict f={f} /></div>
        <Card title={<span className="flex items-center gap-1.5">Probability of each class<InfoTip text={CONFIDENCE_TIP} align="left" /></span>} subtitle="★ on the 90% shortlist">
          <ul className="space-y-2">
            {probs.slice(0, 6).map(p => (
              <li key={p.name}>
                <div className="flex items-center justify-between text-xs">
                  <span className={f.prediction_set.includes(p.name as never) ? 'font-medium text-slate-900' : 'text-slate-500'}>{p.name}{f.prediction_set.includes(p.name as never) && ' ★'}</span>
                  <span className="tabular-nums text-slate-600">{pct(p.value)}</span>
                </div>
                <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-slate-100">
                  <div className="h-full rounded-full" style={{ width: `${p.value * 100}%`, backgroundColor: p.name === 'Normal' ? COLORS.safe : SEVERITY_COLORS[p.name === f.predicted_class ? f.severity : 'Low'] }} />
                </div>
              </li>
            ))}
          </ul>
          {(() => {
            const pNormal = probs.find(p => p.name === 'Normal')?.value ?? 0
            const attack = f.predicted_class !== 'Normal'
            const sure = attack ? 1 - pNormal : pNormal
            return (
              <p className="mt-4 rounded-lg bg-slate-50 px-3 py-2 text-xs text-slate-600">
                <b className="text-slate-800">Attack or normal?</b> The model is <b className="tabular-nums">{prob(sure)}</b> sure this flow is {attack ? 'an attack (any type)' : 'normal traffic'}:
                the probabilities of {attack ? 'the nine attack types' : 'Normal'} added together.
              </p>
            )
          })()}
        </Card>
        <Card className="lg:col-span-2" title={<span className="flex items-center gap-1.5">Why this prediction?<InfoTip align="right" text={`${f.explanation_units}. Each bar shows how much one measurement pushed the model towards ${f.predicted_class} (red) or away from it (blue). Exact TreeSHAP values.`} /></span>}
          subtitle={`Measurements that pushed the model towards ${f.predicted_class} (red) or away from it (blue)`}>
          <div className="h-72 min-w-0">
            <ResponsiveContainer>
              <BarChart data={shap} layout="vertical" margin={{ left: 10, right: 16 }}>
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis type="category" dataKey="name" tick={{ fontSize: 11 }} width={170} />
                <ReferenceLine x={0} stroke="#94a3b8" />
                <Tooltip formatter={(v: number) => v.toFixed(3)} />
                <Bar dataKey="value" name="contribution">{shap.map(d => <Cell key={d.name} fill={d.value > 0 ? '#ef4444' : '#3b82f6'} />)}</Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
        <Card title="Flow measurements" subtitle="Key UNSW-NB15 features of this flow (hover a name for its meaning)" className="lg:col-span-3">
          <dl className="grid grid-cols-1 gap-x-6 gap-y-1.5 text-xs sm:grid-cols-2 lg:grid-cols-3">
            {KEY_FIELDS.map(k => (
              <div key={k} className="flex justify-between gap-2 border-b border-slate-50 py-1">
                <dt className="text-slate-500" title={f.descriptions[k]}>{k}</dt><dd className="truncate font-medium tabular-nums text-slate-800">{String(f.flow[k])}</dd>
              </div>
            ))}
          </dl>
        </Card>
        <div className="lg:col-span-3"><IncidentExplanation f={f} /></div>
      </div>
    </>
  )
}
