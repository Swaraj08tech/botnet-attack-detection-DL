import { useQuery } from '@tanstack/react-query'
import { Activity, AlertTriangle, BarChart3, CheckCircle2, Crosshair, FileText, Layers, ShieldAlert, ShieldCheck, Target, Upload } from 'lucide-react'
import { useState } from 'react'
import { Link, Navigate, useNavigate, useParams } from 'react-router-dom'
import Accuracy from '../components/Accuracy'
import { AttackBars, ConfidenceLevels, SeverityLegend, ThreatSankey, TrafficDonut } from '../components/charts'
import FlowTable from '../components/FlowTable'
import ReportPanel from '../components/ReportPanel'
import ReportCard from '../components/report/ReportCard'
import { Button, Card, ClassBadge, Empty, ErrorBox, InfoTip, KpiCard, PageHeader, RiskBadge, SeverityBadge, Spinner } from '../components/ui'
import { api, type AnalysisResult, type FlowFilters } from '../lib/api'
import { CONFIDENCE_TIP, RISK_STYLE, SEVERITY_COLORS, SEVERITY_ORDER, num, pct, when } from '../lib/format'

function Picker({ current }: { current: number }) {
  const navigate = useNavigate()
  const list = useQuery({ queryKey: ['analyses'], queryFn: api.analyses })
  if (!list.data || list.data.length < 2) return null
  return (
    <select aria-label="Choose analysis" value={current} onChange={e => navigate(`/results/${e.target.value}`)}
      className="max-w-64 rounded-lg border border-slate-300 bg-white px-2.5 py-2 text-sm text-slate-700">
      {list.data.map(a => <option key={a.id} value={a.id}>#{a.id} · {a.dataset_name}</option>)}
    </select>
  )
}

function RiskCard({ r }: { r: AnalysisResult }) {
  const risk = r.analysis.risk
  const s = RISK_STYLE[risk.level]
  return (
    <Card title={<span className="flex items-center gap-1.5">Batch triage priority<InfoTip text={<>Fixed rules, checked from the top:<br />{r.analysis.rules.risk.map(x => <span key={x.level} className="block">• <b>{x.level}</b>: {x.rule}</span>)}<span className="block mt-1">Serious = Critical or High severity flows.</span></>} /></span>}>
      <div className={`rounded-xl ${s.bg} p-4`}>
        <RiskBadge level={risk.level} large />
        <p className={`mt-3 text-sm ${s.color}`}>{risk.explanation}</p>
        <p className="mt-1 text-xs text-slate-500">An application heuristic that orders review work from the predictions; not a model output or measured impact.</p>
      </div>
      <p className="mt-3 text-xs text-slate-500">Rule matched: {risk.rule}.</p>
      <div className="mt-4 grid grid-cols-4 gap-2 text-center">
        {SEVERITY_ORDER.map(sv => (
          <div key={sv} className="rounded-lg border border-slate-100 py-2">
            <p className="text-lg font-semibold tabular-nums" style={{ color: SEVERITY_COLORS[sv] }}>{num(r.analysis.severity_distribution[sv])}</p>
            <p className="text-[11px] text-slate-500">{sv}</p>
          </div>
        ))}
      </div>
    </Card>
  )
}

function SummaryCards({ r, onReview }: { r: AnalysisResult; onReview: () => void }) {
  const s = r.analysis.summary
  const items = [
    { icon: <Crosshair className="h-4 w-4" />, label: 'Most detected attack', value: s.most_detected_attack ? <ClassBadge name={s.most_detected_attack.attack_type} severity={r.analysis.rules.severity[s.most_detected_attack.attack_type]} /> : '–',
      detail: s.most_detected_attack ? `${num(s.most_detected_attack.count)} flows` : 'no attacks' },
    { icon: <Layers className="h-4 w-4" />, label: 'Attack types detected', value: <span className="text-xl font-semibold">{s.attack_types_detected}</span>, detail: 'of 9 attack classes the model knows' },
    { icon: <Target className="h-4 w-4" />, label: 'Highest confidence attack', value: s.highest_confidence_attack ? <ClassBadge name={s.highest_confidence_attack.attack_type} severity={r.analysis.rules.severity[s.highest_confidence_attack.attack_type]} /> : '–',
      detail: s.highest_confidence_attack ? `average confidence ${pct(s.highest_confidence_attack.average_confidence)}` : '' },
    { icon: <AlertTriangle className="h-4 w-4" />, label: 'Flows requiring review', value: <span className="text-xl font-semibold text-amber-600">{num(s.flows_requiring_review)}</span>,
      detail: <>confidence below 70% · <button className="text-brand-600 hover:underline" onClick={onReview}>show them</button></> },
  ]
  return (
    <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
      {items.map(i => (
        <div key={i.label} className="min-w-0 rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
          <p className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-slate-500"><span className="text-slate-400">{i.icon}</span>{i.label}</p>
          <div className="mt-2">{i.value}</div>
          <p className="mt-1 text-xs text-slate-500">{i.detail}</p>
        </div>
      ))}
    </div>
  )
}

export default function Results() {
  const params = useParams()
  const id = Number(params.id)
  const navigate = useNavigate()
  const [reportOpen, setReportOpen] = useState(false)
  const [tableFilter, setTableFilter] = useState<FlowFilters | undefined>(undefined)
  const list = useQuery({ queryKey: ['analyses'], queryFn: api.analyses, enabled: !params.id })
  const result = useQuery({ queryKey: ['analysis', id], queryFn: () => api.analysis(id), enabled: !!params.id })

  if (!params.id) {
    if (list.isLoading) return <Spinner />
    if (list.data?.length) return <Navigate to={`/results/${list.data[0].id}`} replace />
    return (
      <>
        <PageHeader kicker="Analysis" title="Results" />
        <Empty icon={<BarChart3 className="h-10 w-10" />} title="No analysis yet" action={<Link to="/analyze"><Button><Upload className="h-4 w-4" />Analyze Flows</Button></Link>}>
          Upload a network-flow dataset to see which flows the model predicts as attacks, which types they resemble and how confident it is.
        </Empty>
      </>
    )
  }
  if (result.isLoading) return <Spinner label="Loading the analysis…" />
  if (result.error) return <ErrorBox error={result.error} />
  const r = result.data!
  const a = r.analysis
  const showFlows = (f: FlowFilters) => { setTableFilter({ ...f }); setTimeout(() => document.getElementById('flows')?.scrollIntoView({ behavior: 'smooth' }), 50) }

  return (
    <>
      <PageHeader kicker="Network Security Analysis" title={<span className="break-all">{r.dataset_name}</span>}
        description={<span className="flex flex-wrap items-center gap-x-3 gap-y-1">
          <span>Analysis #{r.id} · {when(r.created_at)}</span><span>{num(r.total_flows)} flows</span><span>processed in {r.seconds.toFixed(1)} s</span>
          <span className="inline-flex items-center gap-1 text-emerald-700"><CheckCircle2 className="h-4 w-4" />Completed</span></span>}
        actions={<>
          <Picker current={r.id} />
          <Button variant="secondary" onClick={() => navigate('/analyze')}><Upload className="h-4 w-4" />Analyze Another File</Button>
          <Button onClick={() => setReportOpen(true)}><FileText className="h-4 w-4" />Generate Security Report</Button>
        </>} />

      <details className="group mb-5 rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm shadow-sm">
        <summary className="cursor-pointer select-none font-medium text-slate-800 marker:text-slate-400">How to read this page</summary>
        <ol className="mt-3 list-decimal space-y-1.5 pl-5 text-slate-600">
          <li><b className="text-slate-800">Counts.</b> Each flow is one finished network connection. The model predicts Normal or one of nine attack types for it.</li>
          <li><b className="text-slate-800">Confidence.</b> How sure the model is. It is usually sure whether a flow is an attack; it is less sure about the exact type,
            because some attack types look the same in these measurements.</li>
          <li><b className="text-slate-800">Review.</b> Flows below 70% confidence, or with several types on their shortlist, should be checked by a person first.</li>
          <li><b className="text-slate-800">Next step.</b> Click a flow to see which measurements drove its prediction, or generate the security report for a summary
            with sources and suggested checks. Predictions are not confirmed incidents.</li>
        </ol>
      </details>

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <KpiCard label="Total flows" value={num(a.total_flows)} detail="network flows analysed" icon={<Activity className="h-4 w-4" />} tone="brand" />
        <KpiCard label="Predicted Normal" value={num(a.safe_count)} detail={`${pct(a.safe_percentage)} of all flows`} tone="safe" icon={<ShieldCheck className="h-4 w-4" />}
          tip="Flows the model classified as Normal (no attack). A prediction, not a verified fact." />
        <KpiCard label="Predicted attack" value={num(a.malicious_count)} detail={`${pct(a.malicious_percentage)} of all flows`} tone="threat" icon={<ShieldAlert className="h-4 w-4" />}
          tip="Flows the model classified as one of the nine attack types. Predictions are not confirmed incidents." />
        {a.status_confidence_distribution
          ? <KpiCard label="Sure: attack or normal" value={pct(a.status_confidence_distribution.high / Math.max(a.total_flows, 1))}
              detail={`${num(a.status_confidence_distribution.high)} flows with 90% or more`}
              tip="Share of flows for which the model is at least 90% sure whether the flow is an attack (of any type) or normal traffic. The exact attack type is often less certain; see Prediction Confidence below." />
          : <KpiCard label="Average confidence" value={pct(a.average_confidence)} detail="mean confidence of all predictions" tip={CONFIDENCE_TIP} />}
      </div>

      <div className="mt-6 grid gap-6 lg:grid-cols-5">
        <Card title="Traffic Classification" subtitle="Flows predicted Normal and flows predicted as an attack" className="lg:col-span-3">
          <TrafficDonut safe={a.safe_count} malicious={a.malicious_count} />
        </Card>
        <div className="lg:col-span-2"><RiskCard r={r} /></div>
      </div>

      <div className="mt-6"><SummaryCards r={r} onReview={() => showFlows({ review: true })} /></div>

      <div className="mt-6 grid gap-6 lg:grid-cols-5">
        <Card title="Attack Distribution" subtitle="Predicted attacks by type (bar colour = heuristic severity)" className="lg:col-span-3" actions={<SeverityLegend />}>
          <AttackBars data={a.attack_distribution} />
        </Card>
        <Card title="Prediction Confidence" subtitle={`${num(a.total_flows)} predictions`} className="lg:col-span-2">
          <ConfidenceLevels a={a} />
          <div className="mt-4 rounded-lg bg-amber-50 px-3 py-2.5 text-xs text-amber-900">
            <p><b>{num(a.needs_review_count)} flows need review</b>: the model is less than 70% sure of their exact class.{a.status_confidence_distribution ? ` For ${num(a.status_confidence_distribution.high)} flows it is still at least 90% sure whether they are attacks.` : ''}</p>
            <p className="mt-1"><b>{num(a.attack_type_uncertain_count)} predicted attacks</b> have more than one attack type in their 90% shortlist: likely attacks, but the exact type is uncertain.
              <InfoTip align="right" text="The shortlist is a conformal prediction set: the classes the model cannot rule out, built to contain the true class about 90% of the time. See the Model page." /></p>
          </div>
        </Card>
      </div>

      {a.attack_distribution.length > 0 && (
        <Card className="mt-6" title="Attack details" bodyClassName="p-0">
          <div className="overflow-x-auto">
            <table className="min-w-full text-sm">
              <thead className="bg-slate-50 text-left text-xs uppercase tracking-wide text-slate-500">
                <tr><th className="px-5 py-2.5">Attack type</th><th className="px-3 text-right">Flows</th><th className="px-3 text-right">% of predicted attacks</th>
                  <th className="px-3 text-right">Average confidence</th><th className="px-3">Severity</th><th className="px-3 text-right">Need review</th><th className="px-5" /></tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {a.attack_distribution.map(d => (
                  <tr key={d.attack_type}>
                    <td className="px-5 py-2.5"><ClassBadge name={d.attack_type} severity={d.severity} /></td>
                    <td className="px-3 text-right tabular-nums">{num(d.count)}</td>
                    <td className="px-3 text-right tabular-nums">{pct(d.percentage_of_malicious)}</td>
                    <td className="px-3 text-right tabular-nums">{pct(d.average_confidence)}</td>
                    <td className="px-3"><SeverityBadge severity={d.severity} /></td>
                    <td className="px-3 text-right tabular-nums text-slate-600">{num(d.needs_review)}</td>
                    <td className="px-5 text-right"><button className="text-xs text-brand-600 hover:underline" onClick={() => showFlows({ attack_type: d.attack_type })}>View flows</button></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      <Card className="mt-6" title="Network Threat Flow Visualization" subtitle="How the uploaded traffic was classified: flows → detection model → predicted Normal / predicted attack → attack types (widths = number of flows)">
        <ThreatSankey a={a} />
      </Card>

      <div className="mt-6"><ReportCard analysis={r} onGenerate={() => setReportOpen(true)} /></div>

      {a.evaluation && (
        <Card className="mt-6" title="Check against the labels in this file"
          subtitle="This file contains the true class of each flow, so the predictions can be compared with it. These numbers describe this file only; the model's official performance is on the Model page.">
          <Accuracy evaluation={a.evaluation} />
        </Card>
      )}

      <div id="flows" className="scroll-mt-4">
        <Card className="mt-6" title="Detailed Flow Analysis" subtitle="Every flow with its prediction. Click a row for the explanation of that prediction.">
          <FlowTable analysisId={r.id} a={a} initial={tableFilter} />
        </Card>
      </div>

      {reportOpen && <ReportPanel analysis={r} onClose={() => setReportOpen(false)} />}
    </>
  )
}
