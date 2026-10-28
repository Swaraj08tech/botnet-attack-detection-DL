import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import clsx from 'clsx'
import { AlertTriangle, ChevronDown, ExternalLink, Filter, History, Search, XCircle } from 'lucide-react'
import { useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  api, type ActionState, type ActionStatus, type AnalysisData, type FlowCard, type FlowFilters, type ReportRecord, type Statement,
} from '../../lib/api'
import { num, pct, prob, when } from '../../lib/format'
import { ACTION_LABEL, ORIGIN_LABEL, STATUS_LABEL, safeUrl } from '../../lib/record'
import { useToast } from '../../lib/toast'
import FlowTable from '../FlowTable'
import { Button, Card, ClassBadge, ConfidenceBadge, ErrorBox, InfoTip, RiskBadge, SeverityBadge, Spinner } from '../ui'
import { Cites, Dialog, OriginTag } from './Citations'

const H3 = 'mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500'

// ------------------------------------------------------------------------------------------------ A. Summary
export function SummaryTab({ r }: { r: ReportRecord }) {
  const s = r.summary
  const tiles: [string, string, string, 'model_output' | 'batch_statistic'][] = [
    ['Predicted Normal', num(s.predicted_normal), `${pct(s.predicted_normal_share)} of flows`, 'model_output'],
    ['Predicted attack', num(s.predicted_attack), `${pct(s.predicted_attack_share)} of flows`, 'model_output'],
    ['Needs review', num(s.needs_review), 'confidence below 70%', 'batch_statistic'],
    ['Average confidence', pct(s.average_confidence), 'calibrated probability of the predicted class', 'batch_statistic'],
  ]
  return (
    <div className="space-y-5">
      <p className="text-sm text-slate-700">{s.scope}</p>
      <div>
        <h3 className={H3}>Main findings <span className="normal-case tracking-normal text-slate-400">(the same bullets as the Summary view)</span></h3>
        <ul className="space-y-1.5 text-sm">
          {r.views.summary.overview.bullets.map((b, i) => (
            <li key={i} className="flex gap-2"><span aria-hidden className="text-slate-400">•</span>
              <span>{b.text}<Cites ids={b.cites} /><OriginTag origin={b.origin} />{b.scope && <span className="ml-1 text-[11px] text-slate-400">scope: {b.scope}</span>}</span></li>
          ))}
        </ul>
        <p className="mt-1 text-[11px] text-slate-400">{r.views.summary.overview.method}</p>
      </div>
      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        {tiles.map(([label, value, detail, origin]) => (
          <div key={label} className="rounded-xl border border-slate-200 bg-white p-4">
            <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">{label}</p>
            <p className="mt-1 text-2xl font-semibold tabular-nums text-slate-900">{value}</p>
            <p className="mt-0.5 text-xs text-slate-500">{detail}<OriginTag origin={origin} /></p>
          </div>
        ))}
      </div>
      <div className="grid gap-4 lg:grid-cols-2">
        <div className="rounded-xl border border-slate-200 p-4">
          <p className="flex items-center gap-2 text-sm font-semibold text-slate-900">Batch triage priority <RiskBadge level={s.triage_priority.level} /><OriginTag origin="heuristic" /></p>
          <p className="mt-2 text-sm text-slate-700">{s.triage_priority.meaning}</p>
          <p className="mt-1 text-xs text-slate-500">Rule matched: {s.triage_priority.rule}.</p>
        </div>
        <div className="space-y-3 rounded-xl border border-amber-200 bg-amber-50/60 p-4 text-sm">
          <div><p className="font-semibold text-amber-900">Most important uncertainty</p><p className="text-amber-900/90">{s.most_important_uncertainty}</p></div>
          <div><p className="font-semibold text-amber-900">What to check first</p><p className="text-amber-900/90">{s.next_check}</p></div>
          <p className="text-xs text-amber-900/80">{num(s.attack_status_uncertain)} flows are attack-status uncertain (Normal is still on their shortlist); {num(s.attack_type_uncertain)} are attacks whose type is uncertain.</p>
        </div>
      </div>
      <p className="text-xs text-slate-500">
        Explained in detail: {r.coverage.flows_explained} illustrative flows of {num(r.coverage.flows_total)}; LLM explanations for {r.coverage.attack_types_with_llm} of {r.coverage.attack_types_detected} predicted attack types.
        Predictions are not confirmed incidents.
      </p>
    </div>
  )
}

// ------------------------------------------------------------------------------------------------ B. Detections and uncertainty
const UNCERTAINTY: { key: NonNullable<FlowFilters['uncertainty']>; label: string; count: (r: ReportRecord) => number; tip: string }[] = [
  { key: 'status', label: 'Attack status uncertain', count: r => r.detections.sets.attack_status_uncertain, tip: 'The 90% shortlist contains Normal and at least one attack type: whether it is an attack at all is unclear.' },
  { key: 'type', label: 'Attack type uncertain', count: r => r.detections.sets.attack_type_uncertain, tip: 'Predicted attack, and the shortlist has several attack types but not Normal: an attack, but which type is unclear.' },
  { key: 'none', label: 'Single-class shortlist', count: r => r.detections.sets.singleton, tip: 'Only one class on the shortlist. This is not certainty: the shortlist is built for 90% average coverage.' },
]

export function DetectionsTab({ r, analysis, onOpenFlow, preset }: { r: ReportRecord; analysis: AnalysisData | undefined; onOpenFlow: (flowId: number) => void; preset?: string | null }) {
  const fromLink = UNCERTAINTY.find(u => u.key === preset)
  const [filter, setFilter] = useState<FlowFilters | undefined>(fromLink ? { uncertainty: fromLink.key } : undefined)
  const [label, setLabel] = useState(fromLink ? fromLink.label.toLowerCase() : '')
  const show = (f: FlowFilters, text: string) => {
    setFilter({ ...f }); setLabel(text)
    setTimeout(() => document.getElementById('report-flows')?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 50)
  }
  return (
    <div className="space-y-6">
      <div>
        {!!r.group_findings?.length && <GroupFindings r={r} />}
        <h3 className={H3}>Predicted classes <span className="normal-case tracking-normal text-slate-400">(click a class to list its flows)</span></h3>
        <div className="overflow-x-auto rounded-lg border border-slate-200">
          <table className="min-w-full text-sm">
            <thead className="bg-slate-50 text-left text-xs uppercase tracking-wide text-slate-500">
              <tr><th className="px-3 py-2">Class</th><th className="px-3 text-right">Flows</th><th className="px-3 text-right">Share</th><th className="px-3 text-right">Avg. confidence</th>
                <th className="px-3">Severity <span className="normal-case">(heuristic)</span></th><th className="px-3 text-right">Status uncertain</th><th className="px-3 text-right">Type uncertain</th></tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {r.detections.by_class.map(c => (
                <tr key={c.class} className="hover:bg-slate-50">
                  <td className="px-3 py-2">
                    <button type="button" className="rounded focus:outline-none focus-visible:ring-2 focus-visible:ring-brand-500" aria-label={`Show ${c.class} flows`}
                      onClick={() => show(c.class === 'Normal' ? { traffic: 'safe' } : { attack_type: c.class }, `predicted ${c.class}`)}>
                      <ClassBadge name={c.class} severity={c.severity_heuristic} />
                    </button>
                  </td>
                  <td className="px-3 text-right tabular-nums">{num(c.count)}</td>
                  <td className="px-3 text-right tabular-nums">{pct(c.share)}</td>
                  <td className="px-3 text-right tabular-nums">{pct(c.average_confidence)}</td>
                  <td className="px-3"><SeverityBadge severity={c.severity_heuristic} /></td>
                  <td className="px-3 text-right tabular-nums">{num(c.status_uncertain)}</td>
                  <td className="px-3 text-right tabular-nums">{num(c.type_uncertain)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div>
        <h3 className={H3}>Uncertainty (90% shortlists)</h3>
        <div className="grid gap-3 sm:grid-cols-3">
          {UNCERTAINTY.map(u => (
            <button key={u.key} type="button" onClick={() => show({ uncertainty: u.key }, u.label.toLowerCase())}
              className="rounded-xl border border-slate-200 p-3 text-left hover:border-brand-300 hover:bg-brand-50/40 focus:outline-none focus-visible:ring-2 focus-visible:ring-brand-500">
              <p className="text-xs font-semibold text-slate-500">{u.label}</p>
              <p className="text-xl font-semibold tabular-nums text-slate-900">{num(u.count(r))}</p>
              <p className="mt-1 text-xs text-slate-500">{u.tip}</p>
            </button>
          ))}
        </div>
        <ul className="mt-3 list-disc space-y-1 pl-5 text-xs text-slate-600">
          {[...r.detections.notes, ...r.detections.shift_evidence].map(n => <li key={n}>{n}</li>)}
        </ul>
      </div>

      {r.detections.priority_flows.length > 0 && (
        <div>
          <h3 className={H3}>Flows to look at first <span className="normal-case tracking-normal text-slate-400">(heuristic severity, then confidence: a review order, not confirmed threats)</span></h3>
          <div className="overflow-x-auto rounded-lg border border-slate-200">
            <table className="min-w-full text-sm">
              <thead className="bg-slate-50 text-left text-xs uppercase tracking-wide text-slate-500">
                <tr><th className="px-3 py-2">Flow</th><th className="px-3">Predicted</th><th className="px-3">Confidence</th><th className="px-3">Shortlist</th><th className="px-3">proto / service</th><th className="px-3 text-right">Bytes src / dst</th></tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {r.detections.priority_flows.map(f => (
                  <tr key={f.flow_id} className="hover:bg-slate-50">
                    <td className="px-3 py-2"><button type="button" className="font-medium text-brand-700 hover:underline" onClick={() => onOpenFlow(f.flow_id)}>#{f.flow_index}</button></td>
                    <td className="px-3"><ClassBadge name={f.class} severity={f.severity_heuristic} /></td>
                    <td className="px-3 tabular-nums">{pct(f.confidence)}</td>
                    <td className="px-3 text-xs text-slate-600">{f.prediction_set.join(', ')}</td>
                    <td className="px-3 text-xs text-slate-600">{f.proto} / {f.service}</td>
                    <td className="whitespace-nowrap px-3 text-right tabular-nums text-slate-600">{num(f.sbytes)} / {num(f.dbytes)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      <div id="report-flows" className="scroll-mt-4">
        <h3 className={H3}>Flows {label && <span className="normal-case tracking-normal text-brand-700">· {label}</span>}</h3>
        {analysis ? <FlowTable analysisId={r.provenance.analysis.id} a={analysis} initial={filter} /> : <Spinner label="Loading flows…" />}
      </div>
    </div>
  )
}

/** TreeSHAP over all flows of each main predicted type: how often a measurement is one of a flow's three strongest supporting reasons. */
function GroupFindings({ r }: { r: ReportRecord }) {
  const groups = r.group_findings ?? []
  return (
    <div className="mb-6">
      <h3 className={H3}>Group findings <OriginTag origin="xai_attribution" /></h3>
      <div className="overflow-x-auto rounded-lg border border-slate-200">
        <table className="min-w-full text-sm">
          <thead className="bg-slate-50 text-left text-xs uppercase tracking-wide text-slate-500">
            <tr><th className="px-3 py-2">Predicted type</th><th className="px-3">Flows explained</th><th className="px-3">Most frequent supporting measurements</th><th className="px-3">Most frequent opposing</th></tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {groups.map(g => (
              <tr key={g.class} className="align-top">
                <td className="px-3 py-2"><ClassBadge name={g.class} /></td>
                <td className="whitespace-nowrap px-3 py-2 tabular-nums">{g.status === 'available' ? `${num(g.analysed)} of ${num(g.population)}` : g.status}</td>
                <td className="px-3 py-2">
                  {g.status !== 'available' ? <span className="text-xs text-rose-700">{g.reason}</span> : (
                    <ul className="space-y-0.5">
                      {g.top_supporting?.map(f => <li key={f.feature}><b className="font-medium">{f.name}</b>: {num(f.flows)} of {num(g.analysed)} flows
                        <span className="text-xs text-slate-500"> ({f.value_summary}{f.reference ? `; ${f.reference}` : ''})</span></li>)}
                    </ul>
                  )}
                </td>
                <td className="px-3 py-2 text-xs text-slate-600">{g.most_frequent_opposing?.map(f => `${f.name}: ${num(f.flows)} flows`).join(', ') || '–'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-1 text-xs text-slate-500">{groups.find(g => g.method)?.method}</p>
    </div>
  )
}

/** Quick look at one flow without leaving the report. */
export function FlowDrawer({ analysisId, flowId, onClose }: { analysisId: number; flowId: number; onClose: () => void }) {
  const flow = useQuery({ queryKey: ['flow', analysisId, flowId], queryFn: () => api.flow(analysisId, flowId) })
  const f = flow.data
  return (
    <Dialog label={f ? `Flow #${f.flow_index}` : 'Flow'} onClose={onClose}>
      {flow.isLoading ? <Spinner /> : flow.error ? <ErrorBox error={flow.error} /> : f && (
        <div className="space-y-3 text-sm">
          <div className="flex flex-wrap items-center gap-2"><ClassBadge name={f.predicted_class} severity={f.severity} /><ConfidenceBadge value={f.confidence} level={f.confidence_level} />
            <span className="text-xs text-slate-500">shortlist: {f.prediction_set.join(', ')}</span></div>
          <div>
            <p className={H3}>Probabilities (model output)</p>
            <ul className="space-y-1">
              {Object.entries(f.probabilities).sort((a, b) => b[1] - a[1]).slice(0, 4).map(([k, v]) => (
                <li key={k} className="flex items-center gap-2 text-xs"><span className="w-28">{k}</span>
                  <span className="h-2 flex-1 rounded bg-slate-100"><span className="block h-2 rounded bg-brand-500" style={{ width: `${v * 100}%` }} /></span>
                  <span className="w-12 text-right tabular-nums">{pct(v)}</span></li>
              ))}
            </ul>
          </div>
          <p className="text-xs text-slate-600">{f.proto} / {f.service}, state {f.state}, {num(f.sbytes)} / {num(f.dbytes)} bytes, {num(f.spkts)} / {num(f.dpkts)} packets.</p>
          <Link to={`/results/${analysisId}/flows/${flowId}`} className="inline-flex items-center gap-1 text-sm font-medium text-brand-600 hover:underline">Open the full flow page (all values and the explanation)<ExternalLink className="h-3.5 w-3.5" /></Link>
        </div>
      )}
    </Dialog>
  )
}

// ------------------------------------------------------------------------------------------------ C. Why this prediction
/** Diverging TreeSHAP bars: the top contributions plus the summed remainder, on one scale. */
export function XaiBars({ flow }: { flow: FlowCard }) {
  const x = flow.xai
  const rows = [...x.top.map(c => ({ label: c.name ?? c.feature, sub: `${c.feature} = ${c.value}`, v: c.contribution, ref: c.reference })),
    { label: `${x.other_features} other features`, sub: 'sum of the remaining contributions', v: x.remainder_of_other_features, ref: null }]
  const max = Math.max(...rows.map(r => Math.abs(r.v)), 1e-9)
  return (
    <div role="table" aria-label={`TreeSHAP contributions for flow ${flow.flow_index}`} className="space-y-1.5">
      {rows.map((r, i) => (
        <div role="row" key={i} className="grid grid-cols-[minmax(0,11rem)_1fr_4.5rem] items-center gap-2 text-xs sm:grid-cols-[minmax(0,16rem)_1fr_4.5rem]">
          <div role="cell" className="min-w-0"><p className="truncate font-medium text-slate-800" title={r.label}>{r.label}</p><p className="truncate text-[11px] text-slate-500" title={r.sub}>{r.sub}</p></div>
          <div role="cell" className="relative h-3 rounded bg-slate-50" aria-hidden>
            <span className="absolute inset-y-0 left-1/2 w-px bg-slate-300" />
            <span className={clsx('absolute inset-y-0 rounded', r.v >= 0 ? 'left-1/2 bg-rose-500' : 'right-1/2 bg-sky-500', i === rows.length - 1 && 'opacity-50')}
              style={{ width: `${(Math.abs(r.v) / max) * 50}%` }} />
          </div>
          <div role="cell" className="text-right tabular-nums text-slate-700">{r.v >= 0 ? '+' : '−'}{Math.abs(r.v).toFixed(3)}</div>
        </div>
      ))}
      <p className="pt-1 text-[11px] text-slate-500"><span className="text-rose-600">Red</span> supports {x.class}, <span className="text-sky-600">blue</span> opposes it. Units: {x.units}.</p>
    </div>
  )
}

function StatementList({ items, empty }: { items: Statement[]; empty?: string }) {
  if (!items.length) return empty ? <p className="text-xs text-slate-500">{empty}</p> : null
  return (
    <ul className="space-y-1.5 text-sm">
      {items.map(i => (
        <li key={i.id} className="leading-relaxed text-slate-800">
          <span className="mr-1 text-[10px] font-semibold text-slate-400">{i.id}</span>{i.text}<Cites ids={i.cites} support={i.support} /><OriginTag origin={i.origin} />
          {i.round === 2 && <span className="ml-1 text-[10px] text-slate-400">(repair round)</span>}
        </li>
      ))}
    </ul>
  )
}

export function FlowCardView({ flow, analysisId }: { flow: FlowCard; analysisId: number }) {
  const p = flow.prediction, x = flow.xai, llm = flow.llm
  const [showRejected, setShowRejected] = useState(false)
  const topProbs = Object.entries(p.probabilities).sort((a, b) => b[1] - a[1]).slice(0, 3)
  return (
    <article className="space-y-5" aria-label={`Flow ${flow.flow_index}`}>
      <div className="flex flex-wrap items-center gap-2">
        <h3 className="text-base font-semibold text-slate-900">Flow #{flow.flow_index}</h3>
        <ClassBadge name={p.class} severity={p.severity_heuristic} /><ConfidenceBadge value={p.confidence} level={p.confidence_level} />
        <OriginTag origin="model_output" />
        <Link to={`/results/${analysisId}/flows/${flow.flow_id}`} className="ml-auto inline-flex items-center gap-1 text-xs font-medium text-brand-600 hover:underline">Flow page<ExternalLink className="h-3 w-3" /></Link>
      </div>
      <p className="rounded-lg bg-slate-50 px-3 py-2 text-xs text-slate-600"><b>Why this flow is shown:</b> {flow.selection}</p>
      <div className="grid gap-4 lg:grid-cols-2">
        <div>
          <p className={H3}>Prediction</p>
          <p className="text-sm text-slate-700">90% shortlist: <b>{p.prediction_set.join(', ')}</b>{p.normal_in_set && p.malicious && <span className="ml-1 text-amber-700">(Normal is still possible)</span>}</p>
          <ul className="mt-2 space-y-1">
            {topProbs.map(([k, v]) => <li key={k} className="flex justify-between text-xs"><span>{k}</span><span className="tabular-nums">{prob(v)}</span></li>)}
          </ul>
        </div>
        <div>
          <p className={H3}>Observed values <span className="normal-case tracking-normal text-slate-400">(percentiles from the training rows)</span></p>
          <div className="max-h-48 overflow-y-auto rounded border border-slate-100">
            <table className="min-w-full text-xs">
              <tbody className="divide-y divide-slate-100">
                {flow.values.map(v => <tr key={v.feature}><td className="px-2 py-1 text-slate-600" title={v.definition}>{v.name}</td><td className="px-2 py-1 font-medium tabular-nums">{v.value}</td><td className="px-2 py-1 text-slate-500">{v.reference ?? ''}</td></tr>)}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <div>
        {!!flow.reasoning?.length && (
          <div className="mb-5 rounded-xl border border-violet-200 bg-violet-50/40 p-4">
            <p className="text-sm font-semibold text-violet-900">Reasoning</p>
            <ol className="mt-2 list-decimal space-y-1.5 pl-5 text-sm text-slate-800">
              {flow.reasoning.map(r => (
                <li key={r.slot}><b className="font-medium">{r.title}:</b> {r.text}<Cites ids={r.cites} />
                  {r.origin === 'llm_synthesis' ? <span className="ml-1 rounded bg-violet-100 px-1 text-[10px] font-medium text-violet-700">AI, checked</span> : <OriginTag origin={r.origin} />}</li>
              ))}
            </ol>
          </div>
        )}
        <p className={H3}>Model evidence: TreeSHAP <OriginTag origin="xai_attribution" /></p>
        {x.status === 'available' ? (
          <>
            <XaiBars flow={flow} />
            <p className="mt-2 rounded-lg bg-indigo-50/60 px-3 py-2 text-xs text-indigo-900">
              Exact decomposition: base value {x.base_value.toFixed(3)} + sum of contributions = margin {x.margin.toFixed(3)}; margin ÷ temperature {x.temperature.toFixed(3)} + class offset {x.class_offset} = deployed class score {x.deployed_class_score.toFixed(3)}.
              {x.one_sided && ' All shown contributions point the same way.'} {x.how_to_read}
            </p>
          </>
        ) : <p className="text-xs text-slate-500">TreeSHAP status: {x.status}</p>}
        {flow.interpretation && <p className="mt-2 text-sm text-slate-800"><b>Interpretation.</b> {flow.interpretation}<OriginTag origin="xai_attribution" /></p>}
        {(!!flow.alternatives?.length || flow.missing_confirmation) && (
          <p className="mt-1 text-xs text-slate-600"><b>Alternatives and what is missing.</b>{' '}
            {flow.alternatives?.length ? `Other classes the model did not rule out: ${flow.alternatives.map(a => `${a.class} ${prob(a.probability)}${a.in_set ? ' (on shortlist)' : ''}`).join(', ')}. ` : 'No other class above 5%. '}
            {flow.missing_confirmation}</p>
        )}
        <p className="mt-1 text-xs text-slate-500">LIME: <b>{flow.lime.status}</b>. {flow.lime.reason}</p>
      </div>

      <div>
        <p className={H3}>Evidence facts (computed by code; the LLM may only cite these)</p>
        <ul className="space-y-1 text-xs">
          {Object.entries(flow.facts).map(([k, v]) => <li key={k} className="flex gap-2"><span className="w-16 shrink-0 font-semibold text-slate-500">{k}</span><span className="text-slate-700">{v}</span></li>)}
          {!Object.keys(flow.facts).length && <li className="text-slate-500">No LLM step for this flow, so no evidence ids were assigned.</li>}
        </ul>
      </div>

      {llm ? (
        <div className="space-y-4 rounded-xl border border-violet-200 bg-violet-50/30 p-4">
          <p className="text-xs text-slate-600">
            <b>Local LLM</b> ({llm.model ?? '–'}): status <b>{STATUS_LABEL[llm.status] ?? llm.status}</b> · {llm.checked} statements checked · {llm.removed.length} rejected · {llm.seconds.toFixed(0)} s.
            {llm.note && <> {llm.note}</>}
          </p>
          {!flow.reasoning?.length && (
          <div><p className="mb-2 flex items-center gap-2 text-sm font-semibold text-violet-800">AI explanation: why the model predicted {p.class}<span className="text-[11px] font-normal text-slate-500">(local language model; every number and direction checked against the evidence)</span></p>
            <StatementList items={llm.explanation} empty={llm.status === 'generated' ? 'No explanation sentence passed the checks; the TreeSHAP evidence above is the explanation.' : 'No LLM explanation; the evidence above is exact.'} /></div>
          )}
          {(llm.techniques.length > 0 || llm.context.length > 0) && <div><p className={H3}>ATT&CK techniques and context (from retrieved passages)</p><StatementList items={[...llm.techniques, ...llm.context]} /></div>}
          {llm.actions.length > 0 && <div><p className={H3}>Suggested actions (merged into the Actions tab)</p><StatementList items={llm.actions} /></div>}
          {llm.mapped_techniques.length > 0 && (
            <div>
              <p className={H3}>Class-to-technique mapping <OriginTag origin="heuristic" /></p>
              <ul className="space-y-1 text-xs text-slate-700">
                {llm.mapped_techniques.map(m => <li key={m.class}><b>{m.class}</b>{m.predicted ? ' (predicted)' : ''}: {m.techniques.length ? m.techniques.map(t => `${t.id} ${t.name ?? ''}`.trim()).join('; ') : 'no ATT&CK match'} <span className="text-slate-400">· {m.note}</span></li>)}
              </ul>
            </div>
          )}
          {llm.removed.length > 0 && (
            <div>
              <button type="button" className="inline-flex items-center gap-1 text-xs font-medium text-rose-700 hover:underline" aria-expanded={showRejected} onClick={() => setShowRejected(v => !v)}>
                <ChevronDown className={clsx('h-3.5 w-3.5 transition', showRejected && 'rotate-180')} />{llm.removed.length} statement(s) rejected by the checks
              </button>
              {showRejected && (
                <ul className="mt-2 space-y-1 text-xs text-rose-800">
                  {llm.removed.map((x, k) => <li key={k} className="flex gap-1.5"><XCircle className="mt-0.5 h-3.5 w-3.5 shrink-0" /><span><b>{x.reason}</b> ({x.part}, round {x.round ?? 1}) — “{String(x.text).slice(0, 220)}”</span></li>)}
                </ul>
              )}
            </div>
          )}
        </div>
      ) : <p className="text-xs text-slate-500">No LLM step for this flow (predicted Normal): the evidence above is the explanation.</p>}
      <p className="text-[11px] text-slate-400">{flow.ground_truth_note}</p>
    </article>
  )
}

export function WhyTab({ r, flowId, onFlow }: { r: ReportRecord; flowId: string | null; onFlow: (id: string) => void }) {
  const flow = r.flows.find(f => f.id === flowId) ?? r.flows[0]
  if (!flow) return <p className="text-sm text-slate-500">No flow was explained in this report.</p>
  return (
    <div className="space-y-4">
      <div role="group" aria-label="Illustrative flows" className="flex flex-wrap gap-2">
        {r.flows.map(f => (
          <button key={f.id} type="button" onClick={() => onFlow(f.id)} aria-pressed={f.id === flow.id}
            className={clsx('rounded-lg border px-3 py-1.5 text-left text-xs focus:outline-none focus-visible:ring-2 focus-visible:ring-brand-500',
              f.id === flow.id ? 'border-brand-500 bg-brand-50 text-brand-800' : 'border-slate-200 text-slate-700 hover:bg-slate-50')}>
            <span className="font-semibold">#{f.flow_index}</span> · {f.prediction.class} · {prob(f.prediction.confidence)}
          </button>
        ))}
      </div>
      <FlowCardView flow={flow} analysisId={r.provenance.analysis.id} />
    </div>
  )
}

// ------------------------------------------------------------------------------------------------ D. Sources
export function SourcesTab({ r, onOpen }: { r: ReportRecord; onOpen: (id: string) => void }) {
  const [family, setFamily] = useState('')
  const [text, setText] = useState('')
  const families = useMemo(() => [...new Set(r.sources.register.map(s => s.source))], [r])
  const shown = r.sources.register.filter(s => (!family || s.source === family) && (!text || `${s.id} ${s.location} ${s.text}`.toLowerCase().includes(text.toLowerCase())))
  return (
    <div className="space-y-5">
      <p className="text-sm text-slate-700">{r.sources.note}</p>
      {!!r.sources.documents?.length && (
        <div className="overflow-x-auto rounded-lg border border-slate-200">
          <table className="min-w-full text-xs">
            <thead className="bg-slate-50 text-left uppercase tracking-wide text-slate-500"><tr><th className="px-3 py-2">Source</th><th className="px-3">Publisher and title</th><th className="px-3">Version</th><th className="px-3">Retrieved</th><th className="px-3 text-right">Passages</th></tr></thead>
            <tbody className="divide-y divide-slate-100">
              {r.sources.documents.map(d => (
                <tr key={d.source}><td className="px-3 py-2 font-medium">{d.source}</td>
                  <td className="px-3">{d.publisher} · {safeUrl(d.download_url) ? <a className="text-brand-600 hover:underline" href={safeUrl(d.download_url)!} target="_blank" rel="noopener noreferrer">{d.title}</a> : d.title}</td>
                  <td className="px-3 text-slate-600">{d.version ?? (d.pdf_created ? `PDF created ${d.pdf_created}` : '–')}</td><td className="px-3 text-slate-600">{d.retrieved ?? '–'}</td>
                  <td className="px-3 text-right tabular-nums">{d.passages}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {r.sources.register.length === 0 ? <p className="text-sm text-slate-500">No passages were retrieved (for example, every flow was predicted Normal).</p> : (
        <>
          <div className="flex flex-wrap items-center gap-2">
            <div className="relative min-w-0 flex-1 basis-56">
              <Search className="pointer-events-none absolute left-2.5 top-2.5 h-4 w-4 text-slate-400" aria-hidden />
              <input value={text} onChange={e => setText(e.target.value)} placeholder="Search passages" aria-label="Search passages"
                className="w-full rounded-lg border border-slate-300 py-1.5 pl-8 pr-3 text-sm focus:border-brand-500 focus:outline-none" />
            </div>
            <select aria-label="Source family" value={family} onChange={e => setFamily(e.target.value)} className="rounded-lg border border-slate-300 bg-white px-2.5 py-1.5 text-sm">
              <option value="">All sources ({r.sources.register.length})</option>{families.map(f => <option key={f}>{f}</option>)}
            </select>
          </div>
          <ul className="space-y-2">
            {shown.map(s => (
              <li key={s.id} className="rounded-lg border border-slate-200 p-3">
                <div className="flex flex-wrap items-center gap-2 text-xs">
                  <span className="rounded bg-sky-50 px-1.5 font-semibold text-sky-700 ring-1 ring-inset ring-sky-200">{s.id}</span>
                  <span className="font-medium text-slate-800">{s.source}</span><span className="text-slate-500">{s.location}</span>
                  <span className="text-slate-400">· {s.retrieval.method}{s.retrieval.score !== null ? ` · score ${s.retrieval.score}` : ''} · cited by {s.used_by.join(', ')}</span>
                </div>
                <p className="mt-1.5 line-clamp-3 text-sm text-slate-700">{s.excerpt || s.text}</p>
                <button type="button" className="mt-1 text-xs font-medium text-brand-600 hover:underline" onClick={() => onOpen(s.id)}>Show the full passage and its metadata</button>
              </li>
            ))}
            {!shown.length && <li className="text-sm text-slate-500">No passage matches.</li>}
          </ul>
        </>
      )}
    </div>
  )
}

// ------------------------------------------------------------------------------------------------ E. Actions
const PHASES = ['Verify', 'Contain if confirmed', 'Recover'] as const

function ActionEditor({ reportId, actionId, state }: { reportId: number; actionId: string; state?: ActionState }) {
  const queryClient = useQueryClient()
  const toast = useToast()
  const [status, setStatus] = useState<ActionStatus>(state?.status ?? 'open')
  const [note, setNote] = useState('')
  const [history, setHistory] = useState(false)
  const save = useMutation({
    mutationFn: () => api.updateAction(reportId, actionId, status, note),
    onSuccess: s => {
      queryClient.setQueryData<Record<string, ActionState>>(['report-actions', reportId], old => ({ ...(old ?? {}), [actionId]: s }))
      setNote(''); toast({ tone: 'success', title: `${actionId} saved`, body: `${ACTION_LABEL[s.status]}${s.note ? ` · ${s.note}` : ''}` })
    },
  })
  return (
    <div className="space-y-1.5">
      <div className="flex flex-wrap items-center gap-1.5">
        <select aria-label={`Status of ${actionId}`} value={status} onChange={e => setStatus(e.target.value as ActionStatus)} className="rounded border border-slate-300 bg-white px-1.5 py-1 text-xs">
          {Object.entries(ACTION_LABEL).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
        </select>
        <input aria-label={`Note for ${actionId}`} value={note} onChange={e => setNote(e.target.value)} maxLength={500} placeholder="Note (optional)" className="min-w-0 flex-1 rounded border border-slate-300 px-2 py-1 text-xs" />
        <Button variant="secondary" className="px-2 py-1 text-xs" loading={save.isPending} onClick={() => save.mutate()}>Save</Button>
      </div>
      {save.error && <ErrorBox error={save.error} />}
      {state && (
        <div className="text-[11px] text-slate-500">
          <button type="button" className="inline-flex items-center gap-1 hover:underline" aria-expanded={history} onClick={() => setHistory(v => !v)}>
            <History className="h-3 w-3" />{ACTION_LABEL[state.status]} by {state.by}, {when(state.at)} ({state.history.length} change{state.history.length === 1 ? '' : 's'})
          </button>
          {history && <ul className="mt-1 space-y-0.5 pl-4">{state.history.map((h, i) => <li key={i}>{when(h.at)} · {ACTION_LABEL[h.status]} · {h.by}{h.note ? ` · “${h.note}”` : ''}</li>)}</ul>}
        </div>
      )}
    </div>
  )
}

export function ActionsTab({ r }: { r: ReportRecord }) {
  const [phase, setPhase] = useState<string>('')
  const states = useQuery({ queryKey: ['report-actions', r.report_id], queryFn: () => api.actionStates(r.report_id) })
  const rows = r.actions.rows.filter(a => !phase || a.phase === phase)
  return (
    <div className="space-y-5">
      <p className="flex items-start gap-2 rounded-lg bg-amber-50 px-3 py-2 text-xs text-amber-900"><AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />{r.actions.note}</p>
      {r.actions.rows.length === 0 ? <p className="text-sm text-slate-500">No actions: no flow was predicted as an attack.</p> : (
        <>
          <div role="group" aria-label="Filter by phase" className="flex flex-wrap items-center gap-2 text-xs">
            <Filter className="h-3.5 w-3.5 text-slate-400" aria-hidden />
            {['', ...PHASES].map(p => (
              <button key={p || 'all'} type="button" aria-pressed={phase === p} onClick={() => setPhase(p)}
                className={clsx('rounded-full px-2.5 py-1 ring-1 ring-inset', phase === p ? 'bg-brand-600 text-white ring-brand-600' : 'text-slate-600 ring-slate-200 hover:bg-slate-50')}>
                {p || 'All'} ({p ? r.actions.rows.filter(a => a.phase === p).length : r.actions.rows.length})
              </button>
            ))}
          </div>
          {states.error && <ErrorBox error={states.error} />}
          <ol className="space-y-3">
            {rows.map(a => (
              <li key={a.id} className="rounded-xl border border-slate-200 p-4">
                <div className="flex flex-wrap items-center gap-2 text-xs">
                  <span className="font-semibold text-slate-500">{a.id}</span>
                  <span className={clsx('rounded px-1.5 py-px font-semibold', a.phase === 'Verify' ? 'bg-emerald-50 text-emerald-700' : a.phase === 'Recover' ? 'bg-sky-50 text-sky-700' : 'bg-rose-50 text-rose-700')}>{a.phase}</span>
                  {a.title && <span className="font-semibold text-slate-800">{a.title}</span>}<span className="text-slate-500">{a.scope}</span><OriginTag origin={a.origin} />
                </div>
                <p className="mt-2 text-sm text-slate-800">{a.text}<Cites ids={a.cites} support={a.support} /></p>
                <p className="mt-1 text-xs text-slate-600"><b>Why:</b> {a.reason}</p>
                <dl className="mt-2 grid gap-x-4 gap-y-1 text-xs sm:grid-cols-2">
                  {([['Requires', a.applicability], ['Owner', a.owner], ['Possible disruption', a.disruption], ['How to check', a.validation], ...(a.rollback ? [['Undo', a.rollback]] : [])] as [string, string][]).map(([k, v]) => (
                    <div key={k}><dt className="inline text-slate-500">{k}: </dt><dd className="inline text-slate-700">{v}</dd></div>
                  ))}
                </dl>
                <div className="mt-3 border-t border-slate-100 pt-3">
                  {states.isLoading ? <Spinner label="Loading checklist…" /> : <ActionEditor key={states.data?.[a.id]?.at ?? 'new'} reportId={r.report_id} actionId={a.id} state={states.data?.[a.id]} />}
                </div>
              </li>
            ))}
          </ol>
        </>
      )}
      {r.actions.prevention.length > 0 && (
        <div>
          <h3 className={H3}>General prevention guidance (retrieved; not specific to these flows)</h3>
          <ul className="space-y-1.5 text-sm">
            {r.actions.prevention.map((p, i) => <li key={i} className="text-slate-700"><span className="text-xs text-slate-500">{p.scope}: </span>{p.text}<Cites ids={p.cites} /><OriginTag origin={p.origin} /></li>)}
          </ul>
        </div>
      )}
    </div>
  )
}

// ------------------------------------------------------------------------------------------------ F. Limits, model and provenance
export function LimitsTab({ r }: { r: ReportRecord }) {
  const m = r.appendix.model_card.evaluation
  const prov = r.provenance
  return (
    <div className="space-y-6">
      <div>
        <h3 className={H3}>What this report cannot tell you</h3>
        <ul className="list-disc space-y-1 pl-5 text-sm text-slate-700">{r.limits.unknown.map(u => <li key={u}>{u}</li>)}</ul>
        <p className="mt-2 text-xs text-slate-500">Human review: {r.limits.human_review}</p>
      </div>
      <div>
        <h3 className={H3}>Statement origins</h3>
        <ul className="grid gap-1 text-xs sm:grid-cols-2">
          {Object.entries(r.limits.origins).map(([k, v]) => <li key={k}><OriginTag origin={k as keyof typeof ORIGIN_LABEL} /> <span className="text-slate-600">{v}</span></li>)}
        </ul>
      </div>
      {r.evaluation && (
        <Card title="Labels found in the uploaded file (kept separate)" subtitle={r.evaluation.scope}>
          <div className="grid gap-3 text-sm sm:grid-cols-3">
            <p>Attack vs Normal correct: <b className="tabular-nums">{num(r.evaluation.attack_vs_normal_correct)} / {num(r.evaluation.labelled_flows)}</b></p>
            <p>Exact class correct: <b className="tabular-nums">{num(r.evaluation.exact_class_correct)}</b></p>
            <p>True class on the 90% shortlist: <b className="tabular-nums">{num(r.evaluation.truth_in_set)}</b></p>
          </div>
        </Card>
      )}
      {m && (
        <div>
          <h3 className={H3}>Model benchmark <span className="normal-case tracking-normal text-slate-400">({r.appendix.note})</span></h3>
          <div className="grid gap-3 text-sm sm:grid-cols-2 lg:grid-cols-4">
            {[['Attack detection F1', pct(m.binary.f1)], ['Attack recall', pct(m.binary.recall)], ['False alarms on Normal', pct(m.binary.false_positive_rate)],
              ['10-class accuracy', pct(m.accuracy)], ['Macro F1 (10 classes)', m.macro.f1.toFixed(3)], ['Top-2 accuracy', m.top2_accuracy !== undefined ? pct(m.top2_accuracy) : '–'],
              ['Best possible accuracy on this file', m.ceiling ? pct(m.ceiling.accuracy) : '–'], ['Best possible macro F1', m.ceiling ? m.ceiling.macro_f1.toFixed(3) : '–']].map(([k, v]) => (
              <div key={k} className="rounded-lg border border-slate-200 px-3 py-2"><p className="text-xs text-slate-500">{k}</p><p className="font-semibold tabular-nums text-slate-900">{v}</p></div>
            ))}
          </div>
          <p className="mt-1 text-xs text-slate-500">{m.split}. Full details on the <Link className="text-brand-600 hover:underline" to="/model">Model page</Link>.</p>
        </div>
      )}
      <div>
        <h3 className={H3}>Provenance <InfoTip text="Hashes and versions that identify exactly which input, model, knowledge base and code produced this report." /></h3>
        <dl className="grid grid-cols-1 gap-x-4 gap-y-1 break-words text-xs sm:grid-cols-[12rem_1fr]">
          {([
            ['Report', `#${r.report_id}, schema ${r.schema_version}, template ${r.template_version}, generated ${r.generated_at}`],
            ['Input file sha256', prov.analysis.input_sha256 ?? 'not recorded (analysis created before input hashing)'],
            ['Detector', `${prov.detector.name} (bundle ${prov.detector.bundle_created ?? '–'}, commit ${prov.detector.bundle_source_commit?.slice(0, 10) ?? '–'})`],
            ...Object.entries(prov.detector.files_sha256).map(([k, v]) => [`  ${k}`, v] as [string, string]),
            ['Calibration', `temperature ${prov.calibration.temperature.toFixed(4)}, conformal alpha ${prov.calibration.alpha}`],
            ['Explanation', `${prov.explanation.method}. ${prov.explanation.exactness}`],
            ['Knowledge base', `downloaded ${prov.knowledge.downloaded ?? '–'}, chunks sha256 ${prov.knowledge.chunks_sha256 ?? '–'}`],
            ['LLM', `${prov.llm.model} (${prov.llm.runtime}; ${Object.entries(prov.llm.settings).map(([k, v]) => `${k} ${v}`).join(', ')}). ${prov.llm.note}`],
            ['Application commit', prov.application.code_commit ?? '–'],
            ['Timings (s)', Object.entries(r.timings).map(([k, v]) => `${k} ${v}`).join(' · ')],
          ] as [string, string][]).map(([k, v]) => <div key={k} className="contents"><dt className="whitespace-pre text-slate-500">{k}</dt><dd className="font-mono text-[11px] text-slate-700">{v}</dd></div>)}
        </dl>
      </div>
    </div>
  )
}
