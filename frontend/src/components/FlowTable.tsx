import { keepPreviousData, useQuery } from '@tanstack/react-query'
import { ChevronLeft, ChevronRight, Search, X } from 'lucide-react'
import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api, type AnalysisData, type FlowFilters } from '../lib/api'
import { CONFIDENCE_TIP, num } from '../lib/format'
import { Button, ClassBadge, ConfidenceBadge, ErrorBox, InfoTip, PredictionBadge, SeverityBadge, Spinner } from './ui'

const select = 'rounded-lg border border-slate-300 bg-white px-2.5 py-1.5 text-sm text-slate-700 focus:border-brand-500 focus:outline-none'

/** Searchable, filterable, paginated table of every flow prediction of one analysis (server-side paging). */
export default function FlowTable({ analysisId, a, initial }: { analysisId: number; a: AnalysisData; initial?: FlowFilters }) {
  const navigate = useNavigate()
  const [filters, setFilters] = useState<FlowFilters>({ page: 1, page_size: 25, ...initial })
  const [text, setText] = useState('')
  // A link from elsewhere on the page (e.g. "show them") replaces all filters, so counts match what the link promised.
  useEffect(() => { if (initial) { setText(''); setFilters(f => ({ page: 1, page_size: f.page_size, ...initial })) } }, [initial])
  useEffect(() => {
    const t = setTimeout(() => setFilters(f => (f.search === (text || undefined) ? f : { ...f, search: text || undefined, page: 1 })), 300)
    return () => clearTimeout(t)
  }, [text])
  const flows = useQuery({ queryKey: ['flows', analysisId, filters], queryFn: () => api.flows(analysisId, filters), placeholderData: keepPreviousData })
  const set = (patch: FlowFilters) => setFilters(f => ({ ...f, ...patch, page: 1 }))
  const pages = flows.data ? Math.max(1, Math.ceil(flows.data.total / (filters.page_size ?? 25))) : 1
  const active = !!(filters.traffic || filters.attack_type || filters.severity || filters.confidence || filters.review || filters.search || filters.uncertainty)
  const hasLabels = !!a.evaluation

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <div className="relative min-w-0 flex-1 basis-56">
          <Search className="pointer-events-none absolute left-2.5 top-2.5 h-4 w-4 text-slate-400" aria-hidden />
          <input value={text} onChange={e => setText(e.target.value)} placeholder="Search flow # or attack type" aria-label="Search flows"
            className="w-full rounded-lg border border-slate-300 py-1.5 pl-8 pr-3 text-sm focus:border-brand-500 focus:outline-none" />
        </div>
        <select aria-label="Traffic" className={select} value={filters.traffic ?? ''} onChange={e => set({ traffic: e.target.value || undefined })}>
          <option value="">All traffic</option><option value="safe">Predicted Normal</option><option value="malicious">Predicted attack</option>
        </select>
        <select aria-label="Attack type" className={select} value={filters.attack_type ?? ''} onChange={e => set({ attack_type: e.target.value || undefined, traffic: e.target.value ? undefined : filters.traffic })}>
          <option value="">All attack types</option>{a.attack_distribution.map(d => <option key={d.attack_type} value={d.attack_type}>{d.attack_type} ({num(d.count)})</option>)}
        </select>
        <select aria-label="Severity" className={select} value={filters.severity ?? ''} onChange={e => set({ severity: e.target.value || undefined })}>
          <option value="">All severities</option>{['Critical', 'High', 'Medium', 'Low'].map(s => <option key={s}>{s}</option>)}
        </select>
        <select aria-label="Confidence" className={select} value={filters.confidence ?? ''} onChange={e => set({ confidence: e.target.value || undefined })}>
          <option value="">All confidence</option><option value="high">High (≥ 90%)</option><option value="medium">Medium (70–89.9%)</option><option value="low">Low (&lt; 70%)</option>
        </select>
        <select aria-label="Uncertainty" className={select} value={filters.uncertainty ?? ''} onChange={e => set({ uncertainty: (e.target.value || undefined) as FlowFilters['uncertainty'] })}>
          <option value="">Any shortlist</option><option value="status">Attack status uncertain</option><option value="type">Attack type uncertain</option><option value="none">Single-class shortlist</option>
        </select>
        {filters.review && (
          <button className="inline-flex items-center gap-1 rounded-full bg-amber-50 px-2.5 py-1 text-xs font-medium text-amber-800 ring-1 ring-amber-200" onClick={() => set({ review: undefined })}>
            Needs review (confidence &lt; 70%)<X className="h-3 w-3" />
          </button>
        )}
        {active && <Button variant="ghost" className="px-2 py-1.5 text-xs" onClick={() => { setText(''); setFilters({ page: 1, page_size: filters.page_size }) }}><X className="h-3.5 w-3.5" />Clear</Button>}
      </div>
      <ErrorBox error={flows.error} />
      {flows.isLoading ? <Spinner label="Loading flows…" /> : (
        <div className="overflow-x-auto rounded-lg border border-slate-200">
          <table className="min-w-full text-sm">
            <thead className="bg-slate-50 text-left text-xs uppercase tracking-wide text-slate-500">
              <tr>
                <th className="px-3 py-2.5">Flow</th><th className="px-3">Protocol</th><th className="px-3">Service</th><th className="px-3">State</th>
                <th className="px-3 text-right">Bytes (src / dst)</th><th className="px-3">Prediction</th><th className="px-3">Attack type</th>
                <th className="px-3"><span className="inline-flex items-center gap-1">Confidence<InfoTip text={CONFIDENCE_TIP} /></span></th><th className="px-3">Severity</th>
                {hasLabels && <th className="px-3">True label</th>}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {flows.data?.items.map(f => (
                <tr key={f.id} onClick={() => navigate(`/results/${analysisId}/flows/${f.id}`)} className="cursor-pointer hover:bg-slate-50" title="Open flow details">
                  <td className="px-3 py-2 font-medium tabular-nums text-brand-700">#{f.flow_index}</td>
                  <td className="px-3 text-slate-700">{f.proto}</td><td className="px-3 text-slate-700">{f.service}</td><td className="px-3 text-slate-700">{f.state}</td>
                  <td className="whitespace-nowrap px-3 text-right tabular-nums text-slate-600">{num(f.sbytes)} / {num(f.dbytes)}</td>
                  <td className="px-3"><PredictionBadge malicious={f.prediction === 'Malicious'} /></td>
                  <td className="px-3">{f.attack_type ? <ClassBadge name={f.attack_type} severity={f.severity} /> : <span className="text-xs text-slate-400">–</span>}</td>
                  <td className="px-3"><ConfidenceBadge value={f.confidence} level={f.confidence_level} /></td>
                  <td className="px-3"><SeverityBadge severity={f.severity} /></td>
                  {hasLabels && <td className="px-3 text-xs"><span className={f.ground_truth === f.predicted_class ? 'text-emerald-700' : 'text-rose-600'}>{f.ground_truth ?? '–'}</span></td>}
                </tr>
              ))}
              {flows.data?.items.length === 0 && <tr><td colSpan={hasLabels ? 10 : 9} className="px-3 py-10 text-center text-sm text-slate-500">No flows match these filters.</td></tr>}
            </tbody>
          </table>
        </div>
      )}
      <div className="flex flex-wrap items-center justify-between gap-3 text-sm text-slate-500">
        <span>{flows.data ? `${num(flows.data.total)} flow${flows.data.total === 1 ? '' : 's'}${active ? ' match' : ''}` : ''}</span>
        <div className="flex items-center gap-2">
          <select aria-label="Rows per page" className={select} value={filters.page_size} onChange={e => set({ page_size: Number(e.target.value) })}>
            {[25, 50, 100].map(n => <option key={n} value={n}>{n} / page</option>)}
          </select>
          <Button variant="secondary" className="px-2" disabled={(filters.page ?? 1) <= 1} onClick={() => setFilters(f => ({ ...f, page: (f.page ?? 1) - 1 }))} aria-label="Previous page"><ChevronLeft className="h-4 w-4" /></Button>
          <span className="tabular-nums">Page {filters.page} of {num(pages)}</span>
          <Button variant="secondary" className="px-2" disabled={(filters.page ?? 1) >= pages} onClick={() => setFilters(f => ({ ...f, page: (f.page ?? 1) + 1 }))} aria-label="Next page"><ChevronRight className="h-4 w-4" /></Button>
        </div>
      </div>
      <p className="text-xs text-slate-400">The UNSW-NB15 training and testing files contain no IP addresses or ports, so flows are identified by their row number in the uploaded file.</p>
    </div>
  )
}
