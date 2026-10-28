import { Bar, BarChart, CartesianGrid, Cell, Layer, Pie, PieChart, Rectangle, ResponsiveContainer, Sankey, Tooltip, XAxis, YAxis } from 'recharts'
import { useState } from 'react'
import type { AnalysisData, AttackShare, ConfidenceLevel, Severity } from '../lib/api'
import { COLORS, CONFIDENCE_COLORS, CONFIDENCE_TIP, SEVERITY_COLORS, num, pct } from '../lib/format'
import { InfoTip } from './ui'

const tooltipStyle = { borderRadius: 8, border: '1px solid #e2e8f0', fontSize: 12, boxShadow: '0 4px 12px rgba(15,23,42,.08)' }

/** Safe vs malicious donut: answers "how much traffic is safe and how much is dangerous?" */
export function TrafficDonut({ safe, malicious }: { safe: number; malicious: number }) {
  const total = safe + malicious
  const data = [{ name: 'Predicted Normal', value: safe, color: COLORS.safe }, { name: 'Predicted attack', value: malicious, color: COLORS.threat }]
  return (
    <div className="flex flex-col items-center gap-4 sm:flex-row sm:justify-center">
      <div className="relative h-48 w-48 shrink-0">
        <ResponsiveContainer>
          <PieChart>
            <Pie data={total ? data : [{ name: 'none', value: 1, color: COLORS.grid }]} dataKey="value" innerRadius="64%" outerRadius="100%" startAngle={90} endAngle={-270} stroke="#fff" strokeWidth={2} isAnimationActive={false}>
              {(total ? data : [{ color: COLORS.grid }]).map((d, i) => <Cell key={i} fill={d.color} />)}
            </Pie>
            {total > 0 && <Tooltip contentStyle={tooltipStyle} formatter={(v: number, n: string) => [`${num(v)} flows (${pct(v / total)})`, n]} />}
          </PieChart>
        </ResponsiveContainer>
        <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-2xl font-semibold tabular-nums text-rose-600">{total ? pct(malicious / total) : '–'}</span>
          <span className="text-xs text-slate-500">malicious</span>
        </div>
      </div>
      <ul className="w-full max-w-56 space-y-3 text-sm">
        {data.map(d => (
          <li key={d.name}>
            <div className="flex items-center justify-between gap-3">
              <span className="flex items-center gap-2 text-slate-700"><span className="h-3 w-3 rounded-sm" style={{ backgroundColor: d.color }} />{d.name}</span>
              <span className="font-semibold tabular-nums text-slate-900">{num(d.value)}</span>
            </div>
            <p className="ml-5 text-xs text-slate-500">{total ? pct(d.value / total) : '–'} of {num(total)} flows</p>
          </li>
        ))}
      </ul>
    </div>
  )
}

/** Horizontal bars of detected attack types (never Normal), coloured by severity. */
export function AttackBars({ data, height }: { data: { attack_type: string; count: number; severity: Severity; percentage_of_malicious?: number; average_confidence?: number }[]; height?: number }) {
  if (!data.length) return <p className="py-8 text-center text-sm text-slate-500">No attacks detected.</p>
  return (
    <div style={{ height: height ?? Math.max(160, data.length * 34 + 30) }} className="w-full min-w-0">
      <ResponsiveContainer>
        <BarChart data={data} layout="vertical" margin={{ left: 8, right: 40, top: 4, bottom: 4 }}>
          <CartesianGrid horizontal={false} stroke={COLORS.grid} />
          <XAxis type="number" tick={{ fontSize: 11, fill: '#64748b' }} allowDecimals={false} axisLine={false} tickLine={false} />
          <YAxis type="category" dataKey="attack_type" tick={{ fontSize: 12, fill: '#334155' }} width={110} axisLine={false} tickLine={false} />
          <Tooltip cursor={{ fill: '#f1f5f9' }} contentStyle={tooltipStyle}
            formatter={(v: number, _n, item) => {
              const p = item.payload as AttackShare
              const extra = p.percentage_of_malicious !== undefined ? ` · ${pct(p.percentage_of_malicious)} of malicious · avg. confidence ${pct(p.average_confidence)}` : ''
              return [`${num(v)} flows${extra}`, `${p.severity} severity`]
            }} />
          <Bar dataKey="count" radius={[0, 4, 4, 0]} maxBarSize={22} label={{ position: 'right', fontSize: 11, fill: '#475569' }} isAnimationActive={false}>
            {data.map(d => <Cell key={d.attack_type} fill={SEVERITY_COLORS[d.severity]} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

export function SeverityLegend() {
  return (
    <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-slate-500">
      {(['Critical', 'High', 'Medium', 'Low'] as const).map(s => (
        <span key={s} className="flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-sm" style={{ backgroundColor: SEVERITY_COLORS[s] }} />{s}</span>
      ))}
    </div>
  )
}

/** High / medium / low confidence with the real thresholds. */
/** Two confidences from the same calibrated probabilities: is it an attack at all (attack-class probabilities added together), and is
 *  it this exact type (probability of the predicted class). The first is usually high; the second is lower when attack types look alike. */
export function ConfidenceLevels({ a }: { a: AnalysisData }) {
  const hasStatus = !!a.status_confidence_distribution
  const [mode, setMode] = useState<'status' | 'type'>(hasStatus ? 'status' : 'type')
  const dist = mode === 'status' && a.status_confidence_distribution ? a.status_confidence_distribution : a.confidence_distribution
  const levels: { key: ConfidenceLevel; label: string; rule: string }[] = [
    { key: 'high', label: 'High', rule: a.rules.confidence_levels.high },
    { key: 'medium', label: 'Medium', rule: a.rules.confidence_levels.medium },
    { key: 'low', label: 'Low', rule: a.rules.confidence_levels.low },
  ]
  return (
    <div className="space-y-4">
      {hasStatus && (
        <div role="radiogroup" aria-label="Which confidence" className="inline-flex rounded-lg bg-slate-100 p-0.5 text-xs font-medium">
          {([['status', 'Attack or normal?'], ['type', 'Exact attack type?']] as const).map(([k, label]) => (
            <button key={k} type="button" role="radio" aria-checked={mode === k} onClick={() => setMode(k)}
              className={`rounded-md px-3 py-1.5 focus:outline-none focus-visible:ring-2 focus-visible:ring-brand-500 ${mode === k ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-500 hover:text-slate-700'}`}>
              {label}
            </button>
          ))}
        </div>
      )}
      {levels.map(l => {
        const n = dist[l.key] ?? 0
        const share = a.total_flows ? n / a.total_flows : 0
        return (
          <div key={l.key}>
            <div className="flex items-baseline justify-between gap-3 text-sm">
              <span className="font-medium text-slate-800">{l.label} confidence <span className="font-normal text-slate-500">({l.rule})</span></span>
              <span className="tabular-nums text-slate-700"><b>{num(n)}</b> <span className="text-slate-500">· {pct(share)}</span></span>
            </div>
            <div className="mt-1.5 h-2.5 overflow-hidden rounded-full bg-slate-100">
              <div className="h-full rounded-full transition-all duration-500" style={{ width: `${share * 100}%`, backgroundColor: CONFIDENCE_COLORS[l.key] }} />
            </div>
          </div>
        )
      })}
      <p className="flex items-start gap-1.5 text-xs leading-relaxed text-slate-600"><InfoTip text={CONFIDENCE_TIP} align="left" />
        {mode === 'status'
          ? 'How sure the model is that a flow is an attack of any type, or normal traffic. The model is usually sure about this.'
          : 'How sure the model is about the exact attack type. This is lower because some attack types (for example DoS, Analysis and Backdoor) have the same measurements in this dataset.'}
      </p>
    </div>
  )
}

// ---------------------------------------------------------------------------------------------- Sankey
interface SankeyNodeProps { x: number; y: number; width: number; height: number; index: number; payload: { name: string; value: number; color: string } }
interface SankeyLinkProps {
  sourceX: number; targetX: number; sourceY: number; targetY: number; sourceControlX: number; targetControlX: number; linkWidth: number
  payload: { target: { color: string } }
}

function SankeyNode({ x, y, width, height, payload }: SankeyNodeProps) {
  const right = x > 300
  return (
    <Layer>
      <Rectangle x={x} y={y} width={width} height={Math.max(height, 2)} fill={payload.color} radius={2} />
      <text x={right ? x + width + 6 : x + width + 6} y={y + height / 2} dy={-2} fontSize={12} fill="#1e293b" fontWeight={600}>{payload.name}</text>
      <text x={x + width + 6} y={y + height / 2} dy={12} fontSize={11} fill="#64748b">{num(payload.value)} flows</text>
    </Layer>
  )
}

function SankeyLink({ sourceX, targetX, sourceY, targetY, sourceControlX, targetControlX, linkWidth, payload }: SankeyLinkProps) {
  return (
    <path d={`M${sourceX},${sourceY} C${sourceControlX},${sourceY} ${targetControlX},${targetY} ${targetX},${targetY}`}
      fill="none" stroke={payload.target.color} strokeOpacity={0.28} strokeWidth={Math.max(linkWidth, 1)} />
  )
}

/** Network flows -> detection model -> safe / malicious -> attack types, with the real counts. */
export function ThreatSankey({ a }: { a: AnalysisData }) {
  const nodes: { name: string; color: string }[] = [
    { name: 'Network flows', color: COLORS.neutral }, { name: 'Detection model', color: COLORS.brand },
  ]
  const links: { source: number; target: number; value: number }[] = [{ source: 0, target: 1, value: a.total_flows }]
  if (a.safe_count) {
    // Safe continues to "Normal" in the last column (predicted class); otherwise the chart would push the end node Safe to the far right.
    nodes.push({ name: 'Predicted Normal', color: COLORS.safe }, { name: 'Normal', color: COLORS.safe })
    links.push({ source: 1, target: nodes.length - 2, value: a.safe_count }, { source: nodes.length - 2, target: nodes.length - 1, value: a.safe_count })
  }
  if (a.malicious_count) {
    nodes.push({ name: 'Predicted attack', color: COLORS.threat })
    const mal = nodes.length - 1
    links.push({ source: 1, target: mal, value: a.malicious_count })
    a.attack_distribution.forEach(d => { nodes.push({ name: d.attack_type, color: SEVERITY_COLORS[d.severity] }); links.push({ source: mal, target: nodes.length - 1, value: d.count }) })
  }
  const height = Math.max(300, a.attack_distribution.length * 38 + 40)
  return (
    <div className="w-full min-w-0 overflow-x-auto">
      <div style={{ height, minWidth: 640 }}>
        <ResponsiveContainer>
          <Sankey data={{ nodes, links }} nodePadding={18} nodeWidth={12} margin={{ top: 10, right: 130, bottom: 10, left: 10 }} iterations={64}
            node={(p: SankeyNodeProps) => <SankeyNode {...p} />} link={(p: SankeyLinkProps) => <SankeyLink {...p} />}>
            <Tooltip contentStyle={tooltipStyle} formatter={(v: number) => `${num(v)} flows`} />
          </Sankey>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
