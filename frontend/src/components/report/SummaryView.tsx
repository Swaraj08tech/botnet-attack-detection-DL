import { useQuery } from '@tanstack/react-query'
import { ArrowRight, CheckCircle2, Info, ShieldAlert, Sparkles } from 'lucide-react'
import { api, type ReportRecord } from '../../lib/api'
import { num, pct } from '../../lib/format'
import { ACTION_LABEL, safeUrl } from '../../lib/record'
import { Button, ClassBadge, RiskBadge } from '../ui'
import { Cites } from './Citations'

const PHASE_LABEL = { Verify: 'Verify now', 'Contain if confirmed': 'Contain only if confirmed', Recover: 'Recover and monitor' } as const
const H2 = 'text-base font-semibold text-slate-900'

/** Summary report for students and general users. Every sentence comes from record.views.summary, which the server computes from the
 *  same record as the Detailed view (and the Summary PDF), so the two views cannot disagree. */
export default function SummaryView({ r, onEvidence, onDetailed }: { r: ReportRecord; onEvidence: () => void; onDetailed: () => void }) {
  const v = r.views.summary
  const s = r.summary
  const states = useQuery({ queryKey: ['report-actions', r.report_id], queryFn: () => api.actionStates(r.report_id) })
  const tiles: [string, string][] = [
    ['Flows analysed', num(r.detections.total)], ['Predicted Normal', `${num(s.predicted_normal)} (${pct(s.predicted_normal_share)})`],
    ['Predicted attack', `${num(s.predicted_attack)} (${pct(s.predicted_attack_share)})`], ['Need review', `${num(s.needs_review)} (confidence below 70%)`],
  ]
  const degraded = Object.entries(r.components).filter(([k, c]) => k !== 'pdf' && !['available', 'not run'].includes(c?.status ?? 'available'))
  return (
    <div className="space-y-6 text-[15px] leading-relaxed text-slate-800">
      <p className="rounded-lg bg-slate-50 px-4 py-3 text-sm text-slate-600">{s.scope} The results are predictions of a machine-learning model: they are not
        confirmed incidents and they do not predict future attacks.</p>
      {degraded.length > 0 && (
        <p role="status" className="flex items-start gap-2 rounded-lg bg-amber-50 px-4 py-3 text-sm text-amber-900">
          <ShieldAlert className="mt-0.5 h-4 w-4 shrink-0" />Part of this report could not be produced: {degraded.map(([k, c]) => `${k} ${c.status}`).join(', ')}.
          The sections below show only what is available.</p>
      )}

      <section aria-labelledby="s1">
        <h2 id="s1" className={H2}>1. Overall flow summary</h2>
        <div className="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {tiles.map(([k, val]) => (
            <div key={k} className="rounded-xl border border-slate-200 p-3"><p className="text-xs font-medium text-slate-500">{k}</p><p className="mt-0.5 text-lg font-semibold tabular-nums">{val}</p></div>
          ))}
        </div>
        {s.status_confidence && (
          <div className="mt-3 overflow-x-auto rounded-lg border border-slate-200">
            <table className="min-w-full text-sm">
              <caption className="sr-only">How sure the model is</caption>
              <thead className="bg-slate-50 text-left text-xs text-slate-500">
                <tr><th className="px-3 py-2 font-medium">How sure is the model?</th><th className="px-3 text-right font-medium">High (90%+)</th>
                  <th className="px-3 text-right font-medium">Medium (70–90%)</th><th className="px-3 text-right font-medium">Low (&lt;70%)</th></tr>
              </thead>
              <tbody className="divide-y divide-slate-100 tabular-nums">
                {([['Attack or normal?', s.status_confidence.levels], ['Exact attack type?', r.detections.confidence_levels]] as const).map(([label, lv]) => (
                  <tr key={label}><td className="px-3 py-2 font-medium">{label}</td><td className="px-3 text-right text-emerald-700">{num(lv.high)}</td>
                    <td className="px-3 text-right text-amber-700">{num(lv.medium)}</td><td className="px-3 text-right text-rose-700">{num(lv.low)}</td></tr>
                ))}
              </tbody>
            </table>
            <p className="border-t border-slate-100 px-3 py-2 text-xs text-slate-500">The model is usually sure whether a flow is an attack; it is less sure about the exact type, because some attack types look the same in these measurements.</p>
          </div>
        )}
        <p className="mt-3 flex flex-wrap items-center gap-2 text-sm">Review priority <RiskBadge level={s.triage_priority.level} />
          <span className="text-slate-600">— a fixed rule of this application to decide how soon to review; not a measure of damage.</span></p>
      </section>

      <section aria-labelledby="s2">
        <h2 id="s2" className={H2}>2. What was detected</h2>
        <p className="mt-2 font-medium">{v.detected.question}</p>
        <p>{v.detected.answer}</p>
        {v.detected.groups.length > 0 && (
          <ul className="mt-3 space-y-2">
            {v.detected.groups.map(g => (
              <li key={g.class} className="rounded-lg border border-slate-200 px-3 py-2">
                <p className="flex flex-wrap items-center gap-2"><ClassBadge name={g.class} /><span className="text-sm text-slate-600">{num(g.count)} flows</span></p>
                <p className="mt-1 text-sm">{g.meaning.charAt(0).toUpperCase() + g.meaning.slice(1)}.</p>
                {g.context && <p className="mt-1 text-sm text-slate-600">Public guidance (a possible explanation, not proof): “{g.context.text}”<Cites ids={g.context.cites} /></p>}
              </li>
            ))}
          </ul>
        )}
      </section>

      <section aria-labelledby="s3">
        <h2 id="s3" className={H2}>3. Why the traffic was flagged</h2>
        <ul className="mt-2 list-disc space-y-1.5 pl-5">
          {v.overview.bullets.map((b, i) => <li key={i}>{b.text}<Cites ids={b.cites} /></li>)}
        </ul>
        <p className="mt-2 flex items-start gap-1.5 text-xs text-slate-500"><Info className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden />
          The reasons are the measurements that most often moved the model towards a group. They describe how the model decided, not what caused the traffic.</p>
        {(v.examples ?? []).length > 0 && (
          <div className="mt-4">
            <h3 className="flex items-center gap-2 text-sm font-semibold text-slate-800"><Sparkles className="h-4 w-4 text-violet-600" aria-hidden />Examples: the reasoning for one flow of each main type</h3>
            <p className="text-xs text-slate-500">Points marked “AI, checked” were written by the local language model and every number and direction was checked against the evidence; the others are written by code from the same evidence. Typical values come from the training flows only.</p>
            <div className="mt-2 grid gap-3 lg:grid-cols-3">
              {(v.examples ?? []).map(ex => (
                <div key={ex.flow_id} className="rounded-xl border border-violet-200 bg-violet-50/50 p-3">
                  <p className="flex items-center gap-2 text-sm"><ClassBadge name={ex.class} /><span className="text-xs text-slate-500">example flow #{ex.flow_index}</span></p>
                  <ol className="mt-2 list-decimal space-y-1.5 pl-5 text-sm">
                    {ex.points.map((pt, i) => <li key={i}>{pt.title && <b className="font-medium text-slate-900">{pt.title}: </b>}{pt.text}<Cites ids={pt.cites} />{pt.origin === 'llm_synthesis' ? <span className="ml-1 rounded bg-violet-100 px-1 text-[10px] font-medium text-violet-700">AI, checked</span> : <span className="ml-1 text-[11px] text-slate-500">(written from the model evidence)</span>}</li>)}
                  </ol>
                </div>
              ))}
            </div>
          </div>
        )}
        <div className="mt-3 flex flex-wrap gap-2">
          <Button variant="secondary" onClick={onEvidence}>View evidence</Button>
          <Button variant="secondary" onClick={onDetailed}>Open detailed report<ArrowRight className="h-4 w-4" /></Button>
        </div>
      </section>

      <section aria-labelledby="s4">
        <h2 id="s4" className={H2}>4. What to do</h2>
        {v.actions.length === 0 ? <p className="mt-2 text-sm">No attacks were predicted, so there are no response steps. Spot-check a few low-confidence flows.</p> : (
          <div className="mt-2 space-y-3">
            {(['Verify', 'Contain if confirmed', 'Recover'] as const).map(phase => {
              const rows = v.actions.filter(a => a.phase === phase)
              if (!rows.length) return null
              return (
                <div key={phase}>
                  <h3 className="text-sm font-semibold text-slate-700">{PHASE_LABEL[phase]}</h3>
                  <ul className="mt-1 space-y-2">
                    {rows.map(a => {
                      const st = states.data?.[a.id]
                      return (
                        <li key={a.id} className="rounded-lg border border-slate-200 p-3 text-sm">
                          <p className="flex flex-wrap items-center gap-2 font-medium text-slate-900">{a.what}
                            <span className="text-xs font-normal text-slate-500">{a.id} · {a.owner}</span>
                            {st && <span className="inline-flex items-center gap-1 rounded bg-emerald-50 px-1.5 text-xs font-normal text-emerald-700"><CheckCircle2 className="h-3 w-3" />{ACTION_LABEL[st.status]}</span>}</p>
                          <dl className="mt-1 grid gap-x-4 gap-y-1 sm:grid-cols-3">
                            <div><dt className="text-xs text-slate-500">How to do it</dt><dd>{a.how}<Cites ids={a.cites.filter(c => c.startsWith('S'))} /></dd></div>
                            <div><dt className="text-xs text-slate-500">When it applies</dt><dd>{a.when}</dd></div>
                            <div><dt className="text-xs text-slate-500">How to check success</dt><dd>{a.check}</dd></div>
                          </dl>
                        </li>
                      )
                    })}
                  </ul>
                </div>
              )
            })}
            <p className="text-xs text-slate-500">This application never changes your network. The checklist in the detailed report records what a person decided.</p>
          </div>
        )}
      </section>

      <section aria-labelledby="s5">
        <h2 id="s5" className={H2}>5. How to check the outcome</h2>
        <dl className="mt-2 space-y-1.5 text-sm">
          <div><dt className="inline font-semibold">Evidence you need: </dt><dd className="inline">{v.outcome.evidence_needed}</dd></div>
          <div><dt className="inline font-semibold">Confirms the concern: </dt><dd className="inline">{v.outcome.confirms}</dd></div>
          <div><dt className="inline font-semibold">Weakens the concern: </dt><dd className="inline">{v.outcome.weakens.join('; ')}.</dd></div>
          <div><dt className="inline font-semibold">Fix worked: </dt><dd className="inline">{v.outcome.mitigation_success}</dd></div>
        </dl>
      </section>

      <section aria-labelledby="s6">
        <h2 id="s6" className={H2}>6. Limits and sources</h2>
        <ul className="mt-2 list-disc space-y-1 pl-5 text-sm">{v.limits.map(l => <li key={l}>{l}</li>)}</ul>
        {v.sources.length > 0 && (
          <ul className="mt-3 space-y-1 text-sm">
            {v.sources.map(src => (
              <li key={src.id}><span className="font-semibold text-slate-500">[{src.id}]</span> {src.source},{' '}
                {safeUrl(src.url) ? <a className="text-brand-600 hover:underline" href={safeUrl(src.url)!} target="_blank" rel="noopener noreferrer">{src.location}</a> : src.location}</li>
            ))}
          </ul>
        )}
      </section>
    </div>
  )
}
