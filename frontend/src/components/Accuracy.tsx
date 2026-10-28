import clsx from 'clsx'
import type { ReactNode } from 'react'
import type { Evaluation } from '../lib/api'
import { COLORS, num, pct } from '../lib/format'
import { ClassBadge, InfoTip, ProgressBar } from './ui'

const share = (part: number, whole: number) => (whole ? part / whole : 0)
const tone = (x: number, good = 0.8, ok = 0.5): 'good' | 'warn' | 'bad' => (x >= good ? 'good' : x >= ok ? 'warn' : 'bad')

function Metric({ title, tip, part, whole, unit, reverse = false }: { title: string; tip: ReactNode; part: number; whole: number; unit: string; reverse?: boolean }) {
  const value = share(part, whole)
  const t = reverse ? tone(1 - value, 0.9, 0.7) : tone(value)
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4">
      <p className="flex items-center gap-1.5 text-xs font-medium uppercase tracking-wide text-slate-500">{title}<InfoTip text={tip} /></p>
      <p className={clsx('mt-1 text-3xl font-semibold tabular-nums', { good: 'text-emerald-600', warn: 'text-amber-600', bad: 'text-rose-600' }[t])}>{pct(value)}</p>
      <p className="mt-0.5 text-sm text-slate-600"><span className="font-medium tabular-nums">{num(part)}</span> of {num(whole)} {unit}</p>
      <div className="mt-3"><ProgressBar value={value} tone={t} /></div>
    </div>
  )
}

/** "How accurate was this scan?" for files that carry the true class of every flow. */
export default function Accuracy({ evaluation: e }: { evaluation: Evaluation }) {
  const caught = share(e.attacks_caught, e.attacks), alarms = share(e.normal_flagged, e.normal), exact = share(e.exact_class_correct, e.labelled_flows)
  const rows = [...e.per_class].sort((a, b) => share(b.correct, b.flows) - share(a.correct, a.flows))
  return (
    <div className="space-y-5">
      <p className="rounded-lg bg-slate-50 px-4 py-3 text-sm leading-relaxed text-slate-700">
        This file includes the true class of every flow, so the detector's answers can be checked.{' '}
        {e.attacks > 0 && <>Of <b>{num(e.attacks)}</b> real attacks it flagged <b>{num(e.attacks_caught)}</b> ({pct(caught)}). </>}
        {e.normal > 0 && <>It wrongly flagged <b>{num(e.normal_flagged)}</b> of {num(e.normal)} normal flows ({pct(alarms)}). </>}
        It named the exact class correctly for <b>{pct(exact)}</b> of the flows.
      </p>
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <Metric title="Attacks caught" part={e.attacks_caught} whole={e.attacks} unit="real attacks"
          tip="Of the flows that really were attacks, how many the detector flagged as an attack (of any type). Also called detection rate or recall. Higher is better." />
        <Metric title="False alarms" part={e.normal_flagged} whole={e.normal} unit="normal flows" reverse
          tip="Of the flows that were really normal traffic, how many were wrongly flagged as an attack. Lower is better." />
        <Metric title="Attack or normal right" part={e.attack_vs_normal_correct} whole={e.labelled_flows} unit="flows"
          tip="The simplest check: did the detector correctly say 'attack' or 'normal' for the flow? Higher is better." />
        <Metric title="Exact class right" part={e.exact_class_correct} whole={e.labelled_flows} unit="flows"
          tip="The hardest check: did it name the exact class, for example DoS rather than Exploits? Some classes have identical measurements in this dataset, so this is lower." />
      </div>
      <p className="flex items-center gap-1.5 text-xs text-slate-500">
        The shortlist (90% prediction set) contained the true class for {num(e.truth_in_set)} of {num(e.labelled_flows)} flows ({pct(share(e.truth_in_set, e.labelled_flows))}).
        <InfoTip text="For uncertain flows the detector gives a shortlist of possible classes instead of a single guess. It is designed to contain the true class about 90% of the time on data like its training data." />
      </p>
      <div className="overflow-x-auto">
        <table className="min-w-full text-sm">
          <thead className="text-left text-xs uppercase text-slate-500">
            <tr>
              <th className="py-2 pr-4">True class</th><th className="pr-4">Flows</th><th className="pr-4">Named correctly</th>
              <th className="pr-4">In shortlist</th><th className="pr-4">Flagged as attack</th><th>Most often mistaken for</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {rows.map(c => {
              const r = share(c.correct, c.flows)
              const flaggedShare = share(c.flagged_as_attack, c.flows)
              return (
                <tr key={c.class}>
                  <td className="py-2 pr-4"><ClassBadge name={c.class} /></td>
                  <td className="pr-4 tabular-nums">{num(c.flows)}</td>
                  <td className="pr-4">
                    <div className="flex items-center gap-2">
                      <div className="h-1.5 w-24 overflow-hidden rounded-full bg-slate-100"><div className="h-full rounded-full" style={{ width: `${r * 100}%`, backgroundColor: COLORS.brand }} /></div>
                      <span className="tabular-nums text-slate-700">{num(c.correct)} ({pct(r, 0)})</span>
                    </div>
                  </td>
                  <td className="pr-4 tabular-nums text-slate-600">{pct(share(c.in_set, c.flows), 0)}</td>
                  <td className={clsx('pr-4 tabular-nums', c.class === 'Normal' ? (flaggedShare > 0.1 ? 'text-rose-600' : 'text-slate-600') : (flaggedShare < 0.9 ? 'text-amber-600' : 'text-slate-600'))}>
                    {pct(flaggedShare, 0)}{c.class === 'Normal' && <span className="text-xs text-slate-400"> (false alarms)</span>}
                  </td>
                  <td className="text-slate-600">{c.confused_with ? <span className="inline-flex items-center gap-1.5"><ClassBadge name={c.confused_with} muted /><span className="text-xs tabular-nums">× {num(c.confused_count)}</span></span> : <span className="text-slate-400">–</span>}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      <p className="text-xs text-slate-500">
        Analysis, Backdoor and DoS often have exactly the same measurements as other attack classes in UNSW-NB15. They are still caught as attacks,
        but their exact type is often wrong, which is why the detector shows a shortlist for them.
      </p>
    </div>
  )
}
