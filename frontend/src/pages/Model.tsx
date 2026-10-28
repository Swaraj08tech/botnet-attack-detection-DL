import { useQuery } from '@tanstack/react-query'
import clsx from 'clsx'
import { AlertTriangle, BadgeCheck } from 'lucide-react'
import { useState } from 'react'
import { Card, ClassBadge, ErrorBox, InfoTip, PageHeader, RiskBadge, SeverityBadge, Spinner } from '../components/ui'
import { api, type ModelInfo, type Severity } from '../lib/api'
import { COLORS, num, pct } from '../lib/format'

type Avg = 'macro' | 'weighted'
type Evaluation = NonNullable<ModelInfo['model_card']['evaluation']>
const AVG_LABEL: Record<Avg, string> = { macro: 'Macro average', weighted: 'Weighted average' }
const AVG_HELP: Record<Avg, string> = {
  macro: 'Each of the 10 classes counts equally, so rare attacks (e.g. Worms, 44 test flows) matter as much as common ones.',
  weighted: 'Classes count by their number of flows, so common classes (Normal, Generic, Exploits) dominate.',
}
const METRIC_HELP = {
  accuracy: 'How often the model is correct overall.',
  precision: 'When the model predicts an attack (or a class), how often it is correct.',
  recall: 'How many of the actual attacks (or flows of a class) the model detects.',
  f1: 'Balance between precision and recall (their harmonic mean).',
}

function MetricCards({ cards, accent = 'bg-brand-500' }: { cards: { key: keyof typeof METRIC_HELP; label: string; value: number }[]; accent?: string }) {
  return (
    <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
      {cards.map(c => (
        <div key={c.key} className="rounded-xl border border-slate-200 p-4">
          <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">{c.label}</p>
          <p className="mt-1 text-3xl font-semibold tabular-nums text-slate-900">{pct(c.value)}</p>
          <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-slate-100"><div className={`h-full rounded-full ${accent}`} style={{ width: `${c.value * 100}%` }} /></div>
          <p className="mt-2 text-xs text-slate-500">{METRIC_HELP[c.key]}</p>
        </div>
      ))}
    </div>
  )
}

/** The main task: separating attacks from normal traffic. */
function DetectionHeadline({ e }: { e: Evaluation }) {
  const b = e.binary
  return (
    <Card title="Attack detection: the main task" subtitle={`Attack vs normal traffic · ${e.split}`}>
      <MetricCards accent="bg-emerald-500" cards={[
        { key: 'accuracy', label: 'Accuracy', value: b.accuracy }, { key: 'precision', label: 'Precision', value: b.precision },
        { key: 'recall', label: 'Recall (attacks found)', value: b.recall }, { key: 'f1', label: 'F1 score', value: b.f1 }]} />
      <p className="mt-4 text-xs text-slate-500">All nine attack types are merged into one &quot;attack&quot; class: did the model separate attacks from normal traffic?
        {' '}{pct(b.false_positive_rate)} of normal test flows were flagged as attacks, mostly because the official test file was recorded differently from the
        training file (see &quot;How to read these scores&quot;).</p>
    </Card>
  )
}

function Metrics({ e }: { e: Evaluation }) {
  const [avg, setAvg] = useState<Avg>('macro')
  const m = e[avg]
  return (
    <Card title="Attack type: 10 classes" subtitle="Naming the exact class (Normal or one of nine attack types) · official test set"
      actions={
        <div className="flex rounded-lg border border-slate-200 bg-slate-50 p-0.5 text-xs" role="tablist">
          {(['macro', 'weighted'] as Avg[]).map(k => (
            <button key={k} role="tab" aria-selected={avg === k} onClick={() => setAvg(k)}
              className={clsx('rounded-md px-2.5 py-1 font-medium', avg === k ? 'bg-white text-brand-700 shadow-sm' : 'text-slate-500 hover:text-slate-700')}>{AVG_LABEL[k]}</button>
          ))}
        </div>}>
      <MetricCards cards={[
        { key: 'accuracy', label: 'Accuracy (10-class)', value: e.accuracy }, { key: 'precision', label: 'Precision', value: m.precision },
        { key: 'recall', label: 'Recall', value: m.recall }, { key: 'f1', label: 'F1 score', value: m.f1 }]} />
      <p className="mt-4 text-xs text-slate-500"><b>{AVG_LABEL[avg]}:</b> {AVG_HELP[avg]}</p>
    </Card>
  )
}

/** How to read the scores: what is possible at all on this file, top-2 accuracy and the in-distribution scores. */
function ScoreContext({ e, inDist }: { e: Evaluation; inDist?: ModelInfo['model_card']['in_distribution'] }) {
  if (!e.ceiling) return null
  const c = e.ceiling
  const rows = [
    { label: '10-class accuracy', model: e.accuracy, best: c.accuracy },
    { label: 'Macro F1 (10 classes)', model: e.macro.f1, best: c.macro_f1 },
    { label: 'Attack F1 (attack vs normal)', model: e.binary.f1, best: c.attack_f1 },
  ]
  return (
    <Card title={<span className="flex items-center gap-1.5">How to read these scores<InfoTip text={c.meaning} /></span>}
      subtitle="What is possible at all on the official test file, and how the model does on data like its training data">
      <div className="grid gap-6 lg:grid-cols-5">
        <div className="lg:col-span-3">
          <table className="min-w-full text-sm">
            <thead className="text-left text-xs uppercase tracking-wide text-slate-500">
              <tr><th className="py-2">Metric</th><th className="text-right">Deployed model</th><th className="text-right">Best possible</th><th className="text-right">Share of best</th></tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {rows.map(r => (
                <tr key={r.label}>
                  <td className="py-2.5">{r.label}</td><td className="text-right font-semibold tabular-nums">{pct(r.model)}</td>
                  <td className="text-right tabular-nums text-slate-600">{pct(r.best)}</td>
                  <td className="text-right">
                    <span className="inline-flex items-center gap-2">
                      <span className="h-1.5 w-16 overflow-hidden rounded-full bg-slate-100"><span className="block h-full rounded-full bg-brand-500" style={{ width: `${Math.min(1, r.model / r.best) * 100}%` }} /></span>
                      <span className="tabular-nums">{pct(r.model / r.best, 0)}</span>
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-3 text-xs text-slate-500">
            &quot;Best possible&quot; is a model that already knows the test labels and gives every group of identical flows its most common label; no real model can beat it.
            For the 10 classes the limit is the data itself: identical flows carry different attack labels. For attack vs normal there is almost no such limit, so the
            remaining gap comes from the difference between the training and test files.
          </p>
        </div>
        <div className="space-y-3 lg:col-span-2">
          {e.top2_accuracy !== undefined && (
            <div className="rounded-xl bg-slate-50 p-4">
              <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Top-2 accuracy</p>
              <p className="mt-1 text-2xl font-semibold tabular-nums text-slate-900">{pct(e.top2_accuracy)}</p>
              <p className="text-xs text-slate-500">The true class is one of the two most likely classes. When the first answer is wrong, the second is usually right.</p>
            </div>
          )}
          {inDist && (
            <div className="rounded-xl bg-slate-50 p-4">
              <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Same distribution as training</p>
              <p className="mt-1 text-sm text-slate-700">Attack F1 <b>{pct(inDist.attack_f1)}</b> · accuracy <b>{pct(inDist.accuracy)}</b> · macro F1 <b>{pct(inDist.macro_f1)}</b></p>
              <p className="text-xs text-slate-500">On 17,534 held-out rows of the training file (never used for training). The official test file was recorded differently, which lowers the scores above.</p>
            </div>
          )}
        </div>
      </div>
      <p className="mt-4 rounded-lg bg-amber-50 px-3 py-2 text-xs text-amber-900">Papers reporting 95–99% on UNSW-NB15 usually mix the two official files and split them at random, which removes the
        difference between them. This project keeps the official split on purpose, so those numbers are not comparable.</p>
    </Card>
  )
}

/** Per-class results, official 10 classes or the grouped view (DoS, Analysis and Backdoor as one group), with F1 shown as bars. */
function PerClass({ e, grouped, severity }: { e: Evaluation; grouped: ModelInfo['grouped_evaluation']; severity: Record<string, string> }) {
  const [view, setView] = useState<'classes' | 'grouped'>('classes')
  const rows = view === 'grouped' && grouped
    ? grouped.per_class.map(c => ({ name: c.class, precision: c.precision, recall: c.recall, f1: c.f1, best: undefined as number | undefined, support: c.support, group: c.class === grouped.group }))
    : e.per_class.map(c => ({ name: c.class as string, precision: c.precision, recall: c.recall, f1: c.f1, best: c.ceiling_f1, support: c.support, group: false }))
  const macro = view === 'grouped' && grouped ? grouped.macro_f1 : e.macro.f1
  const accuracy = view === 'grouped' && grouped ? grouped.accuracy : e.accuracy
  return (
    <Card title="Results per class" subtitle="Official test set (82,332 flows the model never saw)" bodyClassName="p-0"
      actions={grouped && (
        <div className="flex rounded-lg border border-slate-200 bg-slate-50 p-0.5 text-xs" role="radiogroup" aria-label="Per-class view">
          {([['classes', '10 classes (official)'], ['grouped', 'Grouped view']] as const).map(([k, label]) => (
            <button key={k} type="button" role="radio" aria-checked={view === k} onClick={() => setView(k)}
              className={clsx('rounded-md px-2.5 py-1 font-medium focus:outline-none focus-visible:ring-2 focus-visible:ring-brand-500', view === k ? 'bg-white text-brand-700 shadow-sm' : 'text-slate-500 hover:text-slate-700')}>{label}</button>
          ))}
        </div>
      )}>
      <div className="flex flex-wrap gap-x-6 gap-y-1 border-b border-slate-100 px-4 py-3 text-sm">
        <span>Accuracy <b className="tabular-nums">{pct(accuracy)}</b></span><span>Macro F1 <b className="tabular-nums">{pct(macro)}</b></span>
        <span className="text-xs text-slate-500">{view === 'grouped' ? 'DoS, Analysis and Backdoor counted as one group' : 'every class counted separately'}</span>
      </div>
      <div className="overflow-x-auto">
        <table className="min-w-full text-sm">
          <thead className="bg-slate-50 text-left text-xs uppercase tracking-wide text-slate-500">
            <tr><th className="px-4 py-2">Class</th><th className="px-3 text-right">Precision</th><th className="px-3 text-right">Recall</th><th className="px-3">F1</th>
              {view === 'classes' && <th className="px-3 text-right" title="F1 of a model that already knows the test labels">Best possible F1</th>}<th className="px-4 text-right">Test flows</th></tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {rows.map(c => (
              <tr key={c.name} className={c.group ? 'bg-brand-50/50' : undefined}>
                <td className="px-4 py-2">{c.group ? <span className="font-medium text-slate-800">{c.name}</span>
                  : <ClassBadge name={c.name} severity={c.name === 'Normal' ? 'None' : severity[c.name] as Severity} />}</td>
                <td className="px-3 text-right tabular-nums">{pct(c.precision)}</td><td className="px-3 text-right tabular-nums">{pct(c.recall)}</td>
                <td className="px-3">
                  <span className="flex items-center gap-2">
                    <span className="relative h-2 w-20 overflow-hidden rounded-full bg-slate-100" aria-hidden>
                      {c.best !== undefined && <span className="absolute inset-y-0 left-0 bg-slate-300" style={{ width: `${c.best * 100}%` }} />}
                      <span className="absolute inset-y-0 left-0 rounded-full bg-brand-500" style={{ width: `${c.f1 * 100}%` }} />
                    </span>
                    <b className="tabular-nums">{pct(c.f1)}</b>
                  </span>
                </td>
                {view === 'classes' && <td className="px-3 text-right tabular-nums text-slate-500">{c.best !== undefined ? pct(c.best) : '–'}</td>}
                <td className="px-4 text-right tabular-nums text-slate-500">{num(c.support)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="space-y-1.5 px-4 py-3 text-xs leading-relaxed text-slate-600">
        {view === 'classes' ? <>
          <p><b>Why are Analysis, Backdoor and DoS low?</b> In UNSW-NB15, many of their flows have exactly the same measurements but different labels.
            No model can tell them apart: even a model that already knows the answers reaches only the grey part of the bar (best possible F1).</p>
          <p>Use the <b>Grouped view</b> to see the results with these three classes counted as one group.</p>
        </> : grouped && <>
          <p><b>Grouped view.</b> {grouped.note}</p>
          <p>{num(grouped.within_group_confusions)} test flows were confused only within the group. This is a different way of counting the same results, not a new
            or better model; the 10-class numbers stay the official scores.</p>
        </>}
      </div>
    </Card>
  )
}

function ConfusionMatrix({ cm }: { cm: { labels: string[]; matrix: number[][] } }) {
  return (
    <Card title="Confusion matrix" subtitle="This matrix shows where the model made correct and incorrect predictions during evaluation (official test set). Rows = true class, columns = predicted class.">
      <div className="overflow-x-auto">
        <table className="mx-auto border-separate border-spacing-0.5 text-[11px]">
          <thead>
            <tr><th className="p-1 text-right text-slate-400">true ↓ / predicted →</th>{cm.labels.map(l => <th key={l} className="w-16 p-1 text-center font-medium text-slate-600" title={l}>{l.slice(0, 6)}</th>)}</tr>
          </thead>
          <tbody>
            {cm.matrix.map((row, i) => {
              const total = row.reduce((s, v) => s + v, 0) || 1
              return (
                <tr key={cm.labels[i]}>
                  <th className="whitespace-nowrap p-1 pr-2 text-right font-medium text-slate-600">{cm.labels[i]}</th>
                  {row.map((v, j) => {
                    const share = v / total
                    const color = i === j ? COLORS.safe : COLORS.threat
                    return (
                      <td key={j} title={`True ${cm.labels[i]}, predicted ${cm.labels[j]}: ${num(v)} flows (${pct(share)} of ${cm.labels[i]})`}
                        className="h-9 w-16 rounded text-center tabular-nums"
                        style={{ backgroundColor: v ? `${color}${Math.round(Math.min(0.12 + share * 0.85, 0.95) * 255).toString(16).padStart(2, '0')}` : '#f8fafc', color: share > 0.45 ? '#fff' : '#334155' }}>
                        {v ? num(v) : '·'}
                      </td>
                    )
                  })}
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      <p className="mt-3 text-xs text-slate-500">Green diagonal cells are correct predictions; red cells are mistakes. Darker = a larger share of that true class. Hover a cell for the exact numbers.
        Most mistakes are between attack types (e.g. Analysis, Backdoor and DoS look identical in this data) or Normal flows flagged as attacks.</p>
    </Card>
  )
}

export default function Model() {
  const model = useQuery({ queryKey: ['model'], queryFn: api.model })
  if (model.isLoading) return <Spinner />
  if (model.error) return <ErrorBox error={model.error} />
  const { model_card: card, calibration, bundle, rules } = model.data!
  const e = card.evaluation
  const classes = card.classes ?? []

  return (
    <>
      <PageHeader kicker="Insights" title="Detection Model" description="What the model is, how well it performs on data it never saw, and how its outputs are turned into confidence, severity and risk." />

      <Card title="About the model">
        <dl className="grid gap-x-8 gap-y-3 text-sm sm:grid-cols-2">
          <div><dt className="text-xs text-slate-500">Model</dt><dd className="font-medium text-slate-900">{card.name}</dd></div>
          <div><dt className="text-xs text-slate-500">Problem type</dt><dd className="text-slate-800">Supervised multiclass classification of network flows ({classes.length} classes)</dd></div>
          <div><dt className="text-xs text-slate-500">Dataset</dt><dd className="text-slate-800">{card.dataset?.name ?? 'UNSW-NB15'}{card.dataset && <> · {num(card.dataset.training_rows)} training / {num(card.dataset.test_rows)} test flows</>}</dd></div>
          <div><dt className="text-xs text-slate-500">Input features</dt><dd className="text-slate-800">{card.input_features ?? 42} flow features ({card.model_columns ?? '–'} model columns after one-hot encoding and 2 engineered byte features)</dd></div>
          <div className="sm:col-span-2"><dt className="mb-1 text-xs text-slate-500">Classes</dt>
            <dd className="flex flex-wrap gap-1.5">{classes.map(c => <ClassBadge key={c} name={c} severity={c === 'Normal' ? 'None' : rules.severity[c] as Severity} />)}</dd></div>
          <div className="sm:col-span-2"><dt className="text-xs text-slate-500">Short description</dt>
            <dd className="text-slate-700">A gradient-boosted decision-tree ensemble (LightGBM) trained with class weights (so rare attacks are not ignored) and importance weights
              (to adapt to the differences between the official training and test files). Its probabilities are calibrated with temperature scaling ({calibration.temperature.toFixed(3)}),
              on held-out flows from the training file, where a confidence of 80% meant that about 80% of such predictions were right. It was chosen after comparing {card.comparison?.rows.length ?? 'several'} models in the research notebooks.</dd></div>
        </dl>
      </Card>

      {e && <div className="mt-6"><DetectionHeadline e={e} /></div>}
      {e && <div className="mt-6"><Metrics e={e} /></div>}
      {e && <div className="mt-6"><ScoreContext e={e} inDist={card.in_distribution} /></div>}
      {e && <div className="mt-6"><ConfusionMatrix cm={e.confusion_matrix} /></div>}

      <div className="mt-6 grid gap-6 lg:grid-cols-2">
        {e && <div className="min-w-0 lg:col-span-2"><PerClass e={e} grouped={model.data!.grouped_evaluation} severity={rules.severity} /></div>}
        {card.comparison && (
          <Card title="Model comparison" subtitle="Every model evaluated once on the official test set (research notebooks 19 and 21)" bodyClassName="p-0">
            <div className="overflow-x-auto">
              <table className="min-w-full text-sm">
                <thead className="bg-slate-50 text-left text-xs uppercase tracking-wide text-slate-500">
                  <tr><th className="px-4 py-2">Model</th><th className="px-3 text-right">Accuracy</th><th className="px-3 text-right">Macro F1</th><th className="px-3 text-right">Attack recall</th><th className="px-4 text-right">False alarms</th></tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {card.comparison.rows.map(r => (
                    <tr key={r.model} className={r.deployed ? 'bg-brand-50/60' : undefined}>
                      <td className="px-4 py-2">{r.model}{r.deployed && <span className="ml-2 rounded bg-brand-600 px-1.5 py-0.5 text-[10px] font-semibold text-white">DEPLOYED</span>}
                        {r.runs > 1 && <span className="ml-1 text-xs text-slate-400">(mean of {r.runs})</span>}</td>
                      <td className="px-3 text-right tabular-nums">{pct(r.accuracy)}</td><td className="px-3 text-right font-medium tabular-nums">{pct(r.macro_f1)}</td>
                      <td className="px-3 text-right tabular-nums">{pct(r.attack_recall)}</td><td className="px-4 text-right tabular-nums">{pct(r.false_positive_rate)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="px-4 py-3 text-xs text-slate-500">The deployed model has the best macro F1 (every class counts equally). Some models have higher accuracy because they predict Normal more often, at the cost of missing rare attacks. Precision was not saved for the comparison runs, so it is not shown.</p>
          </Card>
        )}
      </div>

      <h2 className="mb-3 mt-8 text-sm font-semibold uppercase tracking-wide text-slate-500">How predictions are turned into results</h2>
      <div className="grid gap-6 lg:grid-cols-2">
        <Card title="Prediction confidence" subtitle="Calibrated probability of the predicted class">
          <table className="w-full text-sm">
            <tbody className="divide-y divide-slate-100">
              <tr><td className="py-2 font-medium text-emerald-700">High</td><td>{rules.confidence_levels.high}</td></tr>
              <tr><td className="py-2 font-medium text-amber-700">Medium</td><td>{rules.confidence_levels.medium}</td></tr>
              <tr><td className="py-2 font-medium text-rose-700">Low → needs review</td><td>{rules.confidence_levels.low}</td></tr>
            </tbody>
          </table>
          <p className="mt-3 text-xs text-slate-500">Confidence is {rules.confidence_meaning}.</p>
        </Card>
        <Card title={<span className="flex items-center gap-1.5">Conformal prediction (90% shortlist)<InfoTip align="right" text="Split conformal prediction with class-wise thresholds, fitted on 17,534 held-out calibration flows (research notebook 17)." /></span>}
          subtitle="How 'attack type uncertain' is decided">
          <p className="text-sm text-slate-700">For every flow the model also returns a <b>shortlist</b> of the classes it cannot rule out. The shortlist is built so that it contains the true class
            about {pct(1 - calibration.alpha, 0)} of the time on data like the training data. If a malicious flow's shortlist contains more than one attack type, the flow is reported as
            <b> attack type uncertain</b>: it is an attack, but the exact type needs checking.</p>
          <p className="mt-2 text-xs text-slate-500">On the shifted official test set the shortlist held the true class less often than 90%; this limitation is reported in notebook 21.</p>
        </Card>
        <Card title="Attack severity" subtitle="A fixed rule per attack type (not a model output)" bodyClassName="p-0">
          <table className="min-w-full text-sm">
            <tbody className="divide-y divide-slate-100">
              {(['Critical', 'High', 'Medium', 'Low'] as const).map(s => (
                <tr key={s}>
                  <td className="px-4 py-2.5 align-top"><SeverityBadge severity={s} /></td>
                  <td className="px-3 py-2.5 text-xs text-slate-600"><p className="font-medium text-slate-800">{Object.entries(rules.severity).filter(([, v]) => v === s).map(([c]) => c).join(', ')}</p>{rules.severity_reason[s]}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
        <Card title="Batch triage priority" subtitle="A fixed rule, checked from the top; serious = Critical or High severity flows. Not a model output." bodyClassName="p-0">
          <table className="min-w-full text-sm">
            <tbody className="divide-y divide-slate-100">
              {rules.risk.map(r => <tr key={r.level}><td className="px-4 py-2.5"><RiskBadge level={r.level} /></td><td className="px-3 py-2.5 text-xs text-slate-600">{r.rule}</td></tr>)}
            </tbody>
          </table>
        </Card>
      </div>

      <div className="mt-6 grid gap-6 lg:grid-cols-2">
        <Card title="Known limitations">
          <ul className="space-y-3 text-sm">{card.limitations.map((l, k) => <li key={k} className="flex gap-2"><AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-500" aria-hidden />{l}</li>)}</ul>
        </Card>
        <Card title="Deployed model check" subtitle="The served model is verified against the research model">
          {bundle.verification && (
            <ul className="space-y-2 text-sm">
              {Object.entries(bundle.verification).map(([name, v]) => (
                <li key={name} className="flex gap-2"><BadgeCheck className="mt-0.5 h-4 w-4 shrink-0 text-emerald-500" aria-hidden />
                  Identical predictions to the research detector on {num(v.rows)} {name.replace('_', ' ')} flows (largest difference {v.max_abs_difference}).</li>
              ))}
            </ul>
          )}
          <p className="mt-3 text-xs text-slate-500">Bundle built {bundle.created} from commit {bundle.source_commit?.slice(0, 10)}.</p>
        </Card>
      </div>
    </>
  )
}
