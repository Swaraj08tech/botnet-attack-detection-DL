import { useQuery } from '@tanstack/react-query'
import { BarChart3, BrainCircuit, CheckCircle2, FileSpreadsheet, FileText, FolderOpen, ShieldAlert, Target, Upload } from 'lucide-react'
import { Link, useNavigate } from 'react-router-dom'
import { AttackBars, SeverityLegend, TrafficDonut } from '../components/charts'
import { Button, Card, Empty, ErrorBox, KpiCard, PageHeader, RiskBadge, Spinner } from '../components/ui'
import { api } from '../lib/api'
import { num, pct, when } from '../lib/format'

const PIPELINE = [
  { icon: FileSpreadsheet, title: 'Upload flows', text: 'A CSV of completed network flows (UNSW-NB15 features)' },
  { icon: CheckCircle2, title: 'Validate', text: 'Required features, values and format are checked' },
  { icon: BrainCircuit, title: 'Detect', text: 'LightGBM classifies each flow: Normal or 9 attack types' },
  { icon: BarChart3, title: 'Results', text: 'Predicted Normal vs attack, attack types, confidence, review priority' },
  { icon: FileText, title: 'Report', text: 'Download a PDF security report' },
]

export default function Overview() {
  const navigate = useNavigate()
  const overview = useQuery({ queryKey: ['overview'], queryFn: api.overview })
  if (overview.isLoading) return <Spinner />
  if (overview.error) return <ErrorBox error={overview.error} />
  const o = overview.data!

  return (
    <>
      <PageHeader kicker="Network attack detection" title="Overview"
        description="Upload a file of recorded network flows. A model trained on UNSW-NB15 predicts which flows look like attacks, which attack types they resemble and how confident it is."
        actions={<Link to="/analyze"><Button><Upload className="h-4 w-4" />Analyze Flows</Button></Link>} />

      <Card className="mb-6" bodyClassName="p-4">
        <ol className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
          {PIPELINE.map(({ icon: Icon, title, text }, i) => (
            <li key={title} className="flex gap-3 rounded-lg bg-slate-50 p-3">
              <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-brand-50 text-brand-600"><Icon className="h-4 w-4" /></span>
              <div className="min-w-0"><p className="text-sm font-medium text-slate-900">{i + 1}. {title}</p><p className="text-xs text-slate-500">{text}</p></div>
            </li>
          ))}
        </ol>
      </Card>

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <KpiCard label="Files analyzed" value={num(o.files_analyzed)} detail="datasets uploaded and analysed" icon={<FolderOpen className="h-4 w-4" />} tone="brand" />
        <KpiCard label="Total flows" value={num(o.total_flows)} detail="across all analyses" icon={<BarChart3 className="h-4 w-4" />} />
        <KpiCard label="Total threats" value={num(o.total_threats)} detail={o.total_flows ? `${pct(o.total_threats / o.total_flows)} of all analysed flows` : 'no flows yet'}
          tone="threat" icon={<ShieldAlert className="h-4 w-4" />} tip="Flows classified as one of the nine attack types, summed over all analyses." />
        <KpiCard label="Attack detection F1" value={pct(o.model_attack_f1)} detail={`attack vs normal, official test set · 10-class accuracy ${pct(o.model_accuracy)}`}
          icon={<Target className="h-4 w-4" />} tip="F1 score of separating attacks from normal traffic, measured once on the official UNSW-NB15 test set (82,332 labelled flows the model never saw). Not calculated from your uploads. Details and context on the Model page." />
      </div>

      {o.files_analyzed === 0 ? (
        <div className="mt-6">
          <Empty icon={<Upload className="h-10 w-10" />} title="No datasets analysed yet" action={<Link to="/analyze"><Button>Analyze your first dataset</Button></Link>}>
            Start on the Analyze Flows page: upload a CSV of network flows or use the built-in sample of 300 real UNSW-NB15 test flows.
          </Empty>
        </div>
      ) : (
        <>
          <div className="mt-6 grid gap-6 lg:grid-cols-5">
            <Card title="Traffic Classification" subtitle="All analysed flows: predicted Normal vs predicted attack" className="lg:col-span-2">
              <TrafficDonut safe={o.safe_flows} malicious={o.total_threats} />
            </Card>
            <Card title="Detected Attacks by Type" subtitle="All analyses · predicted attacks only (bar colour = heuristic severity)" className="lg:col-span-3" actions={<SeverityLegend />}>
              <AttackBars data={o.attacks_by_type} />
            </Card>
          </div>
          <Card className="mt-6" title="Recent analyses" subtitle="Click an analysis to open its results" bodyClassName="p-0"
            actions={<Link to="/reports" className="text-xs font-medium text-brand-600 hover:underline">All reports</Link>}>
            <div className="overflow-x-auto">
              <table className="min-w-full text-sm">
                <thead className="bg-slate-50 text-left text-xs uppercase tracking-wide text-slate-500">
                  <tr><th className="px-5 py-2.5">Dataset</th><th className="px-3">Date</th><th className="px-3 text-right">Flows</th><th className="px-3 text-right">Predicted attacks</th><th className="px-3">Triage priority</th></tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {o.recent.map(a => (
                    <tr key={a.id} onClick={() => navigate(`/results/${a.id}`)} className="cursor-pointer hover:bg-slate-50">
                      <td className="max-w-72 truncate px-5 py-2.5 font-medium text-slate-800">{a.dataset_name}</td>
                      <td className="whitespace-nowrap px-3 text-slate-500">{when(a.created_at)}</td>
                      <td className="px-3 text-right tabular-nums">{num(a.total_flows)}</td>
                      <td className="px-3 text-right tabular-nums text-rose-600">{num(a.malicious_count)} <span className="text-slate-400">({pct(a.malicious_percentage)})</span></td>
                      <td className="px-3"><RiskBadge level={a.risk_level} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}
    </>
  )
}
