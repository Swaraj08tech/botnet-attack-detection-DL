import { useQuery } from '@tanstack/react-query'
import { AlertTriangle, ArrowRight } from 'lucide-react'
import { Card, ClassBadge, PageHeader, SeverityBadge } from '../components/ui'
import { api, type Severity } from '../lib/api'
import { pct } from '../lib/format'

// Class descriptions paraphrased from the UNSW-NB15 dataset paper (Moustafa & Slay, 2015).
const CLASSES: { name: string; text: string }[] = [
  { name: 'Normal', text: 'Ordinary network traffic.' },
  { name: 'Generic', text: 'Attacks that work against block ciphers in general, regardless of the cipher’s structure.' },
  { name: 'Exploits', text: 'Use of a known weakness in an operating system or application.' },
  { name: 'Fuzzers', text: 'Random input sent to make a program or network service fail.' },
  { name: 'DoS', text: 'Attempts to make a server or network resource unavailable.' },
  { name: 'Reconnaissance', text: 'Probes and scans that gather information about hosts and services.' },
  { name: 'Analysis', text: 'Port scans, spam and HTML-file probing.' },
  { name: 'Backdoor', text: 'Bypassing security controls to reach a computer or its data.' },
  { name: 'Shellcode', text: 'A small piece of code used as the payload of an exploit.' },
  { name: 'Worms', text: 'Malicious code that copies itself to other computers.' },
]

const PIPELINE = [
  ['Validation', 'The file must contain the 42 UNSW-NB15 flow measurements; types, missing and infinite values are checked.'],
  ['Preparation', 'Protocol, service and state are encoded, and two byte features are added (195 model inputs).'],
  ['Detection', 'A LightGBM model assigns each flow to Normal or one of nine attack types.'],
  ['Confidence', 'Probabilities are calibrated, and each flow gets a 90% shortlist of classes the model cannot rule out.'],
  ['Explanation', 'TreeSHAP shows which measurements raised or lowered the score of the predicted class.'],
  ['Reporting', 'Summary and detailed reports. Text written by the local language model is checked against cited sources; statements that fail are removed.'],
]

const EVALUATION = [
  'Training and test data are kept separate, including identical flows that appear in both files.',
  'Several model types were compared, including neural networks; the final model was chosen on held-out training data.',
  'The official test file was used only for final measurements, never for choosing settings.',
  'Explanations, calibration and the 90% shortlist were checked on held-out data.',
]

export default function About() {
  const model = useQuery({ queryKey: ['model'], queryFn: api.model })
  const severity = model.data?.rules.severity ?? {}
  return (
    <>
      <PageHeader title="About" description="Detection of network attacks in recorded flow data, using a model trained on the UNSW-NB15 dataset, with explanations and reports." />
      <div className="grid gap-6 lg:grid-cols-3">
        <Card title="What the system does" className="lg:col-span-2">
          <div className="space-y-3 text-sm text-slate-700">
            <p>The application analyses files of <b>completed network flows</b>: one row per connection, with 42 measurements such as duration, bytes,
              packets and connection counts. For each flow the model predicts <b>Normal</b> or one of <b>nine attack types</b>, states how confident it is,
              and shows which measurements influenced the prediction.</p>
            <p>It works <b>offline</b>: you upload a file and receive results and a report. It does not capture live traffic. Converting packet captures
              into flow records requires a separate tool, such as Argus or Zeek.</p>
            <p>Results are <b>predictions</b>. They help decide what to check first; they do not confirm an incident.</p>
          </div>
          <ol className="mt-5 flex flex-wrap items-center gap-2 text-xs font-medium text-slate-700">
            {['Upload CSV', 'Validate', 'Detect', 'Review results', 'Report'].map((s, i, all) => (
              <li key={s} className="flex items-center gap-2"><span className="rounded-full bg-brand-50 px-3 py-1 text-brand-700">{s}</span>{i < all.length - 1 && <ArrowRight className="h-3.5 w-3.5 text-slate-400" />}</li>
            ))}
          </ol>
        </Card>
        <Card title="Technology">
          <dl className="space-y-2 text-sm">
            {[['Model', 'LightGBM (Python)'], ['Backend', 'FastAPI, SQLAlchemy, SQLite or PostgreSQL, ReportLab'],
              ['Frontend', 'React, TypeScript, Tailwind CSS, Recharts'], ['Reports', 'Ollama with qwen2.5:7b (runs locally); MITRE ATT&CK, NIST and CISA documents'],
              ['Deployment', 'Docker, GitHub Actions']].map(([k, v]) => (
              <div key={k}><dt className="text-xs text-slate-500">{k}</dt><dd className="text-slate-800">{v}</dd></div>
            ))}
          </dl>
        </Card>
      </div>

      <div className="mt-6 grid gap-6 lg:grid-cols-2">
        <Card title="Dataset: UNSW-NB15" subtitle="Official training (175,341 flows) and testing (82,332 flows) files from UNSW Canberra" bodyClassName="p-0">
          <table className="min-w-full text-sm">
            <tbody className="divide-y divide-slate-100">
              {CLASSES.map(c => (
                <tr key={c.name}>
                  <td className="whitespace-nowrap px-4 py-2 align-top"><ClassBadge name={c.name} severity={c.name === 'Normal' ? 'None' : (severity[c.name] as Severity)} /></td>
                  <td className="px-3 py-2 text-xs text-slate-600">{c.text}</td>
                  <td className="px-4 py-2 align-top">{c.name !== 'Normal' && severity[c.name] && <SeverityBadge severity={severity[c.name] as Severity} />}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="px-4 py-3 text-xs text-slate-500">UNSW-NB15 has no class named "botnet". Behaviour typical of botnets (spreading, backdoors, scanning,
            flooding) falls under Worms, Backdoor, Reconnaissance and DoS. Severity is a fixed rule per class, not a model output.</p>
        </Card>
        <div className="space-y-6">
          <Card title="How it works">
            <ol className="space-y-2.5 text-sm">
              {PIPELINE.map(([k, v], i) => (
                <li key={k} className="flex gap-3"><span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-brand-600 text-xs font-semibold text-white">{i + 1}</span>
                  <div><p className="font-medium text-slate-900">{k}</p><p className="text-xs text-slate-600">{v}</p></div></li>
              ))}
            </ol>
          </Card>
          <Card title="How the model was evaluated">
            <ul className="space-y-1.5 text-sm text-slate-700">{EVALUATION.map(t => <li key={t} className="flex gap-2"><span aria-hidden className="text-slate-400">•</span>{t}</li>)}</ul>
            <p className="mt-3 text-xs text-slate-500">Scores and the full comparison are on the Model page.</p>
          </Card>
        </div>
      </div>

      <Card className="mt-6" title="Limitations">
        <ul className="grid gap-3 text-sm sm:grid-cols-2">
          {['The model was trained and tested on UNSW-NB15 laboratory traffic. Its accuracy on other networks is unknown until it is checked with labelled local traffic.',
            `On the official test file, ${model.data?.model_card.evaluation ? pct(model.data.model_card.evaluation.binary.false_positive_rate, 0) : 'a share'} of Normal flows were flagged as attacks, mainly because the official training and test files differ.`,
            'Analysis, Backdoor and DoS flows often have identical measurements, so their exact type is often uncertain; the 90% shortlist shows this.',
            'The flow files contain no IP addresses, ports or timestamps, so flows are identified by their row number and cannot be traced to hosts.'].map(t => (
            <li key={t} className="flex gap-2"><AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-500" />{t}</li>
          ))}
        </ul>
      </Card>
    </>
  )
}
