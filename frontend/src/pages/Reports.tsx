import { useQuery } from '@tanstack/react-query'
import { Download, Eye, FileText, LayoutList, Upload } from 'lucide-react'
import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import ReportPanel, { PdfPreview } from '../components/ReportPanel'
import { Button, Card, Empty, ErrorBox, PageHeader, RiskBadge, Spinner } from '../components/ui'
import { api, downloadFile, type AnalysisSummary } from '../lib/api'
import { num, pct, when } from '../lib/format'
import { useToast } from '../lib/toast'

export default function Reports() {
  const navigate = useNavigate()
  const toast = useToast()
  const reports = useQuery({ queryKey: ['reports'], queryFn: api.reports })
  const [generate, setGenerate] = useState<AnalysisSummary | null>(null)
  const [preview, setPreview] = useState<AnalysisSummary | null>(null)

  return (
    <>
      <PageHeader kicker="Insights" title="Reports" description="Every analysed dataset and its security report. Open the interactive report (evidence, sources, LLM explanations, action checklist), preview or download the PDF." />
      {reports.isLoading ? <Spinner /> : reports.error ? <ErrorBox error={reports.error} /> : !reports.data?.length ? (
        <Empty icon={<FileText className="h-10 w-10" />} title="No analyses yet" action={<Link to="/analyze"><Button><Upload className="h-4 w-4" />Analyze Flows</Button></Link>}>
          Reports are created from analysed datasets. Analyse a dataset first, then generate its security report.
        </Empty>
      ) : (
        <Card bodyClassName="p-0">
          <div className="overflow-x-auto">
            <table className="min-w-full text-sm">
              <thead className="bg-slate-50 text-left text-xs uppercase tracking-wide text-slate-500">
                <tr><th className="px-5 py-3">Dataset</th><th className="px-3">Date</th><th className="px-3 text-right">Flows</th><th className="px-3 text-right">Predicted attacks</th>
                  <th className="px-3">Triage priority</th><th className="px-3">Report status</th><th className="px-5 text-right">Actions</th></tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {reports.data.map(a => (
                  <tr key={a.id} className="hover:bg-slate-50/60">
                    <td className="max-w-56 px-5 py-3"><button className="block max-w-full truncate text-left font-medium text-slate-800 hover:text-brand-700 hover:underline" onClick={() => navigate(`/results/${a.id}`)}>{a.dataset_name}</button>
                      <p className="text-xs text-slate-400">analysis #{a.id}</p></td>
                    <td className="whitespace-nowrap px-3 text-slate-500">{when(a.created_at)}</td>
                    <td className="px-3 text-right tabular-nums">{num(a.total_flows)}</td>
                    <td className="whitespace-nowrap px-3 text-right tabular-nums text-rose-600">{num(a.malicious_count)} <span className="text-slate-400">({pct(a.malicious_percentage)})</span></td>
                    <td className="px-3"><RiskBadge level={a.risk_level} /></td>
                    <td className="px-3 text-xs">{a.report
                      ? <span className="whitespace-nowrap text-emerald-700" title={`Generated ${when(a.report.created_at)}`}>Generated · {a.report.pages} pages{a.report.has_record ? '' : ' · older PDF-only'}</span>
                      : <span className="whitespace-nowrap text-slate-400">Not generated</span>}</td>
                    <td className="px-5">
                      <div className="flex justify-end gap-1.5">
                        {a.report && <>
                          {a.report.has_record && <Button variant="ghost" className="px-2 py-1" title="Open interactive report" aria-label="Open interactive report" onClick={() => navigate(`/reports/${a.report!.id}`)}><LayoutList className="h-4 w-4" /></Button>}
                          <Button variant="ghost" className="px-2 py-1" title="Preview PDF" aria-label="Preview PDF" onClick={() => setPreview(a)}><Eye className="h-4 w-4" /></Button>
                          <Button variant="ghost" className="px-2 py-1" title="Download PDF" aria-label="Download PDF" onClick={() => downloadFile(api.reportPdfPath(a.report!.id, true)).catch(e => toast({ tone: 'error', title: 'Download failed', body: (e as Error).message }))}>
                            <Download className="h-4 w-4" /></Button>
                        </>}
                        <Button variant={a.report ? 'secondary' : 'primary'} className="px-2.5 py-1 text-xs" onClick={() => setGenerate(a)}><FileText className="h-3.5 w-3.5" />{a.report ? 'Regenerate' : 'Generate'}</Button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
      {generate && <ReportPanel analysis={generate} onClose={() => setGenerate(null)} />}
      {preview?.report && <PdfPreview reportId={preview.report.id} title={`Security report · ${preview.dataset_name}`} onClose={() => setPreview(null)} />}
    </>
  )
}
