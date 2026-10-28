import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import * as client from '../lib/api'
import { ApiError, api, type ActionState, type ReportRecord } from '../lib/api'
import { prob } from '../lib/format'
import { recordToMarkdown, resolveCitation, safeUrl, splitSpan } from '../lib/record'
import { ToastProvider } from '../lib/toast'
import ReportPage from '../pages/ReportPage'
import fixture from './fixtures/record.json'

const base = fixture as unknown as ReportRecord
const clone = (): ReportRecord => JSON.parse(JSON.stringify(base))
let location = ''
function Where() { location = useLocation().search; return null }

function renderReport(record: ReportRecord | Error, path = '/reports/5?view=detailed') {
  vi.spyOn(api, 'reportRecord').mockImplementation(() => (record instanceof Error ? Promise.reject(record) : Promise.resolve(record)))
  vi.spyOn(api, 'actionStates').mockResolvedValue({})
  if (!vi.isMockFunction(api.analysis)) vi.spyOn(api, 'analysis').mockReturnValue(new Promise(() => {}))   // the flow table is not under test here
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={qc}><ToastProvider>
      <MemoryRouter initialEntries={[path]}><Routes><Route path="/reports/:reportId" element={<><ReportPage /><Where /></>} /></Routes></MemoryRouter>
    </ToastProvider></QueryClientProvider>,
  )
}

beforeEach(() => { vi.restoreAllMocks(); try { localStorage.clear() } catch { /* no storage */ } })
afterEach(() => vi.restoreAllMocks())

describe('record helpers', () => {
  it('resolves namespaced evidence ids and source ids, and flags unknown ones', () => {
    const r = clone()
    expect(resolveCitation(r, 'F284.E1')).toMatchObject({ kind: 'evidence', flow: { flow_index: 284 } })
    expect(resolveCitation(r, 'S1')).toMatchObject({ kind: 'source', passage: { location: 'T1498' } })
    expect(resolveCitation(r, 'E1').kind).toBe('missing')                   // un-namespaced ids never resolve to another flow
    expect(resolveCitation(r, 'S999').kind).toBe('missing')
  })
  it('highlights only a span that matches the passage text', () => {
    expect(splitSpan('abc def ghi', 4, 7, 'def')).toEqual(['abc ', 'def', ' ghi'])
    expect(splitSpan('abc def ghi', 4, 7, 'xyz')).toBeNull()
    expect(splitSpan('abc', 2, 9, 'c')).toBeNull()
  })
  it('renders only https links to known reference hosts', () => {
    expect(safeUrl('https://attack.mitre.org/techniques/T1498/')).toBe('https://attack.mitre.org/techniques/T1498/')
    expect(safeUrl('javascript:alert(1)')).toBeNull()
    expect(safeUrl('http://attack.mitre.org/x')).toBeNull()
    expect(safeUrl('https://evil.example.com/')).toBeNull()
  })
  it('never shows a rounded probability as certainty', () => {
    expect(prob(0.9978)).toBe('above 99.5%')
    expect(prob(0.575)).toBe('57.5%')
  })
  it('exports Markdown with the main findings, six parts, citations and the checklist state', () => {
    const states = { A1: { status: 'reviewed', note: '', by: 'a', at: '2026-10-26T10:00:00', history: [] } } as Record<string, ActionState>
    const md = recordToMarkdown(clone(), states)
    for (const h of ['## A. Summary', '## B. Detections and uncertainty', '## C. Why these predictions', '## D. Sources', '## E. Actions', '## F. Limits'])
      expect(md).toContain(h)
    expect(md).toContain(base.views.summary.overview.bullets[1].text)
    expect(md).toContain('Batch triage priority: **CRITICAL** (heuristic')
    expect(md).toContain('[F284.E1] Predicted class: DoS')
    expect(md).toMatch(/\| A1 \| Verify \|.*\| Reviewed \|/)
  })
})

describe('summary and detailed views', () => {
  it('defaults to the Summary view with six sections and one 3-5 bullet explanation of about 100-150 words', async () => {
    renderReport(clone(), '/reports/5')
    expect(await screen.findByRole('radio', { name: /Summary/ })).toHaveAttribute('aria-checked', 'true')
    for (const h of ['1. Overall flow summary', '2. What was detected', '3. Why the traffic was flagged', '4. What to do', '5. How to check the outcome', '6. Limits and sources'])
      expect(screen.getByRole('heading', { name: h })).toBeInTheDocument()
    expect(screen.getByText('Does the analysed traffic show signs of an attack?')).toBeInTheDocument()
    const bullets = within(within(screen.getByRole('region', { name: '3. Why the traffic was flagged' })).getAllByRole('list')[0]).getAllByRole('listitem')
    expect(bullets.length).toBeGreaterThanOrEqual(3)
    expect(bullets.length).toBeLessThanOrEqual(5)
    const words = bullets.map(b => b.textContent ?? '').join(' ').split(/\s+/).filter(Boolean).length
    expect(words).toBeGreaterThanOrEqual(90)
    expect(words).toBeLessThanOrEqual(160)
    const text = bullets.map(b => b.textContent).join(' ')
    expect(text).not.toMatch(/\+\d\.\d{3}|contribution|SHAP|\bsttl\b|ct_srv/)     // no scores, codes or jargon in the main explanation
    expect(text).not.toMatch(/will happen|going to happen|confirmed attack/i)
  })

  it('shows the same findings, counts and action ids in both views', async () => {
    renderReport(clone(), '/reports/5')
    await screen.findByRole('heading', { name: '3. Why the traffic was flagged' })
    const summaryBullets = within(within(screen.getByRole('region', { name: '3. Why the traffic was flagged' })).getAllByRole('list')[0]).getAllByRole('listitem').map(li => li.textContent)
    const summaryActionIds = base.views.summary.actions.map(a => a.id)
    fireEvent.click(screen.getByRole('radio', { name: /Detailed/ }))
    expect(await screen.findByRole('tab', { name: 'Assessment' })).toHaveAttribute('aria-selected', 'true')
    for (const b of base.views.summary.overview.bullets) expect(screen.getAllByText(b.text, { exact: false }).length).toBeGreaterThan(0)
    expect(summaryBullets.map(t => t?.replace(/\s+/g, ' ').trim())).toEqual(base.views.summary.overview.bullets.map(b => b.text))
    const detailedIds = base.actions.rows.map(a => a.id)
    expect(summaryActionIds.every(id => detailedIds.includes(id))).toBe(true)          // the Summary only selects from the same action list
  })

  it('remembers the chosen view, keeps the selected flow when switching, and switches with the arrow keys', async () => {
    renderReport(clone(), '/reports/5?tab=why&flow=F134')
    expect(await screen.findByRole('radio', { name: /Detailed/ })).toHaveAttribute('aria-checked', 'true')    // a deep link opens Detailed
    fireEvent.keyDown(screen.getByRole('radiogroup', { name: 'Report view' }), { key: 'ArrowLeft' })
    expect(await screen.findByRole('radio', { name: /Summary/ })).toHaveAttribute('aria-checked', 'true')
    expect(location).toContain('flow=F134')
    expect(localStorage.getItem('nad.reportView')).toBe('summary')
    fireEvent.click(screen.getByRole('radio', { name: /Detailed/ }))
    expect(await screen.findByRole('article', { name: 'Flow 134' })).toBeInTheDocument()
    expect(localStorage.getItem('nad.reportView')).toBe('detailed')
  })

  it('offers clearly labelled Summary and Detailed PDF downloads for the same report', async () => {
    const download = vi.spyOn(client, 'downloadFile').mockResolvedValue(undefined)
    renderReport(clone(), '/reports/5')
    fireEvent.click(await screen.findByRole('button', { name: /Download Summary PDF/ }))
    fireEvent.click(screen.getByRole('button', { name: /Download Detailed PDF/ }))
    expect(download.mock.calls.map(c => c[0])).toEqual(['/reports/5/pdf?download=true&view=summary', '/reports/5/pdf?download=true'])
  })

  it('opens the evidence of the summary in the detailed view', async () => {
    renderReport(clone(), '/reports/5')
    fireEvent.click(await screen.findByRole('button', { name: 'View evidence' }))
    expect(await screen.findByRole('tab', { name: 'Why this prediction' })).toHaveAttribute('aria-selected', 'true')
    expect(screen.getByRole('table', { name: /TreeSHAP contributions for flow/ })).toBeInTheDocument()
  })

  it('shows both confidences and the checked AI explanation points in plain words', async () => {
    renderReport(clone(), '/reports/5')
    expect(await screen.findByText('Attack or normal?')).toBeInTheDocument()
    expect(screen.getByText('Exact attack type?')).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Examples: the reasoning for one flow of each main type' })).toBeInTheDocument()
    expect(screen.getAllByText('How sure the model is:').length).toBe(base.views.summary.examples.length)
    expect(screen.getAllByText('AI, checked').length).toBeGreaterThan(0)
    const examples = base.views.summary.examples
    expect(examples.length).toBeGreaterThan(0)
    for (const ex of examples) expect(ex.points.length).toBeGreaterThanOrEqual(2)
    const shown = screen.getAllByText(/confident that this flow is/).map(el => el.textContent ?? '')
    expect(shown.some(t => /\d\d% confident/.test(t))).toBe(true)                         // "0.66" shown as "66%"
    expect(document.body.textContent).not.toMatch(/\(contribution [+-]/)                    // no raw scores in the Summary view
    expect(screen.getAllByText('(written from the model evidence)').length).toBeGreaterThan(0)
  })

  it('shows the five reasoning points on each flow card of the detailed view', async () => {
    renderReport(clone(), '/reports/5?view=detailed&tab=why&flow=F134')
    expect(await screen.findByText('Reasoning')).toBeInTheDocument()
    for (const title of ['How sure the model is:', 'Main reasons:', 'How typical the values are:', 'Why not Normal:', 'In this file:'])
      expect(screen.getByText(title)).toBeInTheDocument()
    const reasoning = screen.getByText('Reasoning').parentElement!
    expect(within(reasoning).getByText(/occurs in 40.0% of training Exploits flows and 3.9% of training Normal flows/)).toBeInTheDocument()
  })

  it('keeps component failures visible in the Summary view', async () => {
    const r = clone()
    r.components.llm = { status: 'unavailable', detail: 'evidence_only_llm_unavailable: 2' }
    renderReport(r, '/reports/5')
    expect(await screen.findByText(/Part of this report could not be produced: llm unavailable/)).toBeInTheDocument()
  })
})

describe('detailed report', () => {
  it('moves between tabs with the arrow keys', async () => {
    renderReport(clone())
    const first = await screen.findByRole('tab', { name: 'Assessment' })
    fireEvent.keyDown(first, { key: 'ArrowRight' })
    expect(await screen.findByRole('tab', { name: 'Detections' })).toHaveAttribute('aria-selected', 'true')
    fireEvent.keyDown(screen.getByRole('tab', { name: 'Detections' }), { key: 'End' })
    expect(screen.getByRole('tab', { name: 'Limits & model' })).toHaveAttribute('aria-selected', 'true')
    expect(screen.getByText('What this report cannot tell you')).toBeInTheDocument()
  })

  it('shows group findings with their method, separate from single-flow cards', async () => {
    renderReport(clone(), '/reports/5?tab=detections')
    expect(await screen.findByText('Group findings')).toBeInTheDocument()
    expect(screen.getByText(/Contributions are not added across flows/)).toBeInTheDocument()
  })

  it('explains a flow with exact TreeSHAP evidence, an interpretation, alternatives and the cited passage with its highlighted span', async () => {
    renderReport(clone(), '/reports/5?tab=why&flow=F284')
    expect(await screen.findByText(/Exact decomposition: base value -4.760/)).toBeInTheDocument()
    expect(screen.getByText(/For this flow, the strongest reasons towards DoS were/)).toBeInTheDocument()
    expect(screen.getByText(/Other classes the model did not rule out: Exploits/)).toBeInTheDocument()
    expect(screen.getByText('not generated')).toBeInTheDocument()                                  // LIME is labelled, not faked
    fireEvent.click(screen.getAllByRole('button', { name: 'Open citation S1' }).slice(-1)[0])
    const dialog = await screen.findByRole('dialog', { name: /Source S1/ })
    expect(within(dialog).getByText(/untrusted reference text/)).toBeInTheDocument()
    expect(dialog.querySelector('mark')?.textContent).toMatch(/^Adversaries may perform Network Denial of Service/)
    fireEvent.keyDown(window, { key: 'Escape' })
    await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument())
  })

  it('opens evidence citations of the right flow and lists rejected statements', async () => {
    const r = clone()
    r.flows[1].llm!.removed = [{ part: 'evidence', text: 'The model is 1.00 confident', cites: ['F134.E1'], reason: 'probability written as 1.00', round: 1 }]
    renderReport(r, '/reports/5?tab=why&flow=F134')
    fireEvent.click((await screen.findAllByRole('button', { name: 'Open citation F134.E3' }))[0])
    const dialog = await screen.findByRole('dialog', { name: 'Evidence F134.E3' })
    expect(within(dialog).getByText(/#134/)).toBeInTheDocument()
    fireEvent.click(within(dialog).getByRole('button', { name: 'Close' }))
    const rejected = screen.getByRole('button', { name: /rejected by the checks/ })
    fireEvent.click(rejected)
    expect(rejected).toHaveAttribute('aria-expanded', 'true')
  })

  it('states action-specific requirements: a prediction alone never justifies containment', async () => {
    renderReport(clone(), '/reports/5?tab=actions')
    const cred = (await screen.findByText(/Change system admin passwords/)).closest('li')!
    expect(within(cred).getByText(/Authentication logs or endpoint alerts show that a named account/)).toBeInTheDocument()
    const filtering = screen.getAllByText(/Filter Network Traffic/)[0].closest('li')!
    expect(within(filtering).getByText(/source, destination and service are identified/)).toBeInTheDocument()
    expect(within(filtering).getByText(/Remove the rule or let it expire/)).toBeInTheDocument()
  })

  it('is honest when the LLM was unavailable and offers a retry', async () => {
    const r = clone()
    r.components.llm = { status: 'unavailable', detail: 'evidence_only_llm_unavailable: 2' }
    r.flows.forEach(f => {
      if (f.llm) Object.assign(f.llm, { status: 'evidence_only_llm_unavailable', explanation: [], techniques: [], context: [], actions: [], removed: [], checked: 0 })
      f.reasoning = (f.reasoning ?? []).map(x => ({ ...x, origin: 'xai_attribution', cites: [] }))       // without the LLM every point is written by code
    })
    renderReport(r, '/reports/5?tab=why&flow=F284')
    expect(await screen.findByText(/Retry \(generate again\)/)).toBeInTheDocument()
    expect(screen.getByText('evidence only: local LLM unavailable')).toBeInTheDocument()
    expect(screen.getByText('Reasoning')).toBeInTheDocument()
    expect(screen.queryByText('AI, checked')).not.toBeInTheDocument()
  })

  it('shows an older schema-1 report as PDF-only', async () => {
    renderReport(new ApiError(404, 'This is a schema-1 report (created before report records existed). Open its PDF or generate a new report.'))
    expect(await screen.findByText(/older version of the application \(schema 1\)/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Open the PDF/ })).toBeInTheDocument()
  })

  it('filters the flow list from a class, an uncertainty card and resets', async () => {
    vi.spyOn(api, 'analysis').mockResolvedValue({ analysis: { attack_distribution: [{ attack_type: 'DoS', count: 70 }] } } as never)
    const flows = vi.spyOn(api, 'flows').mockResolvedValue({ total: 0, page: 1, page_size: 25, items: [] })
    renderReport(clone(), '/reports/5?tab=detections')
    fireEvent.click(await screen.findByRole('button', { name: 'Show DoS flows' }))
    await waitFor(() => expect(flows).toHaveBeenLastCalledWith(1, expect.objectContaining({ attack_type: 'DoS', page: 1 })))
    fireEvent.click(screen.getByRole('button', { name: /Attack status uncertain/ }))
    await waitFor(() => expect(flows).toHaveBeenLastCalledWith(1, expect.objectContaining({ uncertainty: 'status' })))
    expect(flows.mock.lastCall![1]).not.toHaveProperty('attack_type')
    fireEvent.click(screen.getByRole('button', { name: /Clear/ }))
    await waitFor(() => expect(flows.mock.lastCall![1]).toEqual({ page: 1, page_size: 25 }))
  })

  it('saves an action status with a note and shows its history', async () => {
    const saved: ActionState = { status: 'reviewed', note: 'checked pcap', by: 'admin@test.local', at: '2026-10-26T10:00:00', history: [{ status: 'reviewed', note: 'checked pcap', by: 'admin@test.local', at: '2026-10-26T10:00:00' }] }
    renderReport(clone(), '/reports/5?tab=actions')
    const update = vi.spyOn(api, 'updateAction').mockResolvedValue(saved)
    const select = await screen.findByRole('combobox', { name: 'Status of A1' })
    fireEvent.change(select, { target: { value: 'reviewed' } })
    fireEvent.change(screen.getByRole('textbox', { name: 'Note for A1' }), { target: { value: 'checked pcap' } })
    fireEvent.click(within(select.closest('li')!).getByRole('button', { name: 'Save' }))
    await waitFor(() => expect(update).toHaveBeenCalledWith(5, 'A1', 'reviewed', 'checked pcap'))
    expect(await screen.findByText(/Reviewed by admin@test.local/)).toBeInTheDocument()
  })
})
