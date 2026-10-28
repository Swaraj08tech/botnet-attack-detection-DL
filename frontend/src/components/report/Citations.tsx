import clsx from 'clsx'
import { ExternalLink, ShieldAlert, X } from 'lucide-react'
import { useContext, useEffect, useRef, type ReactNode } from 'react'
import type { Origin, ReportRecord, Support } from '../../lib/api'
import { CitationContext, ORIGIN_LABEL, resolveCitation, safeUrl, splitSpan } from '../../lib/record'

/** Clickable citation chips. E = evidence computed from the model, S = retrieved source passage. */
export function Cites({ ids, support }: { ids: string[]; support?: Support | null }) {
  const open = useContext(CitationContext)
  if (!ids.length) return null
  return (
    <span className="ml-1 inline-flex flex-wrap gap-0.5 align-baseline">
      {ids.map(id => (
        <button key={id} type="button" onClick={() => open(id, support && support.source === id ? support : null)}
          className={clsx('rounded px-1 py-px text-[10px] font-semibold tabular-nums ring-1 ring-inset hover:underline focus:outline-none focus-visible:ring-2 focus-visible:ring-brand-500',
            id.startsWith('S') ? 'bg-sky-50 text-sky-700 ring-sky-200' : 'bg-slate-100 text-slate-700 ring-slate-200')}
          aria-label={`Open citation ${id}`}>
          {id}
        </button>
      ))}
    </span>
  )
}

const ORIGIN_STYLE: Record<Origin, string> = {
  observed: 'bg-slate-100 text-slate-700 ring-slate-200',
  batch_statistic: 'bg-slate-100 text-slate-700 ring-slate-200',
  model_output: 'bg-indigo-50 text-indigo-700 ring-indigo-200',
  xai_attribution: 'bg-indigo-50 text-indigo-700 ring-indigo-200',
  retrieved_guidance: 'bg-sky-50 text-sky-700 ring-sky-200',
  llm_synthesis: 'bg-violet-50 text-violet-700 ring-violet-200',
  analyst_proposed: 'bg-amber-50 text-amber-800 ring-amber-200',
  heuristic: 'bg-amber-50 text-amber-800 ring-amber-200',
}

/** Where a statement comes from: model, XAI, retrieved source, LLM (checked), heuristic or a proposed next step. */
export function OriginTag({ origin }: { origin: Origin }) {
  return <span className={clsx('ml-1.5 inline-block whitespace-nowrap rounded px-1.5 py-px align-middle text-[10px] font-semibold ring-1 ring-inset', ORIGIN_STYLE[origin])}>{ORIGIN_LABEL[origin]}</span>
}

/** Modal dialog with Escape-to-close and initial focus on the close button. */
export function Dialog({ label, onClose, children, wide }: { label: string; onClose: () => void; children: ReactNode; wide?: boolean }) {
  const close = useRef<HTMLButtonElement>(null)
  useEffect(() => {
    const before = document.activeElement as HTMLElement | null
    close.current?.focus()
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => { window.removeEventListener('keydown', onKey); before?.focus?.() }
  }, [onClose])
  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-ink-900/50 p-4 pt-[8vh] backdrop-blur-[2px]" onMouseDown={e => { if (e.target === e.currentTarget) onClose() }}>
      <div role="dialog" aria-modal="true" aria-label={label} className={clsx('w-full rounded-xl bg-white shadow-2xl', wide ? 'max-w-3xl' : 'max-w-xl')}>
        <div className="flex items-center justify-between gap-3 border-b border-slate-100 px-5 py-3">
          <h2 className="truncate text-sm font-semibold text-slate-900">{label}</h2>
          <button ref={close} type="button" onClick={onClose} aria-label="Close" className="rounded p-1 text-slate-500 hover:bg-slate-100 focus:outline-none focus-visible:ring-2 focus-visible:ring-brand-500"><X className="h-4 w-4" /></button>
        </div>
        <div className="max-h-[75vh] overflow-y-auto px-5 py-4">{children}</div>
      </div>
    </div>
  )
}

function Meta({ rows }: { rows: [string, ReactNode][] }) {
  return (
    <dl className="grid grid-cols-[8rem_1fr] gap-x-3 gap-y-1 text-xs">
      {rows.map(([k, v]) => <div key={k} className="contents"><dt className="text-slate-500">{k}</dt><dd className="min-w-0 break-words text-slate-800">{v ?? '–'}</dd></div>)}
    </dl>
  )
}

/** The exact supporting passage (with the matched span highlighted) or the exact evidence fact behind a citation. */
export function CitationDialog({ record, id, support, onClose }: { record: ReportRecord; id: string; support?: Support | null; onClose: () => void }) {
  const c = resolveCitation(record, id)
  if (c.kind === 'missing') {
    return <Dialog label={`Citation ${id}`} onClose={onClose}><p className="text-sm text-rose-700">This citation does not resolve to any evidence or passage in this report.</p></Dialog>
  }
  if (c.kind === 'evidence') {
    return (
      <Dialog label={`Evidence ${id}`} onClose={onClose}>
        <p className="rounded-lg bg-slate-50 px-3 py-2 text-sm text-slate-800">{c.text}</p>
        <div className="mt-3"><Meta rows={[
          ['Flow', `#${c.flow.flow_index} (${c.flow.id}), predicted ${c.flow.prediction.class}`],
          ['Origin', 'computed by code from the model output and its TreeSHAP attribution (not written by the LLM)'],
          ['Exactness', record.provenance.explanation.exactness],
        ]} /></div>
      </Dialog>
    )
  }
  const p = c.passage
  const parts = support ? splitSpan(p.text, support.start, support.end, support.span) : null
  return (
    <Dialog label={`Source ${id} · ${p.source}, ${p.location}`} onClose={onClose} wide>
      <p className="mb-2 flex items-start gap-1.5 rounded-lg bg-amber-50 px-3 py-2 text-xs text-amber-900"><ShieldAlert className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden />{p.trust}</p>
      <blockquote className="whitespace-pre-wrap rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm leading-relaxed text-slate-800">
        {parts ? <>{parts[0]}<mark className="rounded bg-yellow-200 px-0.5 text-slate-900">{parts[1]}</mark>{parts[2]}</> : p.text}
      </blockquote>
      {support && (
        <p className="mt-2 text-xs text-slate-600">
          {parts ? <>Highlighted: the sentence that supports the statement (characters {support.start}–{support.end}, word overlap {(support.overlap * 100).toFixed(0)}%). </> : <>The recorded supporting span was not found in the passage text. </>}
          <span className="text-slate-500">{support.method}</span>
        </p>
      )}
      <div className="mt-3"><Meta rows={[
        ['Source', p.source], ['Publisher', p.publisher], ['Title', p.title], ['Version', p.version], ['Location', p.location],
        ['Link', safeUrl(p.url) ? <a className="inline-flex items-center gap-1 text-brand-600 hover:underline" href={safeUrl(p.url)!} target="_blank" rel="noopener noreferrer">{p.url}<ExternalLink className="h-3 w-3" aria-hidden /></a> : p.url],
        ['Retrieved', p.retrieved], ['Retrieval', `${p.retrieval.method ?? '–'}${p.retrieval.score !== null ? ` · score ${p.retrieval.score}` : ''}${p.retrieval.query ? ` · query “${p.retrieval.query}”` : ''}`],
        ['Cited by', p.used_by.join(', ')], ['Passage sha256', <code className="text-[11px]">{p.content_sha256}</code>],
        ['Document sha256', p.document_sha256 ? <code className="text-[11px]">{p.document_sha256}</code> : null],
      ]} /></div>
    </Dialog>
  )
}
