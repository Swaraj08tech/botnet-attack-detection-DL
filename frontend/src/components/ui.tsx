import clsx from 'clsx'
import { Check, Info, Loader2, X } from 'lucide-react'
import type { ButtonHTMLAttributes, ReactNode } from 'react'
import type { ConfidenceLevel, RiskLevel, Severity } from '../lib/api'
import { CONFIDENCE_COLORS, CONFIDENCE_LABEL, RISK_STYLE, SEVERITY_COLORS } from '../lib/format'

export function Card({ title, subtitle, actions, children, className, bodyClassName }: {
  title?: ReactNode; subtitle?: ReactNode; actions?: ReactNode; children: ReactNode; className?: string; bodyClassName?: string
}) {
  return (
    <section className={clsx('min-w-0 rounded-xl border border-slate-200 bg-white shadow-sm', className)}>
      {(title || actions) && (
        <header className="flex flex-wrap items-start justify-between gap-3 border-b border-slate-100 px-5 py-4">
          <div className="min-w-0">
            {title && <h2 className="text-sm font-semibold text-slate-900">{title}</h2>}
            {subtitle && <p className="mt-0.5 text-xs text-slate-500">{subtitle}</p>}
          </div>
          {actions}
        </header>
      )}
      <div className={clsx('p-5', bodyClassName)}>{children}</div>
    </section>
  )
}

type Variant = 'primary' | 'secondary' | 'ghost' | 'danger'
export function Button({ variant = 'primary', loading, className, children, ...props }: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant; loading?: boolean }) {
  const styles: Record<Variant, string> = {
    primary: 'bg-brand-600 text-white shadow-sm hover:bg-brand-700 disabled:bg-brand-600/50',
    secondary: 'border border-slate-300 bg-white text-slate-700 hover:bg-slate-50 disabled:opacity-50',
    ghost: 'text-slate-600 hover:bg-slate-100 disabled:opacity-50',
    danger: 'bg-red-600 text-white hover:bg-red-700 disabled:opacity-50',
  }
  return (
    <button {...props} disabled={props.disabled || loading}
      className={clsx('inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-lg px-3.5 py-2 text-sm font-medium transition focus:outline-none focus-visible:ring-2 focus-visible:ring-brand-500 focus-visible:ring-offset-2 disabled:cursor-not-allowed', styles[variant], className)}>
      {loading && <Loader2 className="h-4 w-4 animate-spin" aria-hidden />}
      {children}
    </button>
  )
}

/** A small "i" that explains a value in plain words on hover or keyboard focus. Not rendered at all while hidden (no layout overflow). */
export function InfoTip({ text, align = 'center' }: { text: ReactNode; align?: 'center' | 'left' | 'right' }) {
  return (
    <span className="group relative inline-flex align-middle">
      <button type="button" className="rounded-full text-slate-400 hover:text-slate-600 focus:outline-none focus-visible:text-slate-700" aria-label="What does this mean?">
        <Info className="h-3.5 w-3.5" aria-hidden />
      </button>
      <span role="tooltip" className={clsx('pointer-events-none absolute bottom-full z-30 mb-2 hidden w-64 rounded-lg bg-ink-900 px-3 py-2 text-xs font-normal normal-case leading-relaxed tracking-normal text-white shadow-lg group-focus-within:block group-hover:block',
        align === 'center' && 'left-1/2 -translate-x-1/2', align === 'left' && 'left-0', align === 'right' && 'right-0')}>
        {text}
      </span>
    </span>
  )
}

export function KpiCard({ label, value, detail, tip, tone = 'default', icon }: {
  label: string; value: ReactNode; detail?: ReactNode; tip?: ReactNode; tone?: 'default' | 'safe' | 'threat' | 'warn' | 'brand'; icon?: ReactNode
}) {
  const color = { default: 'text-slate-900', safe: 'text-emerald-600', threat: 'text-rose-600', warn: 'text-amber-600', brand: 'text-brand-600' }[tone]
  const bar = { default: 'bg-slate-300', safe: 'bg-emerald-500', threat: 'bg-rose-500', warn: 'bg-amber-500', brand: 'bg-brand-500' }[tone]
  return (
    <div className="relative min-w-0 overflow-visible rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <span className={clsx('absolute inset-y-3 left-0 w-1 rounded-r', bar)} aria-hidden />
      <div className="flex items-center justify-between gap-2">
        <p className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-slate-500">{label}{tip && <InfoTip text={tip} align="left" />}</p>
        {icon && <span className="text-slate-400">{icon}</span>}
      </div>
      <p className={clsx('mt-1.5 text-2xl font-semibold tabular-nums', color)}>{value}</p>
      {detail && <p className="mt-0.5 text-xs text-slate-500">{detail}</p>}
    </div>
  )
}

export function PageHeader({ title, description, actions, kicker }: { title: ReactNode; description?: ReactNode; actions?: ReactNode; kicker?: ReactNode }) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
      <div className="min-w-0">
        {kicker && <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-brand-600">{kicker}</p>}
        <h1 className="text-xl font-semibold text-slate-900">{title}</h1>
        {description && <div className="mt-1 max-w-3xl text-sm text-slate-500">{description}</div>}
      </div>
      {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
    </div>
  )
}

export function Spinner({ label = 'Loading…' }: { label?: string }) {
  return <div className="flex items-center gap-2 py-10 text-sm text-slate-500"><Loader2 className="h-4 w-4 animate-spin" />{label}</div>
}

export function ErrorBox({ error }: { error: unknown }) {
  if (!error) return null
  return <div role="alert" className="rounded-lg border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">{error instanceof Error ? error.message : String(error)}</div>
}

export function Empty({ icon, title, children, action }: { icon?: ReactNode; title?: string; children?: ReactNode; action?: ReactNode }) {
  return (
    <div className="flex flex-col items-center rounded-xl border border-dashed border-slate-300 bg-white px-6 py-12 text-center">
      {icon && <div className="mb-3 text-slate-400">{icon}</div>}
      {title && <p className="font-medium text-slate-800">{title}</p>}
      {children && <div className="mt-1 max-w-md text-sm text-slate-500">{children}</div>}
      {action && <div className="mt-4">{action}</div>}
    </div>
  )
}

export function ProgressBar({ value, tone = 'brand' }: { value: number | null; tone?: 'brand' | 'good' | 'warn' | 'bad' }) {
  const color = { brand: 'bg-brand-500', good: 'bg-emerald-500', warn: 'bg-amber-500', bad: 'bg-rose-500' }[tone]
  return (
    <div className="h-2 w-full overflow-hidden rounded-full bg-slate-100" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={value === null ? undefined : Math.round(value * 100)}>
      {value === null
        ? <div className={clsx('h-full w-1/3 animate-[slide_1.2s_ease-in-out_infinite] rounded-full', color)} />
        : <div className={clsx('h-full rounded-full transition-all duration-500', color)} style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }} />}
    </div>
  )
}

export type StepState = 'pending' | 'processing' | 'completed' | 'failed'
export interface Step { key: string; label: ReactNode; detail?: ReactNode }

/** Vertical list of real processing stages: completed (tick), processing (spinner), failed (cross), pending (empty). */
export function StepList({ steps, current, failed, done }: { steps: Step[]; current: number; failed?: boolean; done?: boolean }) {
  return (
    <ol className="space-y-3">
      {steps.map((s, i) => {
        const state: StepState = done || i < current ? 'completed' : i === current ? (failed ? 'failed' : 'processing') : 'pending'
        return (
          <li key={s.key} className="flex items-start gap-3" data-state={state}>
            <span className={clsx('mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full',
              state === 'completed' && 'bg-emerald-500 text-white', state === 'processing' && 'bg-brand-50 text-brand-600 ring-1 ring-brand-200',
              state === 'failed' && 'bg-rose-500 text-white', state === 'pending' && 'border border-slate-300 bg-white')}>
              {state === 'completed' && <Check className="h-3.5 w-3.5" aria-hidden />}
              {state === 'processing' && <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden />}
              {state === 'failed' && <X className="h-3.5 w-3.5" aria-hidden />}
            </span>
            <div className="min-w-0 flex-1">
              <p className={clsx('text-sm', state === 'pending' ? 'text-slate-400' : 'text-slate-800', state === 'processing' && 'font-medium')}>
                {s.label}
                <span className="sr-only"> ({state})</span>
              </p>
              {s.detail && state === 'processing' && <div className="mt-1 text-xs text-slate-500">{s.detail}</div>}
            </div>
          </li>
        )
      })}
    </ol>
  )
}

// ---------------------------------------------------------------------------------------------- badges
export function ClassBadge({ name, severity, muted }: { name: string; severity?: Severity; muted?: boolean }) {
  const color = name === 'Normal' ? SEVERITY_COLORS.None : severity ? SEVERITY_COLORS[severity] : '#64748b'
  return (
    <span className={clsx('inline-flex items-center gap-1.5 whitespace-nowrap rounded-full px-2 py-0.5 text-xs font-medium', muted ? 'bg-slate-100 text-slate-600' : 'text-slate-800')}
      style={muted ? undefined : { backgroundColor: `${color}17`, boxShadow: `inset 0 0 0 1px ${color}40` }}>
      <span className="h-2 w-2 rounded-full" style={{ backgroundColor: color }} aria-hidden />
      {name}
    </span>
  )
}

export function PredictionBadge({ malicious }: { malicious: boolean }) {
  return malicious
    ? <span className="inline-flex rounded-full bg-rose-50 px-2 py-0.5 text-xs font-medium text-rose-700 ring-1 ring-inset ring-rose-200">Predicted attack</span>
    : <span className="inline-flex rounded-full bg-emerald-50 px-2 py-0.5 text-xs font-medium text-emerald-700 ring-1 ring-inset ring-emerald-200">Predicted Normal</span>
}

export function SeverityBadge({ severity }: { severity: Severity }) {
  if (severity === 'None') return <span className="text-xs text-slate-400">–</span>
  const color = SEVERITY_COLORS[severity]
  return <span className="inline-flex rounded px-1.5 py-0.5 text-xs font-semibold" style={{ color, backgroundColor: `${color}14` }}>{severity}</span>
}

export function ConfidenceBadge({ value, level }: { value: number; level: ConfidenceLevel }) {
  return (
    <span className="inline-flex items-center gap-2">
      <span className="h-1.5 w-12 overflow-hidden rounded-full bg-slate-100"><span className="block h-full rounded-full" style={{ width: `${value * 100}%`, backgroundColor: CONFIDENCE_COLORS[level] }} /></span>
      <span className="text-xs tabular-nums text-slate-700">{value >= 0.995 ? '>99.5%' : `${(value * 100).toFixed(1)}%`}</span>
      <span className="sr-only">{CONFIDENCE_LABEL[level]} confidence</span>
    </span>
  )
}

export function RiskBadge({ level, large }: { level: RiskLevel; large?: boolean }) {
  const s = RISK_STYLE[level]
  return <span className={clsx('inline-flex items-center rounded-md font-bold tracking-wide ring-1 ring-inset', s.bg, s.color, s.ring, large ? 'px-3 py-1 text-lg' : 'px-2 py-0.5 text-xs')}>{level}</span>
}
