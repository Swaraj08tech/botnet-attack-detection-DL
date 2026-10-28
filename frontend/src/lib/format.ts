import type { ConfidenceLevel, RiskLevel, Severity } from './api'

// One colour system for the whole app (charts, badges, PDF use the same values):
//   safe = green, attack = red family by severity, uncertain / warning = amber, neutral = blue / grey.
export const COLORS = {
  safe: '#10b981', threat: '#e11d48', warn: '#f59e0b', neutral: '#64748b', brand: '#2f58d6', grid: '#e2e8f0',
}
export const SEVERITY_COLORS: Record<Severity, string> = {
  Critical: '#9f1239', High: '#e11d48', Medium: '#f59e0b', Low: '#3b82f6', None: COLORS.safe,
}
export const SEVERITY_ORDER: Exclude<Severity, 'None'>[] = ['Critical', 'High', 'Medium', 'Low']
export const CONFIDENCE_COLORS: Record<ConfidenceLevel, string> = { high: COLORS.safe, medium: COLORS.warn, low: COLORS.threat }
export const CONFIDENCE_LABEL: Record<ConfidenceLevel, string> = { high: 'High', medium: 'Medium', low: 'Low' }
export const RISK_STYLE: Record<RiskLevel, { color: string; bg: string; ring: string }> = {
  LOW: { color: 'text-emerald-700', bg: 'bg-emerald-50', ring: 'ring-emerald-200' },
  MODERATE: { color: 'text-amber-700', bg: 'bg-amber-50', ring: 'ring-amber-200' },
  HIGH: { color: 'text-rose-700', bg: 'bg-rose-50', ring: 'ring-rose-200' },
  CRITICAL: { color: 'text-rose-900', bg: 'bg-rose-100', ring: 'ring-rose-300' },
}

export const CONFIDENCE_TIP = 'Confidence is how sure the model is about its own answer: the probability it gives to the predicted class, adjusted (calibrated) on held-out data so that, for example, 80% means right about 8 times in 10 on similar data. It is not the accuracy of the model.'

export const pct = (x: number | undefined | null, digits = 1) => (x === undefined || x === null || Number.isNaN(x) ? '–' : `${(x * 100).toFixed(digits)}%`)
export const num = (x: number | undefined | null) => (x === undefined || x === null ? '–' : x.toLocaleString())
export const bytes = (n: number) => (n < 1024 ? `${n} B` : n < 1024 ** 2 ? `${(n / 1024).toFixed(1)} KB` : `${(n / 1024 ** 2).toFixed(1)} MB`)
// The server stores UTC times; a time without a zone suffix is UTC, not local time.
export const when = (iso: string) => new Date(/([zZ]|[+-]\d\d:\d\d)$/.test(iso) ? iso : `${iso}Z`).toLocaleString()

export function sortedEntries(record: Record<string, number> = {}) {
  return Object.entries(record).sort((a, b) => b[1] - a[1])
}
/** A calibrated probability: one decimal, but never "100.0%" (rounding is not certainty). */
export const prob = (p: number | undefined | null) => (p === undefined || p === null ? '–' : p >= 0.995 ? 'above 99.5%' : pct(p))
