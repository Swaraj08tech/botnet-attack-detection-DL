import clsx from 'clsx'
import { CheckCircle2, Info, X, XCircle } from 'lucide-react'
import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from 'react'

type Tone = 'success' | 'error' | 'info'
interface Toast { id: number; tone: Tone; title: string; body?: ReactNode }
const ToastContext = createContext<(t: Omit<Toast, 'id'>) => void>(() => {})

/** Short notifications in the bottom-right corner (for example "Report ready"), closed automatically after 6 s. */
export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([])
  const close = useCallback((id: number) => setToasts(list => list.filter(t => t.id !== id)), [])
  const show = useCallback((t: Omit<Toast, 'id'>) => {
    const id = Date.now() + Math.random()
    setToasts(list => [...list.slice(-3), { ...t, id }])
    setTimeout(() => close(id), 6000)
  }, [close])
  const value = useMemo(() => show, [show])
  return (
    <ToastContext.Provider value={value}>
      {children}
      <div className="no-print fixed bottom-4 right-4 z-50 flex w-80 flex-col gap-2" aria-live="polite">
        {toasts.map(t => {
          const Icon = { success: CheckCircle2, error: XCircle, info: Info }[t.tone]
          return (
            <div key={t.id} className="flex gap-3 rounded-xl border border-slate-200 bg-white p-3 shadow-lg">
              <Icon className={clsx('mt-0.5 h-5 w-5 shrink-0', { success: 'text-emerald-500', error: 'text-rose-500', info: 'text-brand-500' }[t.tone])} aria-hidden />
              <div className="min-w-0 flex-1 text-sm"><p className="font-medium text-slate-900">{t.title}</p>{t.body && <div className="mt-0.5 text-slate-600">{t.body}</div>}</div>
              <button onClick={() => close(t.id)} className="h-6 w-6 shrink-0 rounded text-slate-400 hover:text-slate-700" aria-label="Close"><X className="h-4 w-4" /></button>
            </div>
          )
        })}
      </div>
    </ToastContext.Provider>
  )
}

// eslint-disable-next-line react-refresh/only-export-components
export const useToast = () => useContext(ToastContext)
