import clsx from 'clsx'
import { useQuery } from '@tanstack/react-query'
import { BarChart3, Cpu, FileText, Info, LayoutDashboard, LogOut, ShieldCheck, Upload, WifiOff } from 'lucide-react'
import type { LucideIcon } from 'lucide-react'
import { NavLink, Outlet } from 'react-router-dom'
import { api } from '../lib/api'
import { useAuth } from '../lib/auth'

interface NavItem { to: string; label: string; icon: LucideIcon; end?: boolean }
interface NavGroup { title?: string; items: NavItem[] }

const NAV: NavGroup[] = [
  { items: [{ to: '/', label: 'Overview', icon: LayoutDashboard, end: true }] },
  { title: 'Analysis', items: [{ to: '/analyze', label: 'Analyze Flows', icon: Upload }, { to: '/results', label: 'Results', icon: BarChart3 }] },
  { title: 'Insights', items: [{ to: '/model', label: 'Model', icon: Cpu }, { to: '/reports', label: 'Reports', icon: FileText }] },
  { items: [{ to: '/about', label: 'About', icon: Info }] },
]

function Status({ ok, label, offLabel, optional }: { ok: boolean | undefined; label: string; offLabel: string; optional?: boolean }) {
  const color = ok === undefined ? 'bg-slate-500' : ok ? 'bg-emerald-400' : optional ? 'bg-slate-500' : 'bg-rose-400'
  return <p className="flex items-center gap-2"><span className={clsx('h-2 w-2 shrink-0 rounded-full', color)} />{ok ? label : ok === undefined ? 'Checking…' : offLabel}</p>
}

export default function Layout() {
  const { user, logout } = useAuth()
  const health = useQuery({ queryKey: ['health'], queryFn: api.health, refetchInterval: 15_000, retry: 0 })
  const h = health.isError ? { model_loaded: false, report_generator: false, llm_available: false } : health.data

  return (
    <div className="flex h-full overflow-hidden">
      <aside className="no-print hidden w-60 shrink-0 flex-col bg-ink-900 text-slate-300 md:flex">
        <div className="flex items-center gap-2.5 px-5 py-5">
          <ShieldCheck className="h-6 w-6 shrink-0 text-brand-500" aria-hidden />
          <div className="min-w-0">
            <p className="text-sm font-semibold text-white">Attack Detection</p>
            <p className="text-[11px] text-slate-400">Network intrusion · UNSW-NB15</p>
          </div>
        </div>
        <nav className="flex-1 space-y-4 overflow-y-auto px-3 py-2" aria-label="Main">
          {NAV.map((group, g) => (
            <div key={g}>
              {group.title && <p className="mb-1 px-3 text-[10px] font-semibold uppercase tracking-wider text-slate-500">{group.title}</p>}
              <div className="space-y-0.5">
                {group.items.map(({ to, label, icon: Icon, end }) => (
                  <NavLink key={to} to={to} end={end}
                    className={({ isActive }) => clsx('flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition', isActive ? 'bg-white/10 font-medium text-white' : 'hover:bg-white/5 hover:text-white')}>
                    <Icon className="h-4 w-4" aria-hidden />{label}
                  </NavLink>
                ))}
              </div>
            </div>
          ))}
        </nav>
        <div className="space-y-1.5 border-t border-white/10 px-5 py-4 text-[11px]">
          <Status ok={h?.model_loaded} label="Detection Model Ready" offLabel="Detection Model unavailable" />
          <Status ok={h?.report_generator} label="Report Generator Ready" offLabel="Report Generator unavailable" />
          <Status ok={h?.llm_available} label="Local language model online" offLabel="Local language model offline (optional)" optional />
        </div>
      </aside>
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="no-print flex items-center justify-between gap-4 border-b border-slate-200 bg-white px-4 py-3 md:px-6">
          <div className="flex min-w-0 items-center gap-2 text-sm text-slate-500 md:hidden"><ShieldCheck className="h-5 w-5 text-brand-600" />Attack Detection</div>
          <p className="hidden truncate text-xs text-slate-500 md:block">Machine-learning detection of network attacks in uploaded flow datasets (offline analysis)</p>
          <div className="flex shrink-0 items-center gap-3">
            <div className="text-right">
              <p className="text-sm font-medium text-slate-800">{user?.name}</p>
              <p className="text-xs text-slate-500">{user?.email}</p>
            </div>
            <button onClick={logout} className="rounded-lg p-2 text-slate-500 hover:bg-slate-100 hover:text-slate-800" aria-label="Sign out" title="Sign out">
              <LogOut className="h-4 w-4" />
            </button>
          </div>
        </header>
        <nav className="no-print flex gap-1 overflow-x-auto border-b border-slate-200 bg-white px-3 py-2 md:hidden" aria-label="Main mobile">
          {NAV.flatMap(g => g.items).map(({ to, label, end }) => (
            <NavLink key={to} to={to} end={end} className={({ isActive }) => clsx('whitespace-nowrap rounded-md px-3 py-1.5 text-sm', isActive ? 'bg-brand-50 text-brand-700' : 'text-slate-600')}>{label}</NavLink>
          ))}
        </nav>
        {health.isError && (
          <div role="alert" className="no-print flex items-center gap-2 bg-rose-600 px-6 py-2 text-sm text-white">
            <WifiOff className="h-4 w-4 shrink-0" aria-hidden />
            Cannot reach the server. Check that the backend window is still open (or run start-app.cmd again). This message disappears when it is back.
          </div>
        )}
        <main className="relative min-w-0 flex-1 overflow-y-auto px-4 py-6 md:px-8"><div className="mx-auto w-full max-w-7xl"><Outlet /></div></main>
      </div>
    </div>
  )
}
