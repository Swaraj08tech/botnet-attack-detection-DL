import { ShieldCheck } from 'lucide-react'
import { useState, type FormEvent } from 'react'
import { Navigate } from 'react-router-dom'
import { Button, ErrorBox } from '../components/ui'
import { useAuth } from '../lib/auth'

export default function Login() {
  const { user, login } = useAuth()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState<unknown>(null)
  const [busy, setBusy] = useState(false)

  if (user) return <Navigate to="/" replace />

  async function submit(e: FormEvent) {
    e.preventDefault()
    setBusy(true); setError(null)
    try { await login(email, password) } catch (err) { setError(err) } finally { setBusy(false) }
  }

  return (
    <div className="flex min-h-full items-center justify-center bg-gradient-to-br from-ink-900 via-slate-900 to-brand-700 px-4">
      <div className="w-full max-w-sm rounded-2xl bg-white p-8 shadow-xl">
        <div className="mb-6 flex items-center gap-3">
          <ShieldCheck className="h-8 w-8 text-brand-600" aria-hidden />
          <div>
            <h1 className="text-lg font-semibold text-slate-900">Network Attack Detection</h1>
            <p className="text-xs text-slate-500">Sign in to analyse network-flow datasets</p>
          </div>
        </div>
        <form onSubmit={submit} className="space-y-4">
          <label className="block text-sm">
            <span className="font-medium text-slate-700">Email</span>
            <input type="email" required autoComplete="username" value={email} onChange={e => setEmail(e.target.value)}
              className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2 text-sm focus:border-brand-500 focus:outline-none focus:ring-2 focus:ring-brand-500/30" />
          </label>
          <label className="block text-sm">
            <span className="font-medium text-slate-700">Password</span>
            <input type="password" required autoComplete="current-password" value={password} onChange={e => setPassword(e.target.value)}
              className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2 text-sm focus:border-brand-500 focus:outline-none focus:ring-2 focus:ring-brand-500/30" />
          </label>
          <ErrorBox error={error} />
          <Button type="submit" loading={busy} className="w-full">Sign in</Button>
        </form>
      </div>
    </div>
  )
}
