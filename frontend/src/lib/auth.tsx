import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'
import { ApiError, api, setUnauthorizedHandler, tokenStore, type User } from './api'

interface AuthState {
  user: User | null
  loading: boolean
  offline: boolean          // the server could not be reached while checking the saved session
  retry: () => void
  login: (email: string, password: string) => Promise<void>
  logout: () => void
}

const AuthContext = createContext<AuthState | null>(null)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [loading, setLoading] = useState(true)
  const [offline, setOffline] = useState(false)

  const logout = useCallback(() => { tokenStore.clear(); setUser(null) }, [])

  // Check the saved session. Only a 401 means "signed out"; if the server is down, keep the session and offer a retry.
  const check = useCallback(() => {
    if (!tokenStore.get()) { setLoading(false); return }
    setLoading(true)
    api.me()
      .then(u => { setUser(u); setOffline(false) })
      .catch(e => { if (e instanceof ApiError && e.status === 401) tokenStore.clear(); else setOffline(true) })
      .finally(() => setLoading(false))
  }, [])

  useEffect(() => {
    setUnauthorizedHandler(() => setUser(null))
    check()
  }, [check])

  const login = useCallback(async (email: string, password: string) => {
    const res = await api.login(email, password)
    tokenStore.set(res.access_token)
    setUser(res.user)
  }, [])

  const value = useMemo(() => ({ user, loading, offline, retry: check, login, logout }), [user, loading, offline, check, login, logout])
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

// eslint-disable-next-line react-refresh/only-export-components
export function useAuth() {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used inside AuthProvider')
  return ctx
}
