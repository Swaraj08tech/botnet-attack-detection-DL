import { WifiOff } from 'lucide-react'
import { Navigate, Route, Routes } from 'react-router-dom'
import Layout from './components/Layout'
import { Button, Spinner } from './components/ui'
import { useAuth } from './lib/auth'
import About from './pages/About'
import Analyze from './pages/Analyze'
import FlowDetail from './pages/FlowDetail'
import Login from './pages/Login'
import Model from './pages/Model'
import Overview from './pages/Overview'
import ReportPage from './pages/ReportPage'
import Reports from './pages/Reports'
import Results from './pages/Results'

function Protected({ children }: { children: React.ReactNode }) {
  const { user, loading, offline, retry } = useAuth()
  if (loading) return <div className="p-8"><Spinner /></div>
  if (!user && offline) return (
    <div className="flex h-full items-center justify-center p-8">
      <div className="max-w-md rounded-xl border border-slate-200 bg-white p-6 text-center shadow-sm">
        <WifiOff className="mx-auto h-8 w-8 text-rose-500" aria-hidden />
        <p className="mt-3 font-medium text-slate-900">Cannot reach the server</p>
        <p className="mt-1 text-sm text-slate-500">Check that the backend window is still open, or run start-app.cmd again. You are still signed in.</p>
        <Button className="mt-4" onClick={retry}>Try again</Button>
      </div>
    </div>
  )
  if (!user) return <Navigate to="/login" replace />
  return <>{children}</>
}

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route element={<Protected><Layout /></Protected>}>
        <Route index element={<Overview />} />
        <Route path="analyze" element={<Analyze />} />
        <Route path="results" element={<Results />} />
        <Route path="results/:id" element={<Results />} />
        <Route path="results/:id/flows/:flowId" element={<FlowDetail />} />
        <Route path="model" element={<Model />} />
        <Route path="reports" element={<Reports />} />
        <Route path="reports/:reportId" element={<ReportPage />} />
        <Route path="about" element={<About />} />
        {/* old links from the previous version of the app */}
        <Route path="scan" element={<Navigate to="/analyze" replace />} />
        <Route path="alerts/*" element={<Navigate to="/results" replace />} />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}
