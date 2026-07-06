import { useState } from 'react'
import { Navigate } from 'react-router-dom'
import { useAuth } from '../hooks/useAuth'
import { AuthCard } from '../components/AuthCard'

export function LoginPage() {
  const { isAuthenticated, loading, login, register, error, clearError } = useAuth()
  const [mode, setMode] = useState<'login' | 'register'>('login')
  const [submitting, setSubmitting] = useState(false)

  if (!loading && isAuthenticated) {
    return <Navigate to="/app" replace />
  }

  const handleModeChange = (nextMode: 'login' | 'register') => {
    clearError()
    setMode(nextMode)
  }

  const handleSubmit = async (username: string, password: string) => {
    setSubmitting(true)
    try {
      if (mode === 'login') {
        await login(username, password)
      } else {
        await register(username, password)
      }
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center px-4 py-12">
      {loading ? (
        <div className="text-slate-400">Checking session…</div>
      ) : (
        <AuthCard
          mode={mode}
          onModeChange={handleModeChange}
          onSubmit={handleSubmit}
          error={error}
          loading={submitting}
        />
      )}
    </div>
  )
}
