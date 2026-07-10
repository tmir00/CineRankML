import { useState } from 'react'
import { getUserStatus } from '../api/auth'

interface AuthCardProps {
  mode: 'login' | 'register'
  onModeChange: (mode: 'login' | 'register') => void
  onSubmit: (username: string, password: string) => Promise<void>
  error: string | null
  loading: boolean
}

export function AuthCard({ mode, onModeChange, onSubmit, error, loading }: AuthCardProps) {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [usernameHint, setUsernameHint] = useState<string | null>(null)

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault()
    await onSubmit(username.trim(), password)
  }

  const handleUsernameBlur = async () => {
    if (mode !== 'register' || username.trim().length < 3) {
      setUsernameHint(null)
      return
    }
    try {
      const status = await getUserStatus(username.trim())
      setUsernameHint(status.exists ? 'Username already taken' : null)
    } catch {
      setUsernameHint(null)
    }
  }

  return (
    <div className="glass-panel w-full max-w-md p-8 shadow-2xl shadow-purple-950/30">
      <div className="mb-8 text-center">
        <h1 className="text-3xl font-bold text-white">
          Cine<span className="text-gradient">Rank</span>
        </h1>
        <p className="mt-2 text-slate-400">Discover movies ranked for your taste.</p>
      </div>

      <form onSubmit={(event) => void handleSubmit(event)} className="space-y-4">
        <div>
          <label htmlFor="username" className="mb-1.5 block text-sm text-slate-300">
            Username
          </label>
          <input
            id="username"
            type="text"
            autoComplete="username"
            value={username}
            onChange={(event) => setUsername(event.target.value)}
            onBlur={() => void handleUsernameBlur()}
            className="w-full rounded-xl border border-white/10 bg-slate-950/60 px-4 py-3 text-white outline-none transition focus:border-purple-400/50 focus:ring-2 focus:ring-purple-500/20"
            placeholder="alice"
            required
            minLength={3}
          />
          {usernameHint ? <p className="mt-1 text-xs text-amber-400">{usernameHint}</p> : null}
        </div>

        <div>
          <label htmlFor="password" className="mb-1.5 block text-sm text-slate-300">
            Password
          </label>
          <input
            id="password"
            type="password"
            autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            className="w-full rounded-xl border border-white/10 bg-slate-950/60 px-4 py-3 text-white outline-none transition focus:border-purple-400/50 focus:ring-2 focus:ring-purple-500/20"
            placeholder="••••••••"
            required
            minLength={8}
          />
        </div>

        {error ? (
          <div className="rounded-xl border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">
            {error}
          </div>
        ) : null}

        <button
          type="submit"
          disabled={loading || Boolean(usernameHint)}
          className="gradient-accent w-full rounded-xl px-4 py-3 font-medium text-white shadow-lg shadow-purple-900/30 transition hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {loading ? 'Please wait…' : mode === 'login' ? 'Login' : 'Create account'}
        </button>
      </form>

      <div className="mt-6 text-center text-sm text-slate-400">
        {mode === 'login' ? "Don't have an account?" : 'Already have an account?'}{' '}
        <button
          type="button"
          onClick={() => onModeChange(mode === 'login' ? 'register' : 'login')}
          className="font-medium text-purple-300 transition hover:text-purple-200"
        >
          {mode === 'login' ? 'Create account' : 'Login'}
        </button>
      </div>
    </div>
  )
}
