import { useAuth } from '../hooks/useAuth'

export function Navbar() {
  const { user, logout } = useAuth()

  return (
    <header className="sticky top-0 z-40 border-b border-white/10 bg-slate-950/80 backdrop-blur-md">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-4 sm:px-6">
        <div className="text-xl font-semibold tracking-tight text-white">
          Cine<span className="text-gradient">Rank</span>
        </div>

        {user ? (
          <div className="flex items-center gap-4 text-sm">
            <span className="hidden text-slate-300 sm:inline">{user.username}</span>
            <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-slate-300">
              {user.rating_count} rated
            </span>
            <button
              type="button"
              onClick={() => void logout()}
              className="rounded-xl border border-white/10 px-3 py-1.5 text-slate-300 transition hover:border-rose-400/40 hover:text-white"
            >
              Logout
            </button>
          </div>
        ) : null}
      </div>
    </header>
  )
}
