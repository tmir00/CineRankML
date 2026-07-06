import type { ReactNode } from 'react'
import { useAuth } from '../hooks/useAuth'

const MIN_RATINGS = 5

interface ProfileStatusProps {
  ratingsSection?: ReactNode
}

export function ProfileStatus({ ratingsSection }: ProfileStatusProps) {
  const { user } = useAuth()

  if (!user) {
    return null
  }

  const progress = Math.min(user.rating_count / MIN_RATINGS, 1)

  return (
    <section className="glass-panel p-6 sm:p-8">
      <h2 className="text-2xl font-semibold text-white sm:text-3xl">
        Your movie taste, ranked by <span className="text-gradient">AI</span>
      </h2>

      <div className="mt-6 flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <p className="text-sm text-slate-400">Ratings submitted</p>
          <p className="text-3xl font-semibold text-white">{user.rating_count}</p>
        </div>
        <div className="text-left sm:text-right">
          {user.can_recommend ? (
            <span className="inline-flex items-center rounded-full border border-emerald-400/30 bg-emerald-500/10 px-3 py-1 text-sm text-emerald-300">
              Taste profile ready
            </span>
          ) : (
            <p className="max-w-md text-sm text-slate-300">
              Rate at least {MIN_RATINGS} movies to unlock personalized recommendations.
            </p>
          )}
        </div>
      </div>

      <div className="mt-6">
        <div className="mb-2 flex justify-between text-xs text-slate-400">
          <span>Progress to unlock</span>
          <span>
            {user.rating_count} / {MIN_RATINGS}
          </span>
        </div>
        <div className="h-2 overflow-hidden rounded-full bg-slate-800">
          <div
            className="gradient-accent h-full rounded-full transition-all duration-500"
            style={{ width: `${progress * 100}%` }}
          />
        </div>
      </div>

      {ratingsSection ? <div className="mt-8 border-t border-white/10 pt-8">{ratingsSection}</div> : null}
    </section>
  )
}
