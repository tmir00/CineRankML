import { useEffect, useState } from 'react'
import { getHealth } from '../api/auth'
import { EmptyState } from '../components/EmptyState'
import { LoadingState } from '../components/LoadingState'
import { ModelDetailsDrawer } from '../components/ModelDetailsDrawer'
import { MovieSearchBox } from '../components/MovieSearchBox'
import { Navbar } from '../components/Navbar'
import { ProfileStatus } from '../components/ProfileStatus'
import { RecommendationCard } from '../components/RecommendationCard'
import { UserRatingsCarousel } from '../components/UserRatingsCarousel'
import { useAuth } from '../hooks/useAuth'
import { useRecommendations } from '../hooks/useRecommendations'
import { useUserRatings } from '../hooks/useUserRatings'
import type { HealthResponse } from '../types/api'
import { formatNumber } from '../utils/poster'

export function AppPage() {
  const { user, refreshUser } = useAuth()
  const { data, loading, error, topK, refresh } = useRecommendations()
  const userRatings = useUserRatings(Boolean(user))
  const [drawerOpen, setDrawerOpen] = useState(false)
  const [health, setHealth] = useState<HealthResponse | null>(null)

  useEffect(() => {
    void getHealth()
      .then(setHealth)
      .catch(() => setHealth(null))
  }, [])

  const handleRated = () => {
    void refreshUser()
    void userRatings.refresh()
  }

  return (
    <div className="min-h-screen">
      <Navbar />

      <main className="mx-auto max-w-7xl space-y-10 px-4 py-8 sm:px-6">
        <ProfileStatus
          ratingsSection={
            <UserRatingsCarousel
              embedded
              ratings={userRatings.ratings}
              loading={userRatings.loading}
              error={userRatings.error}
              onRatingChange={userRatings.updateRating}
              onDelete={userRatings.removeRating}
              onProfileChange={handleRated}
            />
          }
        />

        {health ? (
          <p className="text-center text-xs text-slate-500">
            Model online · {formatNumber(health.cf_embeddings_loaded)} CF embeddings loaded
          </p>
        ) : null}

        <MovieSearchBox onRated={handleRated} />

        <section className="space-y-4">
          <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <h2 className="text-xl font-semibold text-white">Your recommendations</h2>
              <p className="mt-1 text-sm text-slate-400">
                Personalized picks powered by hybrid collaborative filtering and content embeddings.
              </p>
            </div>
            <div className="flex flex-wrap gap-2">
              <button
                type="button"
                onClick={() => setDrawerOpen(true)}
                className="rounded-xl border border-white/10 px-4 py-2 text-sm text-slate-300 transition hover:border-purple-400/40 hover:text-white"
              >
                Model details
              </button>
              <button
                type="button"
                disabled={!user?.can_recommend || loading}
                onClick={() => void refresh(20)}
                className="gradient-accent rounded-xl px-4 py-2 text-sm font-medium text-white shadow-lg shadow-purple-900/20 transition hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {loading ? 'Refreshing…' : 'Refresh Recommendations'}
              </button>
            </div>
          </div>

          {!user?.can_recommend ? (
            <EmptyState
              title="Recommendations locked"
              description="Rate at least 5 movies to unlock personalized recommendations."
              icon="🔒"
            />
          ) : null}

          {error ? (
            <div className="rounded-xl border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">
              {error}
            </div>
          ) : null}

          {loading ? <LoadingState label="Loading recommendations" /> : null}

          {!loading && user?.can_recommend && data && data.recommendations.length === 0 ? (
            <EmptyState
              title="No recommendations yet"
              description='Click "Refresh Recommendations" to generate your personalized list.'
            />
          ) : null}

          {!loading && data && data.recommendations.length > 0 ? (
            <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5">
              {data.recommendations.map((movie) => (
                <RecommendationCard
                  key={movie.movie_id}
                  movie={movie}
                  recommendMeta={data}
                  onRated={handleRated}
                />
              ))}
            </div>
          ) : null}
        </section>
      </main>

      <ModelDetailsDrawer
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
        recommendData={data}
        topK={topK}
      />
    </div>
  )
}
