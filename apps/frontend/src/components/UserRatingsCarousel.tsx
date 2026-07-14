import { MovieCard } from './MovieCard'
import { EmptyState } from './EmptyState'
import { LoadingState } from './LoadingState'
import type { UserRatingItem } from '../types/api'

interface UserRatingsCarouselProps {
  ratings: UserRatingItem[]
  loading: boolean
  error: string | null
  onRatingChange: (movieId: number, rating: number) => Promise<void>
  onDelete: (movieId: number) => Promise<void>
  onProfileChange?: () => void
  embedded?: boolean
}

export function UserRatingsCarousel({
  ratings,
  loading,
  error,
  onRatingChange,
  onDelete,
  onProfileChange,
  embedded = false,
}: UserRatingsCarouselProps) {
  const handleRatingChange = async (movieId: number, rating: number) => {
    await onRatingChange(movieId, rating)
    onProfileChange?.()
  }

  const handleDelete = async (movieId: number) => {
    await onDelete(movieId)
    onProfileChange?.()
  }

  return (
    <div className="space-y-4">
      <div>
        <h3 className={embedded ? 'text-lg font-semibold text-white' : 'text-xl font-semibold text-white'}>
          Your ratings
        </h3>
        <p className="mt-1 text-sm text-slate-400">
          {ratings.length > 0
            ? `${ratings.length} rated movie${ratings.length === 1 ? '' : 's'} — scroll sideways to browse.`
            : 'Movies you rate will appear here.'}
        </p>
      </div>

      {error ? (
        <div className="rounded-xl border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">
          {error}
        </div>
      ) : null}

      {loading ? <LoadingState label="Loading your ratings" /> : null}

      {!loading && ratings.length === 0 ? (
        <EmptyState
          title="No ratings yet"
          description="Search for movies below and rate them to build your taste profile."
          icon="⭐"
        />
      ) : null}

      {!loading && ratings.length > 0 ? (
        <div className="scrollbar-themed grid auto-cols-[180px] grid-flow-col gap-4 overflow-x-auto pb-2">
          {ratings.map((item) => (
            <div key={item.movie_id} className="h-full min-h-0">
              <MovieCard
                movie={{
                  movie_id: item.movie_id,
                  title: item.title,
                  year: item.year,
                  genres: item.genres,
                  poster_path: item.poster_path,
                }}
                mode="rated"
                initialRating={Math.round(item.rating)}
                onRatingChange={(rating) => handleRatingChange(item.movie_id, rating)}
                onDelete={() => handleDelete(item.movie_id)}
              />
            </div>
          ))}
        </div>
      ) : null}
    </div>
  )
}
