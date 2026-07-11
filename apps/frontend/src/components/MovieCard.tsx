import { useEffect, useState } from 'react'
import { submitRating } from '../api/ratings'
import type { MovieSearchResult } from '../types/api'
import { GenreChips } from './GenreChips'
import { PosterImage } from './PosterImage'
import { StarRating } from './StarRating'

interface MovieCardProps {
  movie: MovieSearchResult
  onRated?: () => void
  mode?: 'search' | 'rated'
  initialRating?: number
  onDelete?: () => void
  onRatingChange?: (rating: number) => Promise<void>
}

export function MovieCard({
  movie,
  onRated,
  mode = 'search',
  initialRating = 0,
  onDelete,
  onRatingChange,
}: MovieCardProps) {
  const [rating, setRating] = useState(initialRating)
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    setRating(initialRating)
  }, [initialRating, movie.movie_id])

  const handleRate = async (value: number) => {
    setSaving(true)
    setError(null)
    try {
      if (mode === 'rated' && onRatingChange) {
        await onRatingChange(value)
      } else {
        await submitRating({ movie_id: movie.movie_id, rating: value })
        onRated?.()
      }
      setRating(value)
      if (mode === 'search') {
        setSaved(true)
        window.setTimeout(() => setSaved(false), 2000)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save rating')
    } finally {
      setSaving(false)
    }
  }

  const handleDelete = () => {
    if (!onDelete) {
      return
    }
    const confirmed = window.confirm(`Remove your rating for "${movie.title}"?`)
    if (confirmed) {
      void onDelete()
    }
  }

  return (
    <article
      className={`group flex flex-col overflow-hidden rounded-2xl border border-white/10 bg-slate-900/80 transition hover:border-purple-400/30 hover:shadow-xl hover:shadow-purple-950/20 ${
        mode === 'rated' ? 'h-full min-h-0 w-[180px] shrink-0 snap-start' : ''
      }`}
    >
      <div className="relative aspect-[2/3] shrink-0 overflow-hidden">
        <PosterImage
          title={movie.title}
          year={movie.year}
          posterPath={movie.poster_path}
          showPoster={movie.show_poster !== false && movie.poster_safe !== false}
        />
        {mode === 'search' && saved ? (
          <span className="absolute right-2 top-2 rounded-full bg-emerald-500/90 px-2 py-1 text-xs font-medium text-white">
            Saved
          </span>
        ) : null}
      </div>

      {mode === 'rated' ? (
        <div className="flex min-h-0 flex-1 flex-col p-4">
          <div className="min-h-[3.25rem]">
            <h3 className="line-clamp-2 font-medium text-white">{movie.title}</h3>
            <p className="text-sm text-slate-400">{movie.year ?? 'Unknown year'}</p>
          </div>
          <div className="mt-3 min-h-[2.75rem]">
            <GenreChips genres={movie.genres} />
          </div>
          <div className="mt-auto space-y-2 pt-3">
            <StarRating value={rating} onChange={(value) => void handleRate(value)} loading={saving} size="sm" />
            {onDelete ? (
              <button
                type="button"
                onClick={handleDelete}
                className="w-full rounded-lg border border-rose-500/30 px-3 py-1.5 text-xs text-rose-200 transition hover:border-rose-400/50 hover:bg-rose-500/10"
              >
                Delete rating
              </button>
            ) : null}
            {error ? <p className="text-xs text-rose-300">{error}</p> : null}
          </div>
        </div>
      ) : (
        <div className="space-y-3 p-4">
          <div>
            <h3 className="line-clamp-2 font-medium text-white">{movie.title}</h3>
            <p className="text-sm text-slate-400">{movie.year ?? 'Unknown year'}</p>
          </div>
          <GenreChips genres={movie.genres} />
          <StarRating value={rating} onChange={(value) => void handleRate(value)} loading={saving} size="sm" />
          {error ? <p className="text-xs text-rose-300">{error}</p> : null}
        </div>
      )}
    </article>
  )
}
