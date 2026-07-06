import { useState } from 'react'
import { submitRating } from '../api/ratings'
import type { Recommendation, RecommendResponse } from '../types/api'
import { GenreChips } from './GenreChips'
import { PosterImage } from './PosterImage'
import { StarRating } from './StarRating'

interface RecommendationCardProps {
  movie: Recommendation
  recommendMeta: RecommendResponse | null
  onRated?: () => void
}

export function RecommendationCard({ movie, recommendMeta, onRated }: RecommendationCardProps) {
  const [rating, setRating] = useState(0)
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleRate = async (value: number) => {
    setSaving(true)
    setError(null)
    try {
      await submitRating({
        movie_id: movie.movie_id,
        rating: value,
        request_id: recommendMeta?.request_id,
        model_version: recommendMeta?.model_version,
        experiment_id: 'exp-main',
      })
      setRating(value)
      setSaved(true)
      onRated?.()
      window.setTimeout(() => setSaved(false), 2000)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save rating')
    } finally {
      setSaving(false)
    }
  }

  return (
    <article className="group relative overflow-hidden rounded-2xl border border-white/10 bg-slate-900/90 transition hover:-translate-y-1 hover:border-purple-400/40 hover:shadow-2xl hover:shadow-purple-950/30">
      <div className="relative aspect-[2/3] overflow-hidden">
        <PosterImage title={movie.title} posterPath={movie.poster_path} />
        <div className="pointer-events-none absolute inset-0 bg-gradient-to-t from-slate-950 via-slate-950/20 to-transparent" />
        <span className="absolute left-3 top-3 rounded-full bg-black/70 px-2.5 py-1 text-xs font-semibold text-white backdrop-blur">
          #{movie.rank_position}
        </span>
        <span className="absolute right-3 top-3 rounded-full border border-cyan-400/30 bg-cyan-500/20 px-2.5 py-1 text-xs font-semibold text-cyan-100 backdrop-blur">
          {movie.predicted_score.toFixed(1)}
        </span>
        {saved ? (
          <span className="absolute bottom-14 right-3 rounded-full bg-emerald-500/90 px-2 py-1 text-xs font-medium text-white">
            Saved
          </span>
        ) : null}
      </div>

      <div className="space-y-3 p-4">
        <div>
          <h3 className="line-clamp-2 font-medium text-white">{movie.title}</h3>
          <p className="text-sm text-slate-400">{movie.year ?? 'Unknown year'}</p>
        </div>
        <GenreChips genres={movie.genres} />
        <StarRating value={rating} onChange={(value) => void handleRate(value)} loading={saving} size="sm" />
        {error ? <p className="text-xs text-rose-300">{error}</p> : null}
      </div>
    </article>
  )
}
