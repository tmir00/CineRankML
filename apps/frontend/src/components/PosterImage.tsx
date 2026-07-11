import { posterUrl } from '../utils/poster'

interface PosterImageProps {
  title: string
  posterPath?: string | null
  showPoster?: boolean
  year?: number | null
  className?: string
}

export function PosterImage({
  title,
  posterPath,
  showPoster = true,
  year,
  className = '',
}: PosterImageProps) {
  const url = showPoster ? posterUrl(posterPath) : null

  if (url) {
    return (
      <img
        src={url}
        alt={`${title} poster`}
        className={`h-full w-full object-cover transition-transform duration-300 group-hover:scale-105 ${className}`}
        loading="lazy"
      />
    )
  }

  return (
    <div
      className={`flex h-full w-full flex-col justify-end bg-gradient-to-br from-slate-900 via-slate-800 to-purple-950 p-4 ${className}`}
    >
      <div className="rounded-2xl border border-white/10 bg-black/20 p-3 backdrop-blur-sm">
        <p className="text-[10px] font-semibold uppercase tracking-[0.2em] text-slate-400">
          {showPoster ? 'Poster unavailable' : 'Poster hidden'}
        </p>
        <h3 className="mt-2 line-clamp-3 text-sm font-semibold text-white">{title}</h3>
        <p className="mt-1 text-xs text-slate-300">{year ?? 'Unknown year'}</p>
      </div>
    </div>
  )
}
