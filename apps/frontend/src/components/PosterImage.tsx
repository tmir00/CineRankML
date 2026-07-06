import { posterUrl } from '../utils/poster'

interface PosterImageProps {
  title: string
  posterPath?: string | null
  className?: string
}

export function PosterImage({ title, posterPath, className = '' }: PosterImageProps) {
  const url = posterUrl(posterPath)

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
      className={`flex h-full w-full flex-col items-center justify-center bg-gradient-to-br from-slate-800 via-purple-950 to-slate-900 ${className}`}
    >
      <span className="text-4xl opacity-60">🎬</span>
      <span className="mt-2 px-3 text-center text-xs text-slate-400">No poster</span>
    </div>
  )
}
