interface GenreChipsProps {
  genres: string[]
  max?: number
}

export function GenreChips({ genres, max = 3 }: GenreChipsProps) {
  const visible = genres.slice(0, max)

  if (visible.length === 0) {
    return null
  }

  return (
    <div className="flex flex-wrap gap-1.5">
      {visible.map((genre) => (
        <span
          key={genre}
          className="rounded-full border border-white/10 bg-white/5 px-2 py-0.5 text-[10px] uppercase tracking-wide text-slate-300"
        >
          {genre}
        </span>
      ))}
    </div>
  )
}
