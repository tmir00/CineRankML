interface LoadingStateProps {
  label?: string
  count?: number
}

export function LoadingState({ label = 'Loading', count = 6 }: LoadingStateProps) {
  return (
    <div className="space-y-3">
      <p className="text-sm text-slate-400">{label}…</p>
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5">
        {Array.from({ length: count }).map((_, index) => (
          <div
            key={index}
            className="aspect-[2/3] animate-pulse rounded-2xl bg-slate-800/80"
          />
        ))}
      </div>
    </div>
  )
}
