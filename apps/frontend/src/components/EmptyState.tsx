interface EmptyStateProps {
  title: string
  description?: string
  icon?: string
}

export function EmptyState({ title, description, icon = '🎬' }: EmptyStateProps) {
  return (
    <div className="glass-panel flex flex-col items-center justify-center px-6 py-12 text-center">
      <div className="mb-4 text-4xl">{icon}</div>
      <h3 className="text-lg font-medium text-white">{title}</h3>
      {description ? <p className="mt-2 max-w-md text-sm text-slate-400">{description}</p> : null}
    </div>
  )
}
