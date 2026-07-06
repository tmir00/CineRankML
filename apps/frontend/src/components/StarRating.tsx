type StarRatingSize = 'sm' | 'md' | 'lg'

interface StarRatingProps {
  value?: number
  onChange?: (rating: number) => void
  disabled?: boolean
  loading?: boolean
  size?: StarRatingSize
}

const sizeClasses: Record<StarRatingSize, string> = {
  sm: 'text-base gap-0.5',
  md: 'text-xl gap-1',
  lg: 'text-2xl gap-1',
}

export function StarRating({
  value = 0,
  onChange,
  disabled = false,
  loading = false,
  size = 'md',
}: StarRatingProps) {
  const isInteractive = Boolean(onChange) && !disabled && !loading

  return (
    <div
      className={`inline-flex items-center ${sizeClasses[size]} ${loading ? 'opacity-60' : ''}`}
      role={isInteractive ? 'radiogroup' : undefined}
      aria-label="Star rating"
    >
      {[1, 2, 3, 4, 5].map((star) => {
        const filled = star <= value
        return (
          <button
            key={star}
            type="button"
            disabled={!isInteractive}
            onClick={() => onChange?.(star)}
            className={`transition-transform ${
              isInteractive ? 'cursor-pointer hover:scale-110' : 'cursor-default'
            } ${filled ? 'text-amber-400' : 'text-slate-600'}`}
            aria-label={`Rate ${star} stars`}
          >
            ★
          </button>
        )
      })}
    </div>
  )
}
