import { useState } from 'react'
import { useMovieSearch } from '../hooks/useMovieSearch'
import { EmptyState } from './EmptyState'
import { LoadingState } from './LoadingState'
import { MovieCard } from './MovieCard'

interface MovieSearchBoxProps {
  onRated?: () => void
}

export function MovieSearchBox({ onRated }: MovieSearchBoxProps) {
  const [query, setQuery] = useState('')
  const { results, loading, error } = useMovieSearch(query)

  return (
    <section className="space-y-4">
      <div>
        <h2 className="text-xl font-semibold text-white">Search & rate movies</h2>
        <p className="mt-1 text-sm text-slate-400">
          Find films you have seen and rate them to build your taste profile.
        </p>
      </div>

      <div className="relative">
        <input
          type="search"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Search by title… e.g. Matrix"
          className="w-full rounded-2xl border border-white/10 bg-slate-900/70 px-5 py-4 text-white outline-none transition placeholder:text-slate-500 focus:border-purple-400/40 focus:ring-2 focus:ring-purple-500/20"
        />
      </div>

      {error ? (
        <div className="rounded-xl border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">
          {error}
        </div>
      ) : null}

      {loading ? <LoadingState label="Searching movies" /> : null}

      {!loading && query.trim() && results.length === 0 ? (
        <EmptyState
          title="No movies found"
          description="Try a different title or check your spelling."
        />
      ) : null}

      {!loading && results.length > 0 ? (
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5">
          {results.map((movie) => (
            <MovieCard key={movie.movie_id} movie={movie} onRated={onRated} />
          ))}
        </div>
      ) : null}

      {!loading && !query.trim() ? (
        <EmptyState
          title="Start searching"
          description="Type a movie title above to find films to rate."
          icon="🔍"
        />
      ) : null}
    </section>
  )
}
