import { useEffect, useState } from 'react'
import { searchMovies } from '../api/movies'
import type { MovieSearchResult } from '../types/api'

export function useMovieSearch(query: string, debounceMs = 300) {
  const [results, setResults] = useState<MovieSearchResult[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const trimmed = query.trim()
    if (!trimmed) {
      setResults([])
      setLoading(false)
      setError(null)
      return
    }

    setLoading(true)
    setError(null)

    const timer = window.setTimeout(() => {
      void (async () => {
        try {
          const response = await searchMovies(trimmed, 20)
          setResults(response.movies)
        } catch (err) {
          setError(err instanceof Error ? err.message : 'Search failed')
          setResults([])
        } finally {
          setLoading(false)
        }
      })()
    }, debounceMs)

    return () => window.clearTimeout(timer)
  }, [query, debounceMs])

  return { results, loading, error }
}
