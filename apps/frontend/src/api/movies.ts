import { apiFetch } from './client'
import type { MovieSearchResponse } from '../types/api'

export function searchMovies(query: string, limit = 20): Promise<MovieSearchResponse> {
  const params = new URLSearchParams({ q: query, limit: String(limit) })
  return apiFetch<MovieSearchResponse>(`/v1/movies/search?${params}`)
}
