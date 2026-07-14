import { useCallback, useState } from 'react'
import { getRecommendations } from '../api/recommendations'
import type { RecommendResponse } from '../types/api'

const MAX_SHOWN_MOVIE_IDS = 60

/** Client nonce for reshuffling retrieval; works on plain HTTP (no randomUUID). */
function newRefreshToken(): string {
  if (globalThis.crypto?.randomUUID) {
    return globalThis.crypto.randomUUID()
  }
  const bytes = new Uint8Array(16)
  globalThis.crypto.getRandomValues(bytes)
  return Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('')
}

/** Append new ids, keep first-seen order, trim to the newest window of maxSize. */
export function appendShownMovieIds(
  existing: number[],
  nextIds: number[],
  maxSize = MAX_SHOWN_MOVIE_IDS,
): number[] {
  const merged: number[] = []
  const seen = new Set<number>()
  for (const id of [...existing, ...nextIds]) {
    if (seen.has(id)) continue
    seen.add(id)
    merged.push(id)
  }
  if (merged.length <= maxSize) return merged
  return merged.slice(merged.length - maxSize)
}

export function useRecommendations() {
  const [data, setData] = useState<RecommendResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [topK, setTopK] = useState(20)
  // Rolling window of last-shown movie ids (oldest → newest), session-only.
  const [shownMovieIds, setShownMovieIds] = useState<number[]>([])

  const refresh = useCallback(async (requestedTopK = 20) => {
    setLoading(true)
    setError(null)
    setTopK(requestedTopK)
    try {
      const response = await getRecommendations({
        ratings: [],
        top_k: requestedTopK,
        refresh_token: newRefreshToken(),
        exclude_movie_ids: shownMovieIds,
      })
      setData(response)
      const returnedIds = response.recommendations.map((item) => item.movie_id)
      setShownMovieIds((prev) => appendShownMovieIds(prev, returnedIds))
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load recommendations')
    } finally {
      setLoading(false)
    }
  }, [shownMovieIds])

  return {
    data,
    loading,
    error,
    topK,
    refresh,
  }
}
