import { useCallback, useState } from 'react'
import { getRecommendations } from '../api/recommendations'
import type { RecommendResponse } from '../types/api'

export function useRecommendations() {
  const [data, setData] = useState<RecommendResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [topK, setTopK] = useState(20)

  const refresh = useCallback(async (requestedTopK = 20) => {
    setLoading(true)
    setError(null)
    setTopK(requestedTopK)
    try {
      const response = await getRecommendations({
        ratings: [],
        top_k: requestedTopK,
        refresh_token: crypto.randomUUID(),
      })
      setData(response)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load recommendations')
    } finally {
      setLoading(false)
    }
  }, [])

  return {
    data,
    loading,
    error,
    topK,
    refresh,
  }
}
