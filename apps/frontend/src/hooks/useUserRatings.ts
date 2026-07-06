import { useCallback, useEffect, useState } from 'react'
import { deleteRating, getUserRatings, submitRating } from '../api/ratings'
import type { UserRatingItem } from '../types/api'

export function useUserRatings(isAuthenticated: boolean) {
  const [ratings, setRatings] = useState<UserRatingItem[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    if (!isAuthenticated) {
      setRatings([])
      return
    }

    setLoading(true)
    setError(null)
    try {
      const response = await getUserRatings()
      setRatings(response.ratings)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load ratings')
    } finally {
      setLoading(false)
    }
  }, [isAuthenticated])

  useEffect(() => {
    void refresh()
  }, [refresh])

  const updateRating = useCallback(async (movieId: number, rating: number) => {
    setRatings((current) =>
      current.map((item) => (item.movie_id === movieId ? { ...item, rating } : item)),
    )
    try {
      await submitRating({ movie_id: movieId, rating })
    } catch (err) {
      await refresh()
      throw err
    }
  }, [refresh])

  const removeRating = useCallback(async (movieId: number) => {
    setRatings((current) => current.filter((item) => item.movie_id !== movieId))
    try {
      await deleteRating(movieId)
    } catch (err) {
      await refresh()
      throw err
    }
  }, [refresh])

  return {
    ratings,
    loading,
    error,
    refresh,
    updateRating,
    removeRating,
  }
}
