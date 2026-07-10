import { apiFetch } from './client'
import type { RecommendResponse } from '../types/api'

export interface RecommendRequest {
  ratings?: Array<{ movie_id: number; rating: number }>
  top_k?: number
}

export function getRecommendations(body: RecommendRequest = {}): Promise<RecommendResponse> {
  return apiFetch<RecommendResponse>('/v1/recommend', {
    method: 'POST',
    body: JSON.stringify({
      ratings: body.ratings ?? [],
      top_k: body.top_k ?? 20,
    }),
  })
}
