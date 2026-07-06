import { apiFetch } from './client'
import type {
  DeleteRatingResponse,
  SubmitRatingRequest,
  SubmitRatingResponse,
  UserRatingsResponse,
} from '../types/api'

export function getUserRatings(): Promise<UserRatingsResponse> {
  return apiFetch<UserRatingsResponse>('/v1/ratings')
}

export function submitRating(body: SubmitRatingRequest): Promise<SubmitRatingResponse> {
  return apiFetch<SubmitRatingResponse>('/v1/ratings', {
    method: 'POST',
    body: JSON.stringify(body),
  })
}

export function deleteRating(movieId: number): Promise<DeleteRatingResponse> {
  return apiFetch<DeleteRatingResponse>(`/v1/ratings/${movieId}`, {
    method: 'DELETE',
  })
}
