export interface UserProfile {
  user_id: number
  username: string
  rating_count: number
  can_recommend: boolean
}

export interface UserStatusResponse {
  exists: boolean
  user_id: number | null
  rating_count: number
  can_recommend: boolean
}

export interface HealthResponse {
  status: string
  model_version: string
  cf_version: string
  input_dim: number
  cf_embeddings_loaded: number
}

export interface MovieSearchResult {
  movie_id: number
  title: string
  year: number | null
  genres: string[]
  poster_path?: string | null
}

export interface MovieSearchResponse {
  movies: MovieSearchResult[]
}

export interface Recommendation {
  movie_id: number
  title: string
  year: number | null
  genres: string[]
  poster_path?: string | null
  predicted_score: number
  rank_position: number
}

export interface RecommendResponse {
  request_id: string
  model_version: string
  recommendations: Recommendation[]
}

export interface SubmitRatingRequest {
  movie_id: number
  rating: number
  request_id?: string
  model_version?: string
  experiment_id?: string
}

export interface SubmitRatingResponse {
  status: string
}

export interface DeleteRatingResponse {
  status: string
}

export interface UserRatingItem {
  movie_id: number
  title: string
  year: number | null
  genres: string[]
  poster_path?: string | null
  rating: number
  rated_at: string
}

export interface UserRatingsResponse {
  ratings: UserRatingItem[]
}

export interface LoginRequest {
  username: string
  password: string
}

export interface RegisterRequest {
  username: string
  password: string
}

export interface LogoutResponse {
  status: string
}
