import { apiFetch } from './client'
import type {
  HealthResponse,
  LoginRequest,
  LogoutResponse,
  RegisterRequest,
  UserProfile,
  UserStatusResponse,
} from '../types/api'

export function getHealth(): Promise<HealthResponse> {
  return apiFetch<HealthResponse>('/health')
}

export function getMe(): Promise<UserProfile> {
  return apiFetch<UserProfile>('/v1/auth/me')
}

export function login(body: LoginRequest): Promise<UserProfile> {
  return apiFetch<UserProfile>('/v1/auth/login', {
    method: 'POST',
    body: JSON.stringify(body),
  })
}

export function register(body: RegisterRequest): Promise<UserProfile> {
  return apiFetch<UserProfile>('/v1/auth/register', {
    method: 'POST',
    body: JSON.stringify(body),
  })
}

export function logout(): Promise<LogoutResponse> {
  return apiFetch<LogoutResponse>('/v1/auth/logout', {
    method: 'POST',
  })
}

export function getUserStatus(username: string): Promise<UserStatusResponse> {
  const params = new URLSearchParams({ username })
  return apiFetch<UserStatusResponse>(`/v1/auth/status?${params}`)
}
