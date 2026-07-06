const BASE_URL = import.meta.env.VITE_RECOMMENDER_API_URL ?? 'http://localhost:8090'

export class ApiError extends Error {
  status: number

  constructor(status: number, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

async function parseErrorBody(response: Response): Promise<string> {
  try {
    const data = await response.json()
    if (typeof data?.detail === 'string') {
      return data.detail
    }
    if (Array.isArray(data?.detail)) {
      return data.detail.map((item: { msg?: string }) => item.msg ?? 'Validation error').join(', ')
    }
    return response.statusText || 'Request failed'
  } catch {
    return response.statusText || 'Request failed'
  }
}

export async function apiFetch<T>(path: string, options: RequestInit = {}): Promise<T> {
  const headers: HeadersInit = {
    ...(options.body ? { 'Content-Type': 'application/json' } : {}),
    ...options.headers,
  }

  const response = await fetch(`${BASE_URL}${path}`, {
    ...options,
    credentials: 'include',
    headers,
  })

  if (!response.ok) {
    throw new ApiError(response.status, await parseErrorBody(response))
  }

  if (response.status === 204) {
    return undefined as T
  }

  return response.json() as Promise<T>
}

export function getApiBaseUrl(): string {
  return BASE_URL
}
