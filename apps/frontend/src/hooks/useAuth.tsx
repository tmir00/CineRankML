import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react'
import type { ReactNode } from 'react'
import * as authApi from '../api/auth'
import { ApiError } from '../api/client'
import type { UserProfile } from '../types/api'

interface AuthContextValue {
  user: UserProfile | null
  loading: boolean
  error: string | null
  isAuthenticated: boolean
  login: (username: string, password: string) => Promise<void>
  register: (username: string, password: string) => Promise<void>
  logout: () => Promise<void>
  refreshUser: () => Promise<void>
  clearError: () => void
}

const AuthContext = createContext<AuthContextValue | null>(null)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<UserProfile | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const refreshUser = useCallback(async () => {
    try {
      const profile = await authApi.getMe()
      setUser(profile)
      setError(null)
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) {
        setUser(null)
      } else {
        setError(err instanceof Error ? err.message : 'Failed to load profile')
      }
    }
  }, [])

  useEffect(() => {
    void (async () => {
      setLoading(true)
      await refreshUser()
      setLoading(false)
    })()
  }, [refreshUser])

  const login = useCallback(async (username: string, password: string) => {
    setError(null)
    try {
      const profile = await authApi.login({ username, password })
      setUser(profile)
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Login failed'
      setError(message)
      throw err
    }
  }, [])

  const register = useCallback(async (username: string, password: string) => {
    setError(null)
    try {
      const profile = await authApi.register({ username, password })
      setUser(profile)
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Registration failed'
      setError(message)
      throw err
    }
  }, [])

  const logout = useCallback(async () => {
    setError(null)
    try {
      await authApi.logout()
    } finally {
      setUser(null)
    }
  }, [])

  const value = useMemo<AuthContextValue>(
    () => ({
      user,
      loading,
      error,
      isAuthenticated: user !== null,
      login,
      register,
      logout,
      refreshUser,
      clearError: () => setError(null),
    }),
    [user, loading, error, login, register, logout, refreshUser],
  )

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth(): AuthContextValue {
  const context = useContext(AuthContext)
  if (!context) {
    throw new Error('useAuth must be used within AuthProvider')
  }
  return context
}
