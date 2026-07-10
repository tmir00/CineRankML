export function posterUrl(posterPath?: string | null): string | null {
  if (!posterPath) {
    return null
  }
  return `https://image.tmdb.org/t/p/w342${posterPath}`
}

export function formatNumber(value: number): string {
  return new Intl.NumberFormat('en-US').format(value)
}
