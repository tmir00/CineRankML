import { useEffect, useState } from 'react'
import { getHealth } from '../api/auth'
import type { HealthResponse, RecommendResponse } from '../types/api'
import { formatNumber } from '../utils/poster'

interface ModelDetailsDrawerProps {
  open: boolean
  onClose: () => void
  recommendData: RecommendResponse | null
  topK: number
}

export function ModelDetailsDrawer({ open, onClose, recommendData, topK }: ModelDetailsDrawerProps) {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (!open) {
      return
    }
    setLoading(true)
    void getHealth()
      .then(setHealth)
      .catch(() => setHealth(null))
      .finally(() => setLoading(false))
  }, [open])

  if (!open) {
    return null
  }

  return (
    <div className="fixed inset-0 z-50 flex justify-end">
      <button
        type="button"
        className="absolute inset-0 bg-black/60 backdrop-blur-sm"
        onClick={onClose}
        aria-label="Close model details"
      />
      <aside className="relative z-10 h-full w-full max-w-md overflow-y-auto border-l border-white/10 bg-slate-950 p-6 shadow-2xl">
        <div className="mb-6 flex items-center justify-between">
          <h2 className="text-lg font-semibold text-white">Model details</h2>
          <button
            type="button"
            onClick={onClose}
            className="rounded-lg border border-white/10 px-3 py-1 text-sm text-slate-300 hover:text-white"
          >
            Close
          </button>
        </div>

        {loading ? <p className="text-sm text-slate-400">Loading health status…</p> : null}

        <dl className="space-y-4 text-sm">
          <DetailRow label="request_id" value={recommendData?.request_id ?? '—'} />
          <DetailRow label="model_version" value={recommendData?.model_version ?? '—'} />
          <DetailRow label="health model_version" value={health?.model_version ?? '—'} />
          <DetailRow label="cf_version" value={health?.cf_version ?? '—'} />
          <DetailRow label="input_dim" value={health ? String(health.input_dim) : '—'} />
          <DetailRow
            label="cf_embeddings_loaded"
            value={health ? formatNumber(health.cf_embeddings_loaded) : '—'}
          />
          <DetailRow label="top_k requested" value={String(topK)} />
        </dl>

        {health ? (
          <p className="mt-8 text-xs text-slate-500">
            Model online · {formatNumber(health.cf_embeddings_loaded)} CF embeddings loaded
          </p>
        ) : null}
      </aside>
    </div>
  )
}

function DetailRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-white/10 bg-slate-900/60 p-3">
      <dt className="text-xs uppercase tracking-wide text-slate-500">{label}</dt>
      <dd className="mt-1 break-all font-mono text-slate-200">{value}</dd>
    </div>
  )
}
