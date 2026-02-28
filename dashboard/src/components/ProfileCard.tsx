import { Link } from 'react-router-dom'
import type { ProfileResponse } from '../types'
import { CHANNEL_COLORS, CHANNEL_LABELS } from '../types'
import StatusBadge from './StatusBadge'

interface Props {
  profile: ProfileResponse
  onStart: (id: string) => void
  onStop: (id: string) => void
}

function formatUptime(secs: number | null): string {
  if (!secs) return ''
  const days = Math.floor(secs / 86400)
  const hours = Math.floor((secs % 86400) / 3600)
  const mins = Math.floor((secs % 3600) / 60)
  if (days > 0) return `${days}d ${hours}h`
  if (hours > 0) return `${hours}h ${mins}m`
  return `${mins}m`
}

export default function ProfileCard({ profile, onStart, onStop }: Props) {
  const channels = profile.config.channels || []
  const provider = profile.config.provider || 'anthropic'
  const model = profile.config.model || 'default'

  return (
    <div className="bg-surface rounded-xl border border-gray-700/50 p-5 hover:border-gray-600/50 transition-colors group">
      <div className="flex items-start justify-between mb-3">
        <Link
          to={`/profile/${profile.id}`}
          className="text-white font-semibold hover:text-accent transition-colors"
        >
          {profile.name}
        </Link>
        <StatusBadge running={profile.status.running} />
      </div>

      <div className="space-y-2 mb-4">
        <div className="flex items-center gap-2 text-xs text-gray-400">
          <span className="text-gray-500">Provider:</span>
          <span className="capitalize">{provider}</span>
        </div>
        <div className="flex items-center gap-2 text-xs text-gray-400">
          <span className="text-gray-500">Model:</span>
          <span className="truncate max-w-[140px]">{model}</span>
        </div>
        {profile.status.running && profile.status.uptime_secs && (
          <div className="flex items-center gap-2 text-xs text-gray-400">
            <span className="text-gray-500">Uptime:</span>
            <span>{formatUptime(profile.status.uptime_secs)}</span>
          </div>
        )}
      </div>

      {channels.length > 0 && (
        <div className="flex flex-wrap gap-1.5 mb-4">
          {channels.map((ch, i) => {
            const type = ch.type as keyof typeof CHANNEL_COLORS
            return (
              <span
                key={i}
                className={`${CHANNEL_COLORS[type] || 'bg-gray-500'} text-white text-[10px] font-bold px-1.5 py-0.5 rounded`}
              >
                {CHANNEL_LABELS[type] || ch.type.toUpperCase().slice(0, 2)}
              </span>
            )
          })}
        </div>
      )}

      <div className="flex gap-2">
        {profile.status.running ? (
          <button
            onClick={() => onStop(profile.id)}
            className="flex-1 px-3 py-1.5 text-xs font-medium rounded-lg bg-red-500/10 text-red-400 hover:bg-red-500/20 border border-red-500/20 transition"
          >
            Stop
          </button>
        ) : (
          <button
            onClick={() => onStart(profile.id)}
            className="flex-1 px-3 py-1.5 text-xs font-medium rounded-lg bg-green-500/10 text-green-400 hover:bg-green-500/20 border border-green-500/20 transition"
          >
            Start
          </button>
        )}
        <Link
          to={`/profile/${profile.id}`}
          className="px-3 py-1.5 text-xs font-medium rounded-lg bg-white/5 text-gray-400 hover:bg-white/10 hover:text-white border border-gray-700/50 transition"
        >
          Configure
        </Link>
      </div>
    </div>
  )
}
