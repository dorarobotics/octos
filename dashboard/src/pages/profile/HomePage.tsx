import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useProfile } from '../../contexts/ProfileContext'
import { useAuth } from '../../contexts/AuthContext'
import GatewayControls from '../../components/GatewayControls'
import ConfirmDialog from '../../components/ConfirmDialog'
import { CHANNEL_COLORS, CHANNEL_LABELS } from '../../types'

export default function HomePage() {
  const { isAdmin } = useAuth()
  const {
    profileId, config, setConfig, status, isOwn, loading,
    startGateway, stopGateway, restartGateway,
    profileName, setProfileName, enabled, setEnabled,
    save, saving, deleteProfile,
  } = useProfile()
  const navigate = useNavigate()
  const [actionLoading, setActionLoading] = useState(false)
  const [deleteOpen, setDeleteOpen] = useState(false)

  const handleStart = async () => {
    setActionLoading(true)
    await startGateway()
    setActionLoading(false)
  }
  const handleStop = async () => {
    setActionLoading(true)
    await stopGateway()
    setActionLoading(false)
  }
  const handleRestart = async () => {
    setActionLoading(true)
    await restartGateway()
    setActionLoading(false)
  }
  const handleDelete = async () => {
    await deleteProfile()
    setDeleteOpen(false)
    navigate('/')
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin w-6 h-6 border-2 border-accent border-t-transparent rounded-full" />
      </div>
    )
  }

  const channels = config.channels || []

  return (
    <div>
      <h1 className="text-2xl font-bold text-white mb-6">Overview</h1>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Gateway Controls */}
        <GatewayControls
          status={status || { running: false, pid: null, started_at: null, uptime_secs: null }}
          loading={actionLoading}
          onStart={handleStart}
          onStop={handleStop}
          onRestart={handleRestart}
        />

        {/* Profile Info */}
        <div className="bg-surface rounded-xl border border-gray-700/50 p-5">
          <h3 className="text-sm font-semibold text-white mb-4">Profile Info</h3>
          <dl className="space-y-3 text-xs">
            <InfoRow label="ID" value={profileId} />
            <InfoRow label="Provider" value={config.provider || 'anthropic'} />
            <InfoRow label="Model" value={config.model || 'default'} />
            <InfoRow
              label="Channels"
              value={channels.length > 0 ? channels.map((c) => c.type).join(', ') : 'None'}
            />
            <InfoRow label="Fallbacks" value={String(config.fallback_models?.length || 0)} />
          </dl>

          {channels.length > 0 && (
            <div className="flex flex-wrap gap-1.5 mt-4">
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
        </div>
      </div>

      {/* Profile Settings */}
      <div className="mt-6 bg-surface rounded-xl border border-gray-700/50 p-5">
        <h3 className="text-sm font-semibold text-white mb-4">Profile Settings</h3>
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1.5">Display Name</label>
            <input
              value={profileName}
              onChange={(e) => setProfileName(e.target.value)}
              className="input max-w-md"
            />
          </div>
          <div>
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={enabled}
                onChange={(e) => setEnabled(e.target.checked)}
                className="w-4 h-4 rounded bg-surface-dark border-gray-600 text-accent focus:ring-accent"
              />
              <span className="text-sm text-gray-400">Auto-start gateway when server starts</span>
            </label>
          </div>
          {isAdmin && !isOwn && (
            <div>
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={config.admin_mode || false}
                  onChange={(e) => setConfig({ ...config, admin_mode: e.target.checked })}
                  className="w-4 h-4 rounded bg-surface-dark border-gray-600 text-accent focus:ring-accent"
                />
                <span className="text-sm text-gray-400">Admin mode (admin-only tools, no shell/file/web)</span>
              </label>
            </div>
          )}
          <div className="flex gap-3 pt-2">
            <button
              onClick={save}
              disabled={saving}
              className="px-5 py-2 text-sm font-medium rounded-lg bg-accent text-white hover:bg-accent-light transition disabled:opacity-50"
            >
              {saving ? 'Saving...' : 'Save'}
            </button>
            {isAdmin && !isOwn && (
              <button
                onClick={() => setDeleteOpen(true)}
                className="px-4 py-2 text-sm font-medium rounded-lg bg-red-500/10 text-red-400 hover:bg-red-500/20 border border-red-500/20 transition"
              >
                Delete Profile
              </button>
            )}
          </div>
        </div>
      </div>

      <ConfirmDialog
        open={deleteOpen}
        title="Delete Profile"
        message={`Are you sure you want to delete "${profileName}"? This will stop the gateway and remove all configuration.`}
        confirmLabel="Delete"
        danger
        onConfirm={handleDelete}
        onCancel={() => setDeleteOpen(false)}
      />
    </div>
  )
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between">
      <dt className="text-gray-500">{label}</dt>
      <dd className="text-gray-300">{value}</dd>
    </div>
  )
}
