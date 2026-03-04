import { useState, useEffect, useCallback } from 'react'
import { api } from '../api'
import { useToast } from '../components/Toast'
import type { MonitorStatus } from '../types'

export default function AdminBotPage() {
  const { toast } = useToast()
  const [loading, setLoading] = useState(true)
  const [monitorStatus, setMonitorStatus] = useState<MonitorStatus>({ watchdog_enabled: false, alerts_enabled: false })

  const loadData = useCallback(async () => {
    try {
      const monitor = await api.monitorStatus().catch(() => ({ watchdog_enabled: false, alerts_enabled: false }))
      setMonitorStatus(monitor)
    } catch (e: any) {
      toast(e.message, 'error')
    } finally {
      setLoading(false)
    }
  }, [toast])

  useEffect(() => {
    loadData()
  }, [loadData])

  const handleToggleWatchdog = async (enabled: boolean) => {
    try {
      const result = await api.toggleWatchdog(enabled)
      setMonitorStatus((prev) => ({ ...prev, watchdog_enabled: result.watchdog_enabled }))
      toast(`Watchdog ${result.watchdog_enabled ? 'enabled' : 'disabled'}`)
    } catch (e: any) {
      toast(e.message, 'error')
    }
  }

  const handleToggleAlerts = async (enabled: boolean) => {
    try {
      const result = await api.toggleAlerts(enabled)
      setMonitorStatus((prev) => ({ ...prev, alerts_enabled: result.alerts_enabled }))
      toast(`Alerts ${result.alerts_enabled ? 'enabled' : 'disabled'}`)
    } catch (e: any) {
      toast(e.message, 'error')
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin w-6 h-6 border-2 border-accent border-t-transparent rounded-full" />
      </div>
    )
  }

  return (
    <div className="max-w-3xl">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-white">Monitor & Watchdog</h1>
        <p className="text-sm text-gray-500 mt-1">
          Controls for the system-wide watchdog and alerting. These apply to all gateway profiles.
        </p>
      </div>

      <div className="bg-surface rounded-xl border border-gray-700/50 p-5 mb-5">
        <h2 className="text-sm font-semibold text-white mb-4">Watchdog & Alerts</h2>
        <div className="space-y-4">
          <Toggle
            label="Watchdog enabled"
            description="Automatically restart crashed gateways"
            checked={monitorStatus.watchdog_enabled}
            onChange={handleToggleWatchdog}
          />
          <Toggle
            label="Alerts enabled"
            description="Send proactive alerts when gateways crash or become unhealthy"
            checked={monitorStatus.alerts_enabled}
            onChange={handleToggleAlerts}
          />
        </div>
      </div>

      <div className="bg-surface rounded-xl border border-gray-700/50 p-5">
        <h2 className="text-sm font-semibold text-white mb-2">Admin Bot Profile</h2>
        <p className="text-sm text-gray-400">
          To set up an admin bot, create a regular profile and enable <strong className="text-white">Admin Mode</strong> in
          its settings. Admin mode restricts the gateway to admin-only tools (profile management,
          monitoring, logs) and uses a built-in admin system prompt.
        </p>
      </div>
    </div>
  )
}

function Toggle({
  label,
  description,
  checked,
  onChange,
}: {
  label: string
  description: string
  checked: boolean
  onChange: (v: boolean) => void
}) {
  return (
    <div className="flex items-center justify-between">
      <div>
        <p className="text-sm text-white">{label}</p>
        <p className="text-xs text-gray-500">{description}</p>
      </div>
      <button
        type="button"
        onClick={() => onChange(!checked)}
        className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${
          checked ? 'bg-accent' : 'bg-gray-600'
        }`}
      >
        <span
          className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white transition-transform ${
            checked ? 'translate-x-4' : 'translate-x-0.5'
          }`}
        />
      </button>
    </div>
  )
}
