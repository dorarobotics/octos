import { useState } from 'react'
import { useProfile } from '../../contexts/ProfileContext'
import CategoryTabs from '../../components/CategoryTabs'
import SaveFooter from '../../components/SaveFooter'
import GatewayTab from '../../components/tabs/GatewayTab'
import EnvVarsEditor from '../../components/EnvVarsEditor'
import LogPanel from '../../components/LogPanel'

const TABS = [
  { key: 'gateway', label: 'Gateway Settings' },
  { key: 'env', label: 'Env Vars' },
  { key: 'logs', label: 'Logs' },
]

export default function SystemPage() {
  const { config, setConfig, save, saving, loading, logStreamUrl } = useProfile()
  const [activeTab, setActiveTab] = useState('gateway')

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin w-6 h-6 border-2 border-accent border-t-transparent rounded-full" />
      </div>
    )
  }

  return (
    <div>
      <h1 className="text-2xl font-bold text-white mb-6">System</h1>
      <div className="bg-surface rounded-xl border border-gray-700/50 overflow-hidden">
        <div className="px-5 pt-4">
          <CategoryTabs tabs={TABS} activeTab={activeTab} onTabChange={setActiveTab} />
        </div>
        <div className="p-5">
          {activeTab === 'gateway' && <GatewayTab config={config} onChange={setConfig} />}
          {activeTab === 'env' && <EnvVarsEditor config={config} onChange={setConfig} />}
          {activeTab === 'logs' && <LogPanel logStreamUrl={logStreamUrl} />}
        </div>
        {activeTab !== 'logs' && <SaveFooter onSave={save} saving={saving} />}
      </div>
    </div>
  )
}
