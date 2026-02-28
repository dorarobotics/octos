import type { ProfileConfig } from '../types'

interface Props {
  config: ProfileConfig
  onChange: (config: ProfileConfig) => void
}

export default function EnvVarsEditor({ config, onChange }: Props) {
  const addEnvVar = () => {
    onChange({ ...config, env_vars: { ...config.env_vars, '': '' } })
  }

  const removeEnvVar = (key: string) => {
    const { [key]: _, ...rest } = config.env_vars
    onChange({ ...config, env_vars: rest })
  }

  const updateEnvVar = (oldKey: string, newKey: string, value: string) => {
    const entries = Object.entries(config.env_vars).map(([k, v]) =>
      k === oldKey ? [newKey, value] : [k, v],
    )
    onChange({ ...config, env_vars: Object.fromEntries(entries) })
  }

  return (
    <div className="space-y-4">
      <p className="text-xs text-gray-500">
        Raw environment variables passed to the gateway process. API keys configured in other tabs
        appear here automatically.
      </p>

      {Object.entries(config.env_vars).map(([key, value], i) => (
        <div key={i} className="flex gap-2 items-start">
          <div className="flex-1">
            <input
              value={key}
              onChange={(e) => updateEnvVar(key, e.target.value, value)}
              placeholder="ANTHROPIC_API_KEY"
              className="input text-xs font-mono"
            />
          </div>
          <div className="flex-[2]">
            <input
              type="password"
              value={value}
              onChange={(e) => updateEnvVar(key, key, e.target.value)}
              placeholder="sk-ant-..."
              className="input text-xs font-mono"
            />
          </div>
          <button
            type="button"
            onClick={() => removeEnvVar(key)}
            className="px-2 py-2 text-xs text-red-400 hover:text-red-300"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
      ))}

      <button
        type="button"
        onClick={addEnvVar}
        className="px-3 py-1.5 text-xs font-medium rounded-lg bg-white/5 text-gray-400 hover:bg-white/10 hover:text-white border border-gray-700/50 transition"
      >
        + Add Environment Variable
      </button>
    </div>
  )
}
