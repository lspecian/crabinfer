import React, { useState } from 'react'
import type { AppSettings } from '../types'

const ROUTING_POLICIES = [
  { value: 'LocalFirst', label: 'Local First', desc: 'Prefer on-device, fallback to cloud' },
  { value: 'CloudFirst', label: 'Cloud First', desc: 'Prefer cloud, fallback to local' },
  { value: 'LocalOnly', label: 'Local Only', desc: 'Never use cloud providers' },
  { value: 'SelfHostedFirst', label: 'Self-Hosted First', desc: 'Prefer LAN servers (Ollama/vLLM)' },
  { value: 'Auto', label: 'Auto', desc: 'Route based on task complexity' },
] as const

const PROVIDERS = [
  { key: 'openai', label: 'OpenAI', placeholder: 'sk-...' },
  { key: 'anthropic', label: 'Anthropic', placeholder: 'sk-ant-...' },
  { key: 'google', label: 'Google AI', placeholder: 'AIza...' },
  { key: 'openrouter', label: 'OpenRouter', placeholder: 'sk-or-...' },
  { key: 'ollama', label: 'Ollama', placeholder: 'http://localhost:11434' },
] as const

interface Props {
  settings: AppSettings
  onUpdate: (update: Partial<AppSettings>) => void
  onClose: () => void
}

export function SettingsPanel({ settings, onUpdate, onClose }: Props) {
  const [editingKey, setEditingKey] = useState<string | null>(null)
  const [keyInput, setKeyInput] = useState('')

  const handleSaveKey = (providerKey: string) => {
    const providers = { ...settings.providers }
    providers[providerKey] = {
      ...providers[providerKey],
      apiKey: keyInput,
    }
    onUpdate({ providers })
    setEditingKey(null)
    setKeyInput('')
  }

  const handleRemoveKey = (providerKey: string) => {
    const providers = { ...settings.providers }
    delete providers[providerKey]
    onUpdate({ providers })
  }

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center">
      <div className="bg-surface-light rounded-2xl w-full max-w-lg max-h-[80vh] overflow-y-auto border border-gray-700 shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between p-5 border-b border-gray-700">
          <h2 className="text-lg font-bold text-white">Settings</h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white transition-colors text-xl leading-none"
          >
            &times;
          </button>
        </div>

        <div className="p-5 space-y-6">
          {/* Routing Policy */}
          <section>
            <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3">
              Routing Policy
            </h3>
            <div className="space-y-1">
              {ROUTING_POLICIES.map((policy) => (
                <button
                  key={policy.value}
                  onClick={() =>
                    onUpdate({ routingPolicy: policy.value as AppSettings['routingPolicy'] })
                  }
                  className={`w-full text-left rounded-lg px-3 py-2 transition-colors ${
                    settings.routingPolicy === policy.value
                      ? 'bg-crab/20 border border-crab/40 text-white'
                      : 'bg-surface hover:bg-surface-lighter text-gray-300'
                  }`}
                >
                  <div className="text-sm font-medium">{policy.label}</div>
                  <div className="text-xs text-gray-500">{policy.desc}</div>
                </button>
              ))}
            </div>
          </section>

          {/* Privacy */}
          <section>
            <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3">
              Privacy
            </h3>
            <div className="space-y-2">
              <label className="flex items-center justify-between bg-surface rounded-lg px-3 py-2 cursor-pointer">
                <div>
                  <div className="text-sm text-white">Privacy Mode</div>
                  <div className="text-xs text-gray-500">Never send data to cloud providers</div>
                </div>
                <input
                  type="checkbox"
                  checked={settings.privacyMode}
                  onChange={(e) => onUpdate({ privacyMode: e.target.checked })}
                  className="accent-crab w-4 h-4"
                />
              </label>
              <label className="flex items-center justify-between bg-surface rounded-lg px-3 py-2 cursor-pointer">
                <div>
                  <div className="text-sm text-white">Data Sovereignty</div>
                  <div className="text-xs text-gray-500">Keep all data on this device</div>
                </div>
                <input
                  type="checkbox"
                  checked={settings.dataSovereignty}
                  onChange={(e) => onUpdate({ dataSovereignty: e.target.checked })}
                  className="accent-crab w-4 h-4"
                />
              </label>
            </div>
          </section>

          {/* Provider API Keys */}
          <section>
            <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3">
              Provider API Keys
            </h3>
            <div className="space-y-2">
              {PROVIDERS.map((provider) => {
                const config = settings.providers[provider.key]
                const hasKey = config?.apiKey && config.apiKey.length > 0
                const isEditing = editingKey === provider.key

                return (
                  <div key={provider.key} className="bg-surface rounded-lg px-3 py-2">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <span className="text-sm text-white">{provider.label}</span>
                        {hasKey && (
                          <span className="text-xs text-green-400 bg-green-400/10 px-1.5 py-0.5 rounded">
                            configured
                          </span>
                        )}
                      </div>
                      <div className="flex gap-1">
                        {hasKey && !isEditing && (
                          <button
                            onClick={() => handleRemoveKey(provider.key)}
                            className="text-xs text-red-400 hover:text-red-300 px-2 py-1 transition-colors"
                          >
                            Remove
                          </button>
                        )}
                        <button
                          onClick={() => {
                            if (isEditing) {
                              setEditingKey(null)
                              setKeyInput('')
                            } else {
                              setEditingKey(provider.key)
                              setKeyInput(config?.apiKey || '')
                            }
                          }}
                          className="text-xs text-gray-400 hover:text-white px-2 py-1 transition-colors"
                        >
                          {isEditing ? 'Cancel' : hasKey ? 'Edit' : 'Add key'}
                        </button>
                      </div>
                    </div>
                    {isEditing && (
                      <div className="flex gap-2 mt-2">
                        <input
                          type="password"
                          value={keyInput}
                          onChange={(e) => setKeyInput(e.target.value)}
                          placeholder={provider.placeholder}
                          className="flex-1 bg-surface-lighter rounded-lg px-3 py-1.5 text-xs text-white placeholder-gray-500 focus:outline-none focus:ring-1 focus:ring-crab"
                          autoFocus
                          onKeyDown={(e) => {
                            if (e.key === 'Enter') handleSaveKey(provider.key)
                            if (e.key === 'Escape') {
                              setEditingKey(null)
                              setKeyInput('')
                            }
                          }}
                        />
                        <button
                          onClick={() => handleSaveKey(provider.key)}
                          disabled={!keyInput.trim()}
                          className="bg-crab hover:bg-crab-light disabled:opacity-30 text-white rounded-lg px-3 py-1.5 text-xs font-medium transition-colors"
                        >
                          Save
                        </button>
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          </section>
        </div>
      </div>
    </div>
  )
}
