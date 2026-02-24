import React from 'react'
import type { DeviceInfo, CatalogModel, DownloadProgress, AgentStatus, AgentFact } from '../types'
import { DeviceInfoPanel } from './DeviceInfoPanel'
import { ModelSelector } from './ModelSelector'
import { AgentStatusPanel } from './AgentStatusPanel'

interface Props {
  device: DeviceInfo | null
  models: CatalogModel[]
  downloading: Record<string, DownloadProgress>
  modelLoaded: boolean
  version: string
  agentStatus: AgentStatus | null
  agentFacts: AgentFact[]
  onDownload: (catalogId: string) => void
  onLoad: (modelPath: string) => void
  onUnload: () => void
  onClearChat: () => void
  onOpenSettings: () => void
  onRemoveFact: (key: string) => void
}

export function Sidebar({
  device,
  models,
  downloading,
  modelLoaded,
  version,
  agentStatus,
  agentFacts,
  onDownload,
  onLoad,
  onUnload,
  onClearChat,
  onOpenSettings,
  onRemoveFact,
}: Props) {
  return (
    <div className="w-72 bg-surface-light border-r border-gray-800 flex flex-col h-full">
      {/* Drag region — clears macOS traffic light buttons */}
      <div className="h-8 shrink-0" style={{ WebkitAppRegion: 'drag' } as React.CSSProperties} />
      {/* Header */}
      <div className="px-4 pb-4 border-b border-gray-800">
        <div className="flex items-center gap-2">
          <span className="text-2xl">🦀</span>
          <div>
            <div className="text-lg font-bold text-white">CrabInfer</div>
            <div className="text-xs text-gray-500">v{version}</div>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        <DeviceInfoPanel device={device} />
        <ModelSelector
          models={models}
          downloading={downloading}
          modelLoaded={modelLoaded}
          onDownload={onDownload}
          onLoad={onLoad}
          onUnload={onUnload}
        />
        <AgentStatusPanel
          status={agentStatus}
          facts={agentFacts}
          onRemoveFact={onRemoveFact}
        />
      </div>

      {/* Footer */}
      <div className="p-4 border-t border-gray-800 space-y-1">
        <button
          onClick={onOpenSettings}
          className="w-full text-center text-xs text-gray-500 hover:text-gray-300 transition-colors py-1"
        >
          Settings
        </button>
        <button
          onClick={onClearChat}
          className="w-full text-center text-xs text-gray-500 hover:text-gray-300 transition-colors py-1"
        >
          Clear conversation
        </button>
      </div>
    </div>
  )
}
