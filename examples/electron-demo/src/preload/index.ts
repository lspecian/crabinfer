import { contextBridge, ipcRenderer } from 'electron'
import type {
  CrabInferAPI,
  DownloadProgress,
  AppSettings,
  AgentToolExecution,
} from '../renderer/types'

const api: CrabInferAPI = {
  // Device
  getDeviceInfo: () => ipcRenderer.invoke('device:info'),

  // Models
  getModelCatalog: () => ipcRenderer.invoke('models:catalog'),
  getRecommendedModels: () => ipcRenderer.invoke('models:recommended'),
  downloadModel: (catalogId: string) => ipcRenderer.invoke('models:download', catalogId),
  onDownloadProgress: (callback: (progress: DownloadProgress) => void) => {
    const handler = (_event: Electron.IpcRendererEvent, progress: DownloadProgress) =>
      callback(progress)
    ipcRenderer.on('models:download-progress', handler)
    return () => ipcRenderer.removeListener('models:download-progress', handler)
  },
  isModelDownloaded: (catalogId: string) => ipcRenderer.invoke('models:is-downloaded', catalogId),
  getModelPath: (catalogId: string) => ipcRenderer.invoke('models:path', catalogId),

  // Inference
  loadModel: (modelPath: string) => ipcRenderer.invoke('inference:load', modelPath),
  unloadModel: () => ipcRenderer.invoke('inference:unload'),
  isModelLoaded: () => ipcRenderer.invoke('inference:is-loaded'),
  complete: (messages, maxTokens) =>
    ipcRenderer.invoke('inference:complete', messages, maxTokens),
  streamStart: (messages, maxTokens) =>
    ipcRenderer.invoke('inference:stream-start', messages, maxTokens),
  onStreamToken: (callback: (token: string, done: boolean) => void) => {
    const handler = (_event: Electron.IpcRendererEvent, token: string, done: boolean) =>
      callback(token, done)
    ipcRenderer.on('inference:stream-token', handler)
    return () => ipcRenderer.removeListener('inference:stream-token', handler)
  },
  stopInference: () => ipcRenderer.invoke('inference:stop'),
  getLastStats: () => ipcRenderer.invoke('inference:last-stats'),

  // Agent
  agentRun: (message: string, maxRounds?: number) =>
    ipcRenderer.invoke('agent:run', message, maxRounds),
  agentGetTools: () => ipcRenderer.invoke('agent:get-tools'),
  agentGetStatus: () => ipcRenderer.invoke('agent:get-status'),
  agentClearConversation: () => ipcRenderer.invoke('agent:clear-conversation'),
  agentGetConversation: () => ipcRenderer.invoke('agent:get-conversation'),
  onAgentToolExecuted: (callback: (exec: AgentToolExecution) => void) => {
    const handler = (_event: Electron.IpcRendererEvent, exec: AgentToolExecution) =>
      callback(exec)
    ipcRenderer.on('agent:tool-executed', handler)
    return () => ipcRenderer.removeListener('agent:tool-executed', handler)
  },
  onAgentConversationCleared: (callback: () => void) => {
    const handler = () => callback()
    ipcRenderer.on('agent:conversation-cleared', handler)
    return () => ipcRenderer.removeListener('agent:conversation-cleared', handler)
  },

  // Facts/Memory
  agentAddFact: (key: string, value: string) => ipcRenderer.invoke('agent:add-fact', key, value),
  agentRemoveFact: (key: string) => ipcRenderer.invoke('agent:remove-fact', key),
  agentGetFacts: () => ipcRenderer.invoke('agent:get-facts'),

  // Settings
  getSettings: () => ipcRenderer.invoke('settings:get'),
  saveSettings: (settings: AppSettings) => ipcRenderer.invoke('settings:save', settings),

  // Window
  getVersion: () => ipcRenderer.invoke('app:version'),
}

contextBridge.exposeInMainWorld('crabinfer', api)
