/**
 * IPC handler registrations — bridges renderer requests to @crabinfer/node.
 *
 * Handles: device, models, inference, agent, settings.
 */

import { ipcMain, type BrowserWindow } from 'electron'
import {
  detectDevice,
  getVersion,
  getModelCatalog,
  getRecommendedModels,
  getDownloadManager,
  loadModel,
  unloadModel,
  isModelLoaded,
  complete,
  streamToRenderer,
  stopInference,
  getLastStats,
} from './inference-bridge'
import { getSettings, saveSettings } from './settings-store'
import { getAgentBridge } from './agent-bridge'

interface Callbacks {
  updateTrayMenu: () => void
  sendNotification: (title: string, body: string) => void
}

export function registerIpcHandlers(win: BrowserWindow, callbacks: Callbacks): void {
  const { updateTrayMenu, sendNotification } = callbacks

  // ─── Device ────────────────────────────────────────────────────────────────
  ipcMain.handle('device:info', () => detectDevice())

  // ─── App ───────────────────────────────────────────────────────────────────
  ipcMain.handle('app:version', () => getVersion())

  // ─── Models ────────────────────────────────────────────────────────────────
  ipcMain.handle('models:catalog', () => {
    const catalog = getModelCatalog()
    const dm = getDownloadManager()
    return catalog.map((m: any) => ({
      ...m,
      category: m.category || 'General',
      isDownloaded: dm.isDownloaded(m.id),
    }))
  })

  ipcMain.handle('models:recommended', () => {
    const models = getRecommendedModels()
    const dm = getDownloadManager()
    return models.map((m: any) => ({
      ...m,
      category: m.category || 'General',
      isDownloaded: dm.isDownloaded(m.id),
    }))
  })

  ipcMain.handle('models:download', async (_event, catalogId: string) => {
    const dm = getDownloadManager()
    const catalog = getModelCatalog()
    const entry = catalog.find((m: any) => m.id === catalogId)
    if (!entry) throw new Error(`Model ${catalogId} not found in catalog`)

    await dm.download(entry, (bytesDownloaded: number, bytesTotal: number, phase: string) => {
      if (!win.isDestroyed()) {
        win.webContents.send('models:download-progress', {
          catalogId,
          bytesDownloaded,
          bytesTotal,
          phase,
        })
      }
    })
  })

  ipcMain.handle('models:is-downloaded', (_event, catalogId: string) => {
    return getDownloadManager().isDownloaded(catalogId)
  })

  ipcMain.handle('models:path', (_event, catalogId: string) => {
    return getDownloadManager().modelPath(catalogId) ?? null
  })

  // ─── Inference ─────────────────────────────────────────────────────────────
  ipcMain.handle('inference:load', async (_event, modelPath: string) => {
    await loadModel(modelPath)
    updateTrayMenu()
  })

  ipcMain.handle('inference:unload', () => {
    unloadModel()
    updateTrayMenu()
  })

  ipcMain.handle('inference:is-loaded', () => {
    return isModelLoaded()
  })

  ipcMain.handle(
    'inference:complete',
    async (_event, messages: { role: string; content: string }[], maxTokens: number) => {
      return complete(messages, maxTokens)
    }
  )

  ipcMain.handle(
    'inference:stream-start',
    async (_event, messages: { role: string; content: string }[], maxTokens: number) => {
      await streamToRenderer(win, messages, maxTokens)
    }
  )

  ipcMain.handle('inference:stop', () => {
    stopInference()
  })

  ipcMain.handle('inference:last-stats', () => {
    return getLastStats()
  })

  // ─── Agent ─────────────────────────────────────────────────────────────────
  ipcMain.handle('agent:get-tools', () => {
    const agent = getAgentBridge()
    return agent.getToolNames()
  })

  ipcMain.handle('agent:get-status', () => {
    const agent = getAgentBridge()
    const status = agent.getStatus()
    return {
      ...status,
      modelLoaded: isModelLoaded(),
    }
  })

  ipcMain.handle(
    'agent:run',
    async (_event, userMessage: string, maxRounds: number = 10) => {
      const agent = getAgentBridge()

      // Use the loaded model for completion
      const getCompletion = async (
        messages: { role: string; content: string }[],
        systemPrompt: string
      ): Promise<string> => {
        const allMessages = [
          { role: 'system', content: systemPrompt },
          ...messages,
        ]
        return complete(allMessages, 2048)
      }

      const result = await agent.runAgent(userMessage, getCompletion, maxRounds)

      // Send tool execution updates to renderer
      if (result.toolExecutions.length > 0) {
        for (const exec of result.toolExecutions) {
          if (!win.isDestroyed()) {
            win.webContents.send('agent:tool-executed', exec)
          }
        }
      }

      // Send notification if window is hidden
      if (!win.isVisible() && result.text) {
        const preview = result.text.length > 100 ? result.text.slice(0, 100) + '...' : result.text
        sendNotification('CrabInfer Assistant', preview)
      }

      // Update tray menu (fact count may have changed)
      updateTrayMenu()

      return result
    }
  )

  ipcMain.handle('agent:clear-conversation', () => {
    const agent = getAgentBridge()
    agent.clearConversation()
    updateTrayMenu()
  })

  ipcMain.handle('agent:get-conversation', () => {
    const agent = getAgentBridge()
    return agent.getConversation()
  })

  // ─── Facts/Memory ──────────────────────────────────────────────────────────
  ipcMain.handle('agent:add-fact', (_event, key: string, value: string) => {
    const agent = getAgentBridge()
    agent.addFact(key, value)
    updateTrayMenu()
  })

  ipcMain.handle('agent:remove-fact', (_event, key: string) => {
    const agent = getAgentBridge()
    const removed = agent.removeFact(key)
    updateTrayMenu()
    return removed
  })

  ipcMain.handle('agent:get-facts', () => {
    const agent = getAgentBridge()
    return agent.getFacts()
  })

  // ─── Settings ──────────────────────────────────────────────────────────────
  ipcMain.handle('settings:get', () => getSettings())
  ipcMain.handle('settings:save', (_event, settings) => saveSettings(settings))
}
