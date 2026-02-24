/**
 * Settings persistence using electron-store.
 * API keys are stored encrypted via Electron's safeStorage when available.
 */

import { safeStorage } from 'electron'

interface StoredSettings {
  routingPolicy: string
  privacyMode: boolean
  dataSovereignty: boolean
  providers: Record<string, { encryptedApiKey?: string; baseUrl?: string; defaultModel?: string }>
}

// Lazy-init: electron-store calls app.getPath() in its constructor,
// which is only available after Electron's app module is ready.
let _store: any = null
function getStore() {
  if (!_store) {
    const Store = require('electron-store')
    _store = new Store({
      defaults: {
        routingPolicy: 'LocalFirst',
        privacyMode: false,
        dataSovereignty: false,
        providers: {},
      },
    })
  }
  return _store
}

/** Get settings with decrypted API keys */
export function getSettings() {
  const raw = getStore().store
  const providers: Record<string, { apiKey: string; baseUrl?: string; defaultModel?: string }> = {}

  for (const [name, config] of Object.entries(raw.providers)) {
    let apiKey = ''
    if (config.encryptedApiKey && safeStorage.isEncryptionAvailable()) {
      try {
        apiKey = safeStorage.decryptString(Buffer.from(config.encryptedApiKey, 'base64'))
      } catch {
        apiKey = ''
      }
    }
    providers[name] = {
      apiKey,
      baseUrl: config.baseUrl,
      defaultModel: config.defaultModel,
    }
  }

  return {
    routingPolicy: raw.routingPolicy as any,
    privacyMode: raw.privacyMode,
    dataSovereignty: raw.dataSovereignty,
    providers,
  }
}

/** Save settings with encrypted API keys */
export function saveSettings(settings: {
  routingPolicy: string
  privacyMode: boolean
  dataSovereignty: boolean
  providers: Record<string, { apiKey: string; baseUrl?: string; defaultModel?: string }>
}): void {
  const stored: StoredSettings = {
    routingPolicy: settings.routingPolicy,
    privacyMode: settings.privacyMode,
    dataSovereignty: settings.dataSovereignty,
    providers: {},
  }

  for (const [name, config] of Object.entries(settings.providers)) {
    let encryptedApiKey: string | undefined
    if (config.apiKey && safeStorage.isEncryptionAvailable()) {
      encryptedApiKey = safeStorage.encryptString(config.apiKey).toString('base64')
    }
    stored.providers[name] = {
      encryptedApiKey,
      baseUrl: config.baseUrl,
      defaultModel: config.defaultModel,
    }
  }

  getStore().store = stored
}
