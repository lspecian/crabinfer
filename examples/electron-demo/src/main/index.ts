/**
 * CrabInfer Menu Bar App — persistent background app with system tray.
 *
 * - Runs as a menu bar app (no dock icon on macOS)
 * - Global keyboard shortcut (Cmd+Shift+Space) to show/hide
 * - System tray with status and context menu
 * - Agent runtime with tool calling, MCP, memory, knowledge
 * - Native notifications for agent responses
 */

import {
  app,
  BrowserWindow,
  Tray,
  Menu,
  globalShortcut,
  nativeImage,
  Notification,
} from 'electron'
import { join } from 'path'
import { registerIpcHandlers } from './ipc-handlers'
import { getAgentBridge } from './agent-bridge'

let mainWindow: BrowserWindow | null = null
let tray: Tray | null = null

function createWindow(): void {
  mainWindow = new BrowserWindow({
    width: 720,
    height: 600,
    minWidth: 480,
    minHeight: 400,
    title: 'CrabInfer Assistant',
    titleBarStyle: 'hiddenInset',
    backgroundColor: '#1a1a2e',
    show: true,
    resizable: true,
    webPreferences: {
      preload: join(__dirname, '../preload/index.js'),
      sandbox: false,
    },
  })

  // Register all IPC handlers (inference + agent + settings)
  registerIpcHandlers(mainWindow, { updateTrayMenu, sendNotification })

  if (process.env.ELECTRON_RENDERER_URL) {
    mainWindow.loadURL(process.env.ELECTRON_RENDERER_URL)
  } else {
    mainWindow.loadFile(join(__dirname, '../renderer/index.html'))
  }

  // Hide instead of close — persistent background app
  mainWindow.on('close', (e) => {
    if (!app.isQuitting) {
      e.preventDefault()
      mainWindow?.hide()
    }
  })

  mainWindow.on('closed', () => {
    mainWindow = null
  })

  // Show window when ready to paint (avoids white flash)
  mainWindow.on('ready-to-show', () => {
    mainWindow?.show()
    mainWindow?.focus()
  })
}

function toggleWindow(): void {
  if (!mainWindow) {
    createWindow()
    mainWindow?.show()
    mainWindow?.focus()
    return
  }

  if (mainWindow.isVisible()) {
    mainWindow.hide()
  } else {
    // Position near tray icon on macOS
    if (tray && process.platform === 'darwin') {
      const trayBounds = tray.getBounds()
      const windowBounds = mainWindow.getBounds()
      const x = Math.round(trayBounds.x + trayBounds.width / 2 - windowBounds.width / 2)
      const y = Math.round(trayBounds.y + trayBounds.height)
      mainWindow.setPosition(x, y, false)
    }
    mainWindow.show()
    mainWindow.focus()
  }
}

function createTray(): void {
  // Create a simple tray icon (16x16 template image for macOS)
  const iconPath = join(__dirname, '../../resources/tray-icon.png')
  let icon: Electron.NativeImage
  try {
    icon = nativeImage.createFromPath(iconPath)
    if (process.platform === 'darwin') {
      icon = icon.resize({ width: 16, height: 16 })
      icon.setTemplateImage(true)
    }
  } catch {
    // Fallback: create a tiny colored square
    icon = nativeImage.createEmpty()
  }

  tray = new Tray(icon)
  tray.setToolTip('CrabInfer Assistant')
  updateTrayMenu()

  tray.on('click', () => {
    toggleWindow()
  })
}

/** Update the tray context menu with current status. */
export function updateTrayMenu(): void {
  if (!tray) return

  const agent = getAgentBridge()
  const status = agent.getStatus()

  const contextMenu = Menu.buildFromTemplate([
    {
      label: `CrabInfer Assistant`,
      enabled: false,
    },
    { type: 'separator' },
    {
      label: status.modelLoaded ? `Model: loaded` : 'No model loaded',
      enabled: false,
    },
    {
      label: `Tools: ${status.toolCount}`,
      enabled: false,
    },
    {
      label: `MCP servers: ${status.mcpServerCount}`,
      enabled: false,
    },
    {
      label: `Facts: ${status.factCount}`,
      enabled: false,
    },
    { type: 'separator' },
    {
      label: 'Show / Hide',
      accelerator: 'CommandOrControl+Shift+Space',
      click: () => toggleWindow(),
    },
    {
      label: 'Clear Conversation',
      click: () => {
        agent.clearConversation()
        if (mainWindow && !mainWindow.isDestroyed()) {
          mainWindow.webContents.send('agent:conversation-cleared')
        }
      },
    },
    { type: 'separator' },
    {
      label: 'Quit',
      accelerator: 'CommandOrControl+Q',
      click: () => {
        app.isQuitting = true
        app.quit()
      },
    },
  ])

  tray.setContextMenu(contextMenu)
}

/** Send a native notification. */
export function sendNotification(title: string, body: string): void {
  if (Notification.isSupported()) {
    const notif = new Notification({ title, body, silent: true })
    notif.on('click', () => {
      toggleWindow()
    })
    notif.show()
  }
}

function registerGlobalShortcut(): void {
  const ret = globalShortcut.register('CommandOrControl+Shift+Space', () => {
    toggleWindow()
  })
  if (!ret) {
    console.warn('Failed to register global shortcut Cmd+Shift+Space')
  }
}

// ─── App lifecycle ──────────────────────────────────────────────────────────

app.whenReady().then(() => {
  createWindow()
  createTray()
  registerGlobalShortcut()
})

app.on('will-quit', () => {
  globalShortcut.unregisterAll()
  // Save agent state on exit
  try {
    getAgentBridge().save()
  } catch {
    // Best-effort save
  }
})

app.on('window-all-closed', () => {
  // Do NOT quit — we're a menu bar app
})

app.on('activate', () => {
  toggleWindow()
})

// Extend app type for isQuitting flag
declare module 'electron' {
  interface App {
    isQuitting: boolean
  }
}
app.isQuitting = false
