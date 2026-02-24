"use strict";
Object.defineProperty(exports, Symbol.toStringTag, { value: "Module" });
const electron = require("electron");
const path = require("path");
const crabinfer = require("@crabinfer/node");
let engine = null;
let downloadManager = null;
let modelArchitecture = "";
let stopRequested = false;
function getDownloadManager() {
  if (!downloadManager) {
    downloadManager = new crabinfer.ModelDownloadManager();
  }
  return downloadManager;
}
function detectDevice() {
  return crabinfer.detectDevice();
}
function getVersion() {
  return crabinfer.version();
}
function getModelCatalog() {
  return crabinfer.modelCatalog();
}
function getRecommendedModels() {
  const device = crabinfer.detectDevice();
  return crabinfer.recommendedModels(device);
}
const METAL_SAFE_ARCHS = /* @__PURE__ */ new Set(["qwen2", "qwen3"]);
async function loadModel(modelPath) {
  const device = crabinfer.detectDevice();
  const ctxLen = device.recommendedContextLength || 2048;
  let arch = "";
  try {
    const meta = crabinfer.peekModelMetadata(modelPath, ctxLen);
    arch = (meta.architecture || "").toLowerCase();
    console.log(`Model metadata: arch=${arch}, params=${meta.parameterCount}, quant=${meta.quantization}`);
  } catch (err) {
    console.warn("Failed to peek model metadata, falling back to CPU:", err);
  }
  const useMetal = device.hasMetalGpu && METAL_SAFE_ARCHS.has(arch);
  engine = new crabinfer.CrabInferEngine({
    modelPath,
    maxTokens: 2048,
    temperature: 0.7,
    topP: 1,
    contextLength: ctxLen,
    useMetal,
    memoryLimitBytes: device.availableMemoryBytes,
    metallibPath: ""
  });
  await engine.loadModel(modelPath);
  try {
    const info = engine.modelInfo();
    modelArchitecture = info.architecture || "";
    console.log(`Model loaded: ${info.architecture}, ${info.parameterCount} params, backend: ${useMetal ? "Metal" : "CPU"}`);
  } catch {
    modelArchitecture = "";
  }
}
function unloadModel() {
  if (engine) {
    engine.unloadModel();
    engine = null;
    modelArchitecture = "";
  }
}
function isModelLoaded() {
  return engine?.isModelLoaded() ?? false;
}
function applyChatTemplate(architecture, messages) {
  const arch = architecture.toLowerCase();
  if (arch === "qwen2" || arch === "qwen3" || arch.includes("qwen")) {
    return chatMLTemplate(messages);
  }
  if (arch === "phi3" || arch.includes("phi")) {
    return phi3Template(messages);
  }
  if (arch.includes("gemma")) {
    return gemmaTemplate(messages);
  }
  return chatMLTemplate(messages);
}
function chatMLTemplate(messages) {
  let prompt = "";
  for (const msg of messages) {
    prompt += `<|im_start|>${msg.role}
${msg.content}<|im_end|>
`;
  }
  prompt += "<|im_start|>assistant\n";
  return prompt;
}
function phi3Template(messages) {
  let prompt = "";
  for (const msg of messages) {
    const tag = msg.role === "system" ? "system" : msg.role === "user" ? "user" : "assistant";
    prompt += `<|${tag}|>
${msg.content}<|end|>
`;
  }
  prompt += "<|assistant|>\n";
  return prompt;
}
function gemmaTemplate(messages) {
  let prompt = "";
  for (const msg of messages) {
    const role = msg.role === "system" ? "user" : msg.role === "assistant" ? "model" : msg.role;
    prompt += `<start_of_turn>${role}
${msg.content}<end_of_turn>
`;
  }
  prompt += "<start_of_turn>model\n";
  return prompt;
}
async function complete(messages, maxTokens) {
  if (!engine) throw new Error("No model loaded");
  const prompt = applyChatTemplate(modelArchitecture, messages);
  return engine.complete(prompt, maxTokens);
}
function stopInference() {
  stopRequested = true;
  if (engine) {
    engine.reset();
  }
}
async function streamToRenderer(win, messages, maxTokens) {
  if (!engine) throw new Error("No model loaded");
  stopRequested = false;
  const prompt = applyChatTemplate(modelArchitecture, messages);
  engine.reset();
  let firstToken = true;
  for (let i = 0; i < maxTokens; i++) {
    if (stopRequested) {
      if (!win.isDestroyed()) {
        win.webContents.send("inference:stream-token", "", true);
      }
      break;
    }
    const token = engine.nextToken(firstToken ? prompt : "");
    firstToken = false;
    if (!token || token.isEndOfSequence) {
      if (!win.isDestroyed()) {
        win.webContents.send("inference:stream-token", "", true);
      }
      break;
    }
    if (!win.isDestroyed()) {
      win.webContents.send("inference:stream-token", token.text, false);
    }
    await new Promise((resolve) => setImmediate(resolve));
  }
  stopRequested = false;
}
function getLastStats() {
  if (!engine) return null;
  const stats = engine.lastStats();
  if (!stats) return null;
  return {
    tokensPerSecond: stats.tokensPerSecond,
    timeToFirstTokenMs: stats.timeToFirstTokenMs,
    totalTimeMs: stats.totalTimeMs,
    peakMemoryBytes: stats.peakMemoryBytes,
    backend: stats.computeBackend,
    provider: "local",
    model: modelArchitecture || "loaded"
  };
}
let _store = null;
function getStore() {
  if (!_store) {
    const Store = require("electron-store");
    _store = new Store({
      defaults: {
        routingPolicy: "LocalFirst",
        privacyMode: false,
        dataSovereignty: false,
        providers: {}
      }
    });
  }
  return _store;
}
function getSettings() {
  const raw = getStore().store;
  const providers = {};
  for (const [name, config] of Object.entries(raw.providers)) {
    let apiKey = "";
    if (config.encryptedApiKey && electron.safeStorage.isEncryptionAvailable()) {
      try {
        apiKey = electron.safeStorage.decryptString(Buffer.from(config.encryptedApiKey, "base64"));
      } catch {
        apiKey = "";
      }
    }
    providers[name] = {
      apiKey,
      baseUrl: config.baseUrl,
      defaultModel: config.defaultModel
    };
  }
  return {
    routingPolicy: raw.routingPolicy,
    privacyMode: raw.privacyMode,
    dataSovereignty: raw.dataSovereignty,
    providers
  };
}
function saveSettings(settings) {
  const stored = {
    routingPolicy: settings.routingPolicy,
    privacyMode: settings.privacyMode,
    dataSovereignty: settings.dataSovereignty,
    providers: {}
  };
  for (const [name, config] of Object.entries(settings.providers)) {
    let encryptedApiKey;
    if (config.apiKey && electron.safeStorage.isEncryptionAvailable()) {
      encryptedApiKey = electron.safeStorage.encryptString(config.apiKey).toString("base64");
    }
    stored.providers[name] = {
      encryptedApiKey,
      baseUrl: config.baseUrl,
      defaultModel: config.defaultModel
    };
  }
  getStore().store = stored;
}
class AgentBridge {
  conversation = [];
  facts = /* @__PURE__ */ new Map();
  constructor() {
  }
  /** Get tool names (empty until ToolRegistry is available). */
  getToolNames() {
    return [];
  }
  /** Get tool count. */
  getToolCount() {
    return 0;
  }
  /** Generate the tools system prompt section. */
  getToolsPrompt() {
    return "";
  }
  /** Parse tool calls from model output text (stub — always empty). */
  parseToolCalls(_text) {
    return [];
  }
  /** Check if text contains tool calls (stub — always false). */
  hasToolCalls(_text) {
    return false;
  }
  /** Extract plain text (stub — returns input unchanged). */
  extractText(text) {
    return text;
  }
  /**
   * Run the agent loop: send message, get completion, return response.
   * In passthrough mode there is no tool execution — single round only.
   */
  async runAgent(userMessage, getCompletion, _maxRounds = 10) {
    this.conversation.push({ role: "user", content: userMessage });
    const systemPrompt = this.buildSystemPrompt();
    const messages = this.conversation.map((m) => ({ role: m.role, content: m.content }));
    const response = await getCompletion(messages, systemPrompt);
    this.conversation.push({ role: "assistant", content: response });
    return { text: response, toolExecutions: [], rounds: 1 };
  }
  /** Build system prompt with facts and instructions. */
  buildSystemPrompt() {
    const parts = [];
    parts.push(
      "You are CrabInfer Assistant, a helpful AI running locally on the user's Mac.",
      "Think step by step. Provide a clear and helpful response.",
      "If you're unsure about something, ask the user for clarification.",
      ""
    );
    if (this.facts.size > 0) {
      parts.push("## Remembered Facts");
      for (const [key, value] of this.facts) {
        parts.push(`- ${key}: ${value}`);
      }
      parts.push("");
    }
    return parts.join("\n");
  }
  /** Add a fact to memory. */
  addFact(key, value) {
    this.facts.set(key, value);
  }
  /** Remove a fact. */
  removeFact(key) {
    return this.facts.delete(key);
  }
  /** Get all facts. */
  getFacts() {
    return Array.from(this.facts.entries()).map(([key, value]) => ({ key, value }));
  }
  /** Get conversation messages. */
  getConversation() {
    return [...this.conversation];
  }
  /** Clear conversation history. */
  clearConversation() {
    this.conversation = [];
  }
  /** Get status for tray menu. */
  getStatus() {
    return {
      modelLoaded: false,
      // Will be updated by inference bridge
      toolCount: 0,
      mcpServerCount: 0,
      factCount: this.facts.size,
      toolNames: []
    };
  }
  /** Save state (conversation + facts) to disk. */
  save() {
  }
}
let agentBridge = null;
function getAgentBridge() {
  if (!agentBridge) {
    agentBridge = new AgentBridge();
  }
  return agentBridge;
}
function registerIpcHandlers(win, callbacks) {
  const { updateTrayMenu: updateTrayMenu2, sendNotification: sendNotification2 } = callbacks;
  electron.ipcMain.handle("device:info", () => detectDevice());
  electron.ipcMain.handle("app:version", () => getVersion());
  electron.ipcMain.handle("models:catalog", () => {
    const catalog = getModelCatalog();
    const dm = getDownloadManager();
    return catalog.map((m) => ({
      ...m,
      category: m.category || "General",
      isDownloaded: dm.isDownloaded(m.id)
    }));
  });
  electron.ipcMain.handle("models:recommended", () => {
    const models = getRecommendedModels();
    const dm = getDownloadManager();
    return models.map((m) => ({
      ...m,
      category: m.category || "General",
      isDownloaded: dm.isDownloaded(m.id)
    }));
  });
  electron.ipcMain.handle("models:download", async (_event, catalogId) => {
    const dm = getDownloadManager();
    const catalog = getModelCatalog();
    const entry = catalog.find((m) => m.id === catalogId);
    if (!entry) throw new Error(`Model ${catalogId} not found in catalog`);
    await dm.download(entry, (bytesDownloaded, bytesTotal, phase) => {
      if (!win.isDestroyed()) {
        win.webContents.send("models:download-progress", {
          catalogId,
          bytesDownloaded,
          bytesTotal,
          phase
        });
      }
    });
  });
  electron.ipcMain.handle("models:is-downloaded", (_event, catalogId) => {
    return getDownloadManager().isDownloaded(catalogId);
  });
  electron.ipcMain.handle("models:path", (_event, catalogId) => {
    return getDownloadManager().modelPath(catalogId) ?? null;
  });
  electron.ipcMain.handle("inference:load", async (_event, modelPath) => {
    await loadModel(modelPath);
    updateTrayMenu2();
  });
  electron.ipcMain.handle("inference:unload", () => {
    unloadModel();
    updateTrayMenu2();
  });
  electron.ipcMain.handle("inference:is-loaded", () => {
    return isModelLoaded();
  });
  electron.ipcMain.handle(
    "inference:complete",
    async (_event, messages, maxTokens) => {
      return complete(messages, maxTokens);
    }
  );
  electron.ipcMain.handle(
    "inference:stream-start",
    async (_event, messages, maxTokens) => {
      await streamToRenderer(win, messages, maxTokens);
    }
  );
  electron.ipcMain.handle("inference:stop", () => {
    stopInference();
  });
  electron.ipcMain.handle("inference:last-stats", () => {
    return getLastStats();
  });
  electron.ipcMain.handle("agent:get-tools", () => {
    const agent = getAgentBridge();
    return agent.getToolNames();
  });
  electron.ipcMain.handle("agent:get-status", () => {
    const agent = getAgentBridge();
    const status = agent.getStatus();
    return {
      ...status,
      modelLoaded: isModelLoaded()
    };
  });
  electron.ipcMain.handle(
    "agent:run",
    async (_event, userMessage, maxRounds = 10) => {
      const agent = getAgentBridge();
      const getCompletion = async (messages, systemPrompt) => {
        const allMessages = [
          { role: "system", content: systemPrompt },
          ...messages
        ];
        return complete(allMessages, 2048);
      };
      const result = await agent.runAgent(userMessage, getCompletion, maxRounds);
      if (result.toolExecutions.length > 0) {
        for (const exec of result.toolExecutions) {
          if (!win.isDestroyed()) {
            win.webContents.send("agent:tool-executed", exec);
          }
        }
      }
      if (!win.isVisible() && result.text) {
        const preview = result.text.length > 100 ? result.text.slice(0, 100) + "..." : result.text;
        sendNotification2("CrabInfer Assistant", preview);
      }
      updateTrayMenu2();
      return result;
    }
  );
  electron.ipcMain.handle("agent:clear-conversation", () => {
    const agent = getAgentBridge();
    agent.clearConversation();
    updateTrayMenu2();
  });
  electron.ipcMain.handle("agent:get-conversation", () => {
    const agent = getAgentBridge();
    return agent.getConversation();
  });
  electron.ipcMain.handle("agent:add-fact", (_event, key, value) => {
    const agent = getAgentBridge();
    agent.addFact(key, value);
    updateTrayMenu2();
  });
  electron.ipcMain.handle("agent:remove-fact", (_event, key) => {
    const agent = getAgentBridge();
    const removed = agent.removeFact(key);
    updateTrayMenu2();
    return removed;
  });
  electron.ipcMain.handle("agent:get-facts", () => {
    const agent = getAgentBridge();
    return agent.getFacts();
  });
  electron.ipcMain.handle("settings:get", () => getSettings());
  electron.ipcMain.handle("settings:save", (_event, settings) => saveSettings(settings));
}
let mainWindow = null;
let tray = null;
function createWindow() {
  mainWindow = new electron.BrowserWindow({
    width: 720,
    height: 600,
    minWidth: 480,
    minHeight: 400,
    title: "CrabInfer Assistant",
    titleBarStyle: "hiddenInset",
    backgroundColor: "#1a1a2e",
    show: true,
    resizable: true,
    webPreferences: {
      preload: path.join(__dirname, "../preload/index.js"),
      sandbox: false
    }
  });
  registerIpcHandlers(mainWindow, { updateTrayMenu, sendNotification });
  if (process.env.ELECTRON_RENDERER_URL) {
    mainWindow.loadURL(process.env.ELECTRON_RENDERER_URL);
  } else {
    mainWindow.loadFile(path.join(__dirname, "../renderer/index.html"));
  }
  mainWindow.on("close", (e) => {
    if (!electron.app.isQuitting) {
      e.preventDefault();
      mainWindow?.hide();
    }
  });
  mainWindow.on("closed", () => {
    mainWindow = null;
  });
  mainWindow.on("ready-to-show", () => {
    mainWindow?.show();
    mainWindow?.focus();
  });
}
function toggleWindow() {
  if (!mainWindow) {
    createWindow();
    mainWindow?.show();
    mainWindow?.focus();
    return;
  }
  if (mainWindow.isVisible()) {
    mainWindow.hide();
  } else {
    if (tray && process.platform === "darwin") {
      const trayBounds = tray.getBounds();
      const windowBounds = mainWindow.getBounds();
      const x = Math.round(trayBounds.x + trayBounds.width / 2 - windowBounds.width / 2);
      const y = Math.round(trayBounds.y + trayBounds.height);
      mainWindow.setPosition(x, y, false);
    }
    mainWindow.show();
    mainWindow.focus();
  }
}
function createTray() {
  const iconPath = path.join(__dirname, "../../resources/tray-icon.png");
  let icon;
  try {
    icon = electron.nativeImage.createFromPath(iconPath);
    if (process.platform === "darwin") {
      icon = icon.resize({ width: 16, height: 16 });
      icon.setTemplateImage(true);
    }
  } catch {
    icon = electron.nativeImage.createEmpty();
  }
  tray = new electron.Tray(icon);
  tray.setToolTip("CrabInfer Assistant");
  updateTrayMenu();
  tray.on("click", () => {
    toggleWindow();
  });
}
function updateTrayMenu() {
  if (!tray) return;
  const agent = getAgentBridge();
  const status = agent.getStatus();
  const contextMenu = electron.Menu.buildFromTemplate([
    {
      label: `CrabInfer Assistant`,
      enabled: false
    },
    { type: "separator" },
    {
      label: status.modelLoaded ? `Model: loaded` : "No model loaded",
      enabled: false
    },
    {
      label: `Tools: ${status.toolCount}`,
      enabled: false
    },
    {
      label: `MCP servers: ${status.mcpServerCount}`,
      enabled: false
    },
    {
      label: `Facts: ${status.factCount}`,
      enabled: false
    },
    { type: "separator" },
    {
      label: "Show / Hide",
      accelerator: "CommandOrControl+Shift+Space",
      click: () => toggleWindow()
    },
    {
      label: "Clear Conversation",
      click: () => {
        agent.clearConversation();
        if (mainWindow && !mainWindow.isDestroyed()) {
          mainWindow.webContents.send("agent:conversation-cleared");
        }
      }
    },
    { type: "separator" },
    {
      label: "Quit",
      accelerator: "CommandOrControl+Q",
      click: () => {
        electron.app.isQuitting = true;
        electron.app.quit();
      }
    }
  ]);
  tray.setContextMenu(contextMenu);
}
function sendNotification(title, body) {
  if (electron.Notification.isSupported()) {
    const notif = new electron.Notification({ title, body, silent: true });
    notif.on("click", () => {
      toggleWindow();
    });
    notif.show();
  }
}
function registerGlobalShortcut() {
  const ret = electron.globalShortcut.register("CommandOrControl+Shift+Space", () => {
    toggleWindow();
  });
  if (!ret) {
    console.warn("Failed to register global shortcut Cmd+Shift+Space");
  }
}
electron.app.whenReady().then(() => {
  createWindow();
  createTray();
  registerGlobalShortcut();
});
electron.app.on("will-quit", () => {
  electron.globalShortcut.unregisterAll();
  try {
    getAgentBridge().save();
  } catch {
  }
});
electron.app.on("window-all-closed", () => {
});
electron.app.on("activate", () => {
  toggleWindow();
});
electron.app.isQuitting = false;
exports.sendNotification = sendNotification;
exports.updateTrayMenu = updateTrayMenu;
