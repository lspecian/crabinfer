"use strict";
const electron = require("electron");
const api = {
  // Device
  getDeviceInfo: () => electron.ipcRenderer.invoke("device:info"),
  // Models
  getModelCatalog: () => electron.ipcRenderer.invoke("models:catalog"),
  getRecommendedModels: () => electron.ipcRenderer.invoke("models:recommended"),
  downloadModel: (catalogId) => electron.ipcRenderer.invoke("models:download", catalogId),
  onDownloadProgress: (callback) => {
    const handler = (_event, progress) => callback(progress);
    electron.ipcRenderer.on("models:download-progress", handler);
    return () => electron.ipcRenderer.removeListener("models:download-progress", handler);
  },
  isModelDownloaded: (catalogId) => electron.ipcRenderer.invoke("models:is-downloaded", catalogId),
  getModelPath: (catalogId) => electron.ipcRenderer.invoke("models:path", catalogId),
  // Inference
  loadModel: (modelPath) => electron.ipcRenderer.invoke("inference:load", modelPath),
  unloadModel: () => electron.ipcRenderer.invoke("inference:unload"),
  isModelLoaded: () => electron.ipcRenderer.invoke("inference:is-loaded"),
  complete: (messages, maxTokens) => electron.ipcRenderer.invoke("inference:complete", messages, maxTokens),
  streamStart: (messages, maxTokens) => electron.ipcRenderer.invoke("inference:stream-start", messages, maxTokens),
  onStreamToken: (callback) => {
    const handler = (_event, token, done) => callback(token, done);
    electron.ipcRenderer.on("inference:stream-token", handler);
    return () => electron.ipcRenderer.removeListener("inference:stream-token", handler);
  },
  stopInference: () => electron.ipcRenderer.invoke("inference:stop"),
  getLastStats: () => electron.ipcRenderer.invoke("inference:last-stats"),
  // Agent
  agentRun: (message, maxRounds) => electron.ipcRenderer.invoke("agent:run", message, maxRounds),
  agentGetTools: () => electron.ipcRenderer.invoke("agent:get-tools"),
  agentGetStatus: () => electron.ipcRenderer.invoke("agent:get-status"),
  agentClearConversation: () => electron.ipcRenderer.invoke("agent:clear-conversation"),
  agentGetConversation: () => electron.ipcRenderer.invoke("agent:get-conversation"),
  onAgentToolExecuted: (callback) => {
    const handler = (_event, exec) => callback(exec);
    electron.ipcRenderer.on("agent:tool-executed", handler);
    return () => electron.ipcRenderer.removeListener("agent:tool-executed", handler);
  },
  onAgentConversationCleared: (callback) => {
    const handler = () => callback();
    electron.ipcRenderer.on("agent:conversation-cleared", handler);
    return () => electron.ipcRenderer.removeListener("agent:conversation-cleared", handler);
  },
  // Facts/Memory
  agentAddFact: (key, value) => electron.ipcRenderer.invoke("agent:add-fact", key, value),
  agentRemoveFact: (key) => electron.ipcRenderer.invoke("agent:remove-fact", key),
  agentGetFacts: () => electron.ipcRenderer.invoke("agent:get-facts"),
  // Settings
  getSettings: () => electron.ipcRenderer.invoke("settings:get"),
  saveSettings: (settings) => electron.ipcRenderer.invoke("settings:save", settings),
  // Window
  getVersion: () => electron.ipcRenderer.invoke("app:version")
};
electron.contextBridge.exposeInMainWorld("crabinfer", api);
