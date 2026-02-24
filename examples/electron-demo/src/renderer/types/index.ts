/** Message in a conversation */
export interface ChatMessage {
  id: string
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp: number
  stats?: ResponseStats
  toolExecutions?: AgentToolExecution[]
}

/** Stats displayed under assistant responses */
export interface ResponseStats {
  tokensPerSecond: number
  timeToFirstTokenMs: number
  totalTimeMs: number
  peakMemoryBytes: number
  backend: string
  provider: string
  model: string
}

/** Device info from @crabinfer/node */
export interface DeviceInfo {
  chipName: string
  chipVariant: string
  totalMemoryBytes: number
  availableMemoryBytes: number
  hasMetalGpu: boolean
  hasNeuralEngine: boolean
  recommendedQuant: string
  maxModelSizeB: number
  recommendedContextLength: number
}

/** Model catalog entry */
export interface CatalogModel {
  id: string
  name: string
  hfRepo: string
  ggufFile: string
  sizeBytes: number
  paramCountB: number
  quant: string
  architecture: string
  category: string
  isDownloaded: boolean
}

/** Download progress update */
export interface DownloadProgress {
  catalogId: string
  bytesDownloaded: number
  bytesTotal: number
  phase: string
}

/** Settings persisted to electron-store */
export interface AppSettings {
  routingPolicy: 'LocalFirst' | 'CloudFirst' | 'LocalOnly' | 'Auto' | 'SelfHostedFirst'
  privacyMode: boolean
  dataSovereignty: boolean
  providers: Record<string, { apiKey: string; baseUrl?: string; defaultModel?: string }>
}

// ─── Agent types ─────────────────────────────────────────────────────────────

/** Record of a tool execution during an agent turn. */
export interface AgentToolExecution {
  toolName: string
  arguments: string
  output: string
  isError: boolean
}

/** Response from agent:run. */
export interface AgentResponse {
  text: string
  toolExecutions: AgentToolExecution[]
  rounds: number
}

/** Agent runtime status. */
export interface AgentStatus {
  modelLoaded: boolean
  toolCount: number
  mcpServerCount: number
  factCount: number
  toolNames: string[]
}

/** A remembered fact. */
export interface AgentFact {
  key: string
  value: string
}

/** Agent conversation message. */
export interface AgentConversationMessage {
  role: 'user' | 'assistant' | 'system' | 'tool'
  content: string
}

// ─── IPC API ─────────────────────────────────────────────────────────────────

/** IPC API exposed via contextBridge */
export interface CrabInferAPI {
  // Device
  getDeviceInfo(): Promise<DeviceInfo>

  // Models
  getModelCatalog(): Promise<CatalogModel[]>
  getRecommendedModels(): Promise<CatalogModel[]>
  downloadModel(catalogId: string): Promise<void>
  onDownloadProgress(callback: (progress: DownloadProgress) => void): () => void
  isModelDownloaded(catalogId: string): Promise<boolean>
  getModelPath(catalogId: string): Promise<string | null>

  // Inference
  loadModel(modelPath: string): Promise<void>
  unloadModel(): Promise<void>
  isModelLoaded(): Promise<boolean>
  complete(messages: { role: string; content: string }[], maxTokens: number): Promise<string>
  streamStart(messages: { role: string; content: string }[], maxTokens: number): Promise<void>
  onStreamToken(callback: (token: string, done: boolean) => void): () => void
  stopInference(): Promise<void>
  getLastStats(): Promise<ResponseStats | null>

  // Agent
  agentRun(message: string, maxRounds?: number): Promise<AgentResponse>
  agentGetTools(): Promise<string[]>
  agentGetStatus(): Promise<AgentStatus>
  agentClearConversation(): Promise<void>
  agentGetConversation(): Promise<AgentConversationMessage[]>
  onAgentToolExecuted(callback: (exec: AgentToolExecution) => void): () => void
  onAgentConversationCleared(callback: () => void): () => void

  // Facts/Memory
  agentAddFact(key: string, value: string): Promise<void>
  agentRemoveFact(key: string): Promise<boolean>
  agentGetFacts(): Promise<AgentFact[]>

  // Settings
  getSettings(): Promise<AppSettings>
  saveSettings(settings: AppSettings): Promise<void>

  // Window
  getVersion(): Promise<string>
}

declare global {
  interface Window {
    crabinfer: CrabInferAPI
  }
}
