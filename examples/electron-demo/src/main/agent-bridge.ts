/**
 * Agent bridge — wraps @crabinfer/node agent runtime for Electron.
 *
 * Provides:
 * - Conversation state management
 * - Fact/memory storage
 * - Status for tray menu
 *
 * Note: Tool registry / MCP / tool-call parsing APIs are not yet exposed by
 * @crabinfer/node, so the agent bridge operates in "passthrough" mode — it
 * forwards prompts to the model and returns the raw response without tool
 * execution.  When those APIs are added to the binding, this file can be
 * updated to enable the full agent loop.
 */

interface AgentMessage {
  role: 'user' | 'assistant' | 'system' | 'tool'
  content: string
}

interface ToolExecution {
  toolName: string
  arguments: string
  output: string
  isError: boolean
}

interface AgentStatus {
  modelLoaded: boolean
  toolCount: number
  mcpServerCount: number
  factCount: number
  toolNames: string[]
}

class AgentBridge {
  private conversation: AgentMessage[] = []
  private facts: Map<string, string> = new Map()

  constructor() {
    // Tool registry / MCP not yet available in the Node binding — agent runs
    // in passthrough mode (no tool execution).
  }

  /** Get tool names (empty until ToolRegistry is available). */
  getToolNames(): string[] {
    return []
  }

  /** Get tool count. */
  getToolCount(): number {
    return 0
  }

  /** Generate the tools system prompt section. */
  getToolsPrompt(): string {
    return ''
  }

  /** Parse tool calls from model output text (stub — always empty). */
  parseToolCalls(_text: string): Array<{ name: string; arguments: string }> {
    return []
  }

  /** Check if text contains tool calls (stub — always false). */
  hasToolCalls(_text: string): boolean {
    return false
  }

  /** Extract plain text (stub — returns input unchanged). */
  extractText(text: string): string {
    return text
  }

  /**
   * Run the agent loop: send message, get completion, return response.
   * In passthrough mode there is no tool execution — single round only.
   */
  async runAgent(
    userMessage: string,
    getCompletion: (messages: { role: string; content: string }[], systemPrompt: string) => Promise<string>,
    _maxRounds = 10
  ): Promise<{ text: string; toolExecutions: ToolExecution[]; rounds: number }> {
    this.conversation.push({ role: 'user', content: userMessage })

    const systemPrompt = this.buildSystemPrompt()
    const messages = this.conversation.map((m) => ({ role: m.role, content: m.content }))
    const response = await getCompletion(messages, systemPrompt)

    this.conversation.push({ role: 'assistant', content: response })
    return { text: response, toolExecutions: [], rounds: 1 }
  }

  /** Build system prompt with facts and instructions. */
  private buildSystemPrompt(): string {
    const parts: string[] = []

    parts.push(
      "You are CrabInfer Assistant, a helpful AI running locally on the user's Mac.",
      'Think step by step. Provide a clear and helpful response.',
      "If you're unsure about something, ask the user for clarification.",
      ''
    )

    // Facts/memory section
    if (this.facts.size > 0) {
      parts.push('## Remembered Facts')
      for (const [key, value] of this.facts) {
        parts.push(`- ${key}: ${value}`)
      }
      parts.push('')
    }

    return parts.join('\n')
  }

  /** Add a fact to memory. */
  addFact(key: string, value: string): void {
    this.facts.set(key, value)
  }

  /** Remove a fact. */
  removeFact(key: string): boolean {
    return this.facts.delete(key)
  }

  /** Get all facts. */
  getFacts(): Array<{ key: string; value: string }> {
    return Array.from(this.facts.entries()).map(([key, value]) => ({ key, value }))
  }

  /** Get conversation messages. */
  getConversation(): AgentMessage[] {
    return [...this.conversation]
  }

  /** Clear conversation history. */
  clearConversation(): void {
    this.conversation = []
  }

  /** Get status for tray menu. */
  getStatus(): AgentStatus {
    return {
      modelLoaded: false, // Will be updated by inference bridge
      toolCount: 0,
      mcpServerCount: 0,
      factCount: this.facts.size,
      toolNames: [],
    }
  }

  /** Save state (conversation + facts) to disk. */
  save(): void {
    // Best-effort save — in a full implementation this would persist to ~/.crabinfer/
    // For the demo, state lives in memory
  }
}

// Singleton
let agentBridge: AgentBridge | null = null

export function getAgentBridge(): AgentBridge {
  if (!agentBridge) {
    agentBridge = new AgentBridge()
  }
  return agentBridge
}
