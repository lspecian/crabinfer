import { useState, useCallback, useEffect } from 'react'
import type { AgentStatus, AgentToolExecution, AgentFact } from '../types'

export function useAgent() {
  const [status, setStatus] = useState<AgentStatus | null>(null)
  const [facts, setFacts] = useState<AgentFact[]>([])
  const [isRunning, setIsRunning] = useState(false)

  // Fetch initial status
  useEffect(() => {
    window.crabinfer.agentGetStatus().then(setStatus)
    window.crabinfer.agentGetFacts().then(setFacts)
  }, [])

  // Listen for tool executions
  useEffect(() => {
    const unsub = window.crabinfer.onAgentToolExecuted(() => {
      // Refresh status after tool execution
      window.crabinfer.agentGetStatus().then(setStatus)
    })
    return unsub
  }, [])

  // Listen for conversation cleared (from tray menu)
  useEffect(() => {
    const unsub = window.crabinfer.onAgentConversationCleared(() => {
      window.crabinfer.agentGetStatus().then(setStatus)
    })
    return unsub
  }, [])

  const runAgent = useCallback(async (message: string) => {
    setIsRunning(true)
    try {
      const result = await window.crabinfer.agentRun(message)
      // Refresh status after run
      const newStatus = await window.crabinfer.agentGetStatus()
      setStatus(newStatus)
      return result
    } finally {
      setIsRunning(false)
    }
  }, [])

  const addFact = useCallback(async (key: string, value: string) => {
    await window.crabinfer.agentAddFact(key, value)
    const updated = await window.crabinfer.agentGetFacts()
    setFacts(updated)
    const newStatus = await window.crabinfer.agentGetStatus()
    setStatus(newStatus)
  }, [])

  const removeFact = useCallback(async (key: string) => {
    await window.crabinfer.agentRemoveFact(key)
    const updated = await window.crabinfer.agentGetFacts()
    setFacts(updated)
    const newStatus = await window.crabinfer.agentGetStatus()
    setStatus(newStatus)
  }, [])

  const clearConversation = useCallback(async () => {
    await window.crabinfer.agentClearConversation()
    const newStatus = await window.crabinfer.agentGetStatus()
    setStatus(newStatus)
  }, [])

  const refreshStatus = useCallback(async () => {
    const newStatus = await window.crabinfer.agentGetStatus()
    setStatus(newStatus)
  }, [])

  return {
    status,
    facts,
    isRunning,
    runAgent,
    addFact,
    removeFact,
    clearConversation,
    refreshStatus,
  }
}
