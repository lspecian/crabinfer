import { useState, useCallback, useEffect, useRef } from 'react'
import type { ChatMessage, ResponseStats } from '../types'

let messageIdCounter = 0
function nextId() {
  return `msg-${++messageIdCounter}-${Date.now()}`
}

export function useChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [isStreaming, setIsStreaming] = useState(false)
  const [modelLoaded, setModelLoaded] = useState(false)
  const streamBufferRef = useRef('')

  // Check model status on mount
  useEffect(() => {
    window.crabinfer.isModelLoaded().then(setModelLoaded)
  }, [])

  // Listen for stream tokens
  useEffect(() => {
    const unsub = window.crabinfer.onStreamToken((token, done) => {
      streamBufferRef.current += token
      const content = streamBufferRef.current

      setMessages((prev) => {
        const last = prev[prev.length - 1]
        if (last?.role === 'assistant') {
          return [...prev.slice(0, -1), { ...last, content }]
        }
        return prev
      })

      if (done) {
        setIsStreaming(false)
        // Fetch stats
        window.crabinfer.getLastStats().then((stats) => {
          if (stats) {
            setMessages((prev) => {
              const last = prev[prev.length - 1]
              if (last?.role === 'assistant') {
                return [...prev.slice(0, -1), { ...last, stats }]
              }
              return prev
            })
          }
        })
      }
    })
    return unsub
  }, [])

  const loadModel = useCallback(async (modelPath: string) => {
    await window.crabinfer.loadModel(modelPath)
    setModelLoaded(true)
  }, [])

  const unloadModel = useCallback(async () => {
    await window.crabinfer.unloadModel()
    setModelLoaded(false)
  }, [])

  const sendMessage = useCallback(
    async (content: string) => {
      if (!modelLoaded || isStreaming) return

      const userMsg: ChatMessage = {
        id: nextId(),
        role: 'user',
        content,
        timestamp: Date.now(),
      }

      const assistantMsg: ChatMessage = {
        id: nextId(),
        role: 'assistant',
        content: '',
        timestamp: Date.now(),
      }

      setMessages((prev) => [...prev, userMsg, assistantMsg])
      setIsStreaming(true)
      streamBufferRef.current = ''

      const systemMsg = {
        role: 'system',
        content: 'You are a helpful AI assistant running locally via CrabInfer. Be concise and helpful.',
      }
      const apiMessages = [
        systemMsg,
        ...[...messages, userMsg].map((m) => ({
          role: m.role,
          content: m.content,
        })),
      ]

      try {
        // Timeout after 120s in case inference hangs
        const timeout = new Promise<never>((_, reject) =>
          setTimeout(() => reject(new Error('Inference timed out after 120s. The model may be too large or unresponsive.')), 120000)
        )
        await Promise.race([
          window.crabinfer.streamStart(apiMessages, 1024),
          timeout,
        ])
      } catch (err) {
        console.error('Inference error:', err)
        setIsStreaming(false)
        setMessages((prev) => {
          const last = prev[prev.length - 1]
          if (last?.role === 'assistant') {
            return [
              ...prev.slice(0, -1),
              { ...last, content: `Error: ${err instanceof Error ? err.message : String(err)}` },
            ]
          }
          return prev
        })
      }
    },
    [messages, modelLoaded, isStreaming]
  )

  const stopInference = useCallback(() => {
    window.crabinfer.stopInference()
    setIsStreaming(false)
  }, [])

  const clearMessages = useCallback(() => {
    setMessages([])
  }, [])

  return {
    messages,
    isStreaming,
    modelLoaded,
    sendMessage,
    clearMessages,
    loadModel,
    unloadModel,
    stopInference,
  }
}
