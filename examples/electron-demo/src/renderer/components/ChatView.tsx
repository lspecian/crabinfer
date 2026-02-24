import React, { useState, useRef, useEffect } from 'react'
import type { ChatMessage } from '../types'
import { MessageBubble } from './MessageBubble'

interface Props {
  messages: ChatMessage[]
  isStreaming: boolean
  modelLoaded: boolean
  onSendMessage: (content: string) => void
  onStop: () => void
}

export function ChatView({ messages, isStreaming, modelLoaded, onSendMessage, onStop }: Props) {
  const [input, setInput] = useState('')
  const scrollRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  // Auto-scroll to bottom
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [messages])

  const handleSubmit = () => {
    const trimmed = input.trim()
    if (!trimmed || isStreaming || !modelLoaded) return
    onSendMessage(trimmed)
    setInput('')
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  return (
    <div className="flex flex-col h-full">
      {/* Drag region — clears macOS traffic light buttons */}
      <div className="h-8 shrink-0" style={{ WebkitAppRegion: 'drag' } as React.CSSProperties} />
      {/* Messages */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 pb-4 space-y-2">
        {messages.length === 0 && (
          <div className="flex items-center justify-center h-full text-gray-500">
            <div className="text-center">
              <div className="text-4xl mb-4">🦀</div>
              <div className="text-lg font-medium">CrabInfer</div>
              <div className="text-sm mt-1">
                {modelLoaded
                  ? 'Model loaded. Start chatting!'
                  : 'Load a model from the sidebar to begin.'}
              </div>
            </div>
          </div>
        )}
        {messages.map((msg) => (
          <MessageBubble key={msg.id} message={msg} />
        ))}
      </div>

      {/* Input */}
      <div className="border-t border-gray-800 p-4">
        <div className="flex gap-2">
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={
              modelLoaded
                ? isStreaming
                  ? 'Generating...'
                  : 'Type a message...'
                : 'Load a model first...'
            }
            disabled={!modelLoaded || isStreaming}
            rows={1}
            className="flex-1 bg-surface-lighter rounded-xl px-4 py-3 text-sm text-white placeholder-gray-500 resize-none focus:outline-none focus:ring-1 focus:ring-crab disabled:opacity-50"
          />
          {isStreaming ? (
            <button
              onClick={onStop}
              className="bg-red-600 hover:bg-red-700 text-white rounded-xl px-4 py-3 text-sm font-medium transition-colors"
            >
              Stop
            </button>
          ) : (
            <button
              onClick={handleSubmit}
              disabled={!input.trim() || !modelLoaded}
              className="bg-crab hover:bg-crab-light disabled:opacity-30 disabled:cursor-not-allowed text-white rounded-xl px-4 py-3 text-sm font-medium transition-colors"
            >
              Send
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
