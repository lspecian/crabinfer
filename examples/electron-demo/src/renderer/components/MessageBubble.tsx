import React, { useState } from 'react'
import type { ChatMessage } from '../types'
import { StatsBar } from './StatsBar'
import { ToolExecutionLog } from './ToolExecutionLog'

interface Props {
  message: ChatMessage
}

/** Split content into thinking trace and visible response. */
function parseThinking(content: string): { thinking: string; response: string } {
  const thinkMatch = content.match(/<think>([\s\S]*?)<\/think>/)
  if (thinkMatch) {
    const thinking = thinkMatch[1].trim()
    const response = content.replace(/<think>[\s\S]*?<\/think>/, '').trim()
    return { thinking, response }
  }
  // Still inside a <think> block (streaming)
  if (content.startsWith('<think>') && !content.includes('</think>')) {
    return { thinking: content.replace('<think>', '').trim(), response: '' }
  }
  return { thinking: '', response: content }
}

export function MessageBubble({ message }: Props) {
  const isUser = message.role === 'user'
  const [showThinking, setShowThinking] = useState(false)

  const { thinking, response } = isUser
    ? { thinking: '', response: message.content }
    : parseThinking(message.content)

  const displayContent = response || (thinking ? '' : message.content)
  const isThinkingInProgress = !isUser && thinking && !response && !message.content.includes('</think>')

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
      <div className={`max-w-[80%] ${isUser ? 'order-2' : 'order-1'}`}>
        {/* Thinking indicator / collapsible */}
        {!isUser && thinking && (
          <button
            onClick={() => setShowThinking(!showThinking)}
            className="text-xs text-gray-500 hover:text-gray-300 mb-1 flex items-center gap-1 transition-colors"
          >
            {isThinkingInProgress ? (
              <span className="animate-pulse">Thinking...</span>
            ) : (
              <>
                <span>{showThinking ? '▼' : '▶'}</span>
                <span>Reasoning ({thinking.split(/\s+/).length} words)</span>
              </>
            )}
          </button>
        )}
        {showThinking && thinking && (
          <div className="bg-surface-lighter/50 rounded-lg px-3 py-2 mb-1 text-xs text-gray-400 max-h-40 overflow-y-auto whitespace-pre-wrap">
            {thinking}
          </div>
        )}
        {/* Main bubble */}
        <div
          className={`rounded-2xl px-4 py-3 ${
            isUser
              ? 'bg-blue-600 text-white rounded-br-md'
              : 'bg-surface-light text-gray-100 rounded-bl-md'
          }`}
        >
          <div className="whitespace-pre-wrap break-words text-sm leading-relaxed">
            {displayContent || (
              <span className="inline-flex items-center gap-1.5 text-gray-400">
                <span className="text-xs">{isThinkingInProgress ? 'Reasoning' : 'Thinking'}</span>
                <span className="inline-flex gap-0.5">
                  <span className="animate-pulse">.</span>
                  <span className="animate-pulse" style={{ animationDelay: '0.2s' }}>.</span>
                  <span className="animate-pulse" style={{ animationDelay: '0.4s' }}>.</span>
                </span>
              </span>
            )}
          </div>
        </div>
        {message.toolExecutions && message.toolExecutions.length > 0 && (
          <ToolExecutionLog executions={message.toolExecutions} />
        )}
        {message.stats && <StatsBar stats={message.stats} />}
      </div>
    </div>
  )
}
