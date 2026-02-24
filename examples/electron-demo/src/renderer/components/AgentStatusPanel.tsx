import React from 'react'
import type { AgentStatus, AgentFact } from '../types'

interface Props {
  status: AgentStatus | null
  facts: AgentFact[]
  onRemoveFact: (key: string) => void
}

export function AgentStatusPanel({ status, facts, onRemoveFact }: Props) {
  if (!status) return null

  return (
    <div className="space-y-3">
      <div className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
        Agent Runtime
      </div>

      {/* Status grid */}
      <div className="grid grid-cols-2 gap-2 text-xs">
        <div className="bg-surface rounded-lg px-2 py-1.5">
          <div className="text-gray-500">Tools</div>
          <div className="text-white font-medium">{status.toolCount}</div>
        </div>
        <div className="bg-surface rounded-lg px-2 py-1.5">
          <div className="text-gray-500">MCP</div>
          <div className="text-white font-medium">{status.mcpServerCount} servers</div>
        </div>
      </div>

      {/* Tool names */}
      {status.toolNames.length > 0 && (
        <div>
          <div className="text-xs text-gray-500 mb-1">Available tools:</div>
          <div className="flex flex-wrap gap-1">
            {status.toolNames.map((name) => (
              <span
                key={name}
                className="inline-block bg-surface text-gray-400 text-[10px] px-1.5 py-0.5 rounded"
              >
                {name}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Facts */}
      {facts.length > 0 && (
        <div>
          <div className="text-xs text-gray-500 mb-1">Remembered facts:</div>
          <div className="space-y-1">
            {facts.map((fact) => (
              <div
                key={fact.key}
                className="flex items-center justify-between bg-surface rounded px-2 py-1 text-xs group"
              >
                <div className="truncate">
                  <span className="text-crab">{fact.key}</span>
                  <span className="text-gray-600 mx-1">=</span>
                  <span className="text-gray-300">{fact.value}</span>
                </div>
                <button
                  onClick={() => onRemoveFact(fact.key)}
                  className="text-gray-600 hover:text-red-400 ml-1 opacity-0 group-hover:opacity-100 transition-opacity"
                  title="Forget this fact"
                >
                  x
                </button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
