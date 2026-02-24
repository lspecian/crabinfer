import React, { useState } from 'react'
import type { AgentToolExecution } from '../types'

interface Props {
  executions: AgentToolExecution[]
}

export function ToolExecutionLog({ executions }: Props) {
  const [expanded, setExpanded] = useState<number | null>(null)

  if (executions.length === 0) return null

  return (
    <div className="mt-2 space-y-1">
      {executions.map((exec, i) => (
        <div key={i} className="text-xs">
          <button
            onClick={() => setExpanded(expanded === i ? null : i)}
            className="flex items-center gap-1.5 w-full text-left hover:bg-surface-lighter rounded px-1.5 py-0.5 transition-colors"
          >
            <span className={exec.isError ? 'text-red-400' : 'text-green-400'}>
              {exec.isError ? 'ERR' : 'OK'}
            </span>
            <span className="text-crab font-mono">{exec.toolName}</span>
            <span className="text-gray-600 ml-auto">
              {expanded === i ? 'v' : '>'}
            </span>
          </button>
          {expanded === i && (
            <div className="ml-4 mt-1 space-y-1">
              <div className="bg-surface rounded p-2 font-mono text-[10px] text-gray-400 overflow-x-auto max-h-32 overflow-y-auto">
                <div className="text-gray-500 mb-1">args:</div>
                <pre className="whitespace-pre-wrap break-all">{exec.arguments}</pre>
              </div>
              <div className="bg-surface rounded p-2 font-mono text-[10px] text-gray-400 overflow-x-auto max-h-32 overflow-y-auto">
                <div className="text-gray-500 mb-1">output:</div>
                <pre className="whitespace-pre-wrap break-all">
                  {exec.output.length > 500 ? exec.output.slice(0, 500) + '...' : exec.output}
                </pre>
              </div>
            </div>
          )}
        </div>
      ))}
    </div>
  )
}
