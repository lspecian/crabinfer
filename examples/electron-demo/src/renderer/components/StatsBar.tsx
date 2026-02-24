import React from 'react'
import type { ResponseStats } from '../types'

interface Props {
  stats: ResponseStats
}

export function StatsBar({ stats }: Props) {
  const memMB = (stats.peakMemoryBytes / 1e6).toFixed(0)

  return (
    <div className="flex items-center gap-3 text-xs text-gray-500 mt-1 px-1">
      <span>{stats.tokensPerSecond.toFixed(1)} tok/s</span>
      <span className="text-gray-700">|</span>
      <span>TTFT {stats.timeToFirstTokenMs.toFixed(0)}ms</span>
      <span className="text-gray-700">|</span>
      <span>{(stats.totalTimeMs / 1000).toFixed(1)}s total</span>
      <span className="text-gray-700">|</span>
      <span>{memMB} MB peak</span>
      <span className="text-gray-700">|</span>
      <span className="text-crab">{stats.backend}</span>
    </div>
  )
}
