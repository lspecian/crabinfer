import React from 'react'
import type { DeviceInfo } from '../types'

interface Props {
  device: DeviceInfo | null
}

export function DeviceInfoPanel({ device }: Props) {
  if (!device) return null

  const memGB = (device.totalMemoryBytes / 1e9).toFixed(0)
  const availGB = (device.availableMemoryBytes / 1e9).toFixed(0)

  return (
    <div className="bg-surface-light rounded-lg p-4 space-y-2">
      <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider">Device</h3>
      <div className="text-lg font-bold text-white">{device.chipName}</div>
      <div className="grid grid-cols-2 gap-2 text-sm">
        <div>
          <span className="text-gray-400">RAM:</span>{' '}
          <span className="text-white">{memGB} GB</span>
          <span className="text-gray-500"> ({availGB} GB free)</span>
        </div>
        <div>
          <span className="text-gray-400">Metal:</span>{' '}
          <span className={device.hasMetalGpu ? 'text-green-400' : 'text-red-400'}>
            {device.hasMetalGpu ? 'Available' : 'Not available'}
          </span>
        </div>
        <div>
          <span className="text-gray-400">Quant:</span>{' '}
          <span className="text-white">{device.recommendedQuant}</span>
        </div>
        <div>
          <span className="text-gray-400">Max model:</span>{' '}
          <span className="text-white">{device.maxModelSizeB}B params</span>
        </div>
      </div>
    </div>
  )
}
