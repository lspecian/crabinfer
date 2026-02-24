import { useState, useEffect } from 'react'
import type { DeviceInfo } from '../types'

export function useDevice() {
  const [device, setDevice] = useState<DeviceInfo | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    window.crabinfer.getDeviceInfo().then((info) => {
      setDevice(info)
      setLoading(false)
    })
  }, [])

  return { device, loading }
}
