import { useState, useEffect, useCallback } from 'react'
import type { AppSettings } from '../types'

const defaultSettings: AppSettings = {
  routingPolicy: 'LocalFirst',
  privacyMode: false,
  dataSovereignty: false,
  providers: {},
}

export function useSettings() {
  const [settings, setSettings] = useState<AppSettings>(defaultSettings)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    window.crabinfer.getSettings().then((s) => {
      setSettings(s)
      setLoading(false)
    })
  }, [])

  const updateSettings = useCallback(async (update: Partial<AppSettings>) => {
    setSettings((prev) => {
      const next = { ...prev, ...update }
      window.crabinfer.saveSettings(next)
      return next
    })
  }, [])

  return { settings, loading, updateSettings }
}
