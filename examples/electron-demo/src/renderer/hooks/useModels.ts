import { useState, useEffect, useCallback } from 'react'
import type { CatalogModel, DownloadProgress } from '../types'

export function useModels() {
  const [models, setModels] = useState<CatalogModel[]>([])
  const [downloading, setDownloading] = useState<Record<string, DownloadProgress>>({})
  const [loading, setLoading] = useState(true)

  const refresh = useCallback(async () => {
    const catalog = await window.crabinfer.getModelCatalog()
    // Sort: downloaded first, then by size (smallest first)
    const sorted = [...catalog].sort((a, b) => {
      if (a.isDownloaded && !b.isDownloaded) return -1
      if (!a.isDownloaded && b.isDownloaded) return 1
      return a.sizeBytes - b.sizeBytes
    })
    setModels(sorted)
    setLoading(false)
  }, [])

  useEffect(() => {
    refresh()
    const unsub = window.crabinfer.onDownloadProgress((progress) => {
      setDownloading((prev) => ({ ...prev, [progress.catalogId]: progress }))
      if (progress.phase === 'complete') {
        setTimeout(() => {
          setDownloading((prev) => {
            const next = { ...prev }
            delete next[progress.catalogId]
            return next
          })
          refresh()
        }, 500)
      }
    })
    return unsub
  }, [refresh])

  const downloadModel = useCallback(
    async (catalogId: string) => {
      setDownloading((prev) => ({
        ...prev,
        [catalogId]: { catalogId, bytesDownloaded: 0, bytesTotal: 0, phase: 'starting' },
      }))
      await window.crabinfer.downloadModel(catalogId)
    },
    []
  )

  return { models, downloading, loading, downloadModel, refresh }
}
