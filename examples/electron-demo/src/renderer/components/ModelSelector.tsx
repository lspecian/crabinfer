import React, { useState } from 'react'
import type { CatalogModel, DownloadProgress } from '../types'

interface Props {
  models: CatalogModel[]
  downloading: Record<string, DownloadProgress>
  modelLoaded: boolean
  onDownload: (catalogId: string) => void
  onLoad: (modelPath: string) => void
  onUnload: () => void
}

export function ModelSelector({
  models,
  downloading,
  modelLoaded,
  onDownload,
  onLoad,
  onUnload,
}: Props) {
  const [loadingModelId, setLoadingModelId] = useState<string | null>(null)
  const [loadedModelId, setLoadedModelId] = useState<string | null>(null)

  const handleModelAction = async (model: CatalogModel) => {
    if (loadedModelId === model.id) {
      // Already loaded — unload
      onUnload()
      setLoadedModelId(null)
      return
    }
    if (model.isDownloaded) {
      setLoadingModelId(model.id)
      try {
        const path = await window.crabinfer.getModelPath(model.id)
        if (path) {
          await onLoad(path)
          setLoadedModelId(model.id)
        }
      } catch (err) {
        console.error('Failed to load model:', err)
      } finally {
        setLoadingModelId(null)
      }
    } else {
      onDownload(model.id)
    }
  }

  const handleUnload = () => {
    onUnload()
    setLoadedModelId(null)
  }

  return (
    <div className="space-y-2">
      <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider">Models</h3>
      {modelLoaded && (
        <button
          onClick={handleUnload}
          className="w-full text-left text-xs bg-red-900/30 hover:bg-red-900/50 text-red-300 rounded-lg px-3 py-2 transition-colors"
        >
          Unload current model
        </button>
      )}
      <div className="space-y-1">
        {models.map((model) => {
          const dl = downloading[model.id]
          const isDownloading = !!dl && dl.phase !== 'complete'
          const isLoading = loadingModelId === model.id
          const isLoaded = modelLoaded && loadedModelId === model.id
          const progress =
            isDownloading && dl.bytesTotal > 0
              ? Math.round((dl.bytesDownloaded / dl.bytesTotal) * 100)
              : 0
          const sizeGB = (model.sizeBytes / 1e9).toFixed(1)

          return (
            <button
              key={model.id}
              onClick={() => handleModelAction(model)}
              disabled={isDownloading || isLoading}
              className={`w-full text-left rounded-lg px-3 py-2 transition-colors disabled:opacity-50 ${
                isLoaded
                  ? 'bg-green-900/30 border border-green-700/50'
                  : 'bg-surface-lighter hover:bg-surface-light/80'
              }`}
            >
              <div className="text-sm font-medium text-white truncate">{model.name}</div>
              <div className="text-xs text-gray-400 flex items-center gap-2">
                <span>{sizeGB} GB</span>
                <span className="text-gray-600">|</span>
                <span>{model.quant}</span>
                {isLoaded && (
                  <span className="text-green-400 ml-auto font-medium">Loaded</span>
                )}
                {isLoading && (
                  <span className="text-yellow-400 ml-auto animate-pulse">Loading...</span>
                )}
                {!isLoaded && !isLoading && model.isDownloaded && !isDownloading && (
                  <span className="text-gray-500 ml-auto">Click to load</span>
                )}
                {isDownloading && (
                  <span className="text-crab ml-auto">{progress}% {dl.phase}</span>
                )}
                {!model.isDownloaded && !isDownloading && (
                  <span className="text-gray-500 ml-auto">Download</span>
                )}
              </div>
              {isDownloading && (
                <div className="mt-1 h-1 bg-surface rounded-full overflow-hidden">
                  <div
                    className="h-full bg-crab rounded-full transition-all"
                    style={{ width: `${progress}%` }}
                  />
                </div>
              )}
              {isLoading && (
                <div className="mt-1 h-1 bg-surface rounded-full overflow-hidden">
                  <div className="h-full bg-yellow-500 rounded-full animate-pulse w-full" />
                </div>
              )}
            </button>
          )
        })}
      </div>
    </div>
  )
}
