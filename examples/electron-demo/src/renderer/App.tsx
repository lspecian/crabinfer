import React, { useState, useEffect } from 'react'
import { useDevice } from './hooks/useDevice'
import { useModels } from './hooks/useModels'
import { useChat } from './hooks/useChat'
import { useSettings } from './hooks/useSettings'
import { useAgent } from './hooks/useAgent'
import { Sidebar } from './components/Sidebar'
import { ChatView } from './components/ChatView'
import { SettingsPanel } from './components/SettingsPanel'

export default function App() {
  const { device } = useDevice()
  const { models, downloading, downloadModel } = useModels()
  const { messages, isStreaming, modelLoaded, sendMessage, clearMessages, loadModel, unloadModel, stopInference } =
    useChat()
  const { settings, updateSettings } = useSettings()
  const { status: agentStatus, facts: agentFacts, removeFact } = useAgent()
  const [version, setVersion] = useState('0.1.0')
  const [showSettings, setShowSettings] = useState(false)

  useEffect(() => {
    window.crabinfer.getVersion().then(setVersion)
  }, [])

  return (
    <div className="flex h-screen bg-surface">
      <Sidebar
        device={device}
        models={models}
        downloading={downloading}
        modelLoaded={modelLoaded}
        version={version}
        agentStatus={agentStatus}
        agentFacts={agentFacts}
        onDownload={downloadModel}
        onLoad={loadModel}
        onUnload={unloadModel}
        onClearChat={clearMessages}
        onOpenSettings={() => setShowSettings(true)}
        onRemoveFact={removeFact}
      />
      <main className="flex-1 flex flex-col">
        <ChatView
          messages={messages}
          isStreaming={isStreaming}
          modelLoaded={modelLoaded}
          onSendMessage={sendMessage}
          onStop={stopInference}
        />
      </main>
      {showSettings && (
        <SettingsPanel
          settings={settings}
          onUpdate={updateSettings}
          onClose={() => setShowSettings(false)}
        />
      )}
    </div>
  )
}
