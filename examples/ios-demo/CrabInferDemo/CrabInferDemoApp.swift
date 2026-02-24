import SwiftUI

@main
struct CrabInferDemoApp: App {
    init() {
        // Register background tasks
        BackgroundTaskManager.shared.registerTasks()
        BackgroundTaskManager.shared.requestNotificationPermission()
    }

    var body: some Scene {
        WindowGroup {
            TabView {
                ContentView()
                    .tabItem {
                        Label("Chat", systemImage: "bubble.left.and.bubble.right")
                    }

                AssistantView()
                    .tabItem {
                        Label("Assistant", systemImage: "brain")
                    }
            }
            .onContinueUserActivity(HandoffManager.conversationActivityType) { activity in
                // Handle Handoff from another device
                if let restored = HandoffManager.shared.restoreConversation(from: activity) {
                    // In a production app, this would pass the restored state
                    // to the AssistantViewModel via an environment object
                    print("Handoff: restored \(restored.messages.count) messages, \(restored.facts.count) facts")
                }
            }
        }
    }
}
