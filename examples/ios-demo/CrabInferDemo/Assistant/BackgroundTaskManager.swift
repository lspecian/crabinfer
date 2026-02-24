import Foundation
import BackgroundTasks
import UserNotifications
import CrabInfer

/// Manages background task registration and execution for the assistant.
///
/// Supports:
/// - Background model downloads (BGProcessingTask)
/// - Quick background queries (BGAppRefreshTask)
/// - Local notifications for completed tasks
final class BackgroundTaskManager {
    static let shared = BackgroundTaskManager()

    /// Background task identifiers
    static let downloadTaskId = "com.crabinfer.model-download"
    static let queryTaskId = "com.crabinfer.background-query"

    private init() {}

    // MARK: - Registration

    /// Register background task handlers. Call from application(_:didFinishLaunchingWithOptions:).
    func registerTasks() {
        BGTaskScheduler.shared.register(
            forTaskWithIdentifier: Self.downloadTaskId,
            using: nil
        ) { task in
            self.handleDownloadTask(task as! BGProcessingTask)
        }

        BGTaskScheduler.shared.register(
            forTaskWithIdentifier: Self.queryTaskId,
            using: nil
        ) { task in
            self.handleQueryTask(task as! BGAppRefreshTask)
        }
    }

    // MARK: - Scheduling

    /// Schedule a background model download.
    func scheduleDownload(catalogId: String) {
        let request = BGProcessingTaskRequest(identifier: Self.downloadTaskId)
        request.requiresNetworkConnectivity = true
        request.requiresExternalPower = false // Allow on battery

        // Store the catalog ID for the task handler
        UserDefaults.standard.set(catalogId, forKey: "bg_download_catalog_id")

        do {
            try BGTaskScheduler.shared.submit(request)
        } catch {
            print("Failed to schedule background download: \(error)")
        }
    }

    /// Schedule a background query (e.g., periodic check or proactive suggestion).
    func scheduleBackgroundQuery(prompt: String) {
        let request = BGAppRefreshTaskRequest(identifier: Self.queryTaskId)
        request.earliestBeginDate = Date(timeIntervalSinceNow: 60) // At least 1 min from now

        UserDefaults.standard.set(prompt, forKey: "bg_query_prompt")

        do {
            try BGTaskScheduler.shared.submit(request)
        } catch {
            print("Failed to schedule background query: \(error)")
        }
    }

    // MARK: - Task Handlers

    private func handleDownloadTask(_ task: BGProcessingTask) {
        task.expirationHandler = {
            // Clean up if task is about to expire
        }

        guard let catalogId = UserDefaults.standard.string(forKey: "bg_download_catalog_id") else {
            task.setTaskCompleted(success: false)
            return
        }

        Task {
            do {
                let downloads = CrabInfer.Downloads()
                guard let entry = CrabInfer.catalog().first(where: { $0.id == catalogId }) else {
                    task.setTaskCompleted(success: false)
                    return
                }

                try await downloads.download(entry)
                sendNotification(
                    title: "Model Downloaded",
                    body: "\(entry.name) is ready to use."
                )
                task.setTaskCompleted(success: true)
            } catch {
                sendNotification(
                    title: "Download Failed",
                    body: "Failed to download model: \(error.localizedDescription)"
                )
                task.setTaskCompleted(success: false)
            }
        }
    }

    private func handleQueryTask(_ task: BGAppRefreshTask) {
        task.expirationHandler = {}

        guard let prompt = UserDefaults.standard.string(forKey: "bg_query_prompt") else {
            task.setTaskCompleted(success: false)
            return
        }

        Task {
            do {
                let apiKey = CrabInfer.Credentials.getApiKey(provider: "openai") ?? ""
                guard !apiKey.isEmpty else {
                    task.setTaskCompleted(success: false)
                    return
                }

                let providerConfig = CrabInfer.cloudConfig(
                    provider: "openai",
                    apiKey: apiKey,
                    model: "gpt-4o"
                )

                let config = CrabInfer.AgentConfig(
                    maxToolRounds: 3,
                    maxTokens: 512,
                    temperature: 0.7
                )

                let agent = try CrabInfer.Agent(provider: providerConfig, config: config)
                let result = try await agent.run(prompt)

                let preview = result.text.count > 200
                    ? String(result.text.prefix(200)) + "..."
                    : result.text

                sendNotification(
                    title: "CrabInfer",
                    body: preview
                )

                task.setTaskCompleted(success: true)
            } catch {
                task.setTaskCompleted(success: false)
            }
        }
    }

    // MARK: - Notifications

    func requestNotificationPermission() {
        UNUserNotificationCenter.current().requestAuthorization(
            options: [.alert, .badge, .sound]
        ) { granted, error in
            if let error {
                print("Notification permission error: \(error)")
            }
        }
    }

    private func sendNotification(title: String, body: String) {
        let content = UNMutableNotificationContent()
        content.title = title
        content.body = body
        content.sound = .default

        let request = UNNotificationRequest(
            identifier: UUID().uuidString,
            content: content,
            trigger: nil // Deliver immediately
        )

        UNUserNotificationCenter.current().add(request)
    }
}
