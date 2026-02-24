import Foundation
#if canImport(UIKit)
import UIKit
#endif

/// Manages Handoff between devices (Mac <-> iPhone/iPad).
///
/// Uses NSUserActivity to allow continuing a conversation
/// from one device to another.
final class HandoffManager {
    static let shared = HandoffManager()

    /// Activity type for assistant conversations.
    static let conversationActivityType = "com.crabinfer.assistant.conversation"

    private init() {}

    /// Create a user activity for the current conversation state.
    ///
    /// Call this whenever the conversation changes to keep Handoff up to date.
    func createConversationActivity(
        messages: [AssistantMessage],
        facts: [(key: String, value: String)]
    ) -> NSUserActivity {
        let activity = NSUserActivity(activityType: Self.conversationActivityType)
        activity.title = "CrabInfer Conversation"
        activity.isEligibleForHandoff = true
        activity.isEligibleForSearch = true

        // Encode conversation state into userInfo
        var userInfo: [String: Any] = [:]

        // Encode messages (last 20 to keep payload small)
        let recentMessages = messages.suffix(20)
        let messageData = recentMessages.map { msg -> [String: Any] in
            [
                "role": msg.role.rawValue,
                "content": msg.content,
                "timestamp": msg.timestamp.timeIntervalSince1970,
            ]
        }
        userInfo["messages"] = messageData

        // Encode facts
        let factData = facts.map { ["key": $0.key, "value": $0.value] }
        userInfo["facts"] = factData

        activity.userInfo = userInfo
        activity.needsSave = true

        return activity
    }

    /// Restore conversation state from a received user activity.
    ///
    /// Returns the messages and facts from the Handoff activity.
    func restoreConversation(
        from activity: NSUserActivity
    ) -> (messages: [AssistantMessage], facts: [(key: String, value: String)])? {
        guard activity.activityType == Self.conversationActivityType else { return nil }
        guard let userInfo = activity.userInfo else { return nil }

        var messages: [AssistantMessage] = []
        var facts: [(key: String, value: String)] = []

        // Decode messages
        if let messageData = userInfo["messages"] as? [[String: Any]] {
            for data in messageData {
                guard let roleStr = data["role"] as? String,
                      let content = data["content"] as? String,
                      let role = AssistantMessage.MessageRole(rawValue: roleStr)
                else { continue }

                messages.append(AssistantMessage(role: role, content: content))
            }
        }

        // Decode facts
        if let factData = userInfo["facts"] as? [[String: String]] {
            for data in factData {
                if let key = data["key"], let value = data["value"] {
                    facts.append((key: key, value: value))
                }
            }
        }

        return (messages, facts)
    }
}
