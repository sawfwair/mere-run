import Foundation
import MereRunRelayKit

/// A chat thread carried over stateless `text.generate` jobs: history lives
/// on the phone, each turn renders the transcript into one prompt and runs it
/// on the fleet. Rendering, budgeting, and think-tag stripping mirror the
/// macOS Studio's `ConversationTranscript` so both shells produce identical
/// prompts for identical threads (a shared home in RelayKit is a follow-up).
@MainActor
final class ChatStore: ObservableObject {
    struct Message: Identifiable, Codable, Equatable {
        enum Role: String, Codable {
            case user
            case assistant
        }

        let id: UUID
        let role: Role
        var content: String
        var failed: Bool

        init(role: Role, content: String, failed: Bool = false) {
            self.id = UUID()
            self.role = role
            self.content = content
            self.failed = failed
        }
    }

    static let budgetChars = 48_000

    @Published private(set) var messages: [Message] = []
    @Published private(set) var awaitingReply = false
    @Published var model = ""
    @Published private(set) var errorMessage: String?

    private let store: RelayStore
    private let threadURL: URL

    init(relay: RelayStore) {
        store = relay
        threadURL = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("MereRun", isDirectory: true)
            .appendingPathComponent("chat-thread.json")
        if let data = try? Data(contentsOf: threadURL),
           let saved = try? JSONDecoder().decode([Message].self, from: data) {
            messages = saved
        }
    }

    func newChat() {
        messages = []
        errorMessage = nil
        persist()
    }

    func send(_ text: String) async {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, !awaitingReply else { return }
        messages.append(Message(role: .user, content: trimmed))
        persist()
        awaitingReply = true
        errorMessage = nil
        defer { awaitingReply = false }
        do {
            let prompt = renderTranscript()
            var arguments: [String: WorkflowValue] = ["prompt": .string(prompt)]
            if !model.isEmpty {
                arguments["model"] = .string(model)
            }
            let job = try await store.submit(kind: "text.generate", arguments: arguments)
            let reply = try await awaitReply(jobID: job.jobID)
            messages.append(Message(role: .assistant, content: reply))
        } catch let error as RelayClientError {
            errorMessage = error.message
            messages.append(Message(role: .assistant, content: error.message, failed: true))
        } catch {
            errorMessage = error.localizedDescription
            messages.append(Message(role: .assistant, content: error.localizedDescription, failed: true))
        }
        persist()
    }

    private func awaitReply(jobID: String) async throws -> String {
        guard let client = store.client else {
            throw RelayClientError("Pair with a relay before chatting.")
        }
        while true {
            let job = try await client.inspect(jobID: jobID)
            switch job.state {
            case .finished:
                let manifest = try await client.manifest(jobID: jobID)
                guard let value = manifest.nodes
                    .flatMap(\.outputs)
                    .first(where: { $0.name == "text" })?
                    .value?.stringValue else {
                    throw RelayClientError("The reply did not include a text output.")
                }
                let cleaned = Self.stripThinkTags(value)
                guard !cleaned.isEmpty else {
                    throw RelayClientError("The model returned an empty reply.")
                }
                return cleaned
            case .failed:
                throw RelayClientError(job.error ?? "The chat run failed on the node.")
            case .cancelled:
                throw RelayClientError("The chat run was cancelled.")
            default:
                try await Task.sleep(for: .seconds(1))
            }
        }
    }

    /// Oldest messages are dropped from what is sent (never from the thread)
    /// when the character budget is exceeded; a lone first turn renders
    /// verbatim so it is byte-identical to a single-shot run.
    private func renderTranscript() -> String {
        let usable = messages.filter { !$0.failed }
        var included: [Message] = []
        var used = 0
        for message in usable.reversed() {
            let cost = message.content.count + 12
            if included.isEmpty || used + cost <= Self.budgetChars {
                included.append(message)
                used += cost
            } else {
                break
            }
        }
        included.reverse()
        if included.count == 1, let only = included.first, only.role == .user {
            return only.content
        }
        return included.map { message in
            switch message.role {
            case .user: "User: \(message.content)"
            case .assistant: "Assistant: \(message.content)"
            }
        }.joined(separator: "\n\n")
    }

    static func stripThinkTags(_ text: String) -> String {
        var result = text.replacingOccurrences(
            of: "<think>[\\s\\S]*?</think>",
            with: "",
            options: .regularExpression
        )
        if !result.contains("<think>"), let close = result.range(of: "</think>") {
            result = String(result[close.upperBound...])
        }
        return result.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func persist() {
        try? JSONEncoder().encode(messages).write(to: threadURL, options: .atomic)
    }
}
