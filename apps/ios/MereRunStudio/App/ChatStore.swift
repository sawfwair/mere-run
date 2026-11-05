import Combine
import Foundation
import MereRunCore
import MereRunRelayKit

/// A chat thread carried over stateless `text.generate` jobs: history lives
/// on the phone, each turn renders the transcript into one prompt and runs it
/// on the fleet. Rendering, budgeting, and think-tag stripping mirror the
/// macOS Studio's `ConversationTranscript` so both shells produce identical
/// prompts for identical threads (a shared home in RelayKit is a follow-up).
@MainActor
final class ChatStore: ObservableObject {
    typealias Message = ChatTurnStore.Message

    static let budgetChars = 48_000

    var messages: [Message] { turns.messages }
    var awaitingReply: Bool { turns.awaitingReply }
    var streamingReply: String? { turns.streamingReply }
    var errorMessage: String? { turns.errorMessage }
    var replyIsLocal: Bool { turns.activeOrigin.map { $0 == .local } ?? runLocally }
    @Published var model = ""
    @Published var runLocally = UserDefaults.standard.bool(forKey: "chat.runLocally") {
        didSet { UserDefaults.standard.set(runLocally, forKey: "chat.runLocally") }
    }

    private let store: RelayStore
    private let turns: ChatTurnStore
    private var turnUpdates: AnyCancellable?
    private var connectionUpdates: AnyCancellable?

    init(relay: RelayStore) {
        store = relay
        let url = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("MereRun", isDirectory: true).appendingPathComponent("chat-thread.json")
        turns = ChatTurnStore(threadURL: url)
        turnUpdates = turns.objectWillChange.sink { [weak self] in self?.objectWillChange.send() }
        connectionUpdates = relay.$connectionID.dropFirst().sink { [weak self] _ in self?.turns.relayConnectionChanged() }
    }

    func newChat() { turns.newChat() }

    func send(_ text: String) async {
        let local = runLocally
        let model = model
        let client = local ? nil : store.client
        await turns.send(text, origin: local ? .local : .remote, generate: { messages, onDelta in
            if local {
                let result = try await LocalEngine.shared.chat(messages: self.nativeMessages(messages), onDelta: onDelta)
                return ChatTurnStore.Reply(text: result.reply, thinking: result.thinking)
            }
            guard let client else { throw RelayClientError("Pair with a relay before chatting.") }
            var arguments: [String: WorkflowValue] = ["prompt": .string(self.renderTranscript(messages))]
            if !model.isEmpty { arguments["model"] = .string(model) }
            let job = try await self.store.submit(kind: "text.generate", arguments: arguments, using: client)
            return try await RemoteChatReply.observe(read: {
                try Task.checkCancellation()
                let reply = try await self.awaitReply(jobID: job.jobID, client: client, onDelta: onDelta)
                return ChatTurnStore.Reply(text: reply)
            }, cancelJob: {
                // A retired connection's credentials refuse cancellation of its old job.
                _ = try? await client.cancel(jobID: job.jobID)
            })
        }, presentError: AppErrorText.presentable)
    }

    private func awaitReply(
        jobID: String, client: RelayWorkflowExecutor, onDelta: @escaping @MainActor (String) -> Void
    ) async throws -> String {
        while true {
            try Task.checkCancellation()
            let job = try await client.inspect(jobID: jobID)
            try Task.checkCancellation()
            if job.state == .running || job.state == .assigned {
                // The worker streams node_output_delta events whose message is
                // the accumulated text so far; render it as it grows.
                if let raw = try? await client.events(jobID: jobID) {
                    try Task.checkCancellation()
                    let delta = RelayEventText.decodedEvents(raw)
                        .last(where: { $0.type == "node_output_delta" })?
                        .message
                    if let delta, !delta.isEmpty {
                        let visible = GeneratedTextFilters.strippingThinking(delta, streaming: true)
                        if !visible.isEmpty { onDelta(visible) }
                    }
                }
            }
            switch job.state {
            case .finished:
                let manifest = try await client.manifest(jobID: jobID)
                try Task.checkCancellation()
                guard let value = manifest.nodes
                    .flatMap(\.outputs)
                    .first(where: { $0.name == "text" })?
                    .value?.stringValue else {
                    throw RelayClientError("The reply did not include a text output.")
                }
                let cleaned = GeneratedTextFilters.strippingThinking(value)
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

    /// The thread as native chat messages for the on-device engine, budgeted
    /// the same way as the rendered transcript: oldest turns drop first.
    private func nativeMessages(_ messages: [Message]) -> [ChatMessage] {
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
        return included.map { message in
            ChatMessage(
                role: message.role == .user ? .user : .assistant,
                content: message.content
            )
        }
    }

    /// Oldest messages are dropped from what is sent (never from the thread)
    /// when the character budget is exceeded; a lone first turn renders
    /// verbatim so it is byte-identical to a single-shot run.
    private func renderTranscript(_ messages: [Message]) -> String {
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

}
