import Combine
import Foundation

/// Owns the phone's persisted conversation and the reply currently being observed.
@MainActor
final class ChatTurnStore: @preconcurrency ObservableObject {
    struct Message: Identifiable, Codable, Equatable, Sendable {
        enum Role: String, Codable, Sendable {
            case user
            case assistant
        }

        let id: UUID
        let role: Role
        var content: String
        var failed: Bool
        /// The model's reasoning for this reply, kept behind a disclosure.
        var thinking: String?

        init(role: Role, content: String, failed: Bool = false, thinking: String? = nil) {
            self.id = UUID()
            self.role = role
            self.content = content
            self.failed = failed
            self.thinking = thinking
        }

        enum CodingKeys: String, CodingKey {
            case id
            case role
            case content
            case failed
            case thinking
        }

        init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            id = try container.decode(UUID.self, forKey: .id)
            role = try container.decode(Role.self, forKey: .role)
            content = try container.decode(String.self, forKey: .content)
            failed = try container.decodeIfPresent(Bool.self, forKey: .failed) ?? false
            thinking = try container.decodeIfPresent(String.self, forKey: .thinking)
        }
    }

    enum Origin: Equatable, Sendable { case local, remote }

    struct Reply: Sendable {
        let text: String
        let thinking: String?
        init(text: String, thinking: String? = nil) { self.text = text; self.thinking = thinking }
    }

    let objectWillChange = ObservableObjectPublisher()

    private struct State {
        var messages: [Message] = []
        var streamingReply: String?
        var errorMessage: String?
    }

    private struct Work {
        let id: UUID
        let origin: Origin
        let task: Task<Reply, Error>
    }

    private var state = State()
    private var work: Work?
    /// An on-device turn retired by New Chat may still hold the engine after observation ends.
    /// The next on-device turn starts generating only once it settles.
    private var retiredLocalTask: Task<Reply, Error>?
    private let threadURL: URL

    var messages: [Message] { state.messages }
    var awaitingReply: Bool { work != nil }
    var streamingReply: String? { state.streamingReply }
    var errorMessage: String? { state.errorMessage }
    var activeOrigin: Origin? { work?.origin }

    init(threadURL: URL) {
        self.threadURL = threadURL
        if let data = try? Data(contentsOf: threadURL), let saved = try? JSONDecoder().decode([Message].self, from: data) {
            state.messages = saved
        }
    }

    func newChat() {
        if let work {
            work.task.cancel()
            if work.origin == .local { retiredLocalTask = work.task }
        }
        work = nil
        state = State()
        publish()
    }

    /// Changing a relay only invalidates remote observation; an on-device reply keeps its owner.
    func relayConnectionChanged() {
        guard work?.origin == .remote else { return }
        work?.task.cancel()
        work = nil
        state.streamingReply = nil
        let message = "The relay connection changed. Send the message again."
        state.errorMessage = message
        state.messages.append(Message(role: .assistant, content: message, failed: true))
        publish()
    }

    func send(
        _ text: String,
        origin: Origin,
        generate: @escaping @MainActor ([Message], @escaping @MainActor (String) -> Void) async throws -> Reply,
        presentError: @MainActor (Error) -> String = { $0.localizedDescription }
    ) async {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, work == nil, !Task.isCancelled else { return }
        let ticket = UUID()
        let messages = state.messages + [Message(role: .user, content: trimmed)]
        let previous = origin == .local ? retiredLocalTask : nil
        if origin == .local { retiredLocalTask = nil }
        let task = Task { [weak self] in
            _ = await previous?.result
            try Task.checkCancellation()
            return try await generate(messages) { [weak self] partial in
                guard let self, self.work?.id == ticket, self.work?.task.isCancelled == false else { return }
                self.state.streamingReply = partial
                self.objectWillChange.send()
            }
        }
        work = Work(id: ticket, origin: origin, task: task)
        state.messages = messages
        state.streamingReply = nil
        state.errorMessage = nil
        publish()
        let result = await withTaskCancellationHandler {
            await task.result
        } onCancel: {
            task.cancel()
        }
        guard work?.id == ticket else { return }
        work = nil
        state.streamingReply = nil
        let settled: Result<Reply, Error> = task.isCancelled ? .failure(CancellationError()) : result
        switch settled {
        case .success(let reply):
            state.messages.append(Message(role: .assistant, content: reply.text, thinking: reply.thinking))
        case .failure(let error):
            let message = error is CancellationError ? "The chat request was cancelled." : presentError(error)
            state.errorMessage = message
            state.messages.append(Message(role: .assistant, content: message, failed: true))
        }
        publish()
    }

    /// Update the entire state and its durable transcript before notifying synchronous observers.
    private func publish() {
        try? JSONEncoder().encode(state.messages).write(to: threadURL, options: .atomic)
        objectWillChange.send()
    }
}
