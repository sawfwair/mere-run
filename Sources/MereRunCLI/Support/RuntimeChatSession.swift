import Foundation
import MereRunCore

protocol RuntimeChatModelLease: Sendable {
    func preparePrompt(_ request: ChatRequest) async throws -> ChatPreparedPrompt
    func chat(_ request: ChatRequest, progressHandler: (@Sendable (ChatProgress) -> Void)?) async throws -> ChatResponse
    func deepseekChatCompletionsURL(progressHandler: (@Sendable (ChatProgress) -> Void)?) async throws -> URL
    func release() async
}

extension RuntimeChatModelLease {
    func preparePrompt(_ request: ChatRequest) async throws -> ChatPreparedPrompt {
        throw MessagesFailure.unsupported("count_tokens")
    }
}

/// One admitted request and its resident model. HTTP response producers retain
/// this session until generation or proxy consumption ends, including errors.
final class RuntimeChatSession: @unchecked Sendable {
    private let originalRequest: ChatRequest
    private var preparedRequest: ChatRequest?
    private var preparation: Task<ChatPromptPreparation, Error>?
    var request: ChatRequest { lock.withLock { preparedRequest ?? originalRequest } }
    var appliedContextEdits: [ChatAppliedContextEdit] { lock.withLock { appliedEdits } }
    var originalInputTokens: Int? { lock.withLock { originalTokens } }
    private var originalTokens: Int?
    private var appliedEdits: [ChatAppliedContextEdit] = []
    let modelID: String
    let engine: RuntimeServingEngine
    let includeUsage: Bool
    /// The model scope check's warnings for the request, sent as response headers.
    let scope: APIModelScope
    var requestID: UUID { admission.requestID }

    private let model: any RuntimeChatModelLease
    private let admission: RuntimeRequestAdmissionLease
    private let lock = NSLock()
    private var completion: Task<Void, Never>?

    init(plan: RuntimeChatPlan, admission: RuntimeRequestAdmissionLease) {
        self.originalRequest = plan.request
        self.modelID = plan.modelID
        self.engine = plan.engine
        self.includeUsage = plan.includeUsage
        self.scope = plan.scope
        self.model = plan.lease
        self.admission = admission
    }

    deinit { _ = completionTask(cancelled: true) }

    func preparePrompt() async throws -> ChatPreparedPrompt {
        try Task.checkCancellation()
        let task = preparationTask()
        let result = try await withTaskCancellationHandler { try await task.value } onCancel: { task.cancel() }
        lock.withLock { preparedRequest = result.request; appliedEdits = result.applied; originalTokens = result.originalInputTokens }
        return result.prompt
    }

    private func preparationTask() -> Task<ChatPromptPreparation, Error> {
        lock.withLock {
            if let preparation { return preparation }
            let model = model
            let request = originalRequest
            let task = Task { try await ChatPromptPreparation.prepare(request) { try await model.preparePrompt($0) } }
            preparation = task
            return task
        }
    }

    func prepareContinuation(_ input: ChatRequest) async throws -> ChatRequest {
        var input = input
        input.preparedPrompt = nil
        let prepared = try await ChatPromptPreparation.prepare(input) { try await model.preparePrompt($0) }
        lock.withLock { appliedEdits += prepared.applied }
        return prepared.request
    }

    func chat(progressHandler: (@Sendable (ChatProgress) -> Void)? = nil) async throws -> ChatResponse {
        if !originalRequest.contextEdits.isEmpty { _ = try await preparePrompt() }
        let admission = admission
        let model = model
        let generate: (ChatRequest, (@Sendable (ChatProgress) -> Void)?) async throws -> ChatResponse = { request, handler in
            try await model.chat(request) { progress in
                admission.observe(progress)
                handler?(progress)
            }
        }
        if let search = request.toolSearch {
            return try await chatWithToolSearch(search, generate: generate, progressHandler: progressHandler)
        }
        return try await generate(request, progressHandler)
    }

    func deepseekChatCompletionsURL() async throws -> URL {
        try Task.checkCancellation()
        return try await model.deepseekChatCompletionsURL(progressHandler: nil)
    }

    func observeClientDisconnect() { admission.observeClientDisconnect() }

    func finish(cancelled: Bool = false) async {
        await completionTask(cancelled: cancelled || Task.isCancelled).value
    }

    private func completionTask(cancelled: Bool) -> Task<Void, Never> {
        lock.withLock {
            if let completion { return completion }
            let model = model
            let admission = admission
            let task = Task {
                await model.release()
                await admission.release(cancelled: cancelled)
            }
            completion = task
            return task
        }
    }
}
