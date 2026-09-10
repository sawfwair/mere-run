import Foundation
import MereRunCore

protocol RuntimeChatModelLease: Sendable {
    func chat(_ request: ChatRequest, progressHandler: (@Sendable (ChatProgress) -> Void)?) async throws -> ChatResponse
    func deepseekChatCompletionsURL(progressHandler: (@Sendable (ChatProgress) -> Void)?) async throws -> URL
    func release() async
}

/// One admitted request and its resident model. HTTP response producers retain
/// this session until generation or proxy consumption ends, including errors.
final class RuntimeChatSession: @unchecked Sendable {
    let request: ChatRequest
    let modelID: String
    let engine: RuntimeServingEngine
    let includeUsage: Bool
    var requestID: UUID { admission.requestID }

    private let model: any RuntimeChatModelLease
    private let admission: RuntimeRequestAdmissionLease
    private let lock = NSLock()
    private var completion: Task<Void, Never>?

    init(plan: RuntimeChatPlan, admission: RuntimeRequestAdmissionLease) {
        self.request = plan.request
        self.modelID = plan.modelID
        self.engine = plan.engine
        self.includeUsage = plan.includeUsage
        self.model = plan.lease
        self.admission = admission
    }

    deinit { _ = completionTask(cancelled: true) }

    func chat(progressHandler: (@Sendable (ChatProgress) -> Void)? = nil) async throws -> ChatResponse {
        let admission = admission
        return try await model.chat(request) { progress in
            admission.observe(progress)
            progressHandler?(progress)
        }
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
