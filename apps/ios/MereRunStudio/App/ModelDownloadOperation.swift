import Foundation

@MainActor
final class ModelDownloadOperation {
    enum State: Equatable {
        case notInstalled
        case downloading(String)
        case ready
    }

    private let store: PendingModelDownloadStore
    private var attempts: [String: UUID] = [:]

    init(store: PendingModelDownloadStore) { self.store = store }

    func run(
        _ pending: PendingModelDownload,
        persistRequest: Bool,
        install: @MainActor (PendingModelDownload, @escaping @Sendable (String) -> Void) async throws -> Void,
        retryable: @MainActor (Error) -> Bool,
        publish: @escaping @MainActor (State) -> Void,
        failed: @MainActor (Error) -> Void
    ) async {
        guard attempts[pending.modelID] == nil else { return }
        let ticket = UUID()
        attempts[pending.modelID] = ticket
        defer { attempts[pending.modelID] = nil }
        publish(.downloading("Starting…"))
        do {
            if persistRequest { try store.save(pending) }
            try await install(pending) { label in
                Task { @MainActor [weak self] in
                    guard self?.attempts[pending.modelID] == ticket else { return }
                    publish(.downloading(label))
                }
            }
            try store.remove(modelID: pending.modelID)
            publish(.ready)
        } catch {
            if !retryable(error) { try? store.remove(modelID: pending.modelID) }
            publish(.notInstalled)
            failed(error)
        }
    }
}
