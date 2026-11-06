import Combine
import Foundation

/// Publishes only the latest refresh for the connection that requested it.
@MainActor
final class RelaySnapshotStore<Value>: @preconcurrency ObservableObject {
    let objectWillChange = ObservableObjectPublisher()

    private struct Snapshot {
        let connectionID: UUID
        var value: Value?
        var errorMessage: String?
    }

    private var snapshot: Snapshot?
    private var requestID: UUID?

    func value(for connectionID: UUID) -> Value? {
        snapshot?.connectionID == connectionID ? snapshot?.value : nil
    }

    func errorMessage(for connectionID: UUID) -> String? {
        snapshot?.connectionID == connectionID ? snapshot?.errorMessage : nil
    }

    func reset() {
        requestID = nil
        snapshot = nil
        objectWillChange.send()
    }

    func load(
        connectionID: UUID,
        read: @MainActor () async throws -> Value,
        isCurrentConnection: @MainActor () -> Bool,
        presentError: @MainActor (Error) -> String = { $0.localizedDescription }
    ) async {
        guard !Task.isCancelled, isCurrentConnection() else { return }
        let ticket = UUID()
        requestID = ticket
        defer {
            if requestID == ticket { requestID = nil }
        }
        if snapshot?.connectionID != connectionID {
            snapshot = Snapshot(connectionID: connectionID)
            objectWillChange.send()
        }
        guard requestID == ticket, !Task.isCancelled, isCurrentConnection() else { return }
        do {
            let value = try await read()
            guard requestID == ticket, !Task.isCancelled, isCurrentConnection() else { return }
            snapshot = Snapshot(connectionID: connectionID, value: value)
        } catch {
            guard requestID == ticket, !Task.isCancelled, isCurrentConnection() else { return }
            snapshot?.errorMessage = presentError(error)
        }
        objectWillChange.send()
    }
}
