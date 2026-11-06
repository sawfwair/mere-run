import Foundation
import XCTest

@MainActor
final class ModelDownloadOperationTests: XCTestCase {
    private var defaults: UserDefaults!
    private var suite: String!
    private var store: PendingModelDownloadStore!

    override func setUp() async throws {
        suite = "ModelDownloadOperationTests.\(UUID())"
        defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        store = PendingModelDownloadStore(defaults: defaults)
    }

    override func tearDown() async throws { defaults.removePersistentDomain(forName: suite) }

    func testInterruptionPreservesDurableIntentForRelaunch() async {
        let download = pending()
        let operation = ModelDownloadOperation(store: store)
        var state: ModelDownloadOperation.State?
        await operation.run(download, persistRequest: true, install: { _, _ in throw URLError(.networkConnectionLost) },
                            retryable: { _ in true }, publish: { state = $0 }, failed: { _ in })
        XCTAssertEqual(store.all(), [download])
        XCTAssertEqual(state, .notInstalled)
    }

    func testPermanentFailureClearsIntent() async {
        let operation = ModelDownloadOperation(store: store)
        await operation.run(pending(), persistRequest: true, install: { _, _ in throw URLError(.serverCertificateUntrusted) },
                            retryable: { _ in false }, publish: { _ in }, failed: { _ in })
        XCTAssertTrue(store.all().isEmpty)
    }

    func testProgressArrivingAfterSuccessCannotRestoreDownloadingState() async {
        let operation = ModelDownloadOperation(store: store)
        var progress: (@Sendable (String) -> Void)?
        var state: ModelDownloadOperation.State?
        await operation.run(pending(), persistRequest: true, install: { _, callback in progress = callback },
                            retryable: { _ in false }, publish: { state = $0 }, failed: { _ in XCTFail("install succeeds") })
        progress?("late")
        await settle()
        XCTAssertEqual(state, .ready)
        XCTAssertTrue(store.all().isEmpty)
    }

    func testOldProgressCannotOverwriteTheNextAttempt() async {
        let operation = ModelDownloadOperation(store: store)
        var oldProgress: (@Sendable (String) -> Void)?
        var state: ModelDownloadOperation.State?
        await operation.run(pending(), persistRequest: true, install: { _, callback in
            oldProgress = callback
            throw URLError(.networkConnectionLost)
        }, retryable: { _ in true }, publish: { state = $0 }, failed: { _ in })
        let gate = DownloadInstallGate()
        let retry = Task {
            await operation.run(pending(), persistRequest: false, install: { _, callback in
                callback("current")
                await gate.pause()
            }, retryable: { _ in true }, publish: { state = $0 }, failed: { _ in XCTFail("retry succeeds") })
        }
        await gate.waitUntilStarted()
        await settle()
        oldProgress?("old")
        await settle()
        XCTAssertEqual(state, .downloading("current"))
        gate.resume()
        await retry.value
        XCTAssertEqual(state, .ready)
    }

    private func pending() -> PendingModelDownload {
        PendingModelDownload(modelID: "image-klein-nano", usageTermsAcknowledged: true, requestedAt: Date(timeIntervalSince1970: 1))
    }

    private func settle() async { for _ in 0..<10 { await Task.yield() } }
}

@MainActor
private final class DownloadInstallGate {
    private var started = false
    private var startWaiter: CheckedContinuation<Void, Never>?
    private var waiter: CheckedContinuation<Void, Never>?
    func pause() async {
        started = true
        startWaiter?.resume()
        startWaiter = nil
        await withCheckedContinuation { waiter = $0 }
    }
    func waitUntilStarted() async {
        if started { return }
        await withCheckedContinuation { startWaiter = $0 }
    }
    func resume() { waiter?.resume(); waiter = nil }
}
