import Foundation
import XCTest
@testable import MereRunCLI
@testable import MereRunCore

final class YuE2JobQueueTests: XCTestCase {
    private actor Barrier {
        var entered = false
        var observer: CheckedContinuation<Void, Never>?
        var release: CheckedContinuation<Void, Never>?
        func execute() async {
            entered = true
            observer?.resume()
            observer = nil
            await withCheckedContinuation { release = $0 }
        }
        func waitUntilEntered() async {
            if !entered { await withCheckedContinuation { observer = $0 } }
        }
        func finish() { release?.resume(); release = nil }
    }

    func testQueueCapacityAndQueuedCancellationDoNotRunTheCancelledJob() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: root) }
        let barrier = Barrier()
        let queue = YuE2JobQueue(root: root, maximumPending: 2) { _, _ in await barrier.execute(); return false }
        let plan = try YuE2GenerationPlan(style: "piano", lyrics: "", planning: .off)
        let first = try await queue.submit(plan)
        await barrier.waitUntilEntered()
        let second = try await queue.submit(plan)
        do { _ = try await queue.submit(plan); XCTFail("Queue must reject overflow") } catch {}
        let cancelled = try await queue.cancel(second.id)
        XCTAssertEqual(cancelled.status, .cancelled)
        await barrier.finish()
        await queue.wait(first.id)
        await queue.wait(second.id)
        let complete = try await queue.get(first.id)
        XCTAssertEqual(complete.status, .succeeded)
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent(second.id.uuidString).path))
    }

    func testServerRequiresAuthenticationForNonLoopbackBinding() throws {
        var command = try YuE2Serve.parse(["--output", "/tmp/yue2-server"])
        command.host = "0.0.0.0"
        XCTAssertThrowsError(try command.validate())
        command.apiKey = "test-secret"
        XCTAssertNoThrow(try command.validate())
    }
}
