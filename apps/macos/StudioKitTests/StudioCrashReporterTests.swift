@testable import StudioKit
import Foundation
import XCTest

@MainActor
final class StudioCrashReporterTests: XCTestCase {
    private var root: URL!
    private var defaults: UserDefaults!
    private var defaultsName: String!
    private var reporter: StudioCrashReporter!
    private var subscriptions: [StudioMetricSubscription] = []
    private var removals: [StudioMetricSubscription] = []

    override func setUp() async throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent("diagnostics-\(UUID())", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defaultsName = "diagnostics-tests-\(UUID())"
        defaults = try XCTUnwrap(UserDefaults(suiteName: defaultsName))
        reporter = StudioCrashReporter(directory: root, defaults: defaults, now: { Date(timeIntervalSince1970: 1_000) },
                                      subscribe: { self.subscriptions.append($0) }, unsubscribe: { self.removals.append($0) })
    }

    override func tearDown() async throws {
        reporter.setCapturing(false)
        reporter = nil
        defaults.removePersistentDomain(forName: defaultsName)
        defaults = nil
        subscriptions = []
        removals = []
        try FileManager.default.removeItem(at: root)
    }

    func testDisablingRejectsQueuedPayloadsBeforeWriting() async throws {
        reporter.setCapturing(true)
        let subscriber = try XCTUnwrap(subscriptions.last)
        let pending = subscriber.receiveEncoded([Data("queued".utf8)], kind: .metrics)
        reporter.setCapturing(false)
        await pending?.value
        XCTAssertFalse(reporter.isCapturing)
        XCTAssertEqual(reporter.storedPayloadCount, 0)
        XCTAssertTrue(try files().isEmpty)
    }

    func testRemovedSubscriberCannotWriteAfterReenablingCapture() async throws {
        reporter.setCapturing(true)
        let old = try XCTUnwrap(subscriptions.last)
        let queued = old.receiveEncoded([Data("queued old".utf8)], kind: .diagnostics)
        reporter.setCapturing(false)
        reporter.setCapturing(true)
        let current = try XCTUnwrap(subscriptions.last)
        let late = old.receiveEncoded([Data("late old".utf8)], kind: .metrics)
        await queued?.value
        await late?.value
        await current.receiveEncoded([Data("current".utf8)], kind: .diagnostics)?.value
        XCTAssertFalse(old === current)
        XCTAssertEqual(reporter.storedPayloadCount, 1)
        XCTAssertEqual(try contents(), ["current"])
    }

    func testDeleteRejectsQueuedReportsAndKeepsCaptureEnabled() async throws {
        reporter.setCapturing(true)
        let subscriber = try XCTUnwrap(subscriptions.last)
        await subscriber.receiveEncoded([Data("stored".utf8)], kind: .metrics)?.value
        XCTAssertEqual(reporter.storedPayloadCount, 1)
        let queued = subscriber.receiveEncoded([Data("queued before delete".utf8)], kind: .diagnostics)
        reporter.deleteStoredPayloads()
        await queued?.value
        XCTAssertTrue(reporter.isCapturing)
        XCTAssertEqual(reporter.storedPayloadCount, 0)
        XCTAssertTrue(try files().isEmpty)
        XCTAssertEqual(subscriptions.count, 1, "deletion must preserve the live native subscription")
        XCTAssertTrue(removals.isEmpty)

        await subscriber.receiveEncoded([Data("received after delete".utf8)], kind: .metrics)?.value
        XCTAssertEqual(reporter.storedPayloadCount, 1)
        XCTAssertEqual(try contents(), ["received after delete"])
    }

    func testEveryPayloadInSameTimestampBatchGetsItsOwnFile() async throws {
        reporter.setCapturing(true)
        let subscriber = try XCTUnwrap(subscriptions.last)
        await subscriber.receiveEncoded([Data("metric one".utf8), Data("metric two".utf8)], kind: .metrics)?.value
        await subscriber.receiveEncoded([Data("crash one".utf8), Data("crash two".utf8)], kind: .diagnostics)?.value
        XCTAssertEqual(reporter.storedPayloadCount, 4)
        XCTAssertEqual(try contents().sorted(), ["crash one", "crash two", "metric one", "metric two"])
    }

    func testDeletionWhileNativeCallbackIsEncodingRejectsThatDelivery() async throws {
        reporter.setCapturing(true)
        let subscriber = try XCTUnwrap(subscriptions.last)
        let (started, continuation) = AsyncStream<Void>.makeStream()
        let resumeEncoding = DispatchSemaphore(value: 0)
        let callback = Task.detached {
            subscriber.receive(kind: .diagnostics) {
                continuation.yield(())
                continuation.finish()
                resumeEncoding.wait()
                return [Data("encoding before delete".utf8)]
            }
        }
        for await _ in started { break }
        reporter.deleteStoredPayloads()
        resumeEncoding.signal()
        let pending = await callback.value
        await pending?.value
        XCTAssertEqual(reporter.storedPayloadCount, 0)
        XCTAssertTrue(try files().isEmpty)
        XCTAssertTrue(reporter.isCapturing)
        await subscriber.receiveEncoded([Data("after delete".utf8)], kind: .diagnostics)?.value
        XCTAssertEqual(try contents(), ["after delete"])
    }

    func testLateCallbackAfterRemovalCannotCreateReports() async throws {
        reporter.setCapturing(true)
        let subscriber = try XCTUnwrap(subscriptions.last)
        reporter.setCapturing(false)
        await subscriber.receiveEncoded([Data("late crash".utf8)], kind: .diagnostics)?.value
        XCTAssertEqual(removals.count, 1)
        XCTAssertTrue(removals.first === subscriber)
        XCTAssertTrue(try files().isEmpty)
        XCTAssertEqual(reporter.storedPayloadCount, 0)
    }

    func testPreferenceChangesSubscribeOnlyOnceAndPersistOptOut() {
        defaults.set(true, forKey: StudioCrashReporter.optInDefaultsKey)
        reporter.applyStoredPreference()
        reporter.applyStoredPreference()
        XCTAssertTrue(reporter.isCapturing)
        XCTAssertEqual(subscriptions.count, 1)
        reporter.setCapturing(false)
        reporter.applyStoredPreference()
        XCTAssertFalse(reporter.isCapturing)
        XCTAssertFalse(defaults.bool(forKey: StudioCrashReporter.optInDefaultsKey))
        XCTAssertEqual(removals.count, 1)
    }

    func testOptOutPreservesStoredReportsUntilExplicitDeletion() async throws {
        reporter.setCapturing(true)
        let subscriber = try XCTUnwrap(subscriptions.last)
        await subscriber.receiveEncoded([Data("retained report".utf8)], kind: .diagnostics)?.value
        reporter.setCapturing(false)
        XCTAssertEqual(reporter.storedPayloadCount, 1)
        XCTAssertEqual(try contents(), ["retained report"])
        try Data("leave this file".utf8).write(to: root.appendingPathComponent("notes.txt"))
        reporter.deleteStoredPayloads()
        XCTAssertEqual(reporter.storedPayloadCount, 0)
        XCTAssertTrue(try files().isEmpty)
        XCTAssertTrue(FileManager.default.fileExists(atPath: root.appendingPathComponent("notes.txt").path))
        XCTAssertFalse(reporter.isCapturing)
        XCTAssertEqual(subscriptions.count, 1)
    }

    private func files() throws -> [URL] {
        try FileManager.default.contentsOfDirectory(at: root, includingPropertiesForKeys: nil).filter { $0.pathExtension == "json" }
    }

    private func contents() throws -> [String] {
        try files().map { try String(contentsOf: $0, encoding: .utf8) }
    }
}
