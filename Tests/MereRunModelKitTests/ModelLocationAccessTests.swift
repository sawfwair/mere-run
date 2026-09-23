import Foundation
import XCTest
import MereRunModelKit

/// Stands in for a volume whose reads block until released, as one does while macOS
/// waits on a removable-volume consent prompt.
private final class StalledVolume: @unchecked Sendable {
    let root: URL
    private let release = DispatchSemaphore(value: 0)
    private let lock = NSLock()
    private var reads: [String] = []

    init(root: URL) {
        self.root = root.standardizedFileURL
    }

    var readCount: Int {
        lock.withLock { reads.count }
    }

    func read(_ url: URL) -> ModelLocationProblem? {
        lock.withLock { reads.append(url.path) }
        if url.standardizedFileURL.path.hasPrefix(root.path) {
            release.wait()
        }
        return nil
    }

    func unblock() {
        release.signal()
    }
}

final class ModelLocationAccessTests: XCTestCase {
    private let local = URL(fileURLWithPath: "/tmp/mere-run-local", isDirectory: true)
    private let stalledRoot = URL(fileURLWithPath: "/Volumes/Stalled/models", isDirectory: true)

    func testStalledLocationIsReportedAtTheDeadlineAndNotReadAgainWhileBlocked() {
        let volume = StalledVolume(root: stalledRoot)
        defer { volume.unblock() }
        let access = ModelLocationAccess(deadline: 0.2, read: volume.read)

        let started = Date()
        let issues = access.issues(for: [local, stalledRoot])
        XCTAssertEqual(issues, [ModelLocationIssue(path: stalledRoot.path, problem: .unresponsive)])
        XCTAssertLessThan(Date().timeIntervalSince(started), 2)

        let repeated = Date()
        XCTAssertEqual(access.issues(for: [local, stalledRoot]), issues)
        XCTAssertLessThan(Date().timeIntervalSince(repeated), 0.1)
        XCTAssertEqual(volume.readCount, 2)
    }

    func testLocationIsResponsiveOnceItsBlockedReadReturns() {
        let volume = StalledVolume(root: stalledRoot)
        let access = ModelLocationAccess(deadline: 0.2, read: volume.read)
        XCTAssertEqual(access.issues(for: [stalledRoot]).map(\.problem), [.unresponsive])

        volume.unblock()
        let giveUp = Date().addingTimeInterval(5)
        while !access.issues(for: [stalledRoot]).isEmpty, Date() < giveUp {
            Thread.sleep(forTimeInterval: 0.01)
        }
        XCTAssertTrue(access.issues(for: [stalledRoot]).isEmpty)
        XCTAssertEqual(volume.readCount, 1)
    }

    func testReadsRepeatAfterTheRecheckInterval() {
        let volume = StalledVolume(root: stalledRoot)
        let access = ModelLocationAccess(deadline: 1, recheckInterval: 0, read: volume.read)
        XCTAssertTrue(access.issues(for: [local]).isEmpty)
        XCTAssertTrue(access.issues(for: [local]).isEmpty)
        XCTAssertEqual(volume.readCount, 2)
    }

    func testIssuesCoverEveryConfiguredLocation() {
        let binding = URL(fileURLWithPath: "/Volumes/Stalled/binding", isDirectory: true)
        let search = URL(fileURLWithPath: "/Volumes/Other/models", isDirectory: true)
        let locations = ModelLocationSnapshot(
            primaryRoot: local,
            searchRoots: [search],
            bindings: [.init(modelID: "image-klein-nano", path: binding.path)]
        )
        XCTAssertEqual(locations.locationRoots.map(\.path), [local.path, binding.path, search.path])

        let volume = StalledVolume(root: URL(fileURLWithPath: "/Volumes/Stalled", isDirectory: true))
        defer { volume.unblock() }
        let access = ModelLocationAccess(deadline: 0.2) { url in
            url.path == search.path ? .denied : volume.read(url)
        }
        XCTAssertEqual(access.issues(in: locations), [
            ModelLocationIssue(path: binding.path, problem: .unresponsive),
            ModelLocationIssue(path: search.path, problem: .denied),
        ])
        XCTAssertEqual(access.unresponsivePaths(in: locations), [binding.path])
    }

    func testReadDirectoryReportsOnlyPermissionErrors() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let locked = root.appendingPathComponent("locked", isDirectory: true)
        try FileManager.default.createDirectory(at: locked, withIntermediateDirectories: true)
        defer {
            try? FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: locked.path)
            try? FileManager.default.removeItem(at: root)
        }
        try FileManager.default.setAttributes([.posixPermissions: 0o000], ofItemAtPath: locked.path)

        XCTAssertNil(ModelLocationAccess.readDirectory(root))
        XCTAssertNil(ModelLocationAccess.readDirectory(root.appendingPathComponent("missing")))
        XCTAssertEqual(ModelLocationAccess.readDirectory(locked), .denied)
    }

    func testResolverSkipsCandidatesUnderAStalledLocation() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let id = ManagedModelID.kleinNano
        let search = root.appendingPathComponent("search", isDirectory: true)
        let installed = search.appendingPathComponent(id.rawValue, isDirectory: true)
        try FileManager.default.createDirectory(at: installed, withIntermediateDirectories: true)
        try MereRunModelManifest(id: id.rawValue).write(to: installed)
        // A valid install that outranks the search root; only the stall keeps lookup out of it.
        let stalledBinding = root.appendingPathComponent("stalled-binding", isDirectory: true)
        try FileManager.default.createDirectory(at: stalledBinding, withIntermediateDirectories: true)
        try MereRunModelManifest(id: id.rawValue).write(to: stalledBinding)

        let volume = StalledVolume(root: stalledBinding)
        defer { volume.unblock() }
        let resolver = InstalledModelResolver(
            locations: ModelLocationSnapshot(
                primaryRoot: root.appendingPathComponent("models"),
                searchRoots: [search],
                bindings: [.init(modelID: id.rawValue, path: stalledBinding.path)]
            ),
            access: ModelLocationAccess(deadline: 0.2, read: volume.read)
        )
        var validated: [URL] = []
        let result = try resolver.resolve(id, descriptor: { InstalledModelDescriptor(id: $0) }) { _, url in
            validated.append(url)
            return true
        }
        XCTAssertEqual(result.candidate.rootURL, installed.standardizedFileURL)
        XCTAssertEqual(validated, [installed.standardizedFileURL])
    }
}
