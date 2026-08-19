import XCTest
@testable import MereRunCore

final class ModelStorageManagerTests: XCTestCase {
    private let fileManager = FileManager.default
    private var temporaryRoot: URL!
    private var applicationSupport: URL!
    private var models: URL!
    private var hub: URL!

    override func setUpWithError() throws {
        temporaryRoot = fileManager.temporaryDirectory
            .appendingPathComponent("mere-run-storage-tests-\(UUID().uuidString)", isDirectory: true)
        applicationSupport = temporaryRoot.appendingPathComponent("MereRun", isDirectory: true)
        models = applicationSupport.appendingPathComponent("models", isDirectory: true)
        hub = applicationSupport.appendingPathComponent("hub", isDirectory: true)
        try fileManager.createDirectory(at: models, withIntermediateDirectories: true)
        try fileManager.createDirectory(at: hub, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        if let temporaryRoot, fileManager.fileExists(atPath: temporaryRoot.path) {
            try fileManager.removeItem(at: temporaryRoot)
        }
    }

    func testExclusiveLegacyPayloadIsReclaimableAfterRemovingInstallLink() throws {
        let unit = legacyUnit(repository: "exclusive")
        let payload = try writePayload(at: unit.appendingPathComponent("model.bin"), bytes: 128)
        try linkModel("model-a", to: payload)
        let manager = try makeManager()

        let usage = try XCTUnwrap(manager.report(modelIDs: ["model-a"]).models.first)
        XCTAssertTrue(usage.installed)
        XCTAssertEqual(usage.referencedBytes, 128)
        XCTAssertEqual(usage.localBytes, 0)
        XCTAssertEqual(usage.reclaimableBytes, 128)
        XCTAssertEqual(usage.sharedBytes, 0)
        XCTAssertTrue(try manager.garbageCollectionPlan().items.isEmpty)

        try fileManager.removeItem(at: models.appendingPathComponent("model-a"))
        let plan = try manager.garbageCollectionPlan(limitingTo: [unit])
        XCTAssertEqual(plan.reclaimableBytes, 128)
        XCTAssertEqual(plan.items.map(\.kind), [.cacheUnit])

        let result = try manager.execute(plan)
        XCTAssertEqual(result.reclaimedBytes, 128)
        XCTAssertFalse(fileManager.fileExists(atPath: unit.path))
    }

    func testSharedPayloadRemainsWhenOneModelIsRemoved() throws {
        let unit = legacyUnit(repository: "shared")
        let payload = try writePayload(at: unit.appendingPathComponent("model.bin"), bytes: 96)
        try linkModel("model-a", to: payload)
        try linkModel("model-b", to: payload)
        let manager = try makeManager()

        let report = try manager.report(modelIDs: ["model-a", "model-b"])
        XCTAssertEqual(report.models.map(\.referencedBytes), [96, 96])
        XCTAssertEqual(report.models.map(\.reclaimableBytes), [0, 0])
        XCTAssertEqual(report.models.map(\.sharedBytes), [96, 96])

        try fileManager.removeItem(at: models.appendingPathComponent("model-a"))
        let plan = try manager.garbageCollectionPlan(limitingTo: [unit])
        XCTAssertEqual(plan.reclaimableBytes, 0)
        XCTAssertTrue(plan.items.isEmpty)
        XCTAssertTrue(fileManager.fileExists(atPath: payload.path))
    }

    func testAllSiblingSymlinkFilesContributeToOwnership() throws {
        let unit = legacyUnit(repository: "many-links")
        let first = try writePayload(at: unit.appendingPathComponent("first.bin"), bytes: 41)
        let second = try writePayload(at: unit.appendingPathComponent("second.bin"), bytes: 59)
        let model = models.appendingPathComponent("model-a", isDirectory: true)
        try fileManager.createDirectory(at: model, withIntermediateDirectories: true)
        try fileManager.createSymbolicLink(
            at: model.appendingPathComponent("first.bin"),
            withDestinationURL: first
        )
        try fileManager.createSymbolicLink(
            at: model.appendingPathComponent("second.bin"),
            withDestinationURL: second
        )

        let manager = try makeManager()
        let usage = try XCTUnwrap(manager.report(modelIDs: ["model-a"]).models.first)
        XCTAssertEqual(usage.referencedBytes, 100)
        XCTAssertEqual(usage.reclaimableBytes, 100)
        XCTAssertEqual(try manager.cacheUnitsReferenced(by: "model-a"), [unit])
    }

    func testLegacyApplicationSupportReferenceKeepsPayloadLive() throws {
        let unit = legacyUnit(repository: "legacy-consumer")
        let payload = try writePayload(at: unit.appendingPathComponent("model.bin"), bytes: 64)
        let legacyRoot = applicationSupport.appendingPathComponent("legacy-runtime", isDirectory: true)
        try fileManager.createDirectory(at: legacyRoot, withIntermediateDirectories: true)
        try fileManager.createSymbolicLink(
            at: legacyRoot.appendingPathComponent("model.bin"),
            withDestinationURL: payload
        )

        let plan = try makeManager().garbageCollectionPlan()
        XCTAssertEqual(plan.reclaimableBytes, 0)
        XCTAssertTrue(plan.items.isEmpty)
    }

    func testExternalPayloadIsReferencedButNeverReportedAsReclaimable() throws {
        let externalRoot = temporaryRoot.appendingPathComponent("external", isDirectory: true)
        let payload = try writePayload(at: externalRoot.appendingPathComponent("model.bin"), bytes: 101)
        try linkModel("model-a", to: payload)

        let usage = try XCTUnwrap(makeManager().report(modelIDs: ["model-a"]).models.first)
        XCTAssertEqual(usage.referencedBytes, 101)
        XCTAssertEqual(usage.externalBytes, 101)
        XCTAssertEqual(usage.reclaimableBytes, 0)
        XCTAssertEqual(usage.sharedBytes, 0)
    }

    func testLiveUnitCollectsStaleIncompleteDownload() throws {
        let unit = legacyUnit(repository: "partial")
        let payload = try writePayload(at: unit.appendingPathComponent("model.bin"), bytes: 80)
        let incomplete = try writePayload(
            at: unit.appendingPathComponent("weights.bin.incomplete"),
            bytes: 32
        )
        try linkModel("model-a", to: payload)

        let plan = try makeManager().garbageCollectionPlan()
        XCTAssertEqual(plan.reclaimableBytes, 32)
        XCTAssertEqual(plan.incompleteDownloadBytes, 32)
        XCTAssertEqual(plan.items.count, 1)
        XCTAssertEqual(plan.items.first?.kind, .incompleteDownload)
        XCTAssertEqual(plan.items.first?.path, incomplete.path)
    }

    func testRevisionedSnapshotAndBlobAreCountedOnceAndCollectedTogether() throws {
        let snapshot = hub
            .appendingPathComponent("snapshots/models/org/revisioned", isDirectory: true)
            .appendingPathComponent(HubSnapshot.revisionKey("commit"), isDirectory: true)
        let blob = try writePayload(
            at: hub.appendingPathComponent("blobs/aa/content", isDirectory: false),
            bytes: 160
        )
        let payload = snapshot.appendingPathComponent("model.bin", isDirectory: false)
        try fileManager.createDirectory(at: snapshot, withIntermediateDirectories: true)
        try fileManager.linkItem(at: blob, to: payload)
        let reference = hub
            .appendingPathComponent("refs/models/org/revisioned", isDirectory: true)
            .appendingPathComponent(HubSnapshot.revisionKey("main") + ".ref")
        try fileManager.createDirectory(at: reference.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data("commit".utf8).write(to: reference)
        try linkModel("model-a", to: payload)
        let manager = try makeManager()

        let report = try manager.report(modelIDs: ["model-a"])
        XCTAssertEqual(report.hubBytes, 166)
        XCTAssertEqual(report.models.first?.referencedBytes, 160)
        XCTAssertEqual(report.models.first?.reclaimableBytes, 160)

        try fileManager.removeItem(at: models.appendingPathComponent("model-a"))
        let plan = try manager.garbageCollectionPlan(limitingTo: [snapshot])
        XCTAssertEqual(plan.reclaimableBytes, 166)
        XCTAssertEqual(Set(plan.items.map(\.kind)), [.cacheUnit, .blob, .reference])

        let result = try manager.execute(plan)
        XCTAssertEqual(result.reclaimedBytes, 166)
        XCTAssertFalse(fileManager.fileExists(atPath: snapshot.path))
        XCTAssertFalse(fileManager.fileExists(atPath: blob.path))
        XCTAssertFalse(fileManager.fileExists(atPath: reference.path))
    }

    func testSymlinkBackedSnapshotKeepsReferencedBlobLiveWithoutHardLinks() throws {
        let snapshot = hub
            .appendingPathComponent("snapshots/models/org/symlinked", isDirectory: true)
            .appendingPathComponent(HubSnapshot.revisionKey("commit"), isDirectory: true)
        let blob = try writePayload(
            at: hub.appendingPathComponent("blobs/bb/content", isDirectory: false),
            bytes: 144
        )
        let payload = snapshot.appendingPathComponent("model.bin", isDirectory: false)
        try fileManager.createDirectory(at: snapshot, withIntermediateDirectories: true)
        try fileManager.createSymbolicLink(at: payload, withDestinationURL: blob)
        try linkModel("model-a", to: payload)

        let manager = try ModelStorageManager(
            modelsDirectory: models,
            hubDirectory: hub,
            applicationSupportDirectory: applicationSupport,
            fileManager: fileManager,
            unreferencedGracePeriod: 0
        )
        let livePlan = try manager.garbageCollectionPlan()
        XCTAssertFalse(livePlan.items.contains { $0.path == blob.path })

        try fileManager.removeItem(at: models.appendingPathComponent("model-a"))
        let removedPlan = try manager.garbageCollectionPlan()
        XCTAssertTrue(removedPlan.items.contains { $0.kind == .blob && $0.path == blob.path })
    }

    func testModelRootSymlinkDoesNotCountHubPayloadAsLocalBytes() throws {
        let unit = legacyUnit(repository: "directory-root")
        _ = try writePayload(at: unit.appendingPathComponent("config.json"), bytes: 40)
        try fileManager.createSymbolicLink(
            at: models.appendingPathComponent("model-a"),
            withDestinationURL: unit
        )

        let usage = try XCTUnwrap(makeManager().report(modelIDs: ["model-a"]).models.first)
        XCTAssertEqual(usage.referencedBytes, 40)
        XCTAssertEqual(usage.localBytes, 0)
        XCTAssertEqual(usage.reclaimableBytes, 40)
    }

    func testGlobalCollectionProtectsRecentlyCreatedUnreferencedCacheUnits() throws {
        let unit = legacyUnit(repository: "recent")
        _ = try writePayload(at: unit.appendingPathComponent("model.bin"), bytes: 72)

        XCTAssertTrue(try makeManager().garbageCollectionPlan().items.isEmpty)

        let noGraceManager = try ModelStorageManager(
            modelsDirectory: models,
            hubDirectory: hub,
            applicationSupportDirectory: applicationSupport,
            fileManager: fileManager,
            unreferencedGracePeriod: 0
        )
        XCTAssertEqual(try noGraceManager.garbageCollectionPlan().reclaimableBytes, 72)
    }

    func testExecutionRechecksReferencesUnderStorageLock() throws {
        let unit = legacyUnit(repository: "recheck")
        let payload = try writePayload(at: unit.appendingPathComponent("model.bin"), bytes: 88)
        try linkModel("model-a", to: payload)
        let manager = try makeManager()
        try fileManager.removeItem(at: models.appendingPathComponent("model-a"))
        let plan = try manager.garbageCollectionPlan(limitingTo: [unit])
        XCTAssertEqual(plan.reclaimableBytes, 88)

        try linkModel("model-b", to: payload)
        let result = try manager.execute(plan)

        XCTAssertEqual(result.reclaimedBytes, 0)
        XCTAssertEqual(result.deletedItemCount, 0)
        XCTAssertTrue(fileManager.fileExists(atPath: payload.path))
    }

    private func makeManager() throws -> ModelStorageManager {
        try ModelStorageManager(
            modelsDirectory: models,
            hubDirectory: hub,
            applicationSupportDirectory: applicationSupport,
            fileManager: fileManager
        )
    }

    private func legacyUnit(repository: String) -> URL {
        hub
            .appendingPathComponent("models/org", isDirectory: true)
            .appendingPathComponent(repository, isDirectory: true)
    }

    @discardableResult
    private func writePayload(at url: URL, bytes: Int) throws -> URL {
        try fileManager.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data(repeating: 1, count: bytes).write(to: url)
        return url
    }

    private func linkModel(_ id: String, to payload: URL) throws {
        let root = models.appendingPathComponent(id, isDirectory: true)
        try fileManager.createDirectory(at: root, withIntermediateDirectories: true)
        try fileManager.createSymbolicLink(
            at: root.appendingPathComponent(payload.lastPathComponent),
            withDestinationURL: payload
        )
    }
}
