import XCTest
import MereRunCore
@testable import MereRunCLI

final class ModelInventoryTests: XCTestCase {
    func testFastInventoryDoesNotMeasureReferencedPayloads() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("mere-run-fast-inventory-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let modelID = ModelResolver.ModelID.ornith35BMLX
        let modelRoot = root.appendingPathComponent(modelID.rawValue, isDirectory: true)
        let nested = modelRoot.appendingPathComponent("nested", isDirectory: true)
        try FileManager.default.createDirectory(at: nested, withIntermediateDirectories: true)
        try Data(repeating: 1, count: 29).write(to: nested.appendingPathComponent("payload.bin"))
        try MereRunModelManifest.template(for: modelID, createdAt: Date(timeIntervalSince1970: 0))
            .write(to: modelRoot)

        let snapshot = ModelInventory.snapshot(
            mode: .fast,
            locations: ModelLocationSnapshot(primaryRoot: root)
        )
        let row = try XCTUnwrap(snapshot.rows.first { $0.id == modelID.rawValue })

        XCTAssertEqual(snapshot.mode, .fast)
        XCTAssertTrue(snapshot.complete)
        XCTAssertGreaterThanOrEqual(snapshot.durationMs, 0)
        XCTAssertEqual(row.status, "installed")
        XCTAssertNil(row.size)
        XCTAssertTrue(row.manifestPresent)
        XCTAssertEqual(row.runtimeAvailable, true)
        XCTAssertEqual(row.verification, .notChecked)
    }

    func testVerifiedInventoryChecksRuntimeWithoutMeasuringSize() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("mere-run-verified-inventory-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let modelID = ModelResolver.ModelID.ornith35BMLX
        let modelRoot = root.appendingPathComponent(modelID.rawValue, isDirectory: true)
        try FileManager.default.createDirectory(at: modelRoot, withIntermediateDirectories: true)
        try MereRunModelManifest.template(for: modelID, createdAt: Date(timeIntervalSince1970: 0))
            .write(to: modelRoot)

        let snapshot = ModelInventory.snapshot(
            mode: .verified,
            locations: ModelLocationSnapshot(primaryRoot: root)
        )
        let row = try XCTUnwrap(snapshot.rows.first { $0.id == modelID.rawValue })

        XCTAssertEqual(row.status, "invalid")
        XCTAssertNil(row.size)
        XCTAssertEqual(row.runtimeAvailable, false)
        XCTAssertEqual(row.verification, .checked)
    }

    func testFastInventoryReportsConfiguredMissingBindingOffline() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("mere-run-offline-inventory-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let modelID = ModelResolver.ModelID.ornith35BMLX
        let locations = ModelLocationSnapshot(
            primaryRoot: root,
            bindings: [
                ModelLocationRegistry.Binding(
                    modelID: modelID.rawValue,
                    path: root.appendingPathComponent("offline-model").path
                ),
            ]
        )
        let snapshot = ModelInventory.snapshot(mode: .fast, locations: locations)
        let row = try XCTUnwrap(snapshot.rows.first { $0.id == modelID.rawValue })

        XCTAssertEqual(row.status, "offline")
        XCTAssertEqual(row.verification, .notChecked)
    }
}
