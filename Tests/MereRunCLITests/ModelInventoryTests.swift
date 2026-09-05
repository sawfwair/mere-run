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
        XCTAssertEqual(snapshot.installedModelIDs, [modelID.rawValue])
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
        XCTAssertTrue(snapshot.installedModelIDs.isEmpty)
    }

    func testVerifiedInstalledIDsIncludeBindingsAndSearchRoots() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("mere-run-external-inventory-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let bindingRoot = root.appendingPathComponent("arbitrary-binding", isDirectory: true)
        let searchRoot = root.appendingPathComponent("catalog", isDirectory: true)
        let searchedModelRoot = searchRoot.appendingPathComponent(
            ModelResolver.ModelID.zetaMax.rawValue,
            isDirectory: true
        )
        try writeMinimalZImageModel(at: bindingRoot, modelID: nil)
        try writeMinimalZImageModel(at: searchedModelRoot, modelID: .zetaMax)

        let snapshot = ModelInventory.snapshot(
            mode: .verified,
            locations: ModelLocationSnapshot(
                primaryRoot: root.appendingPathComponent("primary", isDirectory: true),
                searchRoots: [searchRoot],
                bindings: [
                    .init(modelID: ModelResolver.ModelID.zetaNano.rawValue, path: bindingRoot.path),
                ]
            )
        )

        XCTAssertTrue(
            snapshot.installedModelIDs.contains(ModelResolver.ModelID.zetaNano.rawValue)
        )
        XCTAssertTrue(
            snapshot.installedModelIDs.contains(ModelResolver.ModelID.zetaMax.rawValue)
        )
    }

    private func writeMinimalZImageModel(
        at root: URL,
        modelID: ModelResolver.ModelID?
    ) throws {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        if let modelID {
            try MereRunModelManifest.template(for: modelID, createdAt: Date(timeIntervalSince1970: 0))
                .write(to: root)
        }
        try Data("{}".utf8).write(to: root.appendingPathComponent("model_index.json"))

        for component in ["text_encoder", "transformer", "vae"] {
            let directory = root.appendingPathComponent(component, isDirectory: true)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            try Data("{}".utf8).write(to: directory.appendingPathComponent("config.json"))
            try Data().write(to: directory.appendingPathComponent("model.safetensors"))
        }

        let tokenizer = root.appendingPathComponent("tokenizer", isDirectory: true)
        try FileManager.default.createDirectory(at: tokenizer, withIntermediateDirectories: true)
        for filename in ["tokenizer.json", "tokenizer_config.json", "merges.txt", "vocab.json"] {
            try Data("{}".utf8).write(to: tokenizer.appendingPathComponent(filename))
        }

        let scheduler = root.appendingPathComponent("scheduler", isDirectory: true)
        try FileManager.default.createDirectory(at: scheduler, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: scheduler.appendingPathComponent("scheduler_config.json"))
    }
}
