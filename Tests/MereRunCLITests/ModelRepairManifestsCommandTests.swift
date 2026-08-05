import Foundation
import MereRunCore
import XCTest
@testable import MereRunCLI

final class ModelRepairManifestsCommandTests: XCTestCase {
    func testParsesStructuredPreviewOptions() throws {
        let command = try ModelRepairManifests.parse([
            "--dry-run", "--json", "--accept-model-license",
            "--model", "video-minimax-h3-fl2va-bf16-mlx",
        ])

        XCTAssertTrue(command.dryRun)
        XCTAssertTrue(command.json)
        XCTAssertTrue(command.acceptModelLicense)
        XCTAssertEqual(command.model, "video-minimax-h3-fl2va-bf16-mlx")
    }

    func testRepairCanRecordExplicitRestrictedModelAcceptance() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("ModelRepairManifests.\(UUID().uuidString)", isDirectory: true)
        let modelID = ModelResolver.ModelID.miniMaxH3FL2VABF16MLX
        let modelRoot = root.appendingPathComponent(modelID.rawValue, isDirectory: true)
        try FileManager.default.createDirectory(at: modelRoot, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let command = try ModelRepairManifests.parse([
            "--accept-model-license", "--model", modelID.rawValue,
        ])
        let report = command.makeReport(modelDirs: [root])
        let entry = try XCTUnwrap(report.entries.first { $0.modelID == modelID.rawValue })
        let manifest = try MereRunModelManifest.loadRequired(from: modelRoot)

        XCTAssertEqual(entry.status, .wrote)
        XCTAssertEqual(manifest.usageTermsAcknowledged, true)
    }

    func testPreviewReportIdentifiesMissingManifestWithoutWritingIt() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("ModelRepairManifests.\(UUID().uuidString)", isDirectory: true)
        let modelID = try XCTUnwrap(ModelResolver.ModelID.allCases.first)
        let modelRoot = root.appendingPathComponent(modelID.rawValue, isDirectory: true)
        try FileManager.default.createDirectory(at: modelRoot, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let command = try ModelRepairManifests.parse(["--dry-run"])
        let report = command.makeReport(modelDirs: [root])
        let entry = try XCTUnwrap(report.entries.first { $0.modelID == modelID.rawValue })

        XCTAssertEqual(report.mode, "preview")
        XCTAssertEqual(entry.status, .wouldWrite)
        XCTAssertFalse(FileManager.default.fileExists(atPath: entry.path ?? ""))
        XCTAssertEqual(report.wroteCount, 1)
    }

    func testRepairReportUsesStableSnakeCaseJSONContract() throws {
        let report = ModelRepairManifestsReport(
            mode: "preview",
            wroteCount: 1,
            alreadyCount: 2,
            skippedCount: 3,
            entries: [
                .init(modelID: "image-example", status: .wouldWrite, path: "/tmp/mererun_model.json")
            ]
        )

        let json = try StructuredRunOutput.encode(report)

        XCTAssertTrue(json.contains(#""wrote_count""#))
        XCTAssertTrue(json.contains(#""model_id""#))
        XCTAssertTrue(json.contains(#""would_write""#))
    }
}
