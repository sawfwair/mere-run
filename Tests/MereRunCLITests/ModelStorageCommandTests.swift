import XCTest
import MereRunCore
@testable import MereRunCLI

final class ModelStorageCommandTests: XCTestCase {
    func testModelCommandExposesStorageAndGarbageCollection() {
        let names = Set(Model.configuration.subcommands.map { $0.configuration.commandName })
        XCTAssertTrue(names.contains("storage"))
        XCTAssertTrue(names.contains("gc"))
    }

    func testStorageAndGarbageCollectionFlagsParse() throws {
        XCTAssertTrue(try ModelStorage.parse(["--json"]).json)

        let dryRun = try ModelGarbageCollect.parse([])
        XCTAssertFalse(dryRun.force)
        XCTAssertFalse(dryRun.json)

        let apply = try ModelGarbageCollect.parse(["--force", "--json"])
        XCTAssertTrue(apply.force)
        XCTAssertTrue(apply.json)
    }

    func testRemoveParsesCacheRetentionAndJSON() throws {
        let command = try ModelRemove.parse([
            "image-klein-nano",
            "--force",
            "--keep-cache",
            "--json",
        ])
        XCTAssertTrue(command.force)
        XCTAssertTrue(command.keepCache)
        XCTAssertTrue(command.json)
    }

    func testStorageTextSeparatesPhysicalAndReferencedBytes() {
        let report = ModelStorageReport(
            applicationSupportPath: "/store",
            modelStorePath: "/store/models",
            hubPath: "/store/hub",
            applicationSupportBytes: 1_000,
            modelStoreBytes: 20,
            hubBytes: 900,
            otherApplicationSupportBytes: 80,
            garbageCollectableBytes: 100,
            incompleteDownloadBytes: 25,
            models: [
                ModelStorageModelUsage(
                    id: "model-a",
                    installPath: "/store/models/model-a",
                    installed: true,
                    referencedBytes: 800,
                    localBytes: 0,
                    reclaimableBytes: 600,
                    sharedBytes: 150,
                    externalBytes: 50
                ),
            ]
        )

        let text = ModelStorageCommandOutput.text(report)
        XCTAssertTrue(text.contains("Hub payloads:"))
        XCTAssertTrue(text.contains("Safe to collect:"))
        XCTAssertTrue(text.contains("do not add these values"))
        XCTAssertTrue(text.contains("referenced"))
        XCTAssertTrue(text.contains("reclaimable on removal"))
        XCTAssertTrue(text.contains("shared"))
        XCTAssertTrue(text.contains("external (not managed here)"))
    }

    func testGarbageCollectionJSONIncludesDryRunModeAndPlan() throws {
        let output = ModelGarbageCollectOutput(
            mode: "dry-run",
            plan: ModelStorageGarbagePlan(
                hubPath: "/store/hub",
                reclaimableBytes: 42,
                incompleteDownloadBytes: 42,
                items: [
                    ModelStorageGarbageItem(
                        kind: .incompleteDownload,
                        path: "/store/hub/model.bin.incomplete",
                        logicalBytes: 42
                    ),
                ]
            ),
            result: nil
        )
        let encoded = try ModelStorageCommandOutput.encode(output)
        let decoded = try JSONDecoder().decode(
            ModelGarbageCollectOutput.self,
            from: Data(encoded.utf8)
        )

        XCTAssertEqual(decoded, output)
        XCTAssertTrue(ModelGarbageCollect.text(output).contains("dry run"))
        XCTAssertTrue(ModelGarbageCollect.text(output).contains("--force"))
    }
}
