import XCTest
@testable import MereRunCLI

final class ModelPullCommandParsingTests: XCTestCase {
    func testModelPullParsesHardwareOverride() throws {
        let cmd = try ModelPull.parse([
            "text-code-qwen3",
            "--allow-unsupported",
        ])

        XCTAssertEqual(cmd.target, "text-code-qwen3")
        XCTAssertTrue(cmd.allowUnsupported)
    }

    func testModelCommandExposesCapabilities() {
        let commandNames = Set(Model.configuration.subcommands.map { $0.configuration.commandName })
        XCTAssertTrue(commandNames.contains("capabilities"))
    }

    func testInstallValidationErrorIncludesRetryWithoutUsage() {
        let error = ModelPullInstallError(
            modelID: "image-klein-max",
            modelDir: URL(fileURLWithPath: "/tmp/mere.run/models/image-klein-max"),
            hubRepoID: "black-forest-labs/FLUX.2-klein-4B",
            details: ["No *.safetensors weights found in transformer/"]
        )

        let message = error.localizedDescription
        XCTAssertTrue(message.contains("Model image-klein-max was not installed cleanly."))
        XCTAssertTrue(message.contains("- No *.safetensors weights found in transformer/"))
        XCTAssertTrue(message.contains("Model store: /tmp/mere.run/models/image-klein-max"))
        XCTAssertTrue(message.contains("Retry with: mere.run model pull image-klein-max"))
        XCTAssertTrue(message.contains("Use --force only if you intentionally want to replace a complete install."))
        XCTAssertFalse(message.contains("Usage:"))
    }

    func testDiskPreflightFailsWhenHubCacheCannotFitEstimatedModel() throws {
        XCTAssertThrowsError(
            try ModelPullDiskPreflight.evaluate(
                modelID: "speech-asr-parakeet",
                estimatedDownloadBytes: 2 * ModelPullDiskPreflight.bytesPerGiB,
                hubCacheURL: URL(fileURLWithPath: "/Volumes/Tiny/huggingface"),
                hubCacheAvailableBytes: 1 * ModelPullDiskPreflight.bytesPerGiB,
                modelStoreURL: URL(fileURLWithPath: "/Volumes/Tiny/mere.run"),
                modelStoreAvailableBytes: 5 * ModelPullDiskPreflight.bytesPerGiB
            )
        ) { error in
            let message = String(describing: error)
            XCTAssertTrue(message.contains("Not enough free disk space"))
            XCTAssertTrue(message.contains("MERERUN_HUB_CACHE"))
        }
    }

    func testDiskPreflightWarnsWhenHeadroomWouldBeLow() throws {
        let warnings = try ModelPullDiskPreflight.evaluate(
            modelID: "text-embed-qwen3-0.6b",
            estimatedDownloadBytes: 2 * ModelPullDiskPreflight.bytesPerGiB,
            hubCacheURL: URL(fileURLWithPath: "/Volumes/AlmostFull/huggingface"),
            hubCacheAvailableBytes: 11 * ModelPullDiskPreflight.bytesPerGiB,
            modelStoreURL: URL(fileURLWithPath: "/Volumes/AlmostFull/mere.run"),
            modelStoreAvailableBytes: 11 * ModelPullDiskPreflight.bytesPerGiB
        )

        XCTAssertEqual(warnings.count, 1)
        XCTAssertTrue(warnings[0].contains("may have only about"))
    }
}
