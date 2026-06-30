import XCTest
import MereRunCore
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

    func testModelCapabilitiesParsesJsonRecommendationFlags() throws {
        let cmd = try ModelCapabilities.parse(["--recommended", "--json"])

        XCTAssertTrue(cmd.recommended)
        XCTAssertTrue(cmd.json)
    }

    func testModelCapabilitiesJsonIncludesMachineChatRecommendation() throws {
        let machine = MereRunMachineProfile(
            physicalMemoryBytes: 32 * 1_073_741_824,
            processorName: "M1 Max",
            isAppleSiliconMac: true
        )
        let bands = ManagedModelCapabilityCatalog.recommendedChatBandReports(on: machine)
        let payload = ModelCapabilitiesOutput(
            machine: .init(machine),
            chatBands: bands.map { .init($0, machine: machine) },
            recommendedChatModel: bands
                .first { $0.contains(unifiedMemoryGB: machine.unifiedMemoryGB) }
                .map { .init($0, machine: machine) },
            setupAgent: MereRunAgentModelCatalog
                .recommendation(for: .tier, on: machine)
                .map(ModelCapabilitiesSetupAgent.init),
            recommendedSetupModels: ManagedModelCapabilityCatalog
                .recommendedSetupReports(on: machine)
                .map(ModelCapabilitiesModel.init),
            unavailableRecommendedModelIDs: [],
            models: []
        )

        let data = try JSONEncoder().encode(payload)
        let decoded = try JSONDecoder().decode(ModelCapabilitiesOutput.self, from: data)

        XCTAssertEqual(decoded.machine.unifiedMemoryGB, 32)
        XCTAssertEqual(decoded.recommendedChatModel?.modelID, "text-chat-gemma4-12b-4bit")
        XCTAssertEqual(decoded.recommendedChatModel?.alternateModelIDs, [
            "text-chat-gemma4-turbo",
            "text-chat-q36-nano",
        ])
        XCTAssertTrue(decoded.recommendedChatModel?.currentMachine == true)
        XCTAssertEqual(decoded.setupAgent?.id, "text-chat-gemma4-12b-4bit")
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
