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

    func testBuffaloLPullRequiresExplicitLicenseAcceptance() throws {
        let restricted = try XCTUnwrap(
            ManagedModelCatalog.spec(for: FaceAnalysisResources.modelID)
        )
        let blocked = try ModelPull.parse([FaceAnalysisResources.modelID])
        let accepted = try ModelPull.parse([
            FaceAnalysisResources.modelID,
            "--accept-model-license",
        ])

        XCTAssertNotNil(restricted.usageRestriction)
        let message = try XCTUnwrap(blocked.licenseAcceptanceMessage(for: restricted))
        XCTAssertTrue(message.contains("reviewed and accept these terms"))
        XCTAssertTrue(message.contains("agree to comply with them"))
        XCTAssertTrue(message.contains("Download begins only after this confirmation"))
        XCTAssertNil(accepted.licenseAcceptanceMessage(for: restricted))
    }

    func testPublicOpenModelPullsDoNotRequireLicenseAcceptance() throws {
        let modelIDs = [
            LagunaResources.modelID,
            LagunaResources.xsModelID,
            "image-zimage-nano",
            ModelResolver.ModelID.sortformerDiarization.rawValue,
            ModelResolver.ModelID.cosmos3EdgeMLX.rawValue,
            ModelResolver.ModelID.ltxGemma3TwelveB4Bit.rawValue,
        ]
        for modelID in modelIDs {
            let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: modelID))
            let command = try ModelPull.parse([modelID])

            XCTAssertNil(spec.usageRestriction)
            XCTAssertNil(command.licenseAcceptanceMessage(for: spec))
            XCTAssertFalse(
                CLICommandDisplay.modelPullCommand(for: modelID)
                    .contains("--accept-model-license")
            )
        }
    }

    func testModelListReportsBuffaloLUsageTerms() {
        let lines = ModelList.usageRestrictionLines()

        XCTAssertTrue(lines.contains { line in
            line.contains(FaceAnalysisResources.modelID)
                && line.contains("non-commercial research use")
                && line.contains("Usage terms:")
                && line.contains("deepinsight/insightface#license")
        })
    }

    func testRestrictedModelPreflightRequiresAndPropagatesAcceptance() throws {
        let temp = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: temp) }
        let modelStore = temp.appendingPathComponent("models", isDirectory: true)
        let hubCache = temp.appendingPathComponent("hub", isDirectory: true)

        let blocked = try ModelPull.parse([
            FaceAnalysisResources.modelID,
            "--allow-unsupported",
            "--preflight",
            "--json",
        ]).makePreflightEnvelope(
            hubCacheURL: hubCache,
            modelStoreURL: modelStore,
            diskAvailableBytes: { _ in 100 * ModelPullDiskPreflight.bytesPerGiB }
        )
        XCTAssertEqual(blocked.status, .blocked)
        XCTAssertEqual(blocked.result.models.first?.status, "blocked_usage_terms")
        XCTAssertEqual(blocked.result.models.first?.usageTermsAcknowledged, false)

        let accepted = try ModelPull.parse([
            FaceAnalysisResources.modelID,
            "--allow-unsupported",
            "--accept-model-license",
            "--preflight",
            "--json",
        ]).makePreflightEnvelope(
            hubCacheURL: hubCache,
            modelStoreURL: modelStore,
            diskAvailableBytes: { _ in 100 * ModelPullDiskPreflight.bytesPerGiB }
        )
        XCTAssertNotEqual(accepted.status, .blocked)
        XCTAssertEqual(accepted.result.models.first?.status, "will_download")
        XCTAssertEqual(accepted.result.models.first?.usageTermsAcknowledged, true)
        XCTAssertTrue(accepted.actions.first { $0.id == "pull-model" }?.command?.argv.contains("--accept-model-license") == true)
    }

    func testModelPullParsesPreflightJSONOptions() throws {
        let cmd = try ModelPull.parse([
            "image-zimage-nano",
            "--preflight",
            "--json",
        ])

        XCTAssertEqual(cmd.target, "image-zimage-nano")
        XCTAssertTrue(cmd.preflight)
        XCTAssertTrue(cmd.json)
    }

    func testDisplayedPullGuidanceAddsAcknowledgementOnlyForRestrictedModels() {
        XCTAssertTrue(
            CLICommandDisplay.modelPullCommand(for: FaceAnalysisResources.modelID)
                .hasSuffix("model pull vision-face-buffalo-l --accept-model-license")
        )
        XCTAssertTrue(
            CLICommandDisplay.modelPullCommand(for: "image-klein-nano")
                .hasSuffix("model pull image-klein-nano")
        )
        XCTAssertFalse(
            CLICommandDisplay.modelPullCommand(for: "image-klein-nano")
                .contains("--accept-model-license")
        )
    }

    func testModelPullPreflightReportsPullableModel() throws {
        let temp = try makeTemporaryDirectory()
        defer {
            try? FileManager.default.removeItem(at: temp)
        }
        let modelStore = temp.appendingPathComponent("models", isDirectory: true)
        let hubCache = temp.appendingPathComponent("hub", isDirectory: true)

        let cmd = try ModelPull.parse([
            "image-klein-nano",
            "--allow-unsupported",
            "--preflight",
            "--json",
        ])
        let envelope = cmd.makePreflightEnvelope(
            hubCacheURL: hubCache,
            modelStoreURL: modelStore,
            diskAvailableBytes: { _ in 100 * ModelPullDiskPreflight.bytesPerGiB },
            now: { Date(timeIntervalSince1970: 0) }
        )

        XCTAssertNotEqual(envelope.status, .blocked)
        XCTAssertEqual(envelope.command, ["model", "pull"])
        XCTAssertEqual(envelope.result.mode, "single")
        XCTAssertEqual(envelope.result.models.count, 1)
        XCTAssertEqual(envelope.result.models.first?.id, "image-klein-nano")
        XCTAssertEqual(envelope.result.models.first?.status, "will_download")
        XCTAssertEqual(envelope.result.models.first?.hasDownloadSource, true)
        XCTAssertEqual(envelope.result.models.first?.installed, false)
        XCTAssertEqual(envelope.result.models.first?.runtimeReady, false)
        XCTAssertEqual(envelope.result.models.first?.conversionRequired, false)
        XCTAssertEqual(envelope.result.models.first?.willDownload, true)
        XCTAssertEqual(envelope.result.modelStore.path, modelStore.path)
        XCTAssertEqual(envelope.result.hubCache.path, hubCache.path)

        let pullModel = try XCTUnwrap(envelope.actions.first { $0.id == "pull-model" })
        XCTAssertTrue(pullModel.enabled)
        XCTAssertEqual(pullModel.command?.argv, [
            "mere.run",
            "model",
            "pull",
            "image-klein-nano",
            "--allow-unsupported",
        ])

        let encoded = try StructuredRunOutput.encode(envelope)
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let decoded = try decoder.decode(ModelPullPreflightEnvelope.self, from: Data(encoded.utf8))
        XCTAssertEqual(decoded.result.models.first?.id, "image-klein-nano")
        XCTAssertEqual(decoded.result.models.first?.runtimeReady, false)
        XCTAssertEqual(decoded.result.models.first?.conversionRequired, false)
    }

    func testModelPullPreflightUsesExplicitConversionRequiredStatus() {
        XCTAssertEqual(
            ModelPullPreflightAnalyzer.modelStatus(
                selected: true,
                installed: true,
                conversionRequired: true,
                willDownload: false,
                blockedBySupport: false,
                blockedBySource: false,
                all: false
            ),
            "conversion_required"
        )
    }

    func testModelPullPreflightUsesMissingProcessModelStoreOverride() throws {
        let temp = try makeTemporaryDirectory()
        let fileManager = FileManager.default
        defer {
            MereRunModelPaths.setProcessModelsDirOverride(nil)
            try? fileManager.removeItem(at: temp)
        }

        let missingModelStore = temp.appendingPathComponent("models", isDirectory: true)
        let hubCache = temp.appendingPathComponent("hub", isDirectory: true)
        try fileManager.createDirectory(at: hubCache, withIntermediateDirectories: true)
        MereRunModelPaths.setProcessModelsDirOverride(missingModelStore)

        let cmd = try ModelPull.parse([
            "image-klein-nano",
            "--allow-unsupported",
            "--preflight",
            "--json",
        ])
        let envelope = cmd.makePreflightEnvelope(
            fileManager: fileManager,
            hubCacheURL: hubCache,
            now: { Date(timeIntervalSince1970: 0) }
        )

        XCTAssertFalse(fileManager.fileExists(atPath: missingModelStore.path))
        XCTAssertEqual(envelope.result.modelStore.path, missingModelStore.standardizedFileURL.path)
        XCTAssertEqual(
            envelope.result.models.first?.installPath,
            missingModelStore
                .appendingPathComponent("image-klein-nano", isDirectory: true)
                .standardizedFileURL
                .path
        )
        XCTAssertEqual(envelope.result.models.first?.status, "will_download")
    }

    func testModelPullPreflightBlocksUnknownModel() throws {
        let cmd = try ModelPull.parse([
            "image-not-real",
            "--preflight",
            "--json",
        ])
        let envelope = cmd.makePreflightEnvelope(now: { Date(timeIntervalSince1970: 0) })

        XCTAssertEqual(envelope.status, .blocked)
        XCTAssertTrue(envelope.diagnostics.contains { $0.id == "model_unknown" })

        let pullModel = try XCTUnwrap(envelope.actions.first { $0.id == "pull-model" })
        XCTAssertFalse(pullModel.enabled)
    }

    func testModelPullPreflightSupportsAllMode() throws {
        let cmd = try ModelPull.parse([
            "--all",
            "--preflight",
            "--json",
        ])
        let envelope = cmd.makePreflightEnvelope(now: { Date(timeIntervalSince1970: 0) })

        XCTAssertEqual(envelope.result.mode, "all")
        XCTAssertGreaterThan(envelope.result.models.count, 1)
        XCTAssertTrue(envelope.result.models.contains { $0.hasDownloadSource })
        XCTAssertTrue(envelope.actions.contains { $0.id == "pull-models" })
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
            recommendedCodeModel: ManagedModelCapabilityCatalog
                .recommendedCodeModelReport(on: machine)
                .map(ModelCapabilitiesModel.init),
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
        XCTAssertEqual(decoded.recommendedCodeModel?.id, "text-code-north-mini")
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

    func testModelPullProgressFormatterShowsFractionalPercentSpeedAndETA() {
        var formatter = ModelPullProgressFormatterState()
        let totalBytes: Int64 = 1_000_000_000

        _ = formatter.render(
            modelID: "image-zimage-nano",
            progress: .downloadingBytes(completed: 0, total: totalBytes),
            now: Date(timeIntervalSince1970: 0)
        )
        let line = formatter.render(
            modelID: "image-zimage-nano",
            progress: .downloadingBytes(completed: 4_000_000, total: totalBytes),
            now: Date(timeIntervalSince1970: 2)
        )

        XCTAssertTrue(line.contains("[image-zimage-nano]"))
        XCTAssertTrue(line.contains("0.4%"), line)
        XCTAssertTrue(line.contains(" / "), line)
        XCTAssertTrue(line.contains("/s"), line)
        XCTAssertTrue(line.contains("ETA "), line)
    }

    func testModelPullProgressFormatterHandlesUnknownTotalWithoutETA() {
        var formatter = ModelPullProgressFormatterState()

        _ = formatter.render(
            modelID: "text-chat-gemma4-max",
            progress: .downloadingBytes(completed: 0, total: nil),
            now: Date(timeIntervalSince1970: 0)
        )
        let line = formatter.render(
            modelID: "text-chat-gemma4-max",
            progress: .downloadingBytes(completed: 4_000_000, total: nil),
            now: Date(timeIntervalSince1970: 2)
        )

        XCTAssertTrue(line.contains("[text-chat-gemma4-max]"))
        XCTAssertFalse(line.contains("%"), line)
        XCTAssertFalse(line.contains("ETA"), line)
        XCTAssertTrue(line.contains("/s"), line)
    }

    func testModelPullProgressFormatterResetsSpeedWhenByteCounterRestarts() {
        var formatter = ModelPullProgressFormatterState()

        _ = formatter.render(
            modelID: "image-klein-base",
            progress: .downloadingBytes(completed: 0, total: 100_000_000),
            now: Date(timeIntervalSince1970: 0)
        )
        _ = formatter.render(
            modelID: "image-klein-base",
            progress: .downloadingBytes(completed: 10_000_000, total: 100_000_000),
            now: Date(timeIntervalSince1970: 2)
        )
        let resetLine = formatter.render(
            modelID: "image-klein-base",
            progress: .downloadingBytes(completed: 0, total: 20_000_000),
            now: Date(timeIntervalSince1970: 3)
        )

        XCTAssertTrue(resetLine.contains("0%"), resetLine)
        XCTAssertFalse(resetLine.contains("/s"), resetLine)
        XCTAssertFalse(resetLine.contains("ETA"), resetLine)
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

    func testDiskPreflightRequiredBytesIncludesSafetyMargin() {
        XCTAssertEqual(
            ModelPullDiskPreflight.requiredBytes(estimatedDownloadBytes: 2 * ModelPullDiskPreflight.bytesPerGiB),
            4 * ModelPullDiskPreflight.bytesPerGiB
        )
        XCTAssertEqual(
            ModelPullDiskPreflight.requiredBytes(estimatedDownloadBytes: 20 * ModelPullDiskPreflight.bytesPerGiB),
            24 * ModelPullDiskPreflight.bytesPerGiB
        )
        XCTAssertNil(ModelPullDiskPreflight.requiredBytes(estimatedDownloadBytes: nil))
    }

    func testDiskPreflightDoesNotRequireModelBytesForCompleteCachedSnapshot() throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let cache = root.appendingPathComponent("hub", isDirectory: true)
        let revision = "0123456789abcdef0123456789abcdef01234567"
        let revisionKey = "deb87fabd17715bb31ad4cf4ffb9494eeb15f8d33d85b031a301c64ab3417eaa"
        let repoID = "example/large-text-model"
        let snapshot = cache
            .appendingPathComponent("snapshots/models", isDirectory: true)
            .appendingPathComponent(repoID, isDirectory: true)
            .appendingPathComponent(revisionKey, isDirectory: true)
        try FileManager.default.createDirectory(at: snapshot, withIntermediateDirectories: true)
        for filename in ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"] {
            try Data().write(to: snapshot.appendingPathComponent(filename))
        }
        let spec = ManagedModelSpec(
            id: "text-chat-large-cached-test",
            category: .textChat,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: repoID,
                revision: revision,
                patterns: ["*.json", "*.safetensors"]
            ),
            validationKind: .hfTextChat,
            estimatedDownloadBytes: 1_000 * ModelPullDiskPreflight.bytesPerGiB
        )

        XCTAssertNoThrow(
            try ModelPullDiskPreflight.check(
                spec: spec,
                modelDir: root.appendingPathComponent("models/text-chat-large-cached-test"),
                hubCacheURL: cache,
                warn: { _ in }
            )
        )
    }

    func testDiskPreflightCreditsReclaimablePartialInstallForCompositePull() throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let cache = root.appendingPathComponent("hub", isDirectory: true)
        let modelDir = root.appendingPathComponent("models/video-composite-test", isDirectory: true)
        try FileManager.default.createDirectory(at: cache, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)
        try Data(repeating: 0xA5, count: 4 * 1_048_576).write(
            to: modelDir.appendingPathComponent("existing-support.safetensors")
        )

        let estimate = 20 * ModelPullDiskPreflight.bytesPerGiB
        let spec = ManagedModelSpec(
            id: "video-composite-test",
            category: .video,
            installShape: .structuredRoot,
            hubFallback: HubFallbackConfig(
                repoId: "example/support",
                revision: "0123456789abcdef0123456789abcdef01234567",
                patterns: ["existing-support.safetensors"]
            ),
            mountedHubFallbacks: [
                MountedHubFallbackConfig(
                    destinationPath: "transformer",
                    hubFallback: HubFallbackConfig(
                        repoId: "example/transformer",
                        revision: "fedcba9876543210fedcba9876543210fedcba98",
                        patterns: ["model.safetensors"]
                    )
                ),
            ],
            validationKind: .miniMaxH3MLX,
            estimatedDownloadBytes: estimate
        )
        let reclaimable = ModelPullDiskPreflight.reclaimableLocalBytes(
            in: modelDir,
            onFileSystemContaining: cache
        )

        XCTAssertGreaterThan(reclaimable, 0)
        XCTAssertEqual(
            ModelPullDiskPreflight.estimatedDownloadBytes(
                for: spec,
                modelDir: modelDir,
                force: false,
                hubCacheURL: cache
            ),
            estimate - reclaimable
        )
    }

    func testStructuredPreflightUsesCompleteCachedInklingSnapshotUnlessForced() throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let cache = root.appendingPathComponent("hub", isDirectory: true)
        let modelStore = root.appendingPathComponent("models", isDirectory: true)
        let revisionKey = "c7e23256263a36e914737a8334ee351882fe3de586fc44865a5a54f163fed8eb"
        let snapshot = cache
            .appendingPathComponent("snapshots/models", isDirectory: true)
            .appendingPathComponent(InklingResources.artifactRepoID, isDirectory: true)
            .appendingPathComponent(revisionKey, isDirectory: true)
        try FileManager.default.createDirectory(at: snapshot, withIntermediateDirectories: true)
        for filename in ["config.json", "tokenizer.json", "tokenizer_config.json"] {
            try Data("{}".utf8).write(to: snapshot.appendingPathComponent(filename))
        }
        let shard = "model-00001-of-00001.safetensors"
        try Data().write(to: snapshot.appendingPathComponent(shard))
        let index = #"{"weight_map":{"language_model.model.embed_tokens.weight":"model-00001-of-00001.safetensors"}}"#
        try Data(index.utf8).write(
            to: snapshot.appendingPathComponent("model.safetensors.index.json")
        )

        let cached = try ModelPull.parse([
            InklingResources.modelID,
            "--allow-unsupported",
            "--preflight",
            "--json",
        ]).makePreflightEnvelope(
            hubCacheURL: cache,
            modelStoreURL: modelStore,
            diskAvailableBytes: { _ in 3 * ModelPullDiskPreflight.bytesPerGiB }
        )

        XCTAssertNotEqual(cached.status, .blocked)
        XCTAssertEqual(cached.result.models.first?.estimatedDownloadBytes, 0)
        XCTAssertEqual(
            cached.result.models.first?.estimatedRequiredBytes,
            ModelPullDiskPreflight.safetyMarginBytes
        )
        XCTAssertFalse(cached.diagnostics.contains { $0.id == "hub_cache_space_insufficient" })

        let forced = try ModelPull.parse([
            InklingResources.modelID,
            "--allow-unsupported",
            "--force",
            "--preflight",
            "--json",
        ]).makePreflightEnvelope(
            hubCacheURL: cache,
            modelStoreURL: modelStore,
            diskAvailableBytes: { _ in 3 * ModelPullDiskPreflight.bytesPerGiB }
        )

        XCTAssertEqual(forced.status, .blocked)
        XCTAssertEqual(
            forced.result.models.first?.estimatedDownloadBytes,
            InklingResources.estimatedDownloadBytes
        )
        XCTAssertTrue(forced.diagnostics.contains { $0.id == "hub_cache_space_insufficient" })
    }

    private func makeTemporaryDirectory() throws -> URL {
        let temp = FileManager.default.temporaryDirectory
            .appendingPathComponent("mere-run-model-pull-preflight-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: temp, withIntermediateDirectories: true)
        return temp
    }
}
