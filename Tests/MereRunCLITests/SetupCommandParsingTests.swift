import XCTest
@testable import MereRunCLI
@testable import MereRunCore

final class SetupCommandParsingTests: XCTestCase {
    func testSetupParsesAgentSmallDryRun() throws {
        let command = try Setup.parse([
            "--mode", "agent",
            "--agent-model", "small",
            "--dry-run",
        ])

        XCTAssertEqual(command.mode, .agent)
        XCTAssertEqual(command.agentModel, .small)
        XCTAssertTrue(command.dryRun)
    }

    func testSetupParsesAgentPremierDryRun() throws {
        let command = try Setup.parse([
            "--mode", "agent",
            "--agent-model", "premier",
            "--dry-run",
        ])

        XCTAssertEqual(command.mode, .agent)
        XCTAssertEqual(command.agentModel, .premier)
        XCTAssertTrue(command.dryRun)
    }

    func testSetupParsesBYOAAndManualModes() throws {
        let byoa = try Setup.parse(["--mode", "byoa", "--dry-run"])
        let manual = try Setup.parse(["--mode", "manual", "--dry-run"])

        XCTAssertEqual(byoa.mode, .byoa)
        XCTAssertEqual(manual.mode, .manual)
    }

    func testPiProviderUsesSelectedModel() throws {
        let recommendation = try XCTUnwrap(
            MereRunAgentModelCatalog.recommendation(
                for: .small,
                on: MereRunMachineProfile(
                    physicalMemoryBytes: 16 * 1_073_741_824,
                    processorName: "M1",
                    isAppleSiliconMac: true
                )
            )
        )

        let providerModel = try SetupAgentRuntime.providerModel(for: recommendation)

        XCTAssertEqual(providerModel.id, Q35Resources.ornith9BModelId)
        XCTAssertNotEqual(providerModel.id, CodeGenResources.defaultModelId)
        XCTAssertTrue(providerModel.reasoning)
        XCTAssertTrue(providerModel.toolCall)
    }

    func testDeepseekProviderMatchesServedContextAndStartupTimeout() throws {
        let recommendation = try XCTUnwrap(
            MereRunAgentModelCatalog.recommendation(
                for: .premier,
                on: MereRunMachineProfile(
                    physicalMemoryBytes: 128 * 1_073_741_824,
                    processorName: "M3 Ultra",
                    isAppleSiliconMac: true
                )
            )
        )

        let runtime = try SetupAgentRuntime.runtime(for: recommendation)
        let providerModel = try SetupAgentRuntime.providerModel(for: recommendation)

        XCTAssertEqual(providerModel.contextWindow, DeepseekV4FlashResources.defaultContextLength)
        XCTAssertEqual(providerModel.maxTokens, DeepseekV4FlashResources.defaultContextLength)
        XCTAssertEqual(
            runtime.healthTimeoutSeconds,
            DeepseekV4FlashResources.serverStartupTimeoutSeconds + 30
        )
    }

    func testNorthMiniProviderUsesNativeManagedModelID() throws {
        let recommendation = try XCTUnwrap(
            MereRunAgentModelCatalog
                .allTierRecommendations(
                    on: MereRunMachineProfile(
                        physicalMemoryBytes: 32 * 1_073_741_824,
                        processorName: "M4 Pro",
                        isAppleSiliconMac: true
                    )
                )
                .first { $0.id == NorthMiniCodeResources.modelId }
        )

        let providerModel = try SetupAgentRuntime.providerModel(for: recommendation)

        XCTAssertEqual(providerModel.id, NorthMiniCodeResources.modelId)
        XCTAssertEqual(providerModel.contextWindow, NorthMiniCodeResources.runtimeContextLength)
        XCTAssertEqual(providerModel.maxTokens, NorthMiniCodeResources.maxOutputTokens)
        XCTAssertFalse(providerModel.reasoning)
        XCTAssertFalse(recommendation.isStartableByMereRun)
        XCTAssertEqual(recommendation.servingEngine, .textCode)
    }

    func testGemmaProviderUsesNativeGemmaRuntime() throws {
        let recommendation = try XCTUnwrap(
            MereRunAgentModelCatalog.recommendation(
                for: .tier,
                on: MereRunMachineProfile(
                    physicalMemoryBytes: 32 * 1_073_741_824,
                    processorName: "M2 Max",
                    isAppleSiliconMac: true
                )
            )
        )

        let runtime = try SetupAgentRuntime.runtime(for: recommendation)
        let providerModel = try SetupAgentRuntime.providerModel(for: recommendation)

        XCTAssertEqual(runtime.engine, .textChatGemma4)
        XCTAssertEqual(providerModel.id, Gemma4Resources.twelveB4BitModelId)
        XCTAssertEqual(providerModel.contextWindow, Gemma4Resources.defaultContextLength)
        XCTAssertTrue(recommendation.isStartableByMereRun)
        XCTAssertEqual(recommendation.servingEngine, .textChatGemma4)
    }

    func testQ38ProviderUsesNativeQwenRuntime() throws {
        let runtime = try SetupAgentRuntime.runtime(
            forManagedModelID: Q35Resources.q38TwentySevenBModelId
        )
        let providerModel = runtime.providerModel

        XCTAssertEqual(runtime.engine, .textChatQ36)
        XCTAssertEqual(providerModel.id, Q35Resources.q38TwentySevenBModelId)
        XCTAssertEqual(providerModel.contextWindow, Q35Resources.q38TwentySevenBContextLength)
        XCTAssertEqual(providerModel.maxTokens, 4_096)
        XCTAssertTrue(providerModel.reasoning)
        XCTAssertTrue(providerModel.toolCall)
        XCTAssertTrue(providerModel.supportsReasoningEffort)
        XCTAssertEqual(
            providerModel.thinkingLevelMap?.first { $0.key == "high" }?.value,
            "xhigh"
        )
        XCTAssertEqual(
            providerModel.thinkingLevelMap?.first { $0.key == "max" }?.value,
            "xhigh"
        )
        XCTAssertTrue(runtime.recommendation.isStartableByMereRun)
        XCTAssertEqual(runtime.recommendation.servingEngine, .textChatQ36)
    }

    func testOrnithVisionProviderUsesNativeQwenRuntime() throws {
        let runtime = try SetupAgentRuntime.runtime(
            forManagedModelID: Q35Resources.ornith35BVisionModelId
        )
        let providerModel = runtime.providerModel

        XCTAssertEqual(runtime.engine, .textChatQ36)
        XCTAssertEqual(providerModel.id, Q35Resources.ornith35BVisionModelId)
        XCTAssertEqual(providerModel.contextWindow, Q35Resources.ornith35BMLXContextLength)
        XCTAssertEqual(providerModel.maxTokens, 4_096)
        XCTAssertTrue(providerModel.reasoning)
        XCTAssertTrue(providerModel.toolCall)
        XCTAssertTrue(providerModel.inputModalities.contains("image"))
        XCTAssertTrue(runtime.recommendation.isStartableByMereRun)
        XCTAssertEqual(runtime.recommendation.servingEngine, .textChatQ35)
    }

    func testOrnithMLX4BitVisionProviderUsesNativeQwenRuntime() throws {
        let runtime = try SetupAgentRuntime.runtime(
            forManagedModelID: Q35Resources.ornith35BMLX4BitModelId
        )
        let providerModel = runtime.providerModel

        XCTAssertEqual(runtime.engine, .textChatQ36)
        XCTAssertEqual(providerModel.id, Q35Resources.ornith35BMLX4BitModelId)
        XCTAssertEqual(providerModel.contextWindow, Q35Resources.ornith35BMLXContextLength)
        XCTAssertTrue(providerModel.reasoning)
        XCTAssertTrue(providerModel.toolCall)
        XCTAssertTrue(providerModel.inputModalities.contains("image"))
        XCTAssertTrue(runtime.recommendation.isStartableByMereRun)
        XCTAssertEqual(runtime.recommendation.servingEngine, .textChatQ35)
    }

    func testOrnith35BProviderUsesNativeManagedModelID() throws {
        let recommendation = try XCTUnwrap(
            MereRunAgentModelCatalog
                .allTierRecommendations(
                    on: MereRunMachineProfile(
                        physicalMemoryBytes: 64 * 1_073_741_824,
                        processorName: "M4 Max",
                        isAppleSiliconMac: true
                    )
                )
                .first { $0.id == Ornith35BCodeResources.modelId }
        )

        let providerModel = try SetupAgentRuntime.providerModel(for: recommendation)

        XCTAssertEqual(providerModel.id, Ornith35BCodeResources.modelId)
        XCTAssertEqual(providerModel.contextWindow, Ornith35BCodeResources.runtimeContextLength)
        XCTAssertEqual(providerModel.maxTokens, Ornith35BCodeResources.maxOutputTokens)
        XCTAssertFalse(providerModel.reasoning)
        XCTAssertFalse(recommendation.isStartableByMereRun)
        XCTAssertEqual(recommendation.servingEngine, .textCode)
    }

    func testOrnithProviderUsesNativeQ35ManagedModelID() throws {
        let recommendation = try XCTUnwrap(
            MereRunAgentModelCatalog
                .allTierRecommendations(
                    on: MereRunMachineProfile(
                        physicalMemoryBytes: 24 * 1_073_741_824,
                        processorName: "M4 Pro",
                        isAppleSiliconMac: true
                    )
                )
                .first { $0.id == Q35Resources.ornith9BModelId }
        )

        let runtime = try SetupAgentRuntime.runtime(for: recommendation)
        let providerModel = try SetupAgentRuntime.providerModel(for: recommendation)

        XCTAssertEqual(runtime.engine, .textChatQ36)
        XCTAssertEqual(providerModel.id, Q35Resources.ornith9BModelId)
        XCTAssertEqual(providerModel.contextWindow, Q35Resources.defaultContextLength)
        XCTAssertTrue(providerModel.reasoning)
        XCTAssertTrue(providerModel.toolCall)
        XCTAssertTrue(recommendation.isStartableByMereRun)
        XCTAssertEqual(recommendation.servingEngine, .textChatQ35)
    }

    func testPiProviderCanBeWrittenToIsolatedHome() throws {
        let home = FileManager.default.temporaryDirectory
            .appendingPathComponent("mere-run-pi-home-\(UUID().uuidString)", isDirectory: true)
        defer {
            try? FileManager.default.removeItem(at: home)
        }
        let profile = try XCTUnwrap(
            ManagedModelCatalog.apiProfile(for: Q35Resources.ornith9BModelId)
        )

        let extensionURL = try PiAgentIntegration.writeLocalProviderExtension(
            host: "127.0.0.1",
            port: 8080,
            model: PiProviderModel(
                id: Q35Resources.ornith9BModelId,
                name: "Ornith 9B",
                profile: profile
            ),
            homeDirectory: home,
            persistConfiguration: false
        )

        XCTAssertEqual(
            extensionURL.path,
            home
                .appendingPathComponent(".pi/agent/extensions/mere-run-local-provider.ts")
                .path
        )
        XCTAssertTrue(FileManager.default.fileExists(atPath: extensionURL.path))
        let extensionSource = try String(contentsOf: extensionURL, encoding: .utf8)
        XCTAssertTrue(extensionSource.contains("@earendil-works/pi-coding-agent"))
        XCTAssertTrue(extensionSource.contains("const configuredModels = ["))
        XCTAssertTrue(extensionSource.contains("models: configuredModels"))
        XCTAssertTrue(extensionSource.contains("pi.registerProvider(\"mere-run\""))
        XCTAssertTrue(extensionSource.contains("api: \"openai-completions\""))
        XCTAssertFalse(extensionSource.contains("@earendil-works/pi-ai/compat"))
        XCTAssertFalse(extensionSource.contains("fetch("))
        XCTAssertFalse(extensionSource.contains("discoverModels"))
        XCTAssertTrue(extensionSource.contains("input: [\"text\", \"image\"]"))
        XCTAssertTrue(extensionSource.contains("supportsFinishReason: true"))
    }

    func testAgentStartModelIsOptional() throws {
        let command = try AgentStart.parse([])

        XCTAssertNil(command.model)
    }

    func testAgentServerAdmissionMarkerDetection() throws {
        let logURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("mere-run-agent-server-\(UUID().uuidString).log")
        defer { try? FileManager.default.removeItem(at: logURL) }

        try "API server queued by machine admission.\n"
            .write(to: logURL, atomically: true, encoding: .utf8)
        XCTAssertTrue(PiAgentIntegration.serverQueueObserved(in: logURL))
        XCTAssertFalse(PiAgentIntegration.serverAdmissionObserved(in: logURL))

        try ("API server queued by machine admission.\n" + PiAgentIntegration.serverAdmissionMarker + "\n")
            .write(to: logURL, atomically: true, encoding: .utf8)
        XCTAssertTrue(PiAgentIntegration.serverQueueObserved(in: logURL))
        XCTAssertTrue(PiAgentIntegration.serverAdmissionObserved(in: logURL))
    }

    func testAgentStatusParsesMachineReadableInspection() throws {
        let command = try AgentStatus.parse([
            "--pi-path", "/tmp/pi",
            "--json",
        ])

        XCTAssertEqual(command.piPath, "/tmp/pi")
        XCTAssertTrue(command.json)
    }

    func testAgentStatusSnapshotRoundTrips() throws {
        let snapshot = AgentStatusSnapshot(
            machine: AgentStatusMachine(
                processor: "M4 Max",
                unifiedMemoryGB: 128,
                appleSiliconMac: true,
                linux: false
            ),
            pi: AgentStatusPi(
                installed: true,
                managedInstall: true,
                autoInstallSupported: true,
                path: "/tmp/pi",
                version: "v1.2.3"
            ),
            provider: AgentStatusProvider(
                configured: true,
                host: "127.0.0.1",
                port: 8080,
                modelID: "text-agent-deepseek-v4-flash",
                updatedAt: Date(timeIntervalSince1970: 10),
                configurationPath: "/tmp/provider.json",
                extensionPath: "/tmp/provider.ts"
            ),
            recommendedModelID: "text-agent-deepseek-v4-flash",
            models: [
                AgentStatusModel(
                    id: "text-agent-deepseek-v4-flash",
                    displayName: "DeepSeek V4 Flash",
                    summary: "Premier local agent.",
                    minimumUnifiedMemoryGB: 96,
                    recommendedUnifiedMemoryGB: 128,
                    servingEngine: "deepseek-v4-flash",
                    startableByMereRun: true,
                    sourceConfigurationRequired: false,
                    installed: true,
                    reason: nil
                ),
            ]
        )
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601

        XCTAssertEqual(
            try decoder.decode(AgentStatusSnapshot.self, from: encoder.encode(snapshot)),
            snapshot
        )
    }

    func testAgentStartParsesSelectedModel() throws {
        let command = try AgentStart.parse([
            "--model",
            AgentModelResources.qwen35NineBModelId,
        ])

        XCTAssertEqual(command.model, AgentModelResources.qwen35NineBModelId)
    }

    func testAgentStartParsesInlinePiHarnessArguments() throws {
        let command = try AgentStart.parse([
            "--inline",
            "--working-directory", "/tmp/film",
            "--pi-argument", "--no-extensions",
            "--pi-argument", "--extension",
            "--pi-argument", "/tmp/film-studio.ts",
            "--pi-argument", "@/tmp/film/run.json",
            "--prompt", "Open this film project.",
        ])

        XCTAssertTrue(command.inline)
        XCTAssertEqual(command.workingDirectory, "/tmp/film")
        XCTAssertEqual(
            command.piArguments,
            ["--no-extensions", "--extension", "/tmp/film-studio.ts", "@/tmp/film/run.json"]
        )
        XCTAssertEqual(command.prompt, "Open this film project.")
    }

    func testAgentServerDefaultsGemma4PrefixKVCacheOffButPreservesExplicitOverride() {
        XCTAssertEqual(
            AgentStart.apiServerEnvironment(
                inheriting: [:],
                processID: 123
            )[PiAgentIntegration.agentParentProcessEnvironment],
            "123"
        )
        XCTAssertEqual(
            AgentStart.apiServerEnvironment(
                inheriting: [:],
                processID: 123
            )["MERERUN_GEMMA4_PREFIX_KV_CACHE"],
            "0"
        )
        XCTAssertEqual(
            AgentStart.apiServerEnvironment(
                inheriting: ["MERERUN_GEMMA4_PREFIX_KV_CACHE": "1"]
            )["MERERUN_GEMMA4_PREFIX_KV_CACHE"],
            "1"
        )
    }

    func testAgentParentProcessIDRejectsUnsafeValues() {
        XCTAssertEqual(
            PiAgentIntegration.configuredAgentParentProcessID(
                environment: [PiAgentIntegration.agentParentProcessEnvironment: "123"],
                currentProcessID: 999
            ),
            123
        )
        XCTAssertNil(
            PiAgentIntegration.configuredAgentParentProcessID(
                environment: [PiAgentIntegration.agentParentProcessEnvironment: "1"],
                currentProcessID: 999
            )
        )
        XCTAssertNil(
            PiAgentIntegration.configuredAgentParentProcessID(
                environment: [PiAgentIntegration.agentParentProcessEnvironment: "999"],
                currentProcessID: 999
            )
        )
    }

    func testSetupAgentPromptCarriesBoundedMachineContext() {
        let prompt = SetupAgentPrompt.render(
            userRequest: "Set up this Mac.",
            selectedModelID: DeepseekV4FlashResources.defaultModelId,
            engine: .textChatDeepseekV4Flash,
            modelURL: URL(fileURLWithPath: "/tmp/ds4.gguf"),
            host: "127.0.0.1",
            port: 8080,
            machine: MereRunMachineProfile(
                physicalMemoryBytes: 128 * 1_073_741_824,
                processorName: "M3 Ultra",
                isAppleSiliconMac: true
            )
        )

        XCTAssertTrue(prompt.contains("Machine: M3 Ultra, 128 GB unified memory"))
        XCTAssertTrue(prompt.contains("Pi is already using provider `mere-run`"))
        XCTAssertTrue(prompt.contains("text-agent-deepseek-v4-flash"))
        XCTAssertTrue(prompt.contains("Recommended setup-agent tier for this Mac"))
        XCTAssertTrue(prompt.contains("Selected setup-agent is recommended: true"))
        XCTAssertTrue(prompt.contains("DeepSeek V4 Flash is the preferred premier setup-agent tier"))
        XCTAssertTrue(prompt.contains("smaller tool-capable native agents are alternatives"))
        XCTAssertTrue(prompt.contains("mere.run model capabilities --recommended"))
        XCTAssertTrue(prompt.contains("mere.run model list"))
        XCTAssertTrue(prompt.contains("Do not run demo scripts"))
        XCTAssertTrue(prompt.contains("demo.sh"))
        XCTAssertTrue(prompt.contains("Never pass `--allow-unsupported`"))
    }

    func testAgentStartDoesNotTreatTextOnlyModelAsPiStartable() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("mere-run-agent-start-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer {
            MereRunModelPaths.setProcessModelsDirOverride(nil)
            try? FileManager.default.removeItem(at: root)
        }
        MereRunModelPaths.setProcessModelsDirOverride(root)

        let modelRoot = root.appendingPathComponent(AgentModelResources.qwen35NineBModelId, isDirectory: true)
        try FileManager.default.createDirectory(at: modelRoot, withIntermediateDirectories: true)
        let modelFile = modelRoot.appendingPathComponent(AgentModelResources.qwen35NineBRelativePath)
        try Data("gguf".utf8).write(to: modelFile)

        let recommendation = AgentStart.bestInstalledStartableAgentModel(
            on: MereRunMachineProfile(
                physicalMemoryBytes: 16 * 1_073_741_824,
                processorName: "M1",
                isAppleSiliconMac: true
            )
        )

        XCTAssertNil(recommendation)
    }

    func testAgentStartPrefersInstalledDeepseekOverQwenCode() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("mere-run-agent-start-ds4-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer {
            MereRunModelPaths.setProcessModelsDirOverride(nil)
            try? FileManager.default.removeItem(at: root)
        }
        MereRunModelPaths.setProcessModelsDirOverride(root)

        let qwenFile = root.appendingPathComponent(CodeGenResources.managedRelativePath)
        try Data("qwen".utf8).write(to: qwenFile)

        let deepseekRoot = root.appendingPathComponent(
            DeepseekV4FlashResources.defaultModelId,
            isDirectory: true
        )
        try FileManager.default.createDirectory(at: deepseekRoot, withIntermediateDirectories: true)
        let deepseekFile = deepseekRoot.appendingPathComponent(DeepseekV4FlashResources.imatrixGGUFFile)
        let externalModelRoot = root.appendingPathComponent("external-ds4", isDirectory: true)
        try FileManager.default.createDirectory(at: externalModelRoot, withIntermediateDirectories: true)
        let externalModel = externalModelRoot.appendingPathComponent(DeepseekV4FlashResources.imatrixGGUFFile)
        try Data("ds4".utf8).write(to: externalModel)
        try FileManager.default.createSymbolicLink(at: deepseekFile, withDestinationURL: externalModel)

        let machine = MereRunMachineProfile(
            physicalMemoryBytes: 128 * 1_073_741_824,
            processorName: "M3 Ultra",
            isAppleSiliconMac: true
        )
        let recommendation = AgentStart.bestInstalledStartableAgentModel(on: machine)

        XCTAssertEqual(recommendation?.id, DeepseekV4FlashResources.defaultModelId)
    }

    func testConfiguredAgentModelMustBeInstalledToBeStartable() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("mere-run-agent-config-\(UUID().uuidString)", isDirectory: true)
        defer {
            MereRunModelPaths.setProcessModelsDirOverride(nil)
            try? FileManager.default.removeItem(at: root)
        }
        MereRunModelPaths.setProcessModelsDirOverride(root)

        XCTAssertFalse(
            AgentStart.isInstalledStartableAgentModel(
                AgentModelResources.qwen35NineBModelId,
                on: MereRunMachineProfile(
                    physicalMemoryBytes: 16 * 1_073_741_824,
                    processorName: "M1",
                    isAppleSiliconMac: true
                )
            )
        )
    }

    func testDirectorySizeFollowsSymlinkedModelRoots() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("mere-run-size-\(UUID().uuidString)", isDirectory: true)
        defer {
            try? FileManager.default.removeItem(at: root)
        }
        let target = root.appendingPathComponent("target", isDirectory: true)
        let link = root.appendingPathComponent("link", isDirectory: true)
        try FileManager.default.createDirectory(at: target, withIntermediateDirectories: true)
        try Data(repeating: 1, count: 4).write(to: target.appendingPathComponent("model.safetensors"))
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: target)

        XCTAssertEqual(FileSystemHelper.directorySize(at: link), 4)
    }

    func testDirectorySizeFollowsSymlinkedFilesInsideModelRoots() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("mere-run-size-\(UUID().uuidString)", isDirectory: true)
        defer {
            try? FileManager.default.removeItem(at: root)
        }
        let target = root.appendingPathComponent("target", isDirectory: true)
        let modelRoot = root.appendingPathComponent("text-chat-q36-nano", isDirectory: true)
        try FileManager.default.createDirectory(at: target, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: modelRoot, withIntermediateDirectories: true)
        let weights = target.appendingPathComponent("model.safetensors")
        try Data(repeating: 1, count: 7).write(to: weights)
        try FileManager.default.createSymbolicLink(
            at: modelRoot.appendingPathComponent("model.safetensors"),
            withDestinationURL: weights
        )

        XCTAssertEqual(FileSystemHelper.directorySize(at: modelRoot), 7)
    }

    func testDirectorySizeFollowsSymlinkedDirectoriesInsideModelRoots() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("mere-run-size-\(UUID().uuidString)", isDirectory: true)
        defer {
            try? FileManager.default.removeItem(at: root)
        }
        let target = root.appendingPathComponent("target", isDirectory: true)
        let modelRoot = root.appendingPathComponent("image-ideogram4-sdnq-uint4", isDirectory: true)
        try FileManager.default.createDirectory(at: target, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: modelRoot, withIntermediateDirectories: true)
        try Data(repeating: 1, count: 3).write(to: modelRoot.appendingPathComponent("mererun_model.json"))
        try Data(repeating: 1, count: 11).write(to: target.appendingPathComponent("model.safetensors"))
        try FileManager.default.createSymbolicLink(
            at: modelRoot.appendingPathComponent("transformer", isDirectory: true),
            withDestinationURL: target
        )

        let usage = FileSystemHelper.directoryUsage(at: modelRoot)
        XCTAssertEqual(usage.resolvedBytes, 14)
        XCTAssertEqual(usage.localBytes, 3)
        XCTAssertEqual(usage.symlinkCount, 1)
        XCTAssertEqual(usage.symlinkedDirectoryCount, 1)
        XCTAssertEqual(usage.layoutDescription, "directory with symlinked directories")
        XCTAssertEqual(FileSystemHelper.directorySize(at: modelRoot), 14)
    }

    func testModelInventoryUsesResolvedSizeForSymlinkedModelDirectories() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("mere-run-inventory-\(UUID().uuidString)", isDirectory: true)
        defer {
            MereRunModelPaths.setProcessModelsDirOverride(nil)
            try? FileManager.default.removeItem(at: root)
        }
        MereRunModelPaths.setProcessModelsDirOverride(root)

        let target = root.appendingPathComponent("hub/ideogram-transformer", isDirectory: true)
        let modelRoot = root.appendingPathComponent(ModelResolver.ModelID.ideogram4SDNQUInt4.rawValue, isDirectory: true)
        try FileManager.default.createDirectory(at: target, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: modelRoot, withIntermediateDirectories: true)
        try Data(repeating: 1, count: 13).write(to: modelRoot.appendingPathComponent("legacy-metadata.bin"))
        try Data(repeating: 1, count: 29).write(to: target.appendingPathComponent("diffusion_pytorch_model.safetensors"))
        try FileManager.default.createSymbolicLink(
            at: modelRoot.appendingPathComponent("transformer", isDirectory: true),
            withDestinationURL: target
        )

        let row = try XCTUnwrap(
            ModelInventory.rows().first { $0.id == ModelResolver.ModelID.ideogram4SDNQUInt4.rawValue }
        )
        XCTAssertEqual(row.status, "installed")
        XCTAssertEqual(row.size, ByteCountFormatter.string(fromByteCount: 42, countStyle: .file))
    }

    func testModelInventoryMarksPinnedPlaceholderInstallInvalid() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("mere-run-pinned-inventory-\(UUID().uuidString)", isDirectory: true)
        defer {
            MereRunModelPaths.setProcessModelsDirOverride(nil)
            try? FileManager.default.removeItem(at: root)
        }
        MereRunModelPaths.setProcessModelsDirOverride(root)

        let modelID = ModelResolver.ModelID.visionGeometryMoGe2Small
        let modelRoot = root.appendingPathComponent(modelID.rawValue, isDirectory: true)
        try FileManager.default.createDirectory(at: modelRoot, withIntermediateDirectories: true)
        try Data([0]).write(to: modelRoot.appendingPathComponent("model.onnx"))
        try MereRunModelManifest.template(for: modelID, createdAt: Date(timeIntervalSince1970: 0))
            .write(to: modelRoot)

        let row = try XCTUnwrap(ModelInventory.rows().first { $0.id == modelID.rawValue })
        XCTAssertEqual(row.status, "invalid")
        XCTAssertFalse(row.isInstalled)
    }

    func testModelInventoryMarksManagedRootWithBrokenWeightLinksInvalid() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("mere-run-broken-inventory-\(UUID().uuidString)", isDirectory: true)
        defer {
            MereRunModelPaths.setProcessModelsDirOverride(nil)
            try? FileManager.default.removeItem(at: root)
        }
        MereRunModelPaths.setProcessModelsDirOverride(root)

        let modelID = ModelResolver.ModelID.ornith35BMLX
        let modelRoot = root.appendingPathComponent(modelID.rawValue, isDirectory: true)
        try FileManager.default.createDirectory(at: modelRoot, withIntermediateDirectories: true)
        try MereRunModelManifest.template(for: modelID, createdAt: Date(timeIntervalSince1970: 0))
            .write(to: modelRoot)
        try FileManager.default.createSymbolicLink(
            at: modelRoot.appendingPathComponent("model-00001-of-00004.safetensors"),
            withDestinationURL: root.appendingPathComponent("missing-model-shard.safetensors")
        )

        let row = try XCTUnwrap(ModelInventory.rows().first { $0.id == modelID.rawValue })
        XCTAssertEqual(row.status, "invalid")
        XCTAssertFalse(row.isInstalled)
    }

    func testPiBinaryFinderSkipsDirectoryNamedPi() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("mere-run-pi-finder-\(UUID().uuidString)", isDirectory: true)
        defer {
            try? FileManager.default.removeItem(at: root)
        }
        let packageDir = root.appendingPathComponent("pi", isDirectory: true)
        try FileManager.default.createDirectory(at: packageDir, withIntermediateDirectories: true)
        let binaryURL = packageDir.appendingPathComponent("pi", isDirectory: false)
        try Data("#!/bin/sh\n".utf8).write(to: binaryURL)
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o755],
            ofItemAtPath: binaryURL.path
        )

        let resolved = PiAgentIntegration.findPiBinary(in: root)

        XCTAssertEqual(
            resolved?.resolvingSymlinksInPath(),
            binaryURL.resolvingSymlinksInPath()
        )
    }

    func testPiLaunchExplicitlyKeepsLocalProviderWhenExtensionDiscoveryIsDisabled() {
        let home = URL(fileURLWithPath: "/tmp/mere-run-pi-home", isDirectory: true)

        let arguments = PiTerminalLauncher.arguments(
            modelID: "text-chat-test",
            prompt: "Make a film",
            homeDirectory: home,
            additionalArguments: ["--no-extensions", "--extension", "/tmp/film-studio.ts"]
        )

        XCTAssertEqual(arguments.prefix(4), ["--provider", "mere-run", "--model", "text-chat-test"])
        XCTAssertEqual(
            arguments.dropFirst(4).prefix(2),
            ["--extension", "/tmp/mere-run-pi-home/.pi/agent/extensions/mere-run-local-provider.ts"]
        )
        XCTAssertTrue(arguments.contains("--no-extensions"))
        XCTAssertEqual(arguments.suffix(1), ["Make a film"])
    }
}
