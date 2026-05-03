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

        let providerModel = SetupAgentRuntime.providerModel(for: recommendation)

        XCTAssertEqual(providerModel.id, AgentModelResources.qwen35NineBModelId)
        XCTAssertNotEqual(providerModel.id, CodeGenResources.defaultModelId)
    }

    func testPiProviderCanBeWrittenToIsolatedHome() throws {
        let home = FileManager.default.temporaryDirectory
            .appendingPathComponent("mere-run-pi-home-\(UUID().uuidString)", isDirectory: true)
        defer {
            try? FileManager.default.removeItem(at: home)
        }

        let extensionURL = try PiAgentIntegration.writeLocalProviderExtension(
            host: "127.0.0.1",
            port: 8080,
            model: PiProviderModel(
                id: AgentModelResources.qwen35NineBModelId,
                name: "Qwen3.5 9B",
                contextWindow: 32_768,
                maxTokens: 4_096
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
    }

    func testAgentStartModelIsOptional() throws {
        let command = try AgentStart.parse([])

        XCTAssertNil(command.model)
    }

    func testAgentStartParsesSelectedModel() throws {
        let command = try AgentStart.parse([
            "--model",
            AgentModelResources.qwen35NineBModelId,
        ])

        XCTAssertEqual(command.model, AgentModelResources.qwen35NineBModelId)
    }

    func testAgentStartPrefersInstalledStartableAgentModel() throws {
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

        XCTAssertEqual(recommendation?.id, AgentModelResources.qwen35NineBModelId)
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
}
