import ArgumentParser
import Foundation
import XCTest
@testable import MereRunCLI

final class PluginCommandTests: XCTestCase {
    func testMereRunCLIExposesPluginCommand() {
        let commandNames = Set(MereRunCLI.configuration.subcommands.map { $0.configuration.commandName })

        XCTAssertTrue(commandNames.contains("plugin"))
    }

    func testPluginCommandExposesCatalogSubcommands() {
        let commandNames = Set(Plugin.configuration.subcommands.map { $0.configuration.commandName })

        XCTAssertEqual(commandNames, Set(["list", "info", "install", "doctor", "run", "rollback"]))
    }

    func testDefaultCatalogTargetsPublicPluginRepository() {
        XCTAssertEqual(
            PluginCatalogClient.defaultCatalogURL,
            "https://raw.githubusercontent.com/sawfwair/mere-run-plugins/main/catalog/plugins.v1.json"
        )
    }

    func testCatalogLoadsFromLocalPath() throws {
        let catalogURL = try writeCatalog()

        let catalog = try PluginCatalogClient.load(catalogURL: catalogURL.path)
        let plugin = try catalog.requirePlugin("mere-runpod")
        let install = try plugin.install(channel: "main")

        XCTAssertEqual(catalog.contractVersion, "mere.run/plugin-catalog.v1")
        XCTAssertEqual(catalog.defaultChannel, "main")
        XCTAssertEqual(plugin.name, "RunPod Runner")
        XCTAssertEqual(plugin.package, "mere-runpod")
        XCTAssertEqual(plugin.entrypoint, "mere-runpod")
        XCTAssertEqual(install.manager, "pipx")
        XCTAssertEqual(
            install.spec,
            "git+https://github.com/sawfwair/mere-run-plugins.git@main#subdirectory=packages/mere-runpod"
        )
    }

    func testPluginInfoParsesCatalogAndChannelOptions() throws {
        let command = try PluginInfo.parse([
            "mere-runpod",
            "--catalog-url", "/tmp/catalog.json",
            "--channel", "main",
            "--json",
        ])

        XCTAssertEqual(command.id, "mere-runpod")
        XCTAssertEqual(command.catalogURL, "/tmp/catalog.json")
        XCTAssertEqual(command.channel, "main")
        XCTAssertTrue(command.json)
    }

    func testPluginInstallDefaultsToDryRun() throws {
        let command = try PluginInstall.parse([
            "mere-runpod",
            "--catalog-url", "/tmp/catalog.json",
        ])

        XCTAssertEqual(command.id, "mere-runpod")
        XCTAssertEqual(command.catalogURL, "/tmp/catalog.json")
        XCTAssertNil(command.channel)
        XCTAssertFalse(command.yes)
        XCTAssertFalse(command.force)
    }

    func testPluginInstallCommandRendersPipxForceCommand() {
        let install = PluginCatalogInstall(
            manager: "pipx",
            spec: "git+https://github.com/sawfwair/mere-run-plugins.git@main#subdirectory=packages/mere-runpod",
            ref: "main"
        )

        let command = PluginInstallCommand(install: install, force: true).render()

        XCTAssertEqual(
            command,
            "pipx install --force git+https://github.com/sawfwair/mere-run-plugins.git@main#subdirectory=packages/mere-runpod"
        )
    }

    func testOptionalPluginSetupUsesOnlyVerifiedEntrypointAndFixedVerb() throws {
        var install = PluginCatalogInstall(
            manager: "pipx",
            spec: "git+https://github.com/sawfwair/mere-run-plugins.git@main#subdirectory=packages/mere-computer-use",
            ref: "main"
        )
        var calls: [(String, [String])] = []
        let execute: (String, [String]) throws -> Void = { executable, arguments in
            calls.append((executable, arguments))
        }

        let plugin = PluginCatalogEntry(
            id: "mere-computer-use", name: "Computer Use", description: "", repo: "", package: "mere-computer-use",
            subdirectory: "", entrypoint: "mere-computer-use", capabilities: [], channels: ["main": install]
        )

        try PluginSetupCommand.run(plugin: plugin, install: install, managed: false, execute: execute)
        XCTAssertTrue(calls.isEmpty)
        install.setup = true
        try PluginSetupCommand.run(plugin: plugin, install: install, managed: false, execute: execute)
        XCTAssertEqual(calls.count, 1)
        XCTAssertEqual(calls[0].0, "mere-computer-use")
        XCTAssertEqual(calls[0].1, ["setup", "--yes"])
    }

    func testSignedBundleInstallRunsFixedSetupAfterVerifiedInstall() throws {
        let catalogURL = try writeCatalog(setupCatalogJSON(bundle: true))
        let command = try PluginInstall.parse(["mere-computer-use", "--catalog-url", catalogURL.path, "--yes"])
        var events: [String] = []
        var actions = PluginInstallActions()
        actions.installBundle = { plugin, source, archive, local in
            XCTAssertEqual(plugin.id, "mere-computer-use")
            XCTAssertEqual(source, "https://example.com/mere-computer-use/release.json")
            XCTAssertNil(archive)
            XCTAssertFalse(local)
            events.append("bundle")
        }
        actions.installSource = { _, _ in
            XCTFail("A signed bundle install must not run the source installer")
            throw ValidationError("unexpected")
        }
        actions.registerGraphProvider = { _ in XCTFail("Managed bundles are discovered without registration") }
        actions.execute = { executable, arguments in events.append("\(executable) \(arguments.joined(separator: " "))") }

        try command.run(actions: actions)

        XCTAssertEqual(events, ["bundle", "mere-computer-use setup --yes"])
    }

    func testSignedBundleSetupFailureAdvisesManagedRetry() throws {
        let catalogURL = try writeCatalog(setupCatalogJSON(bundle: true))
        let command = try PluginInstall.parse(["mere-computer-use", "--catalog-url", catalogURL.path, "--yes"])
        var actions = PluginInstallActions()
        actions.installBundle = { _, _, _, _ in }
        actions.execute = { _, _ in throw ValidationError("driver download failed") }

        XCTAssertThrowsError(try command.run(actions: actions)) { error in
            XCTAssertTrue(String(describing: error).contains(
                "Installed mere-computer-use, but setup failed: driver download failed. "
                    + "Retry with \(CLICommandDisplay.command("plugin run mere-computer-use -- setup --yes"))."
            ))
        }
    }

    func testFailedSourceSetupKeepsGraphProviderRegistered() throws {
        let catalogURL = try writeCatalog(setupCatalogJSON(bundle: false))
        let command = try PluginInstall.parse(["mere-computer-use", "--catalog-url", catalogURL.path, "--yes"])
        let manifest = try JSONDecoder().decode(PluginManifest.self, from: Data("""
        {"contractVersion":"mere.run/plugin.v1","name":"mere-computer-use","version":"0.2.0",
         "graphProvider":{"contractVersion":"mere.run/plugin-graph-provider.v1"}}
        """.utf8))
        var events: [String] = []
        var actions = PluginInstallActions()
        actions.installBundle = { _, _, _, _ in XCTFail("A source install must not use the bundle installer") }
        actions.installSource = { plugin, _ in
            XCTAssertEqual(plugin.id, "mere-computer-use")
            events.append("source")
            return manifest
        }
        actions.registerGraphProvider = { events.append("register \($0)") }
        actions.execute = { executable, arguments in
            events.append("\(executable) \(arguments.joined(separator: " "))")
            throw ValidationError("driver download failed")
        }

        XCTAssertThrowsError(try command.run(actions: actions)) { error in
            XCTAssertTrue(String(describing: error).contains("Retry with mere-computer-use setup --yes."))
        }
        XCTAssertEqual(events, ["source", "register mere-computer-use", "mere-computer-use setup --yes"])
    }

    func testEveryDryRunAndInfoShowTheSetupStep() throws {
        let catalogURL = try writeCatalog(setupCatalogJSON(bundle: true))
        let catalog = try PluginCatalogClient.load(catalogURL: catalogURL.path)
        let plugin = try catalog.requirePlugin("mere-computer-use")
        let install = try plugin.install(channel: "main")
        let command = try PluginInstall.parse(["mere-computer-use", "--catalog-url", catalogURL.path])
        let managedSetup = CLICommandDisplay.command("plugin run mere-computer-use -- setup --yes")

        let bundlePlan = command.dryRun(
            plugin: plugin, install: install, bundle: "https://example.com/mere-computer-use/release.json", channel: "main"
        )
        XCTAssertTrue(bundlePlan.contains("  \(managedSetup)"))
        XCTAssertEqual(bundlePlan.last, "  \(command.confirmationCommand(channel: "main"))")

        let sourcePlan = command.dryRun(plugin: plugin, install: install, bundle: nil, channel: "main")
        XCTAssertTrue(sourcePlan.contains("  \(PluginInstallCommand.render(install: install))"))
        XCTAssertTrue(sourcePlan.contains("  mere-computer-use setup --yes"))

        XCTAssertTrue(PluginInfo.summary(plugin: plugin, install: install, channel: "main")
            .contains("  mere-computer-use setup --yes"))

        var withoutSetup = install
        withoutSetup.setup = nil
        XCTAssertFalse(command.dryRun(plugin: plugin, install: withoutSetup, bundle: nil, channel: "main")
            .contains { $0.contains("setup --yes") })
        XCTAssertFalse(PluginInfo.summary(plugin: plugin, install: withoutSetup, channel: "main")
            .contains { $0.contains("setup --yes") })
    }

    func testInstallConfirmationKeepsChannelAndForce() throws {
        let command = try PluginInstall.parse([
            "mere-doc-tools", "--catalog-url", "/tmp/plugin review.json", "--force",
        ])

        XCTAssertEqual(
            command.confirmationCommand(channel: "review"),
            "mere.run plugin install mere-doc-tools --catalog-url '/tmp/plugin review.json' --channel review --yes --force"
        )
    }

    func testForcedInstallRejectsAffectedUVEnvironmentBeforeRunningInstaller() {
        var executed = false
        let command = workflowInstallCommand(force: true)

        XCTAssertThrowsError(try command.run(
            package: "mere-workflow-tools",
            findExecutable: { _ in URL(fileURLWithPath: "/fixture/pipx") },
            capture: pipxCapture(metadata: pipxMetadata(backend: "uv"), version: "1.15.0\n"),
            execute: { _, _ in executed = true }
        )) { error in
            let message = String(describing: error)
            XCTAssertTrue(message.contains("pipx 1.15.0"))
            XCTAssertTrue(message.contains("mere-workflow-tools"))
            XCTAssertTrue(message.contains("1.16.0 or later"))
            XCTAssertTrue(message.contains("brew upgrade pipx"))
            XCTAssertTrue(message.contains("plugin has not been changed"))
        }
        XCTAssertFalse(executed)
    }

    func testForcedInstallAllowsFixedPipxVersionsWithoutChangingArguments() throws {
        let command = workflowInstallCommand(force: true)
        for version in ["1.16.0\n", "1.16.1", "1.100.0", "2.0.0"] {
            var arguments: [String] = []
            try command.run(
                package: "mere-workflow-tools",
                findExecutable: { _ in URL(fileURLWithPath: "/fixture/pipx") },
                capture: pipxCapture(metadata: pipxMetadata(backend: "uv"), version: version),
                execute: { executable, args in
                    XCTAssertEqual(executable, "pipx")
                    arguments = args
                }
            )
            XCTAssertEqual(arguments, ["install", "--force", command.install.spec], version)
        }
    }

    func testNormalInstallDoesNotProbePipxCompatibility() throws {
        let command = workflowInstallCommand(force: false)
        var arguments: [String] = []

        try command.run(
            package: "mere-workflow-tools",
            findExecutable: { _ in URL(fileURLWithPath: "/fixture/pipx") },
            capture: { _, _ in
                XCTFail("A normal install must not probe forced-reinstall compatibility.")
                return Data()
            },
            execute: { _, args in arguments = args }
        )

        XCTAssertEqual(arguments, ["install", command.install.spec])
    }

    func testForcedInstallAllowsFreshPipAndLegacyEnvironmentsWithoutVersionProbe() throws {
        let snapshots = [
            Data(#"{"venvs":{}}"#.utf8),
            pipxMetadata(backend: "pip"),
            pipxMetadata(backend: nil),
            pipxMetadata(backend: "uv", package: "unrelated-tools"),
        ]
        for metadata in snapshots {
            var executed = false
            try workflowInstallCommand(force: true).run(
                package: "mere-workflow-tools",
                findExecutable: { _ in URL(fileURLWithPath: "/fixture/pipx") },
                capture: { executable, arguments in
                    XCTAssertEqual(executable, "pipx")
                    XCTAssertEqual(arguments, ["list", "--json"])
                    return metadata
                },
                execute: { _, _ in executed = true }
            )
            XCTAssertTrue(executed)
        }
    }

    func testForcedUVInstallRequiresARecognizableStableVersion() {
        for version in ["", "unknown", "1.16.0rc1", "1.16.0\nother output"] {
            var executed = false
            XCTAssertThrowsError(try workflowInstallCommand(force: true).run(
                package: "mere-workflow-tools",
                findExecutable: { _ in URL(fileURLWithPath: "/fixture/pipx") },
                capture: pipxCapture(metadata: pipxMetadata(backend: "uv"), version: version),
                execute: { _, _ in executed = true }
            )) { error in
                XCTAssertTrue(String(describing: error).contains("Cannot verify pipx compatibility"))
            }
            XCTAssertFalse(executed)
        }
    }

    func testForcedInstallDoesNotRunAfterMetadataInspectionFails() {
        var executed = false
        XCTAssertThrowsError(try workflowInstallCommand(force: true).run(
            package: "mere-workflow-tools",
            findExecutable: { _ in URL(fileURLWithPath: "/fixture/pipx") },
            capture: pipxCapture(metadata: Data(#"{"venvs":[]}"#.utf8), version: "1.15.0"),
            execute: { _, _ in executed = true }
        )) { error in
            XCTAssertTrue(String(describing: error).contains("Run pipx list --json"))
        }
        XCTAssertFalse(executed)
    }

    func testCatalogSnapshotAddsVerifiedInstallationStateWithoutDroppingCatalogFields() throws {
        let catalog = try PluginCatalogClient.load(catalogURL: writeCatalog().path)

        let snapshot = PluginCatalogSnapshot.make(catalog: catalog) { plugin in
            PluginInstallationInspection(
                installed: true,
                verified: true,
                version: "1.2.3",
                path: "/usr/local/bin/\(plugin.entrypoint)",
                error: nil
            )
        }

        XCTAssertEqual(snapshot.contractVersion, catalog.contractVersion)
        XCTAssertEqual(snapshot.defaultChannel, "main")
        XCTAssertEqual(snapshot.plugins.first?.id, "mere-runpod")
        XCTAssertEqual(snapshot.plugins.first?.installedVersion, "1.2.3")
        XCTAssertEqual(snapshot.plugins.first?.installedPath, "/usr/local/bin/mere-runpod")
        XCTAssertTrue(snapshot.plugins.first?.verified == true)
        XCTAssertTrue(snapshot.plugins.first?.installCommand?.contains("pipx install") == true)
    }

    func testBrokenEditableInstallationReportsMissingSourceAndRepairCommand() throws {
        let catalog = try PluginCatalogClient.load(catalogURL: writeCatalog().path)
        let plugin = try catalog.requirePlugin("mere-runpod")
        let install = try plugin.install(channel: catalog.defaultChannel)
        let metadata = Data(staleEditablePipxJSON.utf8)
        let missingPath = try PluginPipxInstallationMetadata.missingEditableSourcePath(
            for: plugin,
            metadata: metadata,
            fileExists: { _ in false }
        )
        let inspection = PluginInstallationInspection(
            installed: true,
            verified: false,
            version: nil,
            path: "/Users/test/.local/bin/mere-runpod",
            error: "ModuleNotFoundError: No module named 'mere_runpod'"
        )

        let diagnosis = PluginInstallationFailureDiagnosis.make(
            plugin: plugin,
            install: install,
            inspection: inspection,
            missingEditableSourcePath: missingPath
        )

        XCTAssertEqual(missingPath, "/Users/test/mere/mere-plugins/packages/mere-runpod")
        XCTAssertEqual(
            diagnosis.summary,
            "editable source path no longer exists: /Users/test/mere/mere-plugins/packages/mere-runpod"
        )
        XCTAssertNil(diagnosis.verificationError)
        XCTAssertEqual(
            diagnosis.repairCommand,
            "mere.run plugin install mere-runpod --yes --force"
        )
        XCTAssertTrue(diagnosis.rendered.contains("cannot run its doctor"))
    }

    func testGenericVerificationFailureKeepsConciseErrorAndRepairCommand() throws {
        let catalog = try PluginCatalogClient.load(catalogURL: writeCatalog().path)
        let plugin = try catalog.requirePlugin("mere-runpod")
        let install = try plugin.install(channel: catalog.defaultChannel)
        let inspection = PluginInstallationInspection(
            installed: true,
            verified: false,
            version: nil,
            path: "/usr/local/bin/mere-runpod",
            error: "Traceback (most recent call last):\nModuleNotFoundError: No module named 'mere_runpod'"
        )

        let diagnosis = PluginInstallationFailureDiagnosis.make(
            plugin: plugin,
            install: install,
            inspection: inspection,
            missingEditableSourcePath: nil
        )

        XCTAssertEqual(diagnosis.summary, "plugin manifest verification failed")
        XCTAssertEqual(
            diagnosis.verificationError,
            "ModuleNotFoundError: No module named 'mere_runpod'"
        )
        XCTAssertEqual(
            diagnosis.repairCommand,
            "mere.run plugin install mere-runpod --yes --force"
        )
    }

    func testUnknownPluginErrorListsKnownPlugins() throws {
        let catalogURL = try writeCatalog()
        let catalog = try PluginCatalogClient.load(catalogURL: catalogURL.path)

        XCTAssertThrowsError(try catalog.requirePlugin("mere-unknown")) { error in
            let message = String(describing: error)
            XCTAssertTrue(message.contains("Unknown plugin: mere-unknown"))
            XCTAssertTrue(message.contains("mere-runpod"))
        }
    }

    func testGraphProviderCatalogIsPinnedToVerifiedPluginIdentity() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("PluginGraphProviderTests.\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let executable = directory.appendingPathComponent("mere-fixture-tools")
        try fixtureGraphProviderScript.write(to: executable, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: executable.path)

        let provider = try WorkflowGraphProviderRegistry.loadProvider(entrypoint: executable.path)

        XCTAssertEqual(provider.identity.id, "mere-fixture-tools")
        XCTAssertEqual(provider.identity.version, "1.2.3")
        XCTAssertEqual(provider.identity.catalogSHA256.count, 64)
        XCTAssertEqual(provider.nodes.map(\.kind), ["fixture.prepare"])
        XCTAssertEqual(provider.nodes[0].outputs.map(\.type), [.assetDirectory, .json])
        XCTAssertTrue(provider.nodes[0].traits.deterministic)
        XCTAssertEqual(provider.requirement.nodeKinds, ["fixture.prepare"])
    }

    private func writeCatalog(_ json: String = catalogJSON) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("PluginCommandTests.\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let catalogURL = directory.appendingPathComponent("plugins.v1.json")
        try json.write(to: catalogURL, atomically: true, encoding: .utf8)
        addTeardownBlock {
            try? FileManager.default.removeItem(at: directory)
        }
        return catalogURL
    }

    private func workflowInstallCommand(force: Bool) -> PluginInstallCommand {
        PluginInstallCommand(
            install: PluginCatalogInstall(
                manager: "pipx",
                spec: "git+https://github.com/sawfwair/mere-run-plugins.git@main#subdirectory=packages/mere-workflow-tools",
                ref: "main"
            ),
            force: force
        )
    }

    private func pipxCapture(
        metadata: Data,
        version: String
    ) -> (String, [String]) throws -> Data {
        { executable, arguments in
            XCTAssertEqual(executable, "pipx")
            switch arguments {
            case ["list", "--json"]: return metadata
            case ["--version"]: return Data(version.utf8)
            default:
                XCTFail("Unexpected pipx probe: \(arguments)")
                return Data()
            }
        }
    }

    private func pipxMetadata(backend: String?, package: String = "mere-workflow-tools") -> Data {
        let backendField = backend.map { "\"backend\": \"\($0)\"," } ?? ""
        return Data("""
        {"venvs": {"\(package)": {"metadata": {
          \(backendField)
          "main_package": {
            "package": "\(package)",
            "package_or_url": "\(package)",
            "pip_args": []
          }
        }}}}
        """.utf8)
    }
}

private let fixtureGraphProviderScript = #"""
#!/bin/sh
if [ "$1" = "manifest" ]; then
  printf '%s\n' '{"contractVersion":"mere.run/plugin.v1","name":"mere-fixture-tools","version":"1.2.3","graphProvider":{"contractVersion":"mere.run/plugin-graph-provider.v1"}}'
  exit 0
fi
if [ "$1" = "graph" ] && [ "$2" = "catalog" ]; then
  printf '%s\n' '{"contract_version":"mere.run/plugin-graph-provider.v1","provider_id":"mere-fixture-tools","provider_version":"1.2.3","nodes":[{"kind":"fixture.prepare","title":"Prepare fixture","description":"Prepare deterministic fixture data.","category":"fixture","inputs":[{"name":"data","type":"asset_directory","required":true}],"outputs":[{"name":"dataset","type":"asset_directory","optional":false,"content_types":["application/vnd.mere.dataset"]},{"name":"stats","type":"json","optional":false}],"requirements":{"model_ids":[],"accelerator_backends":["cpu"],"minimum_accelerator_memory_bytes":null},"traits":{"deterministic":true,"cacheable":true,"side_effects":"none","supports_progress":true,"supports_previews":false}}]}'
  exit 0
fi
exit 2
"""#

private let staleEditablePipxJSON = #"""
{
  "pipx_spec_version": "0.1",
  "venvs": {
    "mere-runpod": {
      "metadata": {
        "main_package": {
          "package": "mere-runpod",
          "package_or_url": "/Users/test/mere/mere-plugins/packages/mere-runpod",
          "pip_args": ["--editable"]
        }
      }
    }
  }
}
"""#

private let catalogJSON = """
{
  "contractVersion": "mere.run/plugin-catalog.v1",
  "updatedAt": "2026-06-26T11:45:00Z",
  "defaultChannel": "main",
  "plugins": [
    {
      "id": "mere-runpod",
      "name": "RunPod Runner",
      "description": "Run canonical mere.run recipes on user-owned ephemeral RunPod pods.",
      "repo": "https://github.com/sawfwair/mere-run-plugins",
      "package": "mere-runpod",
      "subdirectory": "packages/mere-runpod",
      "entrypoint": "mere-runpod",
      "capabilities": ["remote-runner", "runpod", "train-lora"],
      "channels": {
        "main": {
          "manager": "pipx",
          "ref": "main",
          "spec": "git+https://github.com/sawfwair/mere-run-plugins.git@main#subdirectory=packages/mere-runpod"
        }
      }
    }
  ]
}
"""

private func setupCatalogJSON(bundle: Bool) -> String {
    let bundles = bundle ? #","bundles": {"macos-arm64": "https://example.com/mere-computer-use/release.json"}"# : ""
    return """
    {
      "contractVersion": "mere.run/plugin-catalog.v1",
      "updatedAt": "2026-09-24T00:00:00Z",
      "defaultChannel": "main",
      "plugins": [
        {
          "id": "mere-computer-use",
          "name": "Computer Use",
          "description": "Drive macOS apps through a local driver.",
          "repo": "https://github.com/sawfwair/mere-run-plugins",
          "package": "mere-computer-use",
          "subdirectory": "packages/mere-computer-use",
          "entrypoint": "mere-computer-use",
          "capabilities": ["computer-use"],
          "channels": {
            "main": {
              "manager": "pipx",
              "ref": "main",
              "spec": "git+https://github.com/sawfwair/mere-run-plugins.git@main#subdirectory=packages/mere-computer-use",
              "setup": true\(bundles)
            }
          }
        }
      ]
    }
    """
}
