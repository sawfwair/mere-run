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

        XCTAssertEqual(commandNames, Set(["list", "info", "install", "doctor"]))
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

    private func writeCatalog() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("PluginCommandTests.\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let catalogURL = directory.appendingPathComponent("plugins.v1.json")
        try catalogJSON.write(to: catalogURL, atomically: true, encoding: .utf8)
        addTeardownBlock {
            try? FileManager.default.removeItem(at: directory)
        }
        return catalogURL
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
