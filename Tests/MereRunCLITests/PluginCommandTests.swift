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

    func testUnknownPluginErrorListsKnownPlugins() throws {
        let catalogURL = try writeCatalog()
        let catalog = try PluginCatalogClient.load(catalogURL: catalogURL.path)

        XCTAssertThrowsError(try catalog.requirePlugin("mere-unknown")) { error in
            let message = String(describing: error)
            XCTAssertTrue(message.contains("Unknown plugin: mere-unknown"))
            XCTAssertTrue(message.contains("mere-runpod"))
        }
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
