import ArgumentParser
import Foundation

enum PluginBundleInstaller {
    static var platform: String {
        #if os(macOS) && arch(arm64)
        return "macos-arm64"
        #else
        return "unsupported"
        #endif
    }

    static func install(plugin: PluginCatalogEntry, source: String, archive: String?, allowLocalManifest: Bool = false) throws {
        guard platform == "macos-arm64" else {
            throw ValidationError("This bundle requires macOS on Apple Silicon. Use --source for an explicit pipx installation.")
        }
        let temporary = FileManager.default.temporaryDirectory.appendingPathComponent("mere-plugin-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: false,
                                                attributes: [.posixPermissions: 0o700])
        defer { try? FileManager.default.removeItem(at: temporary) }
        let receipt = temporary.appendingPathComponent("release.json")
        try PluginBundleIO.fetch(source, to: receipt, limit: 65_536, local: allowLocalManifest)
        let data = try Data(contentsOf: receipt)
        let envelope = try JSONDecoder().decode(PluginBundleEnvelope.self, from: data)
        let manifest = try envelope.verified()
        guard manifest.package == plugin.package, manifest.entrypoints[plugin.id] == plugin.entrypoint else {
            throw ValidationError("Signed release identity does not match the catalog plugin.")
        }
        let payload = temporary.appendingPathComponent("bundle.dmg")
        try PluginBundleIO.fetch(archive ?? manifest.artifact.url, to: payload,
                                 limit: manifest.artifact.size, local: archive != nil)
        _ = try PluginBundleStore.standard.install(envelopeData: data, archive: payload,
                                                   package: plugin.package, pluginID: plugin.id)
        print("Installed signed bundle \(plugin.package) \(manifest.version).")
        print("Verified publisher signature, macOS notarization, and bundled plugin entrypoints.")
    }
}

struct PluginRollback: ParsableCommand {
    static let configuration = CommandConfiguration(commandName: "rollback", abstract: "Restore a retained signed plugin bundle.")
    @Argument(help: "Plugin id, for example mere-doc-tools.") var id: String
    @Flag(name: .long, help: "Activate the retained version. Without --yes, print the plan.") var yes = false

    func run() throws {
        guard let package = try PluginBundleStore.standard.package(containing: id) else {
            throw ValidationError("No managed bundle is installed for \(id).")
        }
        guard yes else {
            print("Restore the previous verified bundle for \(package). Run again with --yes to activate it.")
            return
        }
        let manifest = try PluginBundleStore.standard.rollback(package: package)
        print("Restored \(manifest.package) \(manifest.version).")
    }
}

struct PluginRun: ParsableCommand {
    static let configuration = CommandConfiguration(commandName: "run", abstract: "Run an installed plugin without changing PATH.")
    @Argument(help: "Plugin entrypoint, for example mere-doc-tools.") var entrypoint: String
    @Argument(parsing: .captureForPassthrough, help: "Arguments forwarded to the plugin.") var arguments: [String] = []

    func run() throws {
        guard PluginBundleManifest.matches(entrypoint, "^mere-[a-z0-9-]+$") else {
            throw ValidationError("Expected a plugin entrypoint such as mere-doc-tools.")
        }
        try PluginProcess.runExecutable(entrypoint, arguments: arguments.first == "--" ? Array(arguments.dropFirst()) : arguments)
    }
}
