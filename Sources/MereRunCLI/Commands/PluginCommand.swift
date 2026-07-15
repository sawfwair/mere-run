import ArgumentParser
import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

struct Plugin: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "plugin",
        abstract: "Discover and install official mere.run companion plugins.",
        subcommands: [
            PluginList.self,
            PluginInfo.self,
            PluginInstall.self,
            PluginDoctor.self,
        ]
    )
}

struct PluginList: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "list",
        abstract: "List official plugins from the live catalog."
    )

    @Option(name: [.long], help: "Plugin catalog URL or local JSON path.")
    var catalogURL: String?

    @Flag(name: [.long], help: "Print the catalog JSON.")
    var json: Bool = false

    func run() throws {
        let catalog = try PluginCatalogClient.load(catalogURL: catalogURL)
        if json {
            print(try PluginCatalogClient.renderJSON(catalog))
            return
        }

        print("Catalog: \(catalog.source)")
        print("Default channel: \(catalog.defaultChannel)")
        print("")
        for plugin in catalog.plugins {
            let install = try plugin.install(channel: catalog.defaultChannel)
            print("\(plugin.id)")
            print("  \(plugin.description)")
            print("  command: \(plugin.entrypoint)")
            print("  install: \(PluginInstallCommand.render(install: install))")
            print("")
        }
    }
}

struct PluginInfo: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "info",
        abstract: "Show one plugin's catalog entry and install command."
    )

    @Argument(help: "Plugin id, for example mere-runpod.")
    var id: String

    @Option(name: [.long], help: "Plugin catalog URL or local JSON path.")
    var catalogURL: String?

    @Option(name: [.long], help: "Install channel. Defaults to the catalog default channel.")
    var channel: String?

    @Flag(name: [.long], help: "Print the plugin catalog entry as JSON.")
    var json: Bool = false

    func run() throws {
        let catalog = try PluginCatalogClient.load(catalogURL: catalogURL)
        let plugin = try catalog.requirePlugin(id)
        let resolvedChannel = channel ?? catalog.defaultChannel
        let install = try plugin.install(channel: resolvedChannel)
        if json {
            print(try PluginCatalogClient.renderJSON(plugin))
            return
        }

        print("\(plugin.id): \(plugin.name)")
        print(plugin.description)
        print("")
        print("Repository: \(plugin.repo)")
        print("Package: \(plugin.package)")
        print("Entrypoint: \(plugin.entrypoint)")
        print("Capabilities: \(plugin.capabilities.joined(separator: ", "))")
        print("Channel: \(resolvedChannel)")
        print("")
        print("Install command:")
        print("  \(PluginInstallCommand.render(install: install))")
    }
}

struct PluginInstall: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "install",
        abstract: "Install one official plugin using its catalog install command."
    )

    @Argument(help: "Plugin id, for example mere-runpod.")
    var id: String

    @Option(name: [.long], help: "Plugin catalog URL or local JSON path.")
    var catalogURL: String?

    @Option(name: [.long], help: "Install channel. Defaults to the catalog default channel.")
    var channel: String?

    @Flag(name: [.long], help: "Run the install command. Without --yes, print the exact command only.")
    var yes: Bool = false

    @Flag(name: [.long], help: "Pass --force to pipx install.")
    var force: Bool = false

    func run() throws {
        let catalog = try PluginCatalogClient.load(catalogURL: catalogURL)
        let plugin = try catalog.requirePlugin(id)
        let resolvedChannel = channel ?? catalog.defaultChannel
        let install = try plugin.install(channel: resolvedChannel)
        let command = PluginInstallCommand(install: install, force: force)

        if !yes {
            print("Install command:")
            print("  \(command.render())")
            print("")
            print("Run with --yes to execute it:")
            print("  \(CLICommandDisplay.command("plugin install \(ShellCommand.quote(id)) --channel \(ShellCommand.quote(resolvedChannel)) --yes"))")
            return
        }

        try command.run()
        let manifest = try PluginVerifier.verify(entrypoint: plugin.entrypoint)
        guard manifest.name == plugin.id else {
            throw ValidationError(
                "Installed plugin manifest name mismatch: expected \(plugin.id), got \(manifest.name)"
            )
        }
        if manifest.graphProvider != nil {
            try WorkflowGraphProviderRegistry.register(entrypoint: plugin.entrypoint)
        }
        print("Installed \(plugin.id) with \(install.manager).")
        print("Verified \(plugin.entrypoint) manifest version \(manifest.version).")
    }
}

struct PluginDoctor: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "doctor",
        abstract: "Run an installed plugin's doctor command."
    )

    @Argument(help: "Plugin id, for example mere-runpod.")
    var id: String

    @Option(name: [.long], help: "Plugin catalog URL or local JSON path.")
    var catalogURL: String?

    func run() throws {
        let catalog = try PluginCatalogClient.load(catalogURL: catalogURL)
        let plugin = try catalog.requirePlugin(id)
        guard PluginProcess.which(plugin.entrypoint) != nil else {
            let install = try plugin.install(channel: catalog.defaultChannel)
            throw ValidationError(
                """
                Plugin \(plugin.id) is not installed on PATH.
                Install it with: \(PluginInstallCommand.render(install: install))
                """
            )
        }
        try PluginProcess.runExecutable(plugin.entrypoint, arguments: ["doctor"])
    }
}

struct PluginCatalog: Codable {
    let contractVersion: String
    let updatedAt: String
    let defaultChannel: String
    let plugins: [PluginCatalogEntry]
    var source: String = PluginCatalogClient.defaultCatalogURL

    enum CodingKeys: String, CodingKey {
        case contractVersion
        case updatedAt
        case defaultChannel
        case plugins
    }

    func requirePlugin(_ id: String) throws -> PluginCatalogEntry {
        if let plugin = plugins.first(where: { $0.id == id }) {
            return plugin
        }
        let known = plugins.map(\.id).sorted().joined(separator: ", ")
        throw ValidationError("Unknown plugin: \(id). Known plugins: \(known)")
    }
}

struct PluginCatalogEntry: Codable {
    let id: String
    let name: String
    let description: String
    let repo: String
    let package: String
    let subdirectory: String
    let entrypoint: String
    let capabilities: [String]
    let channels: [String: PluginCatalogInstall]

    func install(channel: String) throws -> PluginCatalogInstall {
        guard let install = channels[channel] else {
            let known = channels.keys.sorted().joined(separator: ", ")
            throw ValidationError("Plugin \(id) has no channel '\(channel)'. Available channels: \(known)")
        }
        return install
    }
}

struct PluginCatalogInstall: Codable {
    let manager: String
    let spec: String
    let ref: String?
}

struct PluginManifest: Decodable {
    let contractVersion: String
    let name: String
    let version: String
    let graphProvider: PluginGraphProviderDeclaration?

    enum CodingKeys: String, CodingKey {
        case contractVersion
        case name
        case version
        case graphProvider
    }
}

struct PluginGraphProviderDeclaration: Decodable {
    let contractVersion: String
}

enum PluginCatalogClient {
    static let defaultCatalogURL = "https://raw.githubusercontent.com/sawfwair/mere-run-plugins/main/catalog/plugins.v1.json"

    static func load(catalogURL: String?) throws -> PluginCatalog {
        let source = catalogURL?.trimmingCharacters(in: .whitespacesAndNewlines)
        let resolved = try resolveCatalogURL(source)
        let data = try loadCatalogData(from: resolved)
        let decoder = JSONDecoder()
        var catalog = try decoder.decode(PluginCatalog.self, from: data)
        catalog.source = source?.isEmpty == false ? source! : defaultCatalogURL
        guard catalog.contractVersion == "mere.run/plugin-catalog.v1" else {
            throw ValidationError("Unsupported plugin catalog contract: \(catalog.contractVersion)")
        }
        return catalog
    }

    static func renderJSON<T: Encodable>(_ value: T) throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(value)
        guard let output = String(data: data, encoding: .utf8) else {
            throw ValidationError("Could not render JSON as UTF-8.")
        }
        return output
    }

    private static func resolveCatalogURL(_ raw: String?) throws -> URL {
        guard let raw, !raw.isEmpty else {
            return URL(string: defaultCatalogURL)!
        }
        if raw.hasPrefix("http://") || raw.hasPrefix("https://") || raw.hasPrefix("file://") {
            guard let url = URL(string: raw) else {
                throw ValidationError("Invalid catalog URL: \(raw)")
            }
            return url
        }
        return URL(fileURLWithPath: NSString(string: raw).expandingTildeInPath)
    }

    private static func loadCatalogData(from url: URL) throws -> Data {
        if url.scheme == "http" || url.scheme == "https" {
            return try loadRemoteCatalogData(from: url)
        }
        return try Data(contentsOf: url)
    }

    private static func loadRemoteCatalogData(from url: URL) throws -> Data {
        var request = URLRequest(url: url)
        request.timeoutInterval = 30
        request.cachePolicy = .reloadIgnoringLocalCacheData

        let semaphore = DispatchSemaphore(value: 0)
        let result = PluginCatalogRequestResult()
        let task = URLSession.shared.dataTask(with: request) { data, response, error in
            defer { semaphore.signal() }
            if let error {
                result.store(.failure(error))
                return
            }
            guard let data, let response else {
                result.store(.failure(ValidationError("Plugin catalog request returned no response.")))
                return
            }
            result.store(.success((data, response)))
        }
        task.resume()
        semaphore.wait()

        guard let storedResult = result.load() else {
            throw ValidationError("Plugin catalog request did not complete.")
        }
        let (data, response) = try storedResult.get()
        if let httpResponse = response as? HTTPURLResponse,
           !(200..<300).contains(httpResponse.statusCode) {
            throw ValidationError("Plugin catalog request failed with HTTP \(httpResponse.statusCode): \(url.absoluteString)")
        }
        guard !data.isEmpty else {
            throw ValidationError("Plugin catalog response was empty: \(url.absoluteString)")
        }
        return data
    }
}

final class PluginCatalogRequestResult: @unchecked Sendable {
    private let lock = NSLock()
    private var result: Result<(Data, URLResponse), Error>?

    func store(_ newResult: Result<(Data, URLResponse), Error>) {
        lock.lock()
        result = newResult
        lock.unlock()
    }

    func load() -> Result<(Data, URLResponse), Error>? {
        lock.lock()
        defer { lock.unlock() }
        return result
    }
}

struct PluginInstallCommand {
    let install: PluginCatalogInstall
    let force: Bool

    init(install: PluginCatalogInstall, force: Bool = false) {
        self.install = install
        self.force = force
    }

    static func render(install: PluginCatalogInstall) -> String {
        PluginInstallCommand(install: install).render()
    }

    func render() -> String {
        switch install.manager {
        case "pipx":
            var parts = ["pipx", "install"]
            if force {
                parts.append("--force")
            }
            parts.append(install.spec)
            return parts.map(ShellCommand.quote).joined(separator: " ")
        default:
            return "\(install.manager) \(ShellCommand.quote(install.spec))"
        }
    }

    func run() throws {
        guard install.manager == "pipx" else {
            throw ValidationError("Unsupported plugin install manager: \(install.manager)")
        }
        guard PluginProcess.which("pipx") != nil else {
            throw ValidationError(
                """
                pipx is required to install this plugin.
                Install pipx first, then retry: brew install pipx
                """
            )
        }
        var args = ["install"]
        if force {
            args.append("--force")
        }
        args.append(install.spec)
        try PluginProcess.runExecutable("pipx", arguments: args)
    }
}

enum PluginVerifier {
    static func verify(entrypoint: String) throws -> PluginManifest {
        let data = try PluginProcess.captureExecutable(entrypoint, arguments: ["manifest", "--json"])
        let manifest = try JSONDecoder().decode(PluginManifest.self, from: data)
        guard manifest.contractVersion == "mere.run/plugin.v1" else {
            throw ValidationError("Unsupported plugin manifest contract: \(manifest.contractVersion)")
        }
        return manifest
    }
}

enum PluginProcess {
    static func which(_ name: String) -> URL? {
        if name.contains("/") {
            let url = URL(fileURLWithPath: NSString(string: name).expandingTildeInPath).standardizedFileURL
            guard FileManager.default.isExecutableFile(atPath: url.path) else { return nil }
            return url
        }
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/which")
        process.arguments = [name]
        let output = Pipe()
        process.standardOutput = output
        process.standardError = Pipe()
        try? process.run()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else { return nil }
        let data = output.fileHandleForReading.readDataToEndOfFile()
        guard let raw = String(data: data, encoding: .utf8)?
            .trimmingCharacters(in: .whitespacesAndNewlines),
            !raw.isEmpty else { return nil }
        return URL(fileURLWithPath: raw)
    }

    static func runExecutable(_ executable: String, arguments: [String]) throws {
        guard let url = which(executable) else {
            throw ValidationError("Executable not found on PATH: \(executable)")
        }
        let process = Process()
        process.executableURL = url
        process.arguments = arguments
        process.standardInput = FileHandle.standardInput
        process.standardOutput = FileHandle.standardOutput
        process.standardError = FileHandle.standardError
        try process.run()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else {
            throw ValidationError("\(executable) exited with status \(process.terminationStatus)")
        }
    }

    static func captureExecutable(_ executable: String, arguments: [String]) throws -> Data {
        guard let url = which(executable) else {
            throw ValidationError("Executable not found on PATH: \(executable)")
        }
        let process = Process()
        process.executableURL = url
        process.arguments = arguments
        let output = Pipe()
        process.standardOutput = output
        process.standardError = FileHandle.standardError
        try process.run()
        process.waitUntilExit()
        let data = output.fileHandleForReading.readDataToEndOfFile()
        guard process.terminationStatus == 0 else {
            throw ValidationError("\(executable) exited with status \(process.terminationStatus)")
        }
        return data
    }
}

enum ShellCommand {
    static func quote(_ value: String) -> String {
        if value.isEmpty {
            return "''"
        }
        let safe = CharacterSet(charactersIn: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_+-./:=@#")
        if value.unicodeScalars.allSatisfy({ safe.contains($0) }) {
            return value
        }
        return "'" + value.replacingOccurrences(of: "'", with: "'\\''") + "'"
    }
}
