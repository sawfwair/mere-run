import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif
#if canImport(CryptoKit)
import CryptoKit
#endif
import MereRunCore

struct PiAgentInstallResult {
    let version: String
    let releaseURL: URL
    let binaryURL: URL
}

struct PiProviderModel {
    let id: String
    let name: String
    let contextWindow: Int
    let maxTokens: Int
    /// Whether the model supports a separate "thinking" / reasoning channel.
    let reasoning: Bool
    /// Provider-level OpenAI-compat flags. Matches the shape pi-coding-agent's
    /// provider catalog expects (see the ds4 README "For Pi" section).
    let supportsStore: Bool
    let supportsDeveloperRole: Bool
    let supportsReasoningEffort: Bool
    let supportsUsageInStreaming: Bool
    let supportsStrictMode: Bool
    /// "max_tokens" for legacy OpenAI servers, "max_completion_tokens" for newer.
    let maxTokensField: String
    /// e.g. "deepseek" for the DSML thinking block format. nil → omit field.
    let thinkingFormat: String?
    /// DSML servers require the original `reasoning_content` to be sent back on
    /// follow-up assistant messages so the transcript matches what was sampled.
    let requiresReasoningContentOnAssistantMessages: Bool
    /// Optional map from Pi thinking-level keys (off/minimal/low/medium/high/xhigh)
    /// to the provider's native effort label. `nil` value in JS means the model
    /// should be invoked without a reasoning_effort argument.
    let thinkingLevelMap: [(key: String, value: String?)]?

    init(
        id: String,
        name: String,
        contextWindow: Int,
        maxTokens: Int,
        reasoning: Bool = false,
        supportsStore: Bool = false,
        supportsDeveloperRole: Bool = false,
        supportsReasoningEffort: Bool = false,
        supportsUsageInStreaming: Bool = false,
        supportsStrictMode: Bool = false,
        maxTokensField: String = "max_tokens",
        thinkingFormat: String? = nil,
        requiresReasoningContentOnAssistantMessages: Bool = false,
        thinkingLevelMap: [(key: String, value: String?)]? = nil
    ) {
        self.id = id
        self.name = name
        self.contextWindow = contextWindow
        self.maxTokens = maxTokens
        self.reasoning = reasoning
        self.supportsStore = supportsStore
        self.supportsDeveloperRole = supportsDeveloperRole
        self.supportsReasoningEffort = supportsReasoningEffort
        self.supportsUsageInStreaming = supportsUsageInStreaming
        self.supportsStrictMode = supportsStrictMode
        self.maxTokensField = maxTokensField
        self.thinkingFormat = thinkingFormat
        self.requiresReasoningContentOnAssistantMessages = requiresReasoningContentOnAssistantMessages
        self.thinkingLevelMap = thinkingLevelMap
    }

    static let qwen3CoderNext = PiProviderModel(
        id: CodeGenResources.defaultModelId,
        name: "Qwen3-Coder Next (mere.run)",
        contextWindow: 32768,
        maxTokens: 4096
    )

    /// DeepSeek V4 Flash compat profile from the upstream ds4 README's
    /// "For Pi, add a provider to ~/.pi/agent/models.json" section.
    static let deepseekV4Flash = PiProviderModel(
        id: DeepseekV4FlashResources.defaultModelId,
        name: "DeepSeek V4 Flash (mere.run)",
        contextWindow: DeepseekV4FlashResources.defaultContextLength,
        maxTokens: DeepseekV4FlashResources.defaultContextLength,
        reasoning: true,
        supportsStore: false,
        supportsDeveloperRole: false,
        supportsReasoningEffort: true,
        supportsUsageInStreaming: true,
        supportsStrictMode: false,
        maxTokensField: "max_tokens",
        thinkingFormat: "deepseek",
        requiresReasoningContentOnAssistantMessages: true,
        thinkingLevelMap: [
            ("off", nil),
            ("minimal", "low"),
            ("low", "low"),
            ("medium", "medium"),
            ("high", "high"),
            ("xhigh", "xhigh"),
        ]
    )
}

enum PiAgentIntegration {
    struct ProviderConfiguration: Codable, Hashable {
        let host: String
        let port: Int
        let modelID: String
        let updatedAt: Date
    }

    enum IntegrationError: LocalizedError {
        case applicationSupportUnavailable
        case unsupportedPlatform
        case releaseAssetMissing(String)
        case digestMismatch(expected: String, actual: String)
        case sha256Unavailable
        case extractionFailed(String)
        case piBinaryMissing(URL)
        case piBinaryNotFound
        case serverDidNotBecomeReady(URL)
        case terminalLaunchFailed

        var errorDescription: String? {
            switch self {
            case .applicationSupportUnavailable:
                return "Could not locate Application Support directory."
            case .unsupportedPlatform:
                return "Pi auto-install is currently supported for macOS arm64 and x64 release assets. On Linux, install Pi separately and pass --pi-path or put pi on PATH."
            case .releaseAssetMissing(let name):
                return "Latest Pi release does not include expected asset: \(name)"
            case .digestMismatch(let expected, let actual):
                return "Pi download SHA-256 mismatch. Expected \(expected), got \(actual)."
            case .sha256Unavailable:
                return "SHA-256 verification is unavailable in this build."
            case .extractionFailed(let output):
                return "Failed to extract Pi release: \(output)"
            case .piBinaryMissing(let url):
                return "No executable named pi was found under \(url.path)."
            case .piBinaryNotFound:
                return "Pi is not installed. Run `mere.run agent install-pi` on macOS, or pass --pi-path / put pi on PATH on Linux."
            case .serverDidNotBecomeReady(let url):
                return "mere.run API server did not become ready at \(url.absoluteString)."
            case .terminalLaunchFailed:
                return "Could not open Pi in Terminal.app."
            }
        }
    }

    static var canInstallLatestRelease: Bool {
        #if os(macOS) && (arch(arm64) || arch(x86_64))
        return true
        #else
        return false
        #endif
    }

    static func installLatest(force: Bool = false, progress: (String) -> Void) async throws -> PiAgentInstallResult {
        progress("Fetching latest Pi release metadata")
        let release = try await fetchLatestRelease()
        let assetName = try platformAssetName()
        guard let asset = release.assets.first(where: { $0.name == assetName }) else {
            throw IntegrationError.releaseAssetMissing(assetName)
        }

        let root = try piInstallRoot()
        let versionRoot = root.appendingPathComponent(release.tagName, isDirectory: true)
        let currentLink = root.appendingPathComponent("current", isDirectory: false)
        let fileManager = FileManager.default

        if !force, let existingBinary = findPiBinary(in: versionRoot) {
            try pointCurrentSymlink(currentLink, to: versionRoot, fileManager: fileManager)
            return PiAgentInstallResult(
                version: release.tagName,
                releaseURL: release.htmlURL,
                binaryURL: existingBinary
            )
        }

        try fileManager.createDirectory(at: root, withIntermediateDirectories: true)
        if fileManager.fileExists(atPath: versionRoot.path) {
            try fileManager.removeItem(at: versionRoot)
        }
        try fileManager.createDirectory(at: versionRoot, withIntermediateDirectories: true)

        progress("Downloading \(asset.name)")
        let archiveURL = fileManager.temporaryDirectory
            .appendingPathComponent("mere-run-pi-\(UUID().uuidString).tar.gz", isDirectory: false)
        let (data, _) = try await URLSession.shared.data(from: asset.browserDownloadURL)
        try data.write(to: archiveURL, options: .atomic)

        if let digest = asset.digest?.trimmingCharacters(in: .whitespacesAndNewlines),
           digest.hasPrefix("sha256:") {
            let expected = String(digest.dropFirst("sha256:".count)).lowercased()
            let actual = try sha256Hex(data)
            guard expected == actual else {
                throw IntegrationError.digestMismatch(expected: expected, actual: actual)
            }
        }

        progress("Extracting \(asset.name)")
        try extractTarGzip(archiveURL: archiveURL, destinationURL: versionRoot)
        try? fileManager.removeItem(at: archiveURL)

        guard let binaryURL = findPiBinary(in: versionRoot) else {
            throw IntegrationError.piBinaryMissing(versionRoot)
        }
        try pointCurrentSymlink(currentLink, to: versionRoot, fileManager: fileManager)

        return PiAgentInstallResult(
            version: release.tagName,
            releaseURL: release.htmlURL,
            binaryURL: binaryURL
        )
    }

    static func writeLocalProviderExtension(
        host: String,
        port: Int,
        model: PiProviderModel = .qwen3CoderNext,
        apiKey: String = "mere-run",
        homeDirectory: URL? = nil,
        persistConfiguration: Bool = true,
        fileManager: FileManager = .default
    ) throws -> URL {
        let home = homeDirectory ?? fileManager.homeDirectoryForCurrentUser
        let extensionDir = home
            .appendingPathComponent(".pi/agent/extensions", isDirectory: true)
        try fileManager.createDirectory(at: extensionDir, withIntermediateDirectories: true)

        let extensionURL = extensionDir.appendingPathComponent("mere-run-local-provider.ts", isDirectory: false)
        let baseURL = "http://\(host):\(port)/v1"
        let content = renderPiProviderExtension(model: model, baseURL: baseURL, apiKey: apiKey)
        try content.write(to: extensionURL, atomically: true, encoding: .utf8)
        if persistConfiguration {
            try writeProviderConfiguration(
                ProviderConfiguration(
                    host: host,
                    port: port,
                    modelID: model.id,
                    updatedAt: Date()
                ),
                fileManager: fileManager
            )
        }
        return extensionURL
    }

    /// Render the pi-coding-agent extension TS for a single mere.run provider.
    /// Mirrors the JSON provider catalog shape documented in the ds4 README
    /// ("For Pi, add a provider to ~/.pi/agent/models.json").
    static func renderPiProviderExtension(
        model: PiProviderModel,
        baseURL: String,
        apiKey: String
    ) -> String {
        var providerCompat: [String] = []
        providerCompat.append("supportsStore: \(model.supportsStore)")
        providerCompat.append("supportsDeveloperRole: \(model.supportsDeveloperRole)")
        providerCompat.append("supportsReasoningEffort: \(model.supportsReasoningEffort)")
        providerCompat.append("supportsUsageInStreaming: \(model.supportsUsageInStreaming)")
        providerCompat.append("maxTokensField: \"\(model.maxTokensField)\"")
        providerCompat.append("supportsStrictMode: \(model.supportsStrictMode)")
        if let format = model.thinkingFormat {
            providerCompat.append("thinkingFormat: \"\(format)\"")
        }
        if model.requiresReasoningContentOnAssistantMessages {
            providerCompat.append("requiresReasoningContentOnAssistantMessages: true")
        }
        let compatBlock = providerCompat
            .map { "      \($0)" }
            .joined(separator: ",\n")

        var modelLines: [String] = []
        modelLines.append("id: \"\(model.id)\"")
        modelLines.append("name: \"\(model.name)\"")
        modelLines.append("reasoning: \(model.reasoning)")
        if let map = model.thinkingLevelMap, !map.isEmpty {
            let entries = map.map { entry -> String in
                if let value = entry.value {
                    return "          \(entry.key): \"\(value)\""
                }
                return "          \(entry.key): null"
            }.joined(separator: ",\n")
            modelLines.append("thinkingLevelMap: {\n\(entries)\n        }")
        }
        modelLines.append("input: [\"text\"]")
        modelLines.append("contextWindow: \(model.contextWindow)")
        modelLines.append("maxTokens: \(model.maxTokens)")
        modelLines.append("""
        cost: {
                  input: 0,
                  output: 0,
                  cacheRead: 0,
                  cacheWrite: 0
                }
        """)
        let modelBody = modelLines
            .map { "        \($0)" }
            .joined(separator: ",\n")

        return """
        import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";

        export default function(pi: ExtensionAPI) {
          pi.registerProvider("mere-run", {
            name: "mere.run Local",
            baseUrl: "\(baseURL)",
            api: "openai-completions",
            apiKey: "\(apiKey)",
            compat: {
        \(compatBlock)
            },
            models: [
              {
        \(modelBody)
              }
            ]
          });
        }
        """
    }

    static func mereRunPiHomeDirectory() -> URL {
        MereRunModelPaths.applicationSupportBase
            .appendingPathComponent("agent", isDirectory: true)
            .appendingPathComponent("pi-home", isDirectory: true)
    }

    static func readProviderConfiguration(fileManager: FileManager = .default) -> ProviderConfiguration? {
        let url = providerConfigurationURL()
        guard let data = try? Data(contentsOf: url) else {
            return nil
        }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try? decoder.decode(ProviderConfiguration.self, from: data)
    }

    static func findPiExecutable(explicitPath: String? = nil) -> URL? {
        if let explicitPath, !explicitPath.isEmpty {
            let url = URL(fileURLWithPath: (explicitPath as NSString).expandingTildeInPath)
            return isExecutableRegularFile(url) ? url : nil
        }
        if let installed = installedPiBinaryURL() {
            return installed
        }
        return executableOnPath(named: "pi")
    }

    static func installedPiBinaryURL() -> URL? {
        guard let root = try? piInstallRoot() else { return nil }
        let current = root
            .appendingPathComponent("current", isDirectory: true)
            .resolvingSymlinksInPath()
        return findPiBinary(in: current)
    }

    private static func writeProviderConfiguration(
        _ configuration: ProviderConfiguration,
        fileManager: FileManager
    ) throws {
        let url = providerConfigurationURL()
        try fileManager.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        let data = try encoder.encode(configuration)
        try data.write(to: url, options: .atomic)
    }

    private static func providerConfigurationURL() -> URL {
        MereRunModelPaths.applicationSupportBase
            .appendingPathComponent("agent", isDirectory: true)
            .appendingPathComponent("provider.json", isDirectory: false)
    }

    static func waitForHealth(host: String, port: Int, timeoutSeconds: TimeInterval) async throws {
        let deadline = Date().addingTimeInterval(timeoutSeconds)
        let healthURL = URL(string: "http://\(host):\(port)/health")!
        while Date() < deadline {
            if let (_, response) = try? await URLSession.shared.data(from: healthURL),
               let http = response as? HTTPURLResponse,
               http.statusCode == 200 {
                return
            }
            try await Task.sleep(nanoseconds: 500_000_000)
        }
        throw IntegrationError.serverDidNotBecomeReady(healthURL)
    }

    private static func fetchLatestRelease() async throws -> GitHubRelease {
        let url = URL(string: "https://api.github.com/repos/badlogic/pi-mono/releases/latest")!
        let (data, _) = try await URLSession.shared.data(from: url)
        return try JSONDecoder().decode(GitHubRelease.self, from: data)
    }

    private static func platformAssetName() throws -> String {
        #if os(macOS) && arch(arm64)
        return "pi-darwin-arm64.tar.gz"
        #elseif os(macOS) && arch(x86_64)
        return "pi-darwin-x64.tar.gz"
        #else
            throw IntegrationError.unsupportedPlatform
        #endif
    }

    private static func piInstallRoot() throws -> URL {
        guard let applicationSupport = FileManager.default.urls(
            for: .applicationSupportDirectory,
            in: .userDomainMask
        ).first else {
            throw IntegrationError.applicationSupportUnavailable
        }
        return applicationSupport.appendingPathComponent("MereRun/agents/pi", isDirectory: true)
    }

    private static func pointCurrentSymlink(
        _ currentLink: URL,
        to versionRoot: URL,
        fileManager: FileManager
    ) throws {
        if fileManager.fileExists(atPath: currentLink.path) {
            try fileManager.removeItem(at: currentLink)
        }
        try fileManager.createSymbolicLink(at: currentLink, withDestinationURL: versionRoot)
    }

    static func findPiBinary(in root: URL, fileManager: FileManager = .default) -> URL? {
        guard fileManager.fileExists(atPath: root.path) else { return nil }
        let enumerator = fileManager.enumerator(
            at: root,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        )
        while let candidate = enumerator?.nextObject() as? URL {
            guard candidate.lastPathComponent == "pi" else { continue }
            if isExecutableRegularFile(candidate, fileManager: fileManager) {
                return candidate
            }
        }
        return nil
    }

    static func isExecutableRegularFile(
        _ url: URL,
        fileManager: FileManager = .default
    ) -> Bool {
        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory),
              !isDirectory.boolValue else {
            return false
        }
        return fileManager.isExecutableFile(atPath: url.path)
    }

    private static func executableOnPath(named name: String) -> URL? {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/which")
        process.arguments = [name]

        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = Pipe()
        try? process.run()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else { return nil }

        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        guard let raw = String(data: data, encoding: .utf8)?
            .trimmingCharacters(in: .whitespacesAndNewlines),
            !raw.isEmpty else {
            return nil
        }
        let url = URL(fileURLWithPath: raw)
        return isExecutableRegularFile(url) ? url : nil
    }

    private static func sha256Hex(_ data: Data) throws -> String {
        #if canImport(CryptoKit)
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        #else
        throw IntegrationError.sha256Unavailable
        #endif
    }

    private static func extractTarGzip(archiveURL: URL, destinationURL: URL) throws {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/tar")
        process.arguments = ["-xzf", archiveURL.path, "-C", destinationURL.path]

        let stdout = Pipe()
        let stderr = Pipe()
        process.standardOutput = stdout
        process.standardError = stderr

        try process.run()
        process.waitUntilExit()

        guard process.terminationStatus == 0 else {
            let output = String(data: stdout.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
            let error = String(data: stderr.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
            throw IntegrationError.extractionFailed(output + error)
        }
    }
}

private struct GitHubRelease: Decodable {
    let tagName: String
    let htmlURL: URL
    let assets: [GitHubReleaseAsset]

    enum CodingKeys: String, CodingKey {
        case tagName = "tag_name"
        case htmlURL = "html_url"
        case assets
    }
}

private struct GitHubReleaseAsset: Decodable {
    let name: String
    let browserDownloadURL: URL
    let digest: String?

    enum CodingKeys: String, CodingKey {
        case name
        case browserDownloadURL = "browser_download_url"
        case digest
    }
}
