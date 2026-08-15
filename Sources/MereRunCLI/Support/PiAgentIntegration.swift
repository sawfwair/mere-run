import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif
#if canImport(CryptoKit)
import CryptoKit
#endif
import MereRunCore
#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

struct PiAgentInstallResult {
    let version: String
    let releaseURL: URL
    let binaryURL: URL
}

enum PiProviderModelError: LocalizedError {
    case invalidCatalogProfile(String)

    var errorDescription: String? {
        switch self {
        case .invalidCatalogProfile(let modelID):
            return "Catalog model '\(modelID)' does not define a complete chat API profile."
        }
    }
}

struct PiProviderModel {
    let id: String
    let name: String
    let contextWindow: Int
    let maxTokens: Int
    /// Input modalities Pi can represent for this model.
    let inputModalities: [String]
    /// Whether the model supports a separate "thinking" / reasoning channel.
    let reasoning: Bool
    /// Whether the API/model pair can execute Pi's function tools.
    let toolCall: Bool
    /// Provider-level OpenAI-compat flags. Matches the shape pi-coding-agent's
    /// provider catalog expects (see the ds4 README "For Pi" section).
    let supportsStore: Bool
    let supportsDeveloperRole: Bool
    let supportsReasoningEffort: Bool
    let supportsUsageInStreaming: Bool
    let supportsFinishReason: Bool
    let supportsStrictMode: Bool
    /// "max_tokens" for legacy OpenAI servers, "max_completion_tokens" for newer.
    let maxTokensField: String
    /// e.g. "deepseek" for the DSML thinking block format. nil → omit field.
    let thinkingFormat: String?
    /// DSML servers require the original `reasoning_content` to be sent back on
    /// follow-up assistant messages so the transcript matches what was sampled.
    let requiresReasoningContentOnAssistantMessages: Bool
    /// Optional map from Pi thinking-level keys to the provider's native effort
    /// label. A nil value is rendered as null, which marks that level unsupported.
    let thinkingLevelMap: [(key: String, value: String?)]?

    init(
        id: String,
        name: String,
        profile: ManagedModelAPIProfile
    ) throws {
        guard profile.task == .chatCompletions,
              let contextWindow = profile.contextWindow,
              let maxTokens = profile.maximumOutputTokens else {
            throw PiProviderModelError.invalidCatalogProfile(id)
        }
        let supportedThinkingLevels = Set(profile.thinkingLevels)
        let thinkingLevelMap = profile.reasoning
            ? ManagedModelThinkingLevel.allCases.compactMap { level -> (key: String, value: String?)? in
                guard supportedThinkingLevels.contains(level) else {
                    return (level.rawValue, nil)
                }
                guard let mapped = profile.thinkingLevelMap[level] else {
                    return nil
                }
                return (level.rawValue, mapped.rawValue)
            }
            : nil
        let compatibility = profile.compatibility
        self.id = id
        self.name = name
        self.contextWindow = contextWindow
        self.maxTokens = maxTokens
        self.inputModalities = profile.inputModalities.compactMap { modality in
            switch modality {
            case .text, .image:
                return modality.rawValue
            case .audio, .video, .embedding, .geometry, .threeD:
                return nil
            }
        }
        self.reasoning = profile.reasoning
        self.toolCall = profile.toolCall
        self.supportsStore = compatibility.supportsStore
        self.supportsDeveloperRole = compatibility.supportsDeveloperRole
        self.supportsReasoningEffort = compatibility.supportsReasoningEffort
        self.supportsUsageInStreaming = compatibility.supportsUsageInStreaming
        self.supportsFinishReason = compatibility.supportsFinishReason
        self.supportsStrictMode = compatibility.supportsStrictMode
        self.maxTokensField = compatibility.maxTokensField.rawValue
        self.thinkingFormat = compatibility.thinkingFormat?.rawValue
        self.requiresReasoningContentOnAssistantMessages = compatibility
            .requiresReasoningContentOnAssistantMessages
        self.thinkingLevelMap = thinkingLevelMap
    }
}

enum PiAgentIntegration {
    static let serverQueueMarker = "API server queued by machine admission"
    static let serverAdmissionMarker = "API server admitted by machine admission."
    static let agentParentProcessEnvironment = "MERERUN_AGENT_PARENT_PID"

    static func configuredAgentParentProcessID(
        environment: [String: String] = ProcessInfo.processInfo.environment,
        currentProcessID: Int32 = ProcessInfo.processInfo.processIdentifier
    ) -> Int32? {
        guard let raw = environment[agentParentProcessEnvironment],
              let processID = Int32(raw),
              processID > 1,
              processID != currentProcessID else {
            return nil
        }
        return processID
    }

    static func startAgentParentExitMonitorIfConfigured() {
        guard let parentProcessID = configuredAgentParentProcessID() else { return }
        Thread.detachNewThread {
            while true {
                Thread.sleep(forTimeInterval: 0.5)
                if kill(parentProcessID, 0) != 0, errno != EPERM {
                    _exit(EXIT_SUCCESS)
                }
            }
        }
    }

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
        model: PiProviderModel,
        apiKey: String = "mere-run",
        homeDirectory: URL? = nil,
        persistConfiguration: Bool = true,
        fileManager: FileManager = .default
    ) throws -> URL {
        let extensionURL = localProviderExtensionURL(
            homeDirectory: homeDirectory,
            fileManager: fileManager
        )
        let extensionDir = extensionURL.deletingLastPathComponent()
        try fileManager.createDirectory(at: extensionDir, withIntermediateDirectories: true)

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

    /// Render a current Pi extension with one offline fallback and live model
    /// discovery from mere.run's self-describing `/v1/models` endpoint.
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
        providerCompat.append("supportsFinishReason: \(model.supportsFinishReason)")
        providerCompat.append("maxTokensField: \"\(model.maxTokensField)\"")
        providerCompat.append("supportsStrictMode: \(model.supportsStrictMode)")
        if let format = model.thinkingFormat {
            providerCompat.append("thinkingFormat: \"\(format)\"")
        }
        if model.requiresReasoningContentOnAssistantMessages {
            providerCompat.append("requiresReasoningContentOnAssistantMessages: true")
        }
        let compatBlock = providerCompat
            .map { "        \($0)" }
            .joined(separator: ",\n")

        var modelLines: [String] = []
        modelLines.append("id: \(javascriptStringLiteral(model.id))")
        modelLines.append("name: \(javascriptStringLiteral(model.name))")
        modelLines.append("api: \"openai-completions\" as const")
        modelLines.append("provider: \"mere-run\"")
        modelLines.append("baseUrl")
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
        let inputModalities = model.inputModalities
            .map(javascriptStringLiteral)
            .joined(separator: ", ")
        modelLines.append("input: [\(inputModalities)] as Array<\"text\" | \"image\">")
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
            .map { "      \($0)" }
            .joined(separator: ",\n")

        let fallbackModel: String
        if model.toolCall {
            fallbackModel = [
                "  {",
                modelBody + ",",
                "    compat: {",
                compatBlock,
                "    }",
                "  }",
            ].joined(separator: "\n")
        } else {
            fallbackModel = ""
        }

        let quotedBaseURL = javascriptStringLiteral(baseURL)
        let quotedAPIKey = javascriptStringLiteral(apiKey)

        return """
        import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

        type ThinkingLevel = "off" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max";

        interface MereRunModel {
          id: string;
          name?: string;
          task?: string;
          reasoning?: boolean;
          thinking_levels?: ThinkingLevel[];
          tool_call?: boolean;
          modalities?: { input: string[]; output: string[] };
          limit?: { context: number; output: number };
          openai_compat?: {
            supports_store: boolean;
            supports_developer_role: boolean;
            supports_reasoning_effort: boolean;
            supports_usage_in_streaming: boolean;
            supports_finish_reason: boolean;
            max_tokens_field: "max_tokens" | "max_completion_tokens";
            supports_strict_mode: boolean;
            thinking_format?: "deepseek";
            thinking_level_map?: Partial<Record<ThinkingLevel, string>>;
            requires_reasoning_content_on_assistant_messages: boolean;
          };
        }

        const baseUrl = \(quotedBaseURL);
        const apiKey = \(quotedAPIKey);
        const thinkingLevels: ThinkingLevel[] = [
          "off", "minimal", "low", "medium", "high", "xhigh", "max"
        ];
        const fallbackModels: ReturnType<typeof mapModel>[] = [
        \(fallbackModel)
        ];

        function mapThinkingLevels(model: MereRunModel) {
          if (!model.reasoning || !model.thinking_levels?.length) return undefined;
          const supported = new Set(model.thinking_levels);
          const overrides = model.openai_compat?.thinking_level_map ?? {};
          const result: Partial<Record<ThinkingLevel, string | null>> = {};
          for (const level of thinkingLevels) {
            if (!supported.has(level)) result[level] = null;
            else if (overrides[level]) result[level] = overrides[level];
          }
          return result;
        }

        function mapModel(model: MereRunModel) {
          const compat = model.openai_compat;
          return {
            id: model.id,
            name: model.name ?? model.id,
            api: "openai-completions",
            provider: "mere-run",
            baseUrl,
            reasoning: model.reasoning ?? false,
            thinkingLevelMap: mapThinkingLevels(model),
            input: (model.modalities?.input ?? ["text"])
              .filter((value): value is "text" | "image" => value === "text" || value === "image"),
            cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
            contextWindow: model.limit?.context ?? 32_768,
            maxTokens: model.limit?.output ?? 4_096,
            compat: compat ? {
              supportsStore: compat.supports_store,
              supportsDeveloperRole: compat.supports_developer_role,
              supportsReasoningEffort: compat.supports_reasoning_effort,
              supportsUsageInStreaming: compat.supports_usage_in_streaming,
              supportsFinishReason: compat.supports_finish_reason,
              maxTokensField: compat.max_tokens_field,
              supportsStrictMode: compat.supports_strict_mode,
              thinkingFormat: compat.thinking_format,
              requiresReasoningContentOnAssistantMessages:
                compat.requires_reasoning_content_on_assistant_messages
            } : undefined
          };
        }

        async function discoverModels(
          credential: string,
          signal: AbortSignal
        ): Promise<ReturnType<typeof mapModel>[]> {
          const response = await fetch(`${baseUrl}/models`, {
            headers: { Authorization: `Bearer ${credential}` },
            signal
          });
          if (!response.ok) throw new Error(`mere.run model discovery failed: HTTP ${response.status}`);
          const payload = await response.json() as { data?: MereRunModel[] };
          return (payload.data ?? [])
            .filter((entry) => entry.task === "chat.completions" && entry.tool_call === true)
            .map(mapModel);
        }

        export default async function(pi: ExtensionAPI) {
          const initialModels = await discoverModels(
            apiKey,
            AbortSignal.timeout(2_000)
          ).catch(() => fallbackModels);
          pi.registerProvider("mere-run", {
            name: "mere.run Local",
            baseUrl,
            apiKey,
            api: "openai-completions",
            models: initialModels,
          });
        }
        """
    }

    private static func javascriptStringLiteral(_ value: String) -> String {
        let escaped = value
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
            .replacingOccurrences(of: "\n", with: "\\n")
            .replacingOccurrences(of: "\r", with: "\\r")
            .replacingOccurrences(of: "\t", with: "\\t")
            .replacingOccurrences(of: "\u{2028}", with: "\\u2028")
            .replacingOccurrences(of: "\u{2029}", with: "\\u2029")
        return "\"\(escaped)\""
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

    static func localProviderExtensionURL(
        homeDirectory: URL? = nil,
        fileManager: FileManager = .default
    ) -> URL {
        let home = homeDirectory ?? fileManager.homeDirectoryForCurrentUser
        return home
            .appendingPathComponent(".pi/agent/extensions", isDirectory: true)
            .appendingPathComponent("mere-run-local-provider.ts", isDirectory: false)
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

    static func installedPiVersion() -> String? {
        guard let root = try? piInstallRoot(),
              let binary = installedPiBinaryURL()?.resolvingSymlinksInPath() else {
            return nil
        }
        let rootComponents = root.standardizedFileURL.pathComponents
        let binaryComponents = binary.standardizedFileURL.pathComponents
        guard binaryComponents.count > rootComponents.count,
              Array(binaryComponents.prefix(rootComponents.count)) == rootComponents else {
            return nil
        }
        return binaryComponents[rootComponents.count]
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

    static func providerConfigurationURL() -> URL {
        MereRunModelPaths.applicationSupportBase
            .appendingPathComponent("agent", isDirectory: true)
            .appendingPathComponent("provider.json", isDirectory: false)
    }

    static func waitForHealth(
        host: String,
        port: Int,
        timeoutSeconds: TimeInterval,
        serverProcess: Process? = nil,
        serverLogURL: URL? = nil,
        progressPrefix: String? = nil
    ) async throws {
        var awaitingAdmission = serverProcess != nil && serverLogURL != nil
        var reportedQueue = false
        var deadline = awaitingAdmission ? nil : Date().addingTimeInterval(timeoutSeconds)
        let healthURL = URL(string: "http://\(host):\(port)/health")!
        while deadline.map({ Date() < $0 }) ?? true {
            if let (_, response) = try? await URLSession.shared.data(from: healthURL),
               let http = response as? HTTPURLResponse,
               http.statusCode == 200 {
                return
            }
            if let serverProcess, !serverProcess.isRunning {
                throw IntegrationError.serverDidNotBecomeReady(healthURL)
            }
            if awaitingAdmission,
               !reportedQueue,
               let serverLogURL,
               serverQueueObserved(in: serverLogURL) {
                if let progressPrefix {
                    CLIStderr.write("\(progressPrefix) Local API is queued by machine admission; waiting for permits.\n")
                }
                reportedQueue = true
            }
            if awaitingAdmission,
               let serverLogURL,
               serverAdmissionObserved(in: serverLogURL) {
                awaitingAdmission = false
                deadline = Date().addingTimeInterval(timeoutSeconds)
            }
            try await Task.sleep(nanoseconds: 500_000_000)
        }
        throw IntegrationError.serverDidNotBecomeReady(healthURL)
    }

    static func serverAdmissionObserved(
        in logURL: URL,
        fileManager: FileManager = .default
    ) -> Bool {
        serverLogContains(serverAdmissionMarker, in: logURL, fileManager: fileManager)
    }

    static func serverQueueObserved(
        in logURL: URL,
        fileManager: FileManager = .default
    ) -> Bool {
        serverLogContains(serverQueueMarker, in: logURL, fileManager: fileManager)
    }

    private static func serverLogContains(
        _ marker: String,
        in logURL: URL,
        fileManager: FileManager
    ) -> Bool {
        guard fileManager.fileExists(atPath: logURL.path),
              let data = try? Data(contentsOf: logURL),
              let contents = String(data: data, encoding: .utf8) else {
            return false
        }
        return contents.contains(marker)
    }

    private static func fetchLatestRelease() async throws -> GitHubRelease {
        let url = URL(string: "https://api.github.com/repos/earendil-works/pi/releases/latest")!
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
