import Foundation
import MereRunCore
#if canImport(Darwin)
import Darwin
#else
import Glibc
#endif
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

struct ClaudeCodeConnectionPlan: Encodable {
    let settingsPath: String
    let model: String
    let baseURL: String
    let claudeCodeVersion: String
    let contextTokens: Int
    let outputTokens: Int
    let capabilities: MessagesCapabilities
    let patch: [String: ChatJSONValue]
    let qualification: [String]
}

struct ClaudeCodeServerModel: Decodable {
    let id: String
    let display_name: String?
    let max_input_tokens: Int?
    let max_tokens: Int?
    let local_capabilities: MessagesCapabilities?
}

enum ClaudeCodeConnection {
    static let candidateVersion = "2.1.316"

    static func origin(_ value: String) throws -> URL {
        guard let components = URLComponents(string: value), let scheme = components.scheme,
              ["http", "https"].contains(scheme), components.host != nil,
              components.user == nil, components.password == nil, components.query == nil, components.fragment == nil,
              components.path.isEmpty || components.path == "/", let url = components.url else {
            throw MessagesFailure.invalid("base-url", "supply an HTTP server origin without credentials, query or path")
        }
        return url
    }
    static func validateEnvironmentName(_ name: String) throws {
        guard name.range(of: #"^[A-Za-z_][A-Za-z0-9_]*$"#, options: .regularExpression) != nil else {
            throw MessagesFailure.invalid("api-key-env", "must be a shell environment variable name")
        }
    }
    static let providerFlags = ["CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX", "CLAUDE_CODE_USE_FOUNDRY"]
    static func validateRouting(environment: [String: String], credentialVariable: String) throws {
        for name in providerFlags {
            if let value = environment[name], !["", "0", "false"].contains(value.lowercased()) {
                throw MessagesFailure.invalid(name, "unset this provider selector before using the local Messages connection")
            }
        }
        for name in ["ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"] {
            if let value = environment[name], !value.isEmpty, name != credentialVariable, value != environment[credentialVariable] {
                throw MessagesFailure.invalid(name, "this credential overrides the helper; unset it or select it explicitly with --api-key-env")
            }
        }
    }

    static func installedVersion() throws -> String {
        let process = Process(); process.executableURL = URL(fileURLWithPath: "/usr/bin/env"); process.arguments = ["claude", "--version"]
        let result = try BoundedProcessRunner.run(process, timeout: 10, outputLimitBytes: 1024)
        guard result.status == 0, result.completion == .exited,
              let version = result.stdout.text.split(separator: " ").first, version.contains(".") else {
            throw MessagesFailure.invalid("claude-code", "install Claude Code and ensure claude --version succeeds")
        }
        return String(version)
    }
    static func inspect(origin: URL, model: String, credential: String?, credentialVariable: String = "MERERUN_API_KEY") async throws -> ClaudeCodeServerModel {
        if credential?.contains(where: { $0 == "\r" || $0 == "\n" }) == true {
            throw MessagesFailure.invalid("api-key-env", "\(credentialVariable) must contain a single-line credential")
        }
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = 5
        let session = URLSession(configuration: config, delegate: NoRedirects(), delegateQueue: nil)
        defer { session.invalidateAndCancel() }
        var probe = URLRequest(url: origin.appendingPathComponent("api/hello")); probe.httpMethod = "HEAD"
        if let credential { probe.setValue(credential, forHTTPHeaderField: "x-api-key") }
        let (_, health) = try await session.data(for: probe)
        if let status = (health as? HTTPURLResponse)?.statusCode, [401, 403].contains(status) {
            throw MessagesFailure.invalid("api-key-env", "server rejected authentication; check \(credentialVariable) in the environment used to launch Claude Code")
        }
        guard (health as? HTTPURLResponse)?.statusCode == 200 else { throw MessagesFailure.invalid("server", "connection probe failed; start mere.run api serve explicitly") }
        var request = URLRequest(url: origin.appendingPathComponent("v1/models").appendingPathComponent(model))
        if let credential { request.setValue(credential, forHTTPHeaderField: "x-api-key") }
        let (data, response) = try await session.data(for: request)
        guard (response as? HTTPURLResponse)?.statusCode == 200 else {
            throw MessagesFailure.invalid("model", "model inspection failed; check the model ID, readiness and credential environment variable")
        }
        return try JSONDecoder().decode(ClaudeCodeServerModel.self, from: data)
    }
    static func plan(origin: URL, model: ClaudeCodeServerModel, version: String, settings: URL,
                     credentialVariable: String, credentialPresent: Bool = true) throws -> ClaudeCodeConnectionPlan {
        try validateEnvironmentName(credentialVariable)
        guard let caps = model.local_capabilities, caps.messages, caps.streaming, caps.tools,
              let context = model.max_input_tokens, let output = model.max_tokens, context > 0, output > 0 else {
            throw MessagesFailure.invalid("model", "Claude Code requires Messages streaming, tools and effective limits; choose a capable local adapter")
        }
        var env: [String: ChatJSONValue] = [
            "ANTHROPIC_BASE_URL": .string(origin.absoluteString.trimmingCharacters(in: CharacterSet(charactersIn: "/"))),
            "ANTHROPIC_MODEL": .string(model.id),
            "ANTHROPIC_DEFAULT_OPUS_MODEL": .string(model.id), "ANTHROPIC_DEFAULT_SONNET_MODEL": .string(model.id),
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": .string(model.id), "ANTHROPIC_DEFAULT_FABLE_MODEL": .string(model.id),
            "CLAUDE_CODE_SUBAGENT_MODEL": .string(model.id), "ANTHROPIC_CUSTOM_MODEL_OPTION": .string(model.id),
            "ANTHROPIC_CUSTOM_MODEL_OPTION_NAME": .string(model.display_name ?? model.id),
            "CLAUDE_CODE_MAX_CONTEXT_TOKENS": .string(String(context)), "CLAUDE_CODE_MAX_OUTPUT_TOKENS": .string(String(output))
        ]
        // Never include credential values, including pre-existing environment secrets.
        env["ANTHROPIC_SMALL_FAST_MODEL"] = .string(model.id)
        let patch: [String: ChatJSONValue] = ["model": .string(model.id), "env": .object(env),
                                            "apiKeyHelper": .string("printenv " + credentialVariable)]
        if FileManager.default.fileExists(atPath: settings.path) {
            let existing = try JSONDecoder().decode([String: ChatJSONValue].self, from: Data(contentsOf: settings))
            _ = try merged(existing, patch: patch)
        }
        var qualification = ["End-to-end model qualification is pending; configuration is not evidence of a working agent loop."]
        if !credentialPresent {
            qualification.append("\(credentialVariable) is unset or empty; export it before launching Claude Code. Only the environment-backed helper is stored.")
        }
        if context < 65536 {
            qualification.append("A \(context)-token context can trigger frequent client compaction with the default tool catalog; use a larger-context local model or explicitly raise api serve --context-size to 65536 where supported.")
        }
        if version != candidateVersion { qualification.append("Qualification candidate is Claude Code \(candidateVersion); installed \(version) is unqualified.") }
        if !caps.tool_search { qualification.append("Adapter does not support server tool search; requests for it fail explicitly.") }
        if !caps.thinking.contains("adaptive") { qualification.append("Adaptive thinking is unsupported; select an adapter supporting the client request.") }
        if !caps.thinking.isEmpty, caps.thinking_display?.contains("updates") != true {
            qualification.append("Thinking progress-update display is unsupported; omit thinking.display or use the client's fallback to native reasoning.")
        }
        if !caps.effort.contains("high") {
            qualification.append("Claude Code \(candidateVersion) sends output_config.effort.high by default, which this adapter rejects. Select a local adapter advertising high effort or a client configuration that omits the unsupported field.")
        }
        if !caps.token_counting { qualification.append("Exact token counting is unavailable on this adapter.") }
        return .init(settingsPath: settings.path, model: model.id, baseURL: origin.absoluteString, claudeCodeVersion: version,
                     contextTokens: context, outputTokens: output, capabilities: caps, patch: patch, qualification: qualification)
    }
    static func merged(_ existing: [String: ChatJSONValue], patch: [String: ChatJSONValue]) throws -> [String: ChatJSONValue] {
        var result = existing
        if let current = existing["apiKeyHelper"], current != patch["apiKeyHelper"] {
            throw MessagesFailure.invalid("apiKeyHelper", "existing credential helper differs; review and update it explicitly before applying")
        }
        if let env = existing["env"], env.objectValue == nil { throw MessagesFailure.invalid("env", "existing settings env must be an object") }
        var env = existing["env"]?.objectValue ?? [:]
        for name in providerFlags {
            if let value = env[name], !["", "0", "false"].contains(value.stringValue?.lowercased() ?? "active") {
                throw MessagesFailure.invalid("env.\(name)", "existing settings select another provider; remove that selector explicitly before using local Messages")
            }
        }
        for key in ["ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"] where env[key] != nil {
            throw MessagesFailure.invalid("env.\(key)", "existing credential overrides the helper; move it to the named environment variable before applying")
        }
        for (key, value) in patch["env"]?.objectValue ?? [:] { env[key] = value }
        for (key, value) in patch where key != "env" { result[key] = value }
        result["env"] = .object(env)
        return result
    }
    static func apply(_ plan: ClaudeCodeConnectionPlan, settings: URL) throws {
        let manager = FileManager.default
        try manager.createDirectory(at: settings.deletingLastPathComponent(), withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
        let lockURL = settings.deletingLastPathComponent().appendingPathComponent(".mere-run-connect.lock")
        let descriptor = open(lockURL.path, O_RDWR | O_CREAT | O_NOFOLLOW, 0o600)
        guard descriptor >= 0 else { throw MessagesFailure.invalid("settings", "could not open the merge lock") }
        defer { close(descriptor) }
        guard flock(descriptor, LOCK_EX) == 0 else { throw MessagesFailure.invalid("settings", "could not acquire the merge lock") }
        defer { flock(descriptor, LOCK_UN) }
        let original = manager.fileExists(atPath: settings.path) ? try Data(contentsOf: settings) : nil
        let existing = try original.map { try JSONDecoder().decode([String: ChatJSONValue].self, from: $0) } ?? [:]
        let updated = try merged(existing, patch: plan.patch)
        if updated == existing { return }
        let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        let bytes = try encoder.encode(updated) + Data("\n".utf8)
        let current = manager.fileExists(atPath: settings.path) ? try Data(contentsOf: settings) : nil
        guard current == original else { throw MessagesFailure.invalid("settings", "changed during preparation; retry") }
        try bytes.write(to: settings, options: .atomic)
    }

    private final class NoRedirects: NSObject, URLSessionTaskDelegate, @unchecked Sendable {
        func urlSession(_ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse,
                        newRequest request: URLRequest, completionHandler: @escaping (URLRequest?) -> Void) { completionHandler(nil) }
    }
}
