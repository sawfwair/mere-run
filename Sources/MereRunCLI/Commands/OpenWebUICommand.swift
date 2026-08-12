import ArgumentParser
import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif
import AudioSTT
import AudioTTS
import MereRunCore

struct OpenWebUI: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "open-webui",
        abstract: "Start the optional Open WebUI companion against a local mere.run API.",
        subcommands: [
            OpenWebUIQuickstart.self,
        ],
        defaultSubcommand: OpenWebUIQuickstart.self
    )
}

struct OpenWebUIQuickstart: AsyncParsableCommand {
    static let adminPasswordEnvironmentKey = "MERERUN_OPEN_WEBUI_ADMIN_PASSWORD"

    static let configuration = CommandConfiguration(
        commandName: "quickstart",
        abstract: "Start mere.run, run the official Open WebUI Docker image, and configure the connection."
    )

    @Option(name: [.long], help: "Host for the local mere.run API server.")
    var host: String = "0.0.0.0"

    @Option(name: [.long], help: "Port for the local mere.run API server.")
    var port: Int = 8080

    @Option(name: [.long], help: "Serving engine for the text model. Defaults to the managed model's API engine.")
    var engine: APIEngine?

    @Option(name: [.long], help: "Open WebUI host bind and browser host.")
    var webuiHost: String = "127.0.0.1"

    @Option(name: [.long], help: "Open WebUI host port.")
    var webuiPort: Int = 3000

    @Option(name: [.long], help: "Open WebUI container name.")
    var containerName: String = "open-webui-mere-run"

    @Option(name: [.long], help: "Open WebUI Docker volume name.")
    var volumeName: String = "open-webui-mere-run"

    @Option(name: [.long], help: "Open WebUI Docker image.")
    var image: String = "ghcr.io/open-webui/open-webui:main"

    @Option(name: [.long], help: "Bearer API key for the local mere.run API. Defaults to MERERUN_API_KEY or change-me.")
    var apiKey: String?

    @Option(name: [.long], help: "Installed text chat model to expose as Open WebUI's default chat model.")
    var textModel: String = Gemma4Resources.twelveBModelId

    @Option(name: [.long], help: "Installed vision chat model to expose in Open WebUI's chat selector.")
    var visionModel: String = Gemma4Resources.visionTwelveBModelId

    @Option(name: [.long], help: "Installed embedding model for Open WebUI RAG.")
    var embeddingModel: String = Qwen3EmbeddingCatalog.modelId

    @Option(name: [.long], help: "Installed image-generation model for Open WebUI.")
    var imageModel: String = ModelResolver.ModelID.zetaNano.rawValue

    @Option(name: [.long], help: "Installed text-to-speech model for Open WebUI.")
    var ttsModel: String = Qwen3TTSResources.defaultModelId

    @Option(name: [.long], help: "Installed speech-to-text model for Open WebUI.")
    var sttModel: String = ParakeetResources.defaultModelId

    @Option(name: [.long], help: "OpenAI speech response format for Open WebUI TTS.")
    var ttsFormat: String = "wav"

    @Option(name: [.long], help: "Open WebUI no-auth signin email used while configuring the fresh container.")
    var adminEmail: String = "admin@localhost"

    @Option(
        name: [.long],
        help: "Open WebUI no-auth signin password. Defaults to MERERUN_OPEN_WEBUI_ADMIN_PASSWORD or admin."
    )
    var adminPassword: String?

    @Option(name: [.long], help: "Seconds to wait for mere.run and Open WebUI health checks.")
    var waitSeconds: Int = 180

    @Flag(name: [.long], help: "Pull the configured managed models before starting.")
    var pull: Bool = false

    @Flag(
        name: [.customLong("accept-model-license")],
        help: "Confirm that you reviewed and accept listed third-party model/component terms before downloading restricted configured models."
    )
    var acceptModelLicense: Bool = false

    @Flag(name: [.long], help: "Use an already-running mere.run API server.")
    var skipServer: Bool = false

    @Flag(name: [.long], help: "Use an already-running Open WebUI instance.")
    var skipDocker: Bool = false

    @Flag(name: [.long], help: "Skip Open WebUI admin API configuration.")
    var skipConfigure: Bool = false

    @Flag(name: [.long], help: "Remove any existing Open WebUI container and volume before starting Docker.")
    var reset: Bool = false

    @Flag(name: [.long], help: "Print the planned commands without starting anything.")
    var dryRun: Bool = false

    @Flag(name: [.short, .long], help: "Suppress model pull progress output.")
    var quiet: Bool = false

    func run() async throws {
        let resolvedKey = Self.resolvedAPIKey(explicit: apiKey)
        if dryRun {
            print(try makePlan(apiKey: resolvedKey).render())
            return
        }

        if pull {
            try await pullConfiguredModels()
        }

        let serverProcess: Process?
        if skipServer {
            serverProcess = nil
            CLIStderr.write("[open-webui] Using existing mere.run API at \(localAPIBaseURL.absoluteString)\n")
        } else {
            let started = try startAPIServer(apiKey: resolvedKey)
            serverProcess = started.process
            CLIStderr.write("[open-webui] Loading \(textModel). Server log: \(started.logURL.path)\n")
            try await waitForHTTP200(
                url: serverHealthURL,
                timeoutSeconds: waitSeconds,
                label: "mere.run API"
            )
        }
        defer {
            if let serverProcess, serverProcess.isRunning {
                serverProcess.terminate()
            }
        }

        if skipDocker {
            CLIStderr.write("[open-webui] Using existing Open WebUI at \(openWebUIBaseURL.absoluteString)\n")
        } else {
            try startDockerContainer(apiKey: resolvedKey)
        }

        if !skipConfigure {
            try await waitForHTTP200(
                url: openWebUIHealthURL,
                timeoutSeconds: waitSeconds,
                label: "Open WebUI"
            )
            let configuredModels = try await configureOpenWebUI(apiKey: resolvedKey)
            CLIStderr.write("[open-webui] Configured chat models: \(configuredModels.joined(separator: ", "))\n")
        }

        print("Open WebUI is ready: \(openWebUIBaseURL.absoluteString)")
        print("mere.run API: \(dockerAPIBaseURL.absoluteString)")
        print("Chat models: \(textModel), \(visionModel)")
        print("RAG embeddings: \(embeddingModel)")
        print("Image/TTS/STT: \(imageModel), \(ttsModel), \(sttModel)")
        if serverProcess == nil {
            return
        }
        print("Press Ctrl+C to stop the local mere.run API server.")
        waitForServerProcess()
    }

    static func resolvedAPIKey(explicit: String?, environment: [String: String] = ProcessInfo.processInfo.environment) -> String {
        if let explicit = normalized(explicit) {
            return explicit
        }
        if let fromEnvironment = normalized(environment["MERERUN_API_KEY"]) {
            return fromEnvironment
        }
        return "change-me"
    }

    static func resolvedAdminPassword(
        explicit: String?,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> String {
        if let explicit = normalized(explicit) {
            return explicit
        }
        if let fromEnvironment = normalized(environment[adminPasswordEnvironmentKey]) {
            return fromEnvironment
        }
        return "admin"
    }

    func makePlan(apiKey: String) throws -> OpenWebUIQuickstartPlan {
        let engine = try resolvedEngine()
        return OpenWebUIQuickstartPlan(
            apiKey: apiKey,
            apiServeCommand: apiServeDisplayCommand(engine: engine, apiKey: "$MERERUN_API_KEY"),
            dockerCommand: dockerDisplayCommand(apiKey: "$MERERUN_API_KEY"),
            configureCommand: CLICommandDisplay.command(
                [
                    "open-webui quickstart",
                    "--skip-server",
                    "--skip-docker",
                    "--host \(ShellQuote.quote(host))",
                    "--port \(port)",
                    "--webui-host \(ShellQuote.quote(webuiHost))",
                    "--webui-port \(webuiPort)",
                    "--text-model \(ShellQuote.quote(textModel))",
                    "--vision-model \(ShellQuote.quote(visionModel))",
                    "--api-key \"$MERERUN_API_KEY\"",
                ].joined(separator: " ")
            )
        )
    }

    private var localAPIBaseURL: URL {
        URL(string: "http://\(serverHealthHost):\(port)/v1")!
    }

    private var dockerAPIBaseURL: URL {
        URL(string: "http://host.docker.internal:\(port)/v1")!
    }

    private var serverHealthURL: URL {
        URL(string: "http://\(serverHealthHost):\(port)/health")!
    }

    private var serverHealthHost: String {
        let normalizedHost = host.trimmingCharacters(in: .whitespacesAndNewlines)
        if normalizedHost == "0.0.0.0" || normalizedHost == "::" {
            return "127.0.0.1"
        }
        return normalizedHost.isEmpty ? "127.0.0.1" : normalizedHost
    }

    private var openWebUIBaseURL: URL {
        URL(string: "http://\(browserHost):\(webuiPort)")!
    }

    private var openWebUIHealthURL: URL {
        openWebUIBaseURL.appendingPathComponent("health")
    }

    private var browserHost: String {
        let normalizedHost = webuiHost.trimmingCharacters(in: .whitespacesAndNewlines)
        if normalizedHost == "0.0.0.0" || normalizedHost == "::" {
            return "127.0.0.1"
        }
        return normalizedHost.isEmpty ? "127.0.0.1" : normalizedHost
    }

    private func resolvedEngine() throws -> APIEngine {
        if let engine {
            return engine
        }
        if let spec = ManagedModelCatalog.spec(for: textModel),
           let runtimeEngine = spec.defaultRuntimeServingEngine {
            return try APIEngine(runtimeEngine: runtimeEngine)
        }
        throw ValidationError(
            "Could not infer an API engine for \(textModel). Pass --engine for a local path or custom model."
        )
    }

    private func pullConfiguredModels() async throws {
        if let message = unacknowledgedUsageTermsMessage() {
            throw ValidationError(message)
        }
        for modelID in uniqueModelIDs([textModel, visionModel, embeddingModel, imageModel, ttsModel, sttModel]) {
            guard let spec = ManagedModelCatalog.spec(for: modelID) else {
                CLIStderr.write("[open-webui] Skipping pull for custom model \(modelID)\n")
                continue
            }
            guard spec.hasAnyManagedDownloadSource() else {
                CLIStderr.write("[open-webui] \(modelID) has no managed download source; install it from a local path.\n")
                continue
            }
            if spec.managedRuntimeURL() != nil {
                if !quiet {
                    CLIStderr.write("[open-webui] \(modelID) already available in the unified model catalog\n")
                }
                continue
            }
            if !quiet {
                CLIStderr.write("[open-webui] Pulling \(modelID)\n")
            }
            _ = try await ManagedModelResolver.installManagedModel(
                id: modelID,
                force: false,
                usageTermsAcknowledged: acceptModelLicense,
                progress: { progress in
                    guard !quiet else { return }
                    switch progress {
                    case .downloadingBytes(let completed, let total):
                        let done = ByteCountFormatter.string(fromByteCount: completed, countStyle: .file)
                        if let total, total > 0 {
                            let totalText = ByteCountFormatter.string(fromByteCount: total, countStyle: .file)
                            CLIStderr.write("\r[\(modelID)] \(done) / \(totalText)          ")
                        } else {
                            CLIStderr.write("\r[\(modelID)] \(done)          ")
                        }
                    case .downloadingPercent(let percent, let speed):
                        if let speed, speed > 0 {
                            let speedText = ByteCountFormatter.string(
                                fromByteCount: Int64(speed),
                                countStyle: .file
                            )
                            CLIStderr.write("\r[\(modelID)] \(percent)% (\(speedText)/s)          ")
                        } else {
                            CLIStderr.write("\r[\(modelID)] \(percent)%          ")
                        }
                    case .extracting:
                        CLIStderr.write("\r[\(modelID)] extracting          ")
                    }
                }
            )
            if !quiet {
                CLIStderr.write("\n")
            }
        }
    }

    func unacknowledgedUsageTermsMessage(fileManager: FileManager = .default) -> String? {
        guard !acceptModelLicense else { return nil }
        let restricted = uniqueModelIDs([textModel, visionModel, embeddingModel, imageModel, ttsModel, sttModel])
            .compactMap(ManagedModelCatalog.spec(for:))
            .filter { spec in
                spec.usageRestriction != nil
                    && spec.hasAnyManagedDownloadSource()
                    && spec.managedRuntimeURL(fileManager: fileManager) == nil
            }
        guard !restricted.isEmpty else { return nil }

        let details = restricted.map { spec in
            let restriction = spec.usageRestriction!
            let terms = restriction.terms
                .map { "  - \($0.component): \($0.license) \($0.licenseURL)" }
                .joined(separator: "\n")
            return "- \(spec.id): \(restriction.summary)\n\(terms)"
        }.joined(separator: "\n")
        return """
        Open WebUI --pull includes configured models with third-party usage terms:
        \(details)
        Mere does not determine whether your intended use is permitted. You are responsible for compliance.
        Re-run with --accept-model-license to confirm that you reviewed and accept these terms and agree to comply with them, \
        or pre-install a different model.
        """
    }

    private func startAPIServer(apiKey: String) throws -> (process: Process, logURL: URL) {
        let engine = try resolvedEngine()
        let log = try AgentServerLog.makeLogHandle(prefix: "open-webui-api-server")
        let process = Process()
        process.executableURL = CurrentExecutable.url()
        process.arguments = [
            "api",
            "serve",
            "--engine",
            engine.rawValue,
            "--model",
            textModel,
            "--host",
            host,
            "--port",
            String(port),
            "--api-key",
            apiKey,
        ]
        process.standardInput = AgentServerLog.nullInputHandle()
        process.standardOutput = log.handle
        process.standardError = log.handle
        try process.run()
        CLIStderr.write("[open-webui] Started mere.run API server on \(host):\(port)\n")
        return (process, log.url)
    }

    private func startDockerContainer(apiKey: String) throws {
        _ = try runDocker(["--version"])
        if reset {
            _ = try? runDocker(["rm", "-f", containerName])
            _ = try? runDocker(["volume", "rm", volumeName])
        } else if try dockerContainerExists() {
            throw ValidationError(
                "Docker container \(containerName) already exists. Pass --reset to recreate it or --skip-docker to reuse it."
            )
        }
        let result = try runDocker(dockerRunArguments(apiKey: apiKey))
        CLIStderr.write("[open-webui] Open WebUI container: \(result.stdout.trimmingCharacters(in: .whitespacesAndNewlines))\n")
        CLIStderr.write("[open-webui] Starting at \(openWebUIBaseURL.absoluteString)\n")
    }

    private func dockerContainerExists() throws -> Bool {
        let result = try runDocker(["ps", "-a", "--format", "{{.Names}}"])
        return result.stdout
            .split(whereSeparator: \.isNewline)
            .contains { $0 == containerName }
    }

    private func dockerRunArguments(apiKey: String) -> [String] {
        [
            "run",
            "-d",
            "--name",
            containerName,
            "--restart",
            "unless-stopped",
            "-p",
            "\(webuiHost):\(webuiPort):8080",
            "--add-host=host.docker.internal:host-gateway",
        ] + dockerEnvironmentArguments(apiKey: apiKey)
            + [
                "-v",
                "\(volumeName):/app/backend/data",
                image,
            ]
    }

    private func dockerEnvironmentArguments(apiKey: String) -> [String] {
        dockerEnvironment(apiKey: apiKey)
            .flatMap { ["-e", "\($0.name)=\($0.value)"] }
    }

    private func dockerEnvironment(apiKey: String) -> [OpenWebUIEnvironmentValue] {
        [
            OpenWebUIEnvironmentValue("OPENAI_API_BASE_URL", dockerAPIBaseURL.absoluteString),
            OpenWebUIEnvironmentValue("OPENAI_API_BASE_URLS", dockerAPIBaseURL.absoluteString),
            OpenWebUIEnvironmentValue("OPENAI_API_KEY", apiKey),
            OpenWebUIEnvironmentValue("OPENAI_API_KEYS", apiKey),
            OpenWebUIEnvironmentValue("DEFAULT_MODELS", textModel),
            OpenWebUIEnvironmentValue("DEFAULT_MODEL_PARAMS", OpenWebUIModelParams.defaultJSONString),
            OpenWebUIEnvironmentValue("DEFAULT_MODEL_METADATA", OpenWebUIModelMetadata.defaultJSONString),
            OpenWebUIEnvironmentValue("RAG_EMBEDDING_ENGINE", "openai"),
            OpenWebUIEnvironmentValue("RAG_OPENAI_API_BASE_URL", dockerAPIBaseURL.absoluteString),
            OpenWebUIEnvironmentValue("RAG_OPENAI_API_KEY", apiKey),
            OpenWebUIEnvironmentValue("RAG_EMBEDDING_MODEL", embeddingModel),
            OpenWebUIEnvironmentValue("ENABLE_IMAGE_GENERATION", "True"),
            OpenWebUIEnvironmentValue("ENABLE_IMAGE_EDIT", "False"),
            OpenWebUIEnvironmentValue("IMAGE_GENERATION_ENGINE", "openai"),
            OpenWebUIEnvironmentValue("IMAGE_GENERATION_MODEL", imageModel),
            OpenWebUIEnvironmentValue("IMAGE_SIZE", "1024x1024"),
            OpenWebUIEnvironmentValue("IMAGE_EDIT_ENGINE", "openai"),
            OpenWebUIEnvironmentValue("IMAGE_EDIT_MODEL", "qwen-image-edit"),
            OpenWebUIEnvironmentValue("IMAGE_EDIT_SIZE", "1024x1024"),
            OpenWebUIEnvironmentValue("IMAGES_OPENAI_API_BASE_URL", dockerAPIBaseURL.absoluteString),
            OpenWebUIEnvironmentValue("IMAGES_OPENAI_API_KEY", apiKey),
            OpenWebUIEnvironmentValue("IMAGES_EDIT_OPENAI_API_BASE_URL", dockerAPIBaseURL.absoluteString),
            OpenWebUIEnvironmentValue("IMAGES_EDIT_OPENAI_API_KEY", apiKey),
            OpenWebUIEnvironmentValue("AUDIO_TTS_ENGINE", "openai"),
            OpenWebUIEnvironmentValue("AUDIO_TTS_MODEL", ttsModel),
            OpenWebUIEnvironmentValue("AUDIO_TTS_VOICE", "nova"),
            OpenWebUIEnvironmentValue("AUDIO_TTS_OPENAI_API_BASE_URL", dockerAPIBaseURL.absoluteString),
            OpenWebUIEnvironmentValue("AUDIO_TTS_OPENAI_API_KEY", apiKey),
            OpenWebUIEnvironmentValue("AUDIO_TTS_OPENAI_PARAMS", "{\"response_format\":\"\(ttsFormat)\"}"),
            OpenWebUIEnvironmentValue("AUDIO_STT_ENGINE", "openai"),
            OpenWebUIEnvironmentValue("AUDIO_STT_MODEL", sttModel),
            OpenWebUIEnvironmentValue("AUDIO_STT_OPENAI_API_BASE_URL", dockerAPIBaseURL.absoluteString),
            OpenWebUIEnvironmentValue("AUDIO_STT_OPENAI_API_KEY", apiKey),
            OpenWebUIEnvironmentValue("ENABLE_PERSISTENT_CONFIG", "False"),
            OpenWebUIEnvironmentValue("WEBUI_AUTH", "False"),
        ]
    }

    private func configureOpenWebUI(apiKey: String) async throws -> [String] {
        let availableModelIDs = try await fetchMereRunModelIDs(apiKey: apiKey)
        let requestedChatModels = uniqueModelIDs([textModel, visionModel])
        let selectedChatModels = requestedChatModels.filter { availableModelIDs.contains($0) }
        let chatModels = selectedChatModels.isEmpty ? requestedChatModels : selectedChatModels
        let token = try await fetchOpenWebUIToken()
        try await postOpenWebUIJSON(
            path: "/openai/config/update",
            token: token,
            payload: OpenWebUIOpenAIConfigPayload(
                apiBaseURL: dockerAPIBaseURL.absoluteString,
                apiKey: apiKey,
                modelIDs: chatModels
            )
        )
        try await postOpenWebUIJSON(
            path: "/api/v1/configs/models",
            token: token,
            payload: OpenWebUIModelsConfigPayload(
                defaultModels: textModel,
                modelOrderList: chatModels
            )
        )
        try await postOpenWebUIJSON(
            path: "/api/v1/models/import",
            token: token,
            payload: OpenWebUIModelsImportPayload(
                models: wrappers(for: chatModels)
            )
        )
        let openWebUIModels: OpenWebUIModelListPayload = try await getOpenWebUIJSON(
            path: "/api/models",
            token: token
        )
        let listedIDs = Set(openWebUIModels.data.map(\.id))
        let missing = chatModels.filter { !listedIDs.contains($0) }
        if !missing.isEmpty {
            throw ValidationError(
                "Open WebUI model list is missing configured ids: \(missing.joined(separator: ", "))"
            )
        }
        return chatModels
    }

    private func fetchMereRunModelIDs(apiKey: String) async throws -> Set<String> {
        var request = URLRequest(url: localAPIBaseURL.appendingPathComponent("models"))
        request.addValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        let data = try await send(request: request, label: "mere.run /v1/models")
        let payload = try JSONDecoder().decode(OpenWebUIModelListPayload.self, from: data)
        return Set(payload.data.map(\.id))
    }

    private func fetchOpenWebUIToken() async throws -> String {
        let payload = OpenWebUIAuthSigninRequest(
            email: adminEmail,
            password: Self.resolvedAdminPassword(explicit: adminPassword)
        )
        let response: OpenWebUIAuthSigninResponse = try await postOpenWebUIJSON(
            path: "/api/v1/auths/signin",
            token: nil,
            payload: payload
        )
        guard let token = Self.normalized(response.token) else {
            throw ValidationError("Open WebUI signin did not return a token.")
        }
        return token
    }

    @discardableResult
    private func postOpenWebUIJSON<Payload: Encodable>(
        path: String,
        token: String?,
        payload: Payload
    ) async throws -> Data {
        let url = try joinedURL(base: openWebUIBaseURL, path: path)
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.addValue("application/json", forHTTPHeaderField: "Content-Type")
        if let token {
            request.addValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        request.httpBody = try JSONEncoder().encode(payload)
        return try await send(request: request, label: "Open WebUI \(path)")
    }

    private func postOpenWebUIJSON<Response: Decodable, Payload: Encodable>(
        path: String,
        token: String?,
        payload: Payload
    ) async throws -> Response {
        let data = try await postOpenWebUIJSON(path: path, token: token, payload: payload)
        return try JSONDecoder().decode(Response.self, from: data)
    }

    private func getOpenWebUIJSON<Response: Decodable>(path: String, token: String) async throws -> Response {
        let url = try joinedURL(base: openWebUIBaseURL, path: path)
        var request = URLRequest(url: url)
        request.addValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        let data = try await send(request: request, label: "Open WebUI \(path)")
        return try JSONDecoder().decode(Response.self, from: data)
    }

    private func waitForHTTP200(url: URL, timeoutSeconds: Int, label: String) async throws {
        let deadline = Date().addingTimeInterval(TimeInterval(timeoutSeconds))
        while Date() < deadline {
            var request = URLRequest(url: url)
            request.timeoutInterval = 5
            if let (_, response) = try? await URLSession.shared.data(for: request),
               let http = response as? HTTPURLResponse,
               http.statusCode == 200 {
                return
            }
            try await Task.sleep(nanoseconds: 1_000_000_000)
        }
        throw ValidationError("Timed out waiting for \(label) at \(url.absoluteString).")
    }

    private func send(request: URLRequest, label: String) async throws -> Data {
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw ValidationError("\(label) returned a non-HTTP response.")
        }
        guard (200..<300).contains(http.statusCode) else {
            let text = String(data: data.prefix(500), encoding: .utf8) ?? ""
            throw ValidationError("\(label) returned HTTP \(http.statusCode). \(text)")
        }
        return data
    }

    private func joinedURL(base: URL, path: String) throws -> URL {
        var rawBase = base.absoluteString
        while rawBase.hasSuffix("/") {
            rawBase.removeLast()
        }
        let normalizedPath = path.hasPrefix("/") ? path : "/\(path)"
        guard let url = URL(string: rawBase + normalizedPath) else {
            throw ValidationError("Invalid URL path \(path) for \(base.absoluteString).")
        }
        return url
    }

    private func wrappers(for chatModels: [String]) -> [OpenWebUIModelWrapper] {
        chatModels.map { modelID in
            let isVision = modelID == visionModel
            return OpenWebUIModelWrapper(
                id: modelID,
                baseModelID: modelID,
                name: "mere.run \(modelID)",
                meta: OpenWebUIModelMetadata(
                    capabilities: OpenWebUIModelCapabilities.default(vision: isVision),
                    description: "mere.run serves \(modelID) locally."
                ),
                params: OpenWebUIModelParams.default,
                isActive: true
            )
        }
    }

    private func uniqueModelIDs(_ ids: [String]) -> [String] {
        var seen: Set<String> = []
        var ordered: [String] = []
        for id in ids {
            guard let normalized = Self.normalized(id),
                  !seen.contains(normalized) else {
                continue
            }
            seen.insert(normalized)
            ordered.append(normalized)
        }
        return ordered
    }

    private static func normalized(_ rawValue: String?) -> String? {
        guard let trimmed = rawValue?.trimmingCharacters(in: .whitespacesAndNewlines),
              !trimmed.isEmpty else {
            return nil
        }
        return trimmed
    }

    private func apiServeDisplayCommand(engine: APIEngine, apiKey: String) -> String {
        CLICommandDisplay.command(
            [
                "api serve",
                "--engine \(ShellQuote.quote(engine.rawValue))",
                "--model \(ShellQuote.quote(textModel))",
                "--host \(ShellQuote.quote(host))",
                "--port \(port)",
                "--api-key \(apiKey)",
            ].joined(separator: " ")
        )
    }

    private func dockerDisplayCommand(apiKey: String) -> String {
        let args = dockerRunArguments(apiKey: apiKey)
        var lines: [String] = ["docker run -d \\"]
        var index = 2
        while index < args.count {
            let item = args[index]
            let next = index + 1 < args.count ? args[index + 1] : nil
            if item == "-e", let next {
                lines.append("  -e \(ShellQuote.quote(next)) \\")
                index += 2
            } else if item == "-p" || item == "--name" || item == "--restart" || item == "-v",
                      let next {
                lines.append("  \(item) \(ShellQuote.quote(next)) \\")
                index += 2
            } else if item.hasPrefix("--add-host=") {
                lines.append("  \(ShellQuote.quote(item)) \\")
                index += 1
            } else {
                lines.append("  \(ShellQuote.quote(item))")
                index += 1
            }
        }
        return lines.joined(separator: "\n")
    }

    private func runDocker(_ arguments: [String]) throws -> OpenWebUIProcessResult {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = ["docker"] + arguments
        process.standardInput = AgentServerLog.nullInputHandle()
        let stdout = Pipe()
        let stderr = Pipe()
        process.standardOutput = stdout
        process.standardError = stderr
        try process.run()
        process.waitUntilExit()
        let stdoutText = String(data: stdout.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
        let stderrText = String(data: stderr.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
        guard process.terminationStatus == 0 else {
            throw ValidationError("docker \(arguments.joined(separator: " ")) failed: \(stderrText)")
        }
        return OpenWebUIProcessResult(stdout: stdoutText, stderr: stderrText)
    }

    private func waitForServerProcess() -> Never {
        while true {
            Thread.sleep(forTimeInterval: 3600)
        }
    }
}

struct OpenWebUIQuickstartPlan {
    let apiKey: String
    let apiServeCommand: String
    let dockerCommand: String
    let configureCommand: String

    func render() -> String {
        """
        mere.run Open WebUI quickstart plan

        1. Export the API key:
           export MERERUN_API_KEY=\(ShellQuote.quote(apiKey))

        2. Start the local API:
           \(apiServeCommand)

        3. Start the official Open WebUI container:
        \(dockerCommand)

        4. Configure an already-running Open WebUI instance:
           \(configureCommand)

        Run without --dry-run to execute the Docker path.
        """
    }
}

private struct OpenWebUIEnvironmentValue {
    let name: String
    let value: String

    init(_ name: String, _ value: String) {
        self.name = name
        self.value = value
    }
}

private struct OpenWebUIProcessResult {
    let stdout: String
    let stderr: String
}

private struct OpenWebUIModelListPayload: Decodable {
    let data: [OpenWebUIListedModel]
}

private struct OpenWebUIListedModel: Decodable {
    let id: String
}

private struct OpenWebUIAuthSigninRequest: Encodable {
    let email: String
    let password: String
}

private struct OpenWebUIAuthSigninResponse: Decodable {
    let token: String?
}

private struct OpenWebUIOpenAIConfigPayload: Encodable {
    let enableOpenAIAPI: Bool
    let openAIAPIBaseURLs: [String]
    let openAIAPIKeys: [String]
    let openAIAPIConfigs: [String: OpenWebUIOpenAIConnectionConfig]

    init(apiBaseURL: String, apiKey: String, modelIDs: [String]) {
        self.enableOpenAIAPI = true
        self.openAIAPIBaseURLs = [apiBaseURL]
        self.openAIAPIKeys = [apiKey]
        self.openAIAPIConfigs = [
            "0": OpenWebUIOpenAIConnectionConfig(enable: true, modelIDs: modelIDs),
        ]
    }

    enum CodingKeys: String, CodingKey {
        case enableOpenAIAPI = "ENABLE_OPENAI_API"
        case openAIAPIBaseURLs = "OPENAI_API_BASE_URLS"
        case openAIAPIKeys = "OPENAI_API_KEYS"
        case openAIAPIConfigs = "OPENAI_API_CONFIGS"
    }
}

private struct OpenWebUIOpenAIConnectionConfig: Encodable {
    let enable: Bool
    let modelIDs: [String]

    enum CodingKeys: String, CodingKey {
        case enable
        case modelIDs = "model_ids"
    }
}

private struct OpenWebUIModelsConfigPayload: Encodable {
    let defaultModels: String
    let defaultPinnedModels: String
    let modelOrderList: [String]
    let defaultModelMetadata: OpenWebUIModelMetadata
    let defaultModelParams: OpenWebUIModelParams

    init(defaultModels: String, modelOrderList: [String]) {
        self.defaultModels = defaultModels
        self.defaultPinnedModels = defaultModels
        self.modelOrderList = modelOrderList
        self.defaultModelMetadata = .default
        self.defaultModelParams = .default
    }

    enum CodingKeys: String, CodingKey {
        case defaultModels = "DEFAULT_MODELS"
        case defaultPinnedModels = "DEFAULT_PINNED_MODELS"
        case modelOrderList = "MODEL_ORDER_LIST"
        case defaultModelMetadata = "DEFAULT_MODEL_METADATA"
        case defaultModelParams = "DEFAULT_MODEL_PARAMS"
    }
}

private struct OpenWebUIModelsImportPayload: Encodable {
    let models: [OpenWebUIModelWrapper]
}

private struct OpenWebUIModelWrapper: Encodable {
    let id: String
    let baseModelID: String
    let name: String
    let meta: OpenWebUIModelMetadata
    let params: OpenWebUIModelParams
    let isActive: Bool

    enum CodingKeys: String, CodingKey {
        case id
        case baseModelID = "base_model_id"
        case name
        case meta
        case params
        case isActive = "is_active"
    }
}

private struct OpenWebUIModelMetadata: Codable {
    let capabilities: OpenWebUIModelCapabilities
    let description: String?

    static let `default` = OpenWebUIModelMetadata(
        capabilities: .default(vision: false),
        description: nil
    )

    static var defaultJSONString: String {
        encodeCompact(OpenWebUIModelMetadata.default)
    }
}

private struct OpenWebUIModelCapabilities: Codable {
    let fileContext: Bool
    let vision: Bool
    let fileUpload: Bool
    let webSearch: Bool
    let imageGeneration: Bool
    let codeInterpreter: Bool
    let terminal: Bool
    let citations: Bool
    let statusUpdates: Bool
    let builtinTools: Bool

    static func `default`(vision: Bool) -> OpenWebUIModelCapabilities {
        OpenWebUIModelCapabilities(
            fileContext: true,
            vision: vision,
            fileUpload: true,
            webSearch: false,
            imageGeneration: true,
            codeInterpreter: false,
            terminal: false,
            citations: true,
            statusUpdates: true,
            builtinTools: true
        )
    }

    enum CodingKeys: String, CodingKey {
        case fileContext = "file_context"
        case vision
        case fileUpload = "file_upload"
        case webSearch = "web_search"
        case imageGeneration = "image_generation"
        case codeInterpreter = "code_interpreter"
        case terminal
        case citations
        case statusUpdates = "status_updates"
        case builtinTools = "builtin_tools"
    }
}

private struct OpenWebUIModelParams: Codable {
    let functionCalling: String

    static let `default` = OpenWebUIModelParams(functionCalling: "native")

    static var defaultJSONString: String {
        encodeCompact(OpenWebUIModelParams.default)
    }

    enum CodingKeys: String, CodingKey {
        case functionCalling = "function_calling"
    }
}

private func encodeCompact<Value: Encodable>(_ value: Value) -> String {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
    guard let data = try? encoder.encode(value),
          let text = String(data: data, encoding: .utf8) else {
        return "{}"
    }
    return text
}

private extension APIEngine {
    init(runtimeEngine: RuntimeServingEngine) throws {
        switch runtimeEngine.canonical {
        case .textCode:
            self = .textCode
        case .textChatKlein:
            self = .textChatKlein
        case .textChatGemma4:
            self = .textChatGemma4
        case .textChatLaguna:
            self = .textChatLaguna
        case .textChatQ36:
            self = .textChatQ36
        case .textChatQ35:
            self = .textChatQ36
        case .textChatLFM2:
            self = .textChatLFM2
        case .textChatDeepseekV4Flash:
            self = .textChatDeepseekV4Flash
        case .textChatMuseGlimmer:
            self = .textChatMuseGlimmer
        case .textChatNemotronH:
            self = .textChatNemotronH
        }
    }
}

private enum ShellQuote {
    static func quote(_ value: String) -> String {
        if value.isEmpty {
            return "''"
        }
        let safeScalars = CharacterSet(charactersIn: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_./:=@%+,$")
        if value.unicodeScalars.allSatisfy({ safeScalars.contains($0) }) {
            return value
        }
        return "'" + value.replacingOccurrences(of: "'", with: "'\\''") + "'"
    }
}
