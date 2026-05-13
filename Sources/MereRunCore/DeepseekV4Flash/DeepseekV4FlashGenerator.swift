import Foundation

/// Premier-tier agent runtime backed by the vendored DeepSeek V4 Flash engine.
///
/// Unlike the other ChatGenerators in this module (which load weights
/// in-process via MLX or llama.cpp), this generator wraps the bundled
/// `ds4-server` binary as a child process. The binary exposes an
/// OpenAI-compatible HTTP surface, so the generator simply proxies one
/// `/v1/chat/completions` POST per call over loopback.
///
/// Lifecycle:
/// - `prepare(...)` resolves the binary + GGUF, downloads the GGUF if
///   missing, spawns `ds4-server`, and waits until it answers `/v1/models`.
/// - Subsequent `chat(...)` calls reuse the same subprocess.
/// - `shutdown()` (and the actor's deinit) terminate the subprocess.
public actor DeepseekV4FlashGenerator: ChatGenerator {
    private let modelId: String
    private var process: Process?
    private var port: Int?
    private var stderrBuffer: String = ""
    private var stderrTask: Task<Void, Never>?
    private var loadStartedAt: Date?
    private var loadSeconds: Double = 0

    public init(modelId: String = DeepseekV4FlashResources.defaultModelId) {
        self.modelId = modelId
    }

    deinit {
        if let process, process.isRunning {
            process.terminate()
        }
    }

    public func chat(
        _ request: ChatRequest,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> ChatResponse {
        try await chat(request, modelPath: nil, progressHandler: progressHandler)
    }

    public func chat(
        _ request: ChatRequest,
        modelPath: String?,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> ChatResponse {
        try await ensureRunning(modelPath: modelPath, progressHandler: progressHandler)
        guard let port else {
            throw DeepseekV4FlashError.serverFailedToStart("no port allocated")
        }

        let payload = OpenAIChatRequest(
            model: modelId,
            messages: request.messages.map { OpenAIChatMessage(role: $0.role.rawValue, content: $0.content) },
            temperature: request.temperature,
            top_p: request.topP,
            max_tokens: request.maxTokens,
            stream: false
        )

        var urlRequest = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
        urlRequest.httpMethod = "POST"
        urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        urlRequest.httpBody = try JSONEncoder().encode(payload)
        urlRequest.timeoutInterval = 600

        let prefillStart = Date()
        let (data, response) = try await URLSession.shared.data(for: urlRequest)
        let httpStatus = (response as? HTTPURLResponse)?.statusCode ?? 0
        guard (200..<300).contains(httpStatus) else {
            let body = String(data: data, encoding: .utf8) ?? "<binary>"
            throw DeepseekV4FlashError.requestFailed("HTTP \(httpStatus): \(body)")
        }

        let decoded = try JSONDecoder().decode(OpenAIChatResponse.self, from: data)
        guard let choice = decoded.choices.first else {
            throw DeepseekV4FlashError.requestFailed("response has no choices")
        }
        let elapsed = Date().timeIntervalSince(prefillStart)
        return ChatResponse(
            response: choice.message?.content ?? choice.delta?.content ?? "",
            tokensGenerated: decoded.usage?.completion_tokens ?? 0,
            timing: ChatTiming(
                loadSeconds: loadSeconds,
                prefillSeconds: 0,
                decodeSeconds: elapsed
            )
        )
    }

    public func prepare(
        modelPath: String? = nil,
        progressHandler: (@Sendable (ChatProgress) -> Void)? = nil
    ) async throws {
        try await ensureRunning(modelPath: modelPath, progressHandler: progressHandler)
    }

    public func chatCompletionsURL(
        modelPath: String? = nil,
        progressHandler: (@Sendable (ChatProgress) -> Void)? = nil
    ) async throws -> URL {
        try await ensureRunning(modelPath: modelPath, progressHandler: progressHandler)
        guard let port,
              let url = URL(string: "http://127.0.0.1:\(port)/v1/chat/completions") else {
            throw DeepseekV4FlashError.serverFailedToStart("no port allocated")
        }
        return url
    }

    public func shutdown() {
        if let process, process.isRunning {
            process.terminate()
        }
        stderrTask?.cancel()
        stderrTask = nil
        process = nil
        port = nil
    }

    // MARK: - Lifecycle

    private func ensureRunning(
        modelPath: String?,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws {
        if let process, process.isRunning {
            return
        }
        if process != nil {
            process = nil
            port = nil
        }
        if let port, await isDS4ServerReady(port: port) {
            return
        }

        loadStartedAt = Date()
        progressHandler?(ChatProgress(stage: .loadingModel, message: "Locating ds4-server binary"))
        let binary = try DeepseekV4FlashBinary.locate(.server)

        progressHandler?(ChatProgress(stage: .loadingModel, message: "Resolving DeepSeek V4 Flash GGUF"))
        let ggufURL = try await resolveGGUF(explicitPath: modelPath, progressHandler: progressHandler)
        let lockURL = MereRunModelPaths.modelDir(DeepseekV4FlashResources.defaultModelId)
            .appendingPathComponent("mere-run-ds4-server.lock")

        if let attachedPort = await attachToExistingServer(lockURL: lockURL) {
            port = attachedPort
            loadSeconds = loadStartedAt.map { Date().timeIntervalSince($0) } ?? 0
            progressHandler?(ChatProgress(
                stage: .loadingModel,
                message: "Attached to existing ds4-server on 127.0.0.1:\(attachedPort)"
            ))
            return
        }

        let chosenPort = try pickFreeLoopbackPort()
        let kvDir = try makeKVDiskDir()

        let proc = Process()
        proc.executableURL = binary
        // ds4 looks for `metal/*.metal` shader sources relative to cwd, so root
        // the process at vendor/ds4 (binary's parent) where those files live.
        proc.currentDirectoryURL = binary.deletingLastPathComponent()
        proc.arguments = [
            "-m", ggufURL.path,
            "--host", "127.0.0.1",
            "--port", String(chosenPort),
            "--ctx", String(DeepseekV4FlashResources.defaultContextLength),
            "--kv-disk-dir", kvDir.path,
            "--kv-disk-space-mb", "8192",
        ]
        // Scope the ds4-server singleton lock to mere.run so it doesn't collide
        // with a user-launched `ds4` on the side. ds4 honors DS4_LOCK_FILE.
        var env = ProcessInfo.processInfo.environment
        env["DS4_LOCK_FILE"] = lockURL.path
        proc.environment = env

        let stderrPipe = Pipe()
        proc.standardError = stderrPipe
        proc.standardOutput = FileHandle.nullDevice
        stderrBuffer = ""
        let stderrReader = stderrPipe.fileHandleForReading
        stderrTask?.cancel()
        stderrTask = Task.detached { [weak self] in
            while !Task.isCancelled {
                let data = stderrReader.availableData
                if data.isEmpty {
                    try? await Task.sleep(nanoseconds: 200_000_000)
                    continue
                }
                if let chunk = String(data: data, encoding: .utf8) {
                    await self?.appendStderr(chunk)
                }
            }
        }

        progressHandler?(ChatProgress(
            stage: .loadingModel,
            message: "Starting ds4-server (this loads ~81 GB from disk)"
        ))
        try proc.run()
        process = proc
        port = chosenPort

        try await waitUntilReady(port: chosenPort, process: proc)
        loadSeconds = loadStartedAt.map { Date().timeIntervalSince($0) } ?? 0
        progressHandler?(ChatProgress(
            stage: .loadingModel,
            message: "ds4-server ready on 127.0.0.1:\(chosenPort)"
        ))
    }

    private func appendStderr(_ chunk: String) {
        stderrBuffer += chunk
        // Cap memory: keep the most recent 32 KiB.
        if stderrBuffer.count > 32 * 1024 {
            stderrBuffer = String(stderrBuffer.suffix(32 * 1024))
        }
    }

    private func waitUntilReady(port: Int, process: Process) async throws {
        let deadline = Date().addingTimeInterval(DeepseekV4FlashResources.serverStartupTimeoutSeconds)
        let url = URL(string: "http://127.0.0.1:\(port)/v1/models")!
        var request = URLRequest(url: url)
        request.timeoutInterval = 5

        while Date() < deadline {
            if !process.isRunning {
                throw DeepseekV4FlashError.serverExited(
                    process.terminationStatus,
                    stderr: stderrBuffer
                )
            }
            do {
                let (_, response) = try await URLSession.shared.data(for: request)
                if let http = response as? HTTPURLResponse, http.statusCode == 200 {
                    return
                }
            } catch {
                // Connection refused, etc. — server still booting.
            }
            try await Task.sleep(nanoseconds: 500_000_000)
        }

        process.terminate()
        throw DeepseekV4FlashError.serverFailedToStart(
            "/v1/models did not respond within 5 minutes. Last stderr:\n\(stderrBuffer)"
        )
    }

    private func attachToExistingServer(lockURL: URL) async -> Int? {
        guard let pidText = try? String(contentsOf: lockURL, encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines),
            let pid = Int32(pidText),
            let command = commandLine(forPID: pid),
            command.contains("ds4-server"),
            let port = portArgument(in: command),
            await isDS4ServerReady(port: port) else {
            return nil
        }
        return port
    }

    private func commandLine(forPID pid: Int32) -> String? {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/ps")
        process.arguments = ["-p", String(pid), "-o", "command="]
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = Pipe()
        do {
            try process.run()
            process.waitUntilExit()
        } catch {
            return nil
        }
        guard process.terminationStatus == 0 else { return nil }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        return String(data: data, encoding: .utf8)?
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func portArgument(in command: String) -> Int? {
        let pieces = command.split(whereSeparator: { $0 == " " || $0 == "\t" })
        guard let index = pieces.firstIndex(of: "--port") else { return nil }
        let valueIndex = pieces.index(after: index)
        guard valueIndex < pieces.endIndex else { return nil }
        return Int(pieces[valueIndex])
    }

    private func isDS4ServerReady(port: Int) async -> Bool {
        guard let url = URL(string: "http://127.0.0.1:\(port)/v1/models") else {
            return false
        }
        var request = URLRequest(url: url)
        request.timeoutInterval = 2
        do {
            let (_, response) = try await URLSession.shared.data(for: request)
            return (response as? HTTPURLResponse)?.statusCode == 200
        } catch {
            return false
        }
    }

    private func resolveGGUF(
        explicitPath: String?,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> URL {
        if let explicitPath {
            let url = URL(fileURLWithPath: explicitPath)
            if FileManager.default.fileExists(atPath: url.path) {
                return url
            }
            throw DeepseekV4FlashError.modelNotFound(url)
        }

        // 1. Try the canonical alias path (<modelDir>/<id>.gguf).
        let aliasURL = MereRunModelPaths.resolveModelFile(
            relativePath: DeepseekV4FlashResources.managedRelativePath,
            validator: { FileManager.default.fileExists(atPath: $0.path) }
        )
        if FileManager.default.fileExists(atPath: aliasURL.path) {
            return aliasURL
        }

        // 2. Try imatrix first (preferred per upstream README), then legacy,
        //    then any other .gguf in the install dir.
        let modelDir = MereRunModelPaths.modelDir(DeepseekV4FlashResources.defaultModelId)
        let imatrixURL = modelDir.appendingPathComponent(DeepseekV4FlashResources.imatrixGGUFFile)
        if FileManager.default.fileExists(atPath: imatrixURL.path) {
            return imatrixURL
        }
        let legacyURL = modelDir.appendingPathComponent(DeepseekV4FlashResources.legacyGGUFFile)
        if FileManager.default.fileExists(atPath: legacyURL.path) {
            return legacyURL
        }
        if let preferred = preferredGGUF(in: modelDir) {
            return preferred
        }

        // 3. Lazy-download via the standard HubFallbackConfig pipeline.
        do {
            let fileURL = try await PretrainedModelLoader.fromPretrainedFile(
                modelPath: nil,
                modelId: DeepseekV4FlashResources.defaultModelId,
                defaultModelIds: [DeepseekV4FlashResources.defaultModelId],
                relativePath: DeepseekV4FlashResources.managedRelativePath,
                hubFallback: DeepseekV4FlashResources.hubFallbackConfig,
                validate: { url, manager in
                    manager.fileExists(atPath: url.path) ? [] : [url]
                },
                progress: { event in
                    switch event {
                    case .downloading(let percent):
                        progressHandler?(ChatProgress(
                            stage: .loadingModel,
                            message: "Downloading DS4 GGUF (~81 GB): \(percent)%"
                        ))
                    case .extracting:
                        progressHandler?(ChatProgress(stage: .loadingModel, message: "Extracting"))
                    }
                }
            )
            return fileURL
        } catch let error as PretrainedModelLoader.LoadError {
            throw DeepseekV4FlashError.downloadFailed(error.localizedDescription)
        } catch {
            throw DeepseekV4FlashError.downloadFailed(error.localizedDescription)
        }
    }

    /// Returns the most preferred .gguf in `directory` — files whose name
    /// contains "imatrix" win over those that do not, matching the upstream
    /// README's recommendation to prefer the imatrix variant.
    private func preferredGGUF(in directory: URL) -> URL? {
        let fileManager = FileManager.default
        guard let enumerator = fileManager.enumerator(
            at: directory,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else {
            return nil
        }
        var imatrix: URL?
        var fallback: URL?
        for case let url as URL in enumerator {
            guard url.pathExtension.lowercased() == "gguf",
                  let values = try? url.resourceValues(forKeys: [.isRegularFileKey]),
                  values.isRegularFile == true else {
                continue
            }
            if url.lastPathComponent.lowercased().contains("imatrix") {
                imatrix = url
                break
            }
            if fallback == nil {
                fallback = url
            }
        }
        return imatrix ?? fallback
    }

    private func makeKVDiskDir() throws -> URL {
        let base = MereRunModelPaths.modelDir(DeepseekV4FlashResources.defaultModelId)
            .appendingPathComponent("kv-disk", isDirectory: true)
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        return base
    }

    private func pickFreeLoopbackPort() throws -> Int {
        let sock = socket(AF_INET, SOCK_STREAM, 0)
        guard sock >= 0 else {
            throw DeepseekV4FlashError.serverFailedToStart("socket() failed")
        }
        defer { close(sock) }

        var addr = sockaddr_in()
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = 0
        addr.sin_addr.s_addr = inet_addr("127.0.0.1")

        let bindResult = withUnsafePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                bind(sock, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        guard bindResult == 0 else {
            throw DeepseekV4FlashError.serverFailedToStart("bind() failed: \(errno)")
        }

        var bound = sockaddr_in()
        var boundLen = socklen_t(MemoryLayout<sockaddr_in>.size)
        let nameResult = withUnsafeMutablePointer(to: &bound) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                getsockname(sock, $0, &boundLen)
            }
        }
        guard nameResult == 0 else {
            throw DeepseekV4FlashError.serverFailedToStart("getsockname() failed: \(errno)")
        }
        return Int(UInt16(bigEndian: bound.sin_port))
    }
}

// Wire types `OpenAIChatRequest`, `OpenAIChatMessage`, and `OpenAIChatResponse`
// are defined in `MereRunCore/CodeGen/OpenAITypes.swift` and reused here.
