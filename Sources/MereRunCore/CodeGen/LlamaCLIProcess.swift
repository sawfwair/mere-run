#if os(Linux)
import Foundation

enum LlamaCLIProcessError: LocalizedError {
    case executableNotFound
    case executionFailed(status: Int32, stderr: String)
    case emptyResponse(stderr: String)

    var errorDescription: String? {
        switch self {
        case .executableNotFound:
            return "llama-cli was not found. Set MERERUN_LLAMA_CLI or use the packaged Linux runtime."
        case .executionFailed(let status, let stderr):
            let detail = stderr.trimmingCharacters(in: .whitespacesAndNewlines)
            return "llama-cli exited with status \(status).\(detail.isEmpty ? "" : " \(detail)")"
        case .emptyResponse(let stderr):
            let detail = stderr.trimmingCharacters(in: .whitespacesAndNewlines)
            return "llama-cli produced no response.\(detail.isEmpty ? "" : " \(detail)")"
        }
    }
}

struct LlamaCLIProcess {
    let executableURL: URL

    struct Performance: Sendable, Hashable {
        var promptTokensPerSecond: Double?
        var generationTokensPerSecond: Double?
    }

    static func discover(
        environment: [String: String] = ProcessInfo.processInfo.environment,
        arguments: [String] = CommandLine.arguments,
        fileManager: FileManager = .default
    ) -> LlamaCLIProcess? {
        let candidates = executableCandidates(environment: environment, arguments: arguments)
        for candidate in candidates where fileManager.isExecutableFile(atPath: candidate.path) {
            return LlamaCLIProcess(executableURL: candidate)
        }
        return nil
    }

    static func executableCandidates(environment: [String: String], arguments: [String]) -> [URL] {
        var candidates: [URL] = []

        if let override = environment["MERERUN_LLAMA_CLI"], !override.isEmpty {
            candidates.append(URL(fileURLWithPath: override).standardizedFileURL)
        }

        if let executablePath = arguments.first, !executablePath.isEmpty {
            let executableURL = URL(fileURLWithPath: executablePath).standardizedFileURL
            let root = executableURL.deletingLastPathComponent()
            candidates.append(root.appendingPathComponent("llama-cli"))
            candidates.append(root.appendingPathComponent("bin/llama-cli"))
            candidates.append(root.appendingPathComponent("libexec/llama-cli"))
        }

        candidates.append(URL(fileURLWithPath: "/usr/lib/mere-run/llama-cli"))

        var seen: Set<String> = []
        var unique: [URL] = []
        for candidate in candidates {
            let path = candidate.path
            if seen.insert(path).inserted {
                unique.append(candidate)
            }
        }
        return unique
    }

    func chat(
        request: ChatRequest,
        modelPath: String,
        progressHandler: (@Sendable (ChatProgress) -> Void)? = nil
    ) async throws -> ChatResponse {
        try await Task.detached(priority: .userInitiated) {
            let invocation = Self.invocation(for: request, modelPath: modelPath)
            progressHandler?(ChatProgress(stage: .generating, message: "Generating..."))
            let result = try run(arguments: invocation.arguments)
            let response = Self.extractResponse(from: result.stdout)
            let performance = Self.extractPerformance(from: result.stdout)
            guard !response.isEmpty else {
                throw LlamaCLIProcessError.emptyResponse(stderr: Self.tail(result.stderr))
            }
            progressHandler?(ChatProgress(stage: .generating, message: response))
            return ChatResponse(
                response: response,
                tokensGenerated: max(1, response.split(whereSeparator: { $0.isWhitespace }).count),
                timing: performance.chatTiming
            )
        }.value
    }

    private func run(arguments: [String]) throws -> (stdout: String, stderr: String) {
        let process = Process()
        process.executableURL = executableURL
        process.arguments = arguments
        process.environment = processEnvironment()

        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe

        try process.run()
        process.waitUntilExit()

        let stdout = String(data: stdoutPipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
        let stderr = String(data: stderrPipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
        guard process.terminationStatus == 0 else {
            throw LlamaCLIProcessError.executionFailed(status: process.terminationStatus, stderr: Self.tail(stderr))
        }
        return (stdout, stderr)
    }

    private func processEnvironment() -> [String: String] {
        var environment = ProcessInfo.processInfo.environment
        let root = executableURL.deletingLastPathComponent()
        let packagedLib = root.appendingPathComponent("lib").path
        let existing = environment["LD_LIBRARY_PATH"]
        if FileManager.default.fileExists(atPath: packagedLib) {
            environment["LD_LIBRARY_PATH"] = [packagedLib, existing]
                .compactMap { $0 }
                .filter { !$0.isEmpty }
                .joined(separator: ":")
        }
        return environment
    }

    private static func invocation(for request: ChatRequest, modelPath: String) -> (arguments: [String], prompt: String) {
        let promptParts = promptParts(from: request.messages)
        var arguments = [
            "-m", modelPath,
            "-p", promptParts.prompt,
            "-n", String(max(1, request.maxTokens)),
            "-ngl", String(linuxGPULayerCount()),
            "-st",
            "--simple-io",
            "--no-display-prompt",
            "--temp", String(request.temperature),
            "--top-p", String(request.topP),
        ]
        if let system = promptParts.system, !system.isEmpty {
            arguments.insert(contentsOf: ["-sys", system], at: 4)
        }
        return (arguments, promptParts.prompt)
    }

    private static func promptParts(from messages: [ChatMessage]) -> (system: String?, prompt: String) {
        let system = messages
            .filter { $0.role == .system }
            .map(\.content)
            .joined(separator: "\n\n")

        let conversationalMessages = messages.filter { $0.role != .system }
        if conversationalMessages.count == 1, let only = conversationalMessages.first, only.role == .user {
            return (system.isEmpty ? nil : system, only.content)
        }

        let prompt = conversationalMessages.map { message in
            switch message.role {
            case .assistant:
                return "Assistant: \(message.content)"
            case .tool:
                return "Tool: \(message.content)"
            case .user:
                return "User: \(message.content)"
            case .system:
                return message.content
            }
        }.joined(separator: "\n\n")

        return (system.isEmpty ? nil : system, prompt)
    }

    private static func linuxGPULayerCount(environment: [String: String] = ProcessInfo.processInfo.environment) -> Int {
        if let rawValue = environment["MERERUN_LLAMA_GPU_LAYERS"] {
            let trimmed = rawValue.trimmingCharacters(in: .whitespacesAndNewlines)
            if let parsed = Int(trimmed) {
                return max(0, parsed)
            }
        }

        if environment["MERERUN_LINUX_ACCEL"]?.lowercased() == "cuda" {
            return 999
        }

        return 0
    }

    static func extractResponse(from stdout: String) -> String {
        let lines = stdout.components(separatedBy: .newlines)
        var responseLines: [String] = []
        var seenPrompt = false

        for line in lines {
            let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.hasPrefix("[ Prompt:") || trimmed == "Exiting..." {
                break
            }
            if line.hasPrefix("> ") {
                seenPrompt = true
                responseLines.removeAll()
                continue
            }
            if seenPrompt {
                responseLines.append(line)
            }
        }

        let response = responseLines
            .joined(separator: "\n")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if !response.isEmpty {
            return response
        }

        return lines
            .filter { line in
                let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
                return !trimmed.isEmpty &&
                    !trimmed.hasPrefix("[ Prompt:") &&
                    trimmed != "Loading model..." &&
                    trimmed != "Exiting..." &&
                    !trimmed.hasPrefix("build") &&
                    !trimmed.hasPrefix("model") &&
                    !trimmed.hasPrefix("modalities") &&
                    !trimmed.hasPrefix("available commands:") &&
                    !trimmed.hasPrefix("/") &&
                    !trimmed.hasPrefix("▄▄") &&
                    !trimmed.hasPrefix("██") &&
                    !trimmed.hasPrefix("▀▀")
            }
            .joined(separator: "\n")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    static func extractPerformance(from stdout: String) -> Performance {
        let pattern = #"\[\s*Prompt:\s*([0-9.]+)\s*t/s\s*\|\s*Generation:\s*([0-9.]+)\s*t/s\s*\]"#
        guard let regex = try? NSRegularExpression(pattern: pattern) else {
            return Performance()
        }

        let range = NSRange(stdout.startIndex..<stdout.endIndex, in: stdout)
        guard let match = regex.matches(in: stdout, range: range).last else {
            return Performance()
        }

        func value(at index: Int) -> Double? {
            let nsRange = match.range(at: index)
            guard let range = Range(nsRange, in: stdout) else { return nil }
            return Double(stdout[range])
        }

        return Performance(
            promptTokensPerSecond: value(at: 1),
            generationTokensPerSecond: value(at: 2)
        )
    }

    private static func tail(_ text: String, maxLines: Int = 40) -> String {
        text.components(separatedBy: .newlines).suffix(maxLines).joined(separator: "\n")
    }
}

private extension LlamaCLIProcess.Performance {
    var chatTiming: ChatTiming? {
        guard promptTokensPerSecond != nil || generationTokensPerSecond != nil else {
            return nil
        }
        return ChatTiming(
            prefillTokensPerSecond: promptTokensPerSecond,
            decodeTokensPerSecond: generationTokensPerSecond
        )
    }
}
#endif
