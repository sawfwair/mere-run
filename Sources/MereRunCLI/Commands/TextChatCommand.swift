import ArgumentParser
import Foundation
import MereRunCore
#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

// MARK: - Text Chat Command

struct TextChat: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "chat",
        abstract: "Run local chat with text chat models.",
        discussion: """
        Auto-downloads the selected model on first use. The default is
        text-chat-q36-nano on Apple Silicon (MLX) and text-chat-q36-nano-gguf on
        Linux CUDA (llama.cpp) — the fastest strong chat model on each platform.
        Known model IDs:
          - text-chat-q36-nano (Qwen3.6-35B-A3B OptiQ 4-bit, default on Apple Silicon)
          - text-chat-q36-nano-gguf (Qwen3.6-35B-A3B GGUF, default on Linux CUDA)
          - text-agent-ornith-9b (Ornith 1.0 9B OptiQ, experimental coding-agent target)
          - text-chat-gemma4-12b (Gemma 4 12B dense native Swift runtime)
          - text-chat-gemma4-12b-4bit (Gemma 4 12B MLX 4-bit native Swift runtime)
          - text-chat-gemma4-turbo (Gemma 4 26B-A4B NVFP4 native Swift runtime)
          - text-chat-gemma4 (Gemma 4 31B; large/slow, kept for compatibility)
          - text-chat-gemma4-max (Gemma 4 31B native Swift runtime)
          - text-chat-gemma4-nano (Gemma 4 4B native Swift runtime)
          - text-chat-lfm25-a1b-8bit (LiquidAI LFM2.5 8B-A1B MLX 8-bit native Swift runtime)
          - text-chat-psi-agent
        Models are cached under ~/Library/Application Support/MereRun/models/<model-id>.
        Thinking output is hidden by default; pass --thinking to include it.
        Use --models-root or MERERUN_MODELS_DIR to override the model store path.
        """
    )

    @Option(name: [.customShort("p"), .long], help: "User prompt.")
    var prompt: String

    @Option(name: [.long], help: "Optional image path for vision-capable chat models such as vision-chat-gemma4-12b.")
    var image: String?

    @Option(name: [.customShort("s"), .customLong("system")], help: "System prompt.")
    var systemPrompt: String?

    @Option(name: [.long], help: "Max new tokens.")
    var maxTokens: Int = 2048

    @Option(name: [.long], help: "Temperature.")
    var temperature: Double = 0.7

    @Option(name: [.long], help: "Top-p.")
    var topP: Double = 0.9

    @Option(name: [.long], help: "Quantize the Gemma4 KV cache to this many bits. Supports integer widths for uniform/polar and integer/.5 widths for turboquant.")
    var kvBits: Double?

    @Option(name: [.long], help: "Gemma4 KV cache quantization backend: uniform, polar, or turboquant.")
    var kvQuantScheme: String?

    @Option(name: [.long], help: "Gemma4 KV cache quantization group size.")
    var kvGroupSize: Int?

    @Option(name: [.long], help: "Gemma4 token offset at which KV cache quantization begins.")
    var quantizedKVStart: Int?

    @Option(name: [.customShort("m"), .long], help: "Override model root directory (skips auto-download).")
    var modelRoot: String?

    /// Hardware-aware default chat model. Picks the strongest chat model whose
    /// minimum unified memory fits the machine (via MereRunMachineProfile +
    /// the capability catalog), and the right engine per platform: Qwen3.6-35B-A3B
    /// as MLX on Apple Silicon (~64 tok/s on M4 Max) or GGUF/llama.cpp on Linux
    /// CUDA (~68 tok/s on GB10, vs ~13 for MLX there). Below the A3B memory tier
    /// it steps down to Gemma 4 12B 4-bit, then nano as the final fallback.
    static var defaultChatModelId: String {
        let machine = MereRunMachineProfile.current
        func fits(_ id: String) -> Bool {
            guard let descriptor = ManagedModelCapabilityCatalog.descriptor(for: id) else { return false }
            return machine.unifiedMemoryGB >= descriptor.minimumUnifiedMemoryGB
        }

        let isCUDA: Bool
        #if os(Linux)
        isCUDA = ProcessInfo.processInfo.environment["MERERUN_LINUX_ACCEL"]?.lowercased() == "cuda"
        #else
        isCUDA = false
        #endif

        #if os(Linux)
        let a3b = isCUDA ? "text-chat-q36-nano-gguf" : Q35Resources.q36NanoModelId
        #else
        let a3b = Q35Resources.q36NanoModelId
        #endif
        if fits(a3b) { return a3b }
        if fits(Gemma4Resources.twelveB4BitModelId) { return Gemma4Resources.twelveB4BitModelId }
        return Gemma4Resources.nanoModelId
    }

    static func backendDescription(for modelID: String) -> String {
        let normalizedModelID = modelID.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if ManagedModelCatalog.spec(for: normalizedModelID)?.validationKind == .codegenGGUF {
            return "llama.cpp/GGUF"
        }
        return NativeMLXRuntime.backendDescription
    }

    @Option(name: [.long], help: "Canonical model id. Default: text-chat-q36-nano (Apple Silicon) / text-chat-q36-nano-gguf (Linux CUDA). Others: text-agent-ornith-9b, text-chat-gemma4[-12b|-12b-4bit|-turbo|-max|-nano], text-chat-lfm25-a1b-8bit, text-chat-psi-agent.")
    var model: String = TextChat.defaultChatModelId

    @Flag(name: [.customLong("thinking"), .customLong("show-thinking")], help: "Show model reasoning output.")
    var thinking: Bool = false

    @Flag(name: [.customLong("stats")], help: "Print generation timing and tokens/sec.")
    var stats: Bool = false

    @Flag(name: [.customLong("stream")], help: "Stream generated text to stdout as tokens arrive.")
    var stream: Bool = false

    @Option(name: [.customLong("tools")], help: "Comma-separated built-in tool names: write_file, shell_exec.")
    var tools: String?

    @Flag(name: [.customLong("tool-loop")], help: "Enable agentic tool loop: generate → execute tool calls → feed results back → continue.")
    var toolLoop: Bool = false

    @Option(name: [.customLong("sandbox-dir")], help: "Working directory for tool execution (default: temp dir).")
    var sandboxDir: String?

    @Flag(name: [.customLong("allow-shell-exec")], help: "Allow the model to execute shell commands when shell_exec is enabled.")
    var allowShellExec: Bool = false

    @Flag(name: [.customLong("allow-absolute-tool-paths")], help: "Allow write_file to target absolute paths outside the sandbox.")
    var allowAbsoluteToolPaths: Bool = false

    @Flag(name: [.customLong("auto-approve-tools")], help: "Execute tool calls without interactive confirmation.")
    var autoApproveTools: Bool = false

    @Flag(name: [.short, .long], help: "Suppress progress output.")
    var quiet: Bool = false

    func run() async throws {
        try MLXBundleSupport.ensureAvailable(quiet: quiet)

        let imageReference: String?
        if let image, !image.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            let trimmedImage = image.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmedImage.lowercased().hasPrefix("data:image/") {
                imageReference = trimmedImage
            } else {
                let imageURL = URL(fileURLWithPath: trimmedImage).standardizedFileURL
                guard FileManager.default.fileExists(atPath: imageURL.path) else {
                    throw ValidationError("Image file not found: \(imageURL.path)")
                }
                imageReference = imageURL.path
            }
        } else {
            imageReference = nil
        }

        var messages: [ChatMessage] = []
        if let systemPrompt, !systemPrompt.isEmpty {
            messages.append(ChatMessage(role: .system, content: systemPrompt))
        }
        messages.append(ChatMessage(role: .user, content: prompt, imageUrl: imageReference))

        let toolDefs: [ToolDefinition]? = try tools.flatMap { raw in
            let names = raw.split(separator: ",").map { String($0).trimmingCharacters(in: .whitespaces) }
            return try BuiltinTools.resolve(names: names)
        }
        if toolDefs?.contains(where: { $0.name == "shell_exec" }) == true, !allowShellExec {
            throw ValidationError("The 'shell_exec' tool requires --allow-shell-exec.")
        }

        let request = ChatRequest(
            messages: messages,
            maxTokens: maxTokens,
            temperature: temperature,
            topP: topP,
            showThinking: thinking,
            tools: toolDefs
        )

        let streamingOutput = StreamingChatOutput(enabled: stream)
        let progressHandler: (@Sendable (ChatProgress) -> Void)?
        if quiet {
            progressHandler = nil
        } else {
            progressHandler = { progress in
                if streamingOutput.write(progress: progress) {
                    return
                }
                CLIStderr.write("[\(progress.stage.rawValue)] \(progress.message ?? "")\n")
            }
        }

        let startTime = Date()
        let normalizedModelId = model.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if !quiet {
            CLIStderr.write("[runtime] text backend: \(Self.backendDescription(for: normalizedModelId))\n")
        }
        var lastGemma4MTPStats: Gemma4MTPStats?

        let chatOnce: (ChatRequest) async throws -> ChatResponse = { req in
            if normalizedModelId == Psi3ChatResources.defaultModelId {
                let generator = Psi3ChatGenerator(modelId: Psi3ChatResources.defaultModelId)
                return try await generator.chat(req, modelPath: self.modelRoot, progressHandler: progressHandler)
            } else if Gemma4Resources.handles(modelSpec: normalizedModelId) {
                let effectiveModelId = normalizedModelId.isEmpty ? Gemma4Resources.defaultModelId : normalizedModelId
                let kvQuantization = try self.resolveGemma4KVCacheQuantization(for: effectiveModelId)
                let generator = Gemma4Generator(
                    modelId: effectiveModelId,
                    kvCacheQuantization: kvQuantization
                )
                let response = try await generator.chat(req, modelPath: self.modelRoot, progressHandler: progressHandler)
                lastGemma4MTPStats = await generator.mtpStats()
                return response
            } else if ManagedModelCatalog.spec(for: normalizedModelId)?.validationKind == .codegenGGUF {
                // GGUF chat models run through the llama.cpp engine (the same path
                // `text code` uses). On Linux CUDA this is the GB10-optimized
                // llama.cpp runtime, which has fast quantized-MoE kernels MLX lacks.
                let generator = CodeGenGenerator(modelId: normalizedModelId)
                return try await generator.chat(req, modelPath: self.modelRoot, progressHandler: progressHandler)
            } else if LFM2Resources.handles(modelSpec: normalizedModelId) {
                let effectiveModelId = normalizedModelId.isEmpty ? LFM2Resources.defaultModelId : normalizedModelId
                let generator = LFM2Generator(modelId: effectiveModelId)
                return try await generator.chat(req, modelPath: self.modelRoot, progressHandler: progressHandler)
            } else {
                let effectiveModelId = normalizedModelId.isEmpty ? Q35Resources.defaultModelId : normalizedModelId
                let generator = Q35Generator(modelId: effectiveModelId)
                return try await generator.chat(req, modelPath: self.modelRoot, progressHandler: progressHandler)
            }
        }

        if toolLoop, let toolDefs, !toolDefs.isEmpty {
            let sandbox: URL
            if let sandboxDir {
                sandbox = URL(fileURLWithPath: sandboxDir).standardizedFileURL
            } else {
                sandbox = FileManager.default.temporaryDirectory
                    .appendingPathComponent("mererun-tools-\(ProcessInfo.processInfo.processIdentifier)")
            }
            try FileManager.default.createDirectory(at: sandbox, withIntermediateDirectories: true)
            if !quiet { CLIStderr.write("[tool-loop] Sandbox: \(sandbox.path)\n") }
            let toolPolicy = BuiltinTools.ToolExecutionPolicy(
                sandboxDir: sandbox,
                allowShellExec: allowShellExec,
                allowAbsolutePaths: allowAbsoluteToolPaths
            )

            var loopMessages = messages
            let maxIterations = 10

            for iteration in 0..<maxIterations {
                var req = request
                req.messages = loopMessages

                let result = try await chatOnce(req)

                guard let calls = result.toolCalls, !calls.isEmpty else {
                    if stream && streamingOutput.hasWritten {
                        streamingOutput.finishLine()
                    } else {
                        print(cleanResponse(result.response, showThinking: thinking))
                    }
                    return
                }

                // Show the model's response (may contain text before/after tool calls)
                let textBeforeTools = result.response
                    .replacingOccurrences(of: "<\\|tool_call>.*?<tool_call\\|>", with: "", options: .regularExpression)
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                if !textBeforeTools.isEmpty {
                    CLIStderr.write(cleanResponse(textBeforeTools, showThinking: thinking) + "\n")
                }

                loopMessages.append(ChatMessage(role: .assistant, content: result.response))

                for call in calls {
                    if !quiet { CLIStderr.write("[tool] \(call.name)(\(call.arguments.map { "\($0.key)=\($0.value.prefix(80))" }.joined(separator: ", ")))\n") }
                    let approved = BuiltinTools.canAutoApprove(call, autoApproveTools: autoApproveTools)
                        || confirmToolCall(
                            call,
                            sandbox: sandbox,
                            autoApproveToolsRequested: autoApproveTools
                        )
                    let output: String
                    if approved {
                        do {
                            output = try BuiltinTools.execute(call, policy: toolPolicy)
                        } catch {
                            output = "Error: \(error.localizedDescription)"
                        }
                    } else {
                        output = "Denied: tool execution was not approved."
                    }
                    if !quiet { CLIStderr.write("[tool] → \(output.prefix(200))\n") }
                    loopMessages.append(ChatMessage(role: .tool, content: output))
                }

                if !quiet { CLIStderr.write("[tool-loop] Iteration \(iteration + 1)/\(maxIterations)\n") }
            }

            CLIStderr.write("[tool-loop] Hit iteration limit (\(maxIterations))\n")
        } else {
            let result = try await chatOnce(request)

            let elapsed = Date().timeIntervalSince(startTime)
            if stats {
                let e2eTps = elapsed > 0 ? Double(result.tokensGenerated) / elapsed : 0
                if let timing = result.timing {
                    let decodeTps = timing.decodeTokensPerSecond
                        ?? (timing.decodeSeconds > 0 ? Double(result.tokensGenerated) / timing.decodeSeconds : 0)
                    var line = String(
                        format: "time=%.2fs load=%.2fs prefill=%.2fs decode=%.2fs tokens=%d decode_tps=%.2f e2e_tps=%.2f",
                        elapsed,
                        timing.loadSeconds,
                        timing.prefillSeconds,
                        timing.decodeSeconds,
                        result.tokensGenerated,
                        decodeTps,
                        e2eTps
                    )
                    if let prefillTps = timing.prefillTokensPerSecond {
                        line += String(format: " prefill_tps=%.2f", prefillTps)
                    }
                    CLIStderr.write("\(line)\n")
                    if let mtp = lastGemma4MTPStats {
                        CLIStderr.write(Self.formatGemma4MTPStats(mtp) + "\n")
                    }
                } else {
                    let line = String(format: "time=%.2fs tokens=%d tps=%.2f", elapsed, result.tokensGenerated, e2eTps)
                    CLIStderr.write("\(line)\n")
                    if let mtp = lastGemma4MTPStats {
                        CLIStderr.write(Self.formatGemma4MTPStats(mtp) + "\n")
                    }
                }
            }

            if stream && streamingOutput.hasWritten {
                streamingOutput.finishLine()
            } else {
                print(cleanResponse(result.response, showThinking: thinking))
            }
        }
    }

    static func formatGemma4MTPStats(_ stats: Gemma4MTPStats) -> String {
        let state: String
        if stats.active {
            state = "active"
        } else if stats.available {
            state = stats.enabled ? "available" : "disabled"
        } else {
            state = stats.enabled ? "unavailable" : "disabled"
        }
        let reason = stats.reason.map { " reason=\($0)" } ?? ""
        return "mtp=\(state) block=\(stats.blockSize) threshold=\(stats.threshold) rounds=\(stats.rounds) drafted=\(stats.draftedTokens) accepted=\(stats.acceptedTokens) rejected=\(stats.rejectedTokens)\(reason)"
    }

    func cleanResponse(_ response: String, showThinking: Bool) -> String {
        guard !showThinking else { return response }
        var cleaned = response.replacingOccurrences(
            of: "(?is)<think>.*?</think>",
            with: "",
            options: .regularExpression
        )
        cleaned = cleaned.replacingOccurrences(
            of: "(?is)<think>.*\\z",
            with: "",
            options: .regularExpression
        )
        cleaned = cleaned.replacingOccurrences(
            of: "(?i)</think>",
            with: "",
            options: .regularExpression
        )
        return cleaned.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    func resolveGemma4KVCacheQuantization(for modelId: String) throws -> Gemma4KVCacheQuantization {
        let usesTurboDefaults = Gemma4Resources.usesTurboDefaults(modelSpec: modelId)
            && Gemma4Resources.supportsDefaultTurboKVQuantization
        let defaultScheme = usesTurboDefaults
            ? Gemma4Resources.defaultTurboKVQuantizationScheme.rawValue
            : Gemma4Resources.defaultKVQuantizationScheme.rawValue
        let scheme = try parseGemma4KVQuantizationScheme(kvQuantScheme ?? defaultScheme)

        return Gemma4KVCacheQuantization(
            bits: kvBits ?? (usesTurboDefaults ? Gemma4Resources.defaultTurboKVBits : nil),
            scheme: scheme,
            groupSize: kvGroupSize ?? Gemma4Resources.defaultKVGroupSize,
            quantizedStart: quantizedKVStart ?? (usesTurboDefaults
                ? Gemma4Resources.defaultTurboQuantizedKVStart
                : Gemma4Resources.defaultQuantizedKVStart)
        )
    }

    private func parseGemma4KVQuantizationScheme(_ raw: String) throws -> Gemma4KVQuantizationScheme {
        guard let scheme = Gemma4KVQuantizationScheme(
            rawValue: raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        ) else {
            throw ValidationError("Unsupported --kv-quant-scheme '\(raw)'. Expected 'uniform', 'polar', or 'turboquant'.")
        }
        return scheme
    }

    private func confirmToolCall(
        _ call: ToolCall,
        sandbox: URL,
        autoApproveToolsRequested: Bool = false
    ) -> Bool {
        guard Self.stdinIsInteractive() else {
            if autoApproveToolsRequested && call.name == "shell_exec" {
                CLIStderr.write("[tool] Denied shell_exec: this tool always requires interactive approval, even when --auto-approve-tools is set.\n")
            } else {
                CLIStderr.write("[tool] Denied \(call.name): stdin is not interactive. Re-run with --auto-approve-tools to allow non-interactive execution.\n")
            }
            return false
        }

        let args = call.arguments
            .sorted { $0.key < $1.key }
            .map { "\($0.key)=\($0.value)" }
            .joined(separator: ", ")
        CLIStderr.write("[tool] Approve \(call.name)(\(args)) in \(sandbox.path)? [y/N] ")

        guard let line = readLine(strippingNewline: true)?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased() else {
            return false
        }
        return line == "y" || line == "yes"
    }

    private static func stdinIsInteractive() -> Bool {
        CLIStdin.isInteractive()
    }
}

private final class StreamingChatOutput: @unchecked Sendable {
    let enabled: Bool

    private let lock = NSLock()
    private var wroteOutput = false

    init(enabled: Bool) {
        self.enabled = enabled
    }

    var hasWritten: Bool {
        lock.lock()
        defer { lock.unlock() }
        return wroteOutput
    }

    func write(progress: ChatProgress) -> Bool {
        guard enabled, progress.stage == .generating else {
            return false
        }

        let text = progress.message ?? ""
        guard !text.isEmpty, text != "Generating..." else {
            return true
        }

        lock.lock()
        defer { lock.unlock() }
        CLIStdout.write(text)
        wroteOutput = true
        return true
    }

    func finishLine() {
        lock.lock()
        defer { lock.unlock() }
        CLIStdout.write("\n")
    }
}
