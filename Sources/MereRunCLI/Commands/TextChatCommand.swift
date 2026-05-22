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
        Auto-downloads the selected model on first use.
        Known model IDs:
          - text-chat-gemma4 (Gemma 4 default alias, currently 31B)
          - text-chat-gemma4-turbo (Gemma 4 26B-A4B NVFP4 native Swift runtime)
          - text-chat-gemma4-max (Gemma 4 31B native Swift runtime)
          - text-chat-gemma4-nano (Gemma 4 4B native Swift runtime)
          - text-chat-q35-nano (Qwen3.5-35B-A3B 4-bit)
          - text-chat-q35
          - text-chat-psi-agent
        Models are cached under ~/Library/Application Support/MereRun/models/<model-id>.
        Thinking output is hidden by default; pass --thinking to include it.
        Use --models-root or MERERUN_MODELS_DIR to override the model store path.
        """
    )

    @Option(name: [.customShort("p"), .long], help: "User prompt.")
    var prompt: String

    @Option(name: [.customShort("s"), .customLong("system")], help: "System prompt.")
    var systemPrompt: String?

    @Option(name: [.long], help: "Max new tokens.")
    var maxTokens: Int = 2048

    @Option(name: [.long], help: "Temperature.")
    var temperature: Double = 0.7

    @Option(name: [.long], help: "Top-p.")
    var topP: Double = 0.9

    @Option(name: [.long], help: "Quantize the Gemma4 KV cache to this many bits. Supports integer widths for uniform and integer/.5 widths for turboquant.")
    var kvBits: Double?

    @Option(name: [.long], help: "Gemma4 KV cache quantization backend: uniform or turboquant.")
    var kvQuantScheme: String?

    @Option(name: [.long], help: "Gemma4 KV cache quantization group size.")
    var kvGroupSize: Int?

    @Option(name: [.long], help: "Gemma4 token offset at which KV cache quantization begins.")
    var quantizedKVStart: Int?

    @Option(name: [.customShort("m"), .long], help: "Override model root directory (skips auto-download).")
    var modelRoot: String?

    @Option(name: [.long], help: "Canonical model id: text-chat-gemma4 (default alias), text-chat-gemma4-turbo, text-chat-gemma4-max, text-chat-gemma4-nano, text-chat-q35, text-chat-q35-nano, or text-chat-psi-agent.")
    var model: String = Gemma4Resources.defaultModelId

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

        var messages: [ChatMessage] = []
        if let systemPrompt, !systemPrompt.isEmpty {
            messages.append(ChatMessage(role: .system, content: systemPrompt))
        }
        messages.append(ChatMessage(role: .user, content: prompt))

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
                    let decodeTps = timing.decodeSeconds > 0
                        ? Double(result.tokensGenerated) / timing.decodeSeconds
                        : 0
                    let line = String(
                        format: "time=%.2fs load=%.2fs prefill=%.2fs decode=%.2fs tokens=%d decode_tps=%.2f e2e_tps=%.2f",
                        elapsed,
                        timing.loadSeconds,
                        timing.prefillSeconds,
                        timing.decodeSeconds,
                        result.tokensGenerated,
                        decodeTps,
                        e2eTps
                    )
                    CLIStderr.write("\(line)\n")
                } else {
                    let line = String(format: "time=%.2fs tokens=%d tps=%.2f", elapsed, result.tokensGenerated, e2eTps)
                    CLIStderr.write("\(line)\n")
                }
            }

            if stream && streamingOutput.hasWritten {
                streamingOutput.finishLine()
            } else {
                print(cleanResponse(result.response, showThinking: thinking))
            }
        }
    }

    private func cleanResponse(_ response: String, showThinking: Bool) -> String {
        guard !showThinking else { return response }
        var cleaned = response.replacingOccurrences(
            of: "(?is)<think>.*?</think>",
            with: "",
            options: .regularExpression
        )
        cleaned = cleaned.replacingOccurrences(of: "<think>", with: "")
        cleaned = cleaned.replacingOccurrences(of: "</think>", with: "")
        return cleaned.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    func resolveGemma4KVCacheQuantization(for modelId: String) throws -> Gemma4KVCacheQuantization {
        let usesTurboDefaults = Gemma4Resources.usesTurboDefaults(modelSpec: modelId)
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
            throw ValidationError("Unsupported --kv-quant-scheme '\(raw)'. Expected 'uniform' or 'turboquant'.")
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
