import ArgumentParser
import Foundation
import MereRunCore
#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

// MARK: - Text Chat Command

enum TextChatResponseFormat: String, CaseIterable, ExpressibleByArgument {
    case text
    case jsonObject = "json_object"
}

struct TextChat: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "chat",
        abstract: "Run local chat with text chat models.",
        discussion: """
        Auto-downloads the selected model on first use. The default is
        text-chat-gemma4-12b-4bit on Apple Silicon (MLX) and text-chat-q36-nano-gguf on
        Linux CUDA (llama.cpp).
        Known model IDs:
          - text-chat-gemma4-12b-4bit (Gemma 4 12B MLX 4-bit, default on Apple Silicon)
          - text-chat-q36-nano (Qwen3.6-35B-A3B OptiQ 4-bit)
          - text-chat-bonsai-27b-1bit (Bonsai 27B packed 1-bit Qwen3.6 vision/reasoning model)
          - text-chat-bonsai-27b-2bit (Ternary Bonsai 27B packed 2-bit Qwen3.6 vision/reasoning model)
          - text-chat-q36-nano-gguf (Qwen3.6-35B-A3B GGUF, default on Linux CUDA)
          - text-agent-ornith-9b (Ornith 1.0 9B OptiQ, experimental coding-agent target)
          - text-agent-ornith-35b-mlx (Ornith 1.0 35B MLX Q4, local converted coding-agent target)
          - text-chat-gemma4-12b (Gemma 4 12B dense native Swift runtime)
          - text-chat-gemma4-12b-4bit (Gemma 4 12B MLX 4-bit native Swift runtime)
          - text-chat-gemma4-turbo (Gemma 4 26B-A4B NVFP4 native Swift runtime)
          - text-chat-gemma4 (Gemma 4 31B; large/slow, kept for compatibility)
          - text-chat-gemma4-max (Gemma 4 31B native Swift runtime)
          - text-chat-gemma4-nano (Gemma 4 4B native Swift runtime)
          - text-chat-laguna-s-2-1 (Poolside Laguna S 2.1 118B-A8B NVFP4 with DFlash)
          - text-chat-laguna-xs-2-1 (Poolside Laguna XS 2.1 33B-A3B NVFP4)
          - text-chat-inkling-small (Inkling-Small 276B-A12B mixed MLX: routed-expert q2, non-routed BF16)
          - text-chat-lfm25-2.6b-4bit (LiquidAI LFM2.5 2.6B dense MLX 4-bit native Swift runtime)
          - text-chat-lfm25-a1b-8bit (LiquidAI LFM2.5 8B-A1B MLX 8-bit native Swift runtime)
          - text-chat-psi-agent
        Models are cached under ~/Library/Application Support/MereRun/models/<model-id>.
        Thinking output is hidden by default; pass --thinking to include it.
        Bonsai 27B and R1-style lanes (text-agent-ornith-*) generate with thinking enabled even
        when it is hidden; pass --no-thinking to disable reasoning generation.
        Use --models-root or MERERUN_MODELS_DIR to override the model store path.
        """
    )

    @Option(name: [.customShort("p"), .long], help: "User prompt.")
    var prompt: String

    @Option(name: [.long], help: "Optional image path for vision-capable chat models such as Bonsai 27B or vision-chat-gemma4-12b.")
    var image: String?

    @Option(name: [.customShort("s"), .customLong("system")], help: "System prompt.")
    var systemPrompt: String?

    @Option(name: [.long], help: "Max new tokens.")
    var maxTokens: Int = 2048

    @Option(
        name: [.customLong("context-size")],
        help: "Maximum prompt plus generation context. Bonsai supports 262144 tokens; Inkling-Small advertises 1048576 and defaults to 32768."
    )
    var contextSize: Int?

    @Option(name: [.long], help: "Temperature. Default: 0.7, or the model's published value where one exists (Laguna/Ornith: 1.0).")
    var temperature: Double?

    @Option(name: [.long], help: "Top-p. Default: 0.9, or the model's published value where one exists (Laguna: 1.0; Bonsai/Ornith: 0.95).")
    var topP: Double?

    @Option(name: [.customLong("top-k")], help: "Top-k sampling cutoff. Default: no cutoff, or the model's published value where one exists (Laguna/Bonsai/Ornith: 20).")
    var topK: Int?

    @Option(name: [.customLong("min-p")], help: "Min-p cutoff relative to the most likely token. Default: 0 (disabled), or 0.02 for Laguna S 2.1.")
    var minP: Double?

    @Option(
        name: [.customLong("reasoning-effort")],
        help: "Inkling-Small reasoning effort from 0 through 0.99. Default: 0.9."
    )
    var reasoningEffort: Double?

    @Option(name: [.long], help: "Quantize the KV cache to this many bits. Qwen-family supports affine 4 or 8; Gemma4 also supports its model-specific schemes.")
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
        let isCUDA: Bool
        #if os(Linux)
        isCUDA = ProcessInfo.processInfo.environment["MERERUN_LINUX_ACCEL"]?.lowercased() == "cuda"
        #else
        isCUDA = false
        #endif

        return defaultChatModelId(on: machine, linuxCUDA: isCUDA)
    }

    static func defaultChatModelId(on machine: MereRunMachineProfile, linuxCUDA: Bool = false) -> String {
        func fits(_ id: String) -> Bool {
            guard let descriptor = ManagedModelCapabilityCatalog.descriptor(for: id) else { return false }
            return machine.unifiedMemoryGB >= descriptor.minimumUnifiedMemoryGB
        }

        let a3b = machine.isLinux
            ? (linuxCUDA ? "text-chat-q36-nano-gguf" : Q35Resources.q36NanoModelId)
            : Gemma4Resources.twelveB4BitModelId
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

    static func ttftSeconds(for timing: ChatTiming) -> Double? {
        guard let firstTokenSeconds = timing.firstTokenSeconds else { return nil }
        return timing.loadSeconds
            + timing.prefillSeconds
            + (timing.cacheConversionSeconds ?? 0)
            + firstTokenSeconds
    }

    @Option(name: [.long], help: "Canonical model id. Default: text-chat-gemma4-12b-4bit (Apple Silicon) / text-chat-q36-nano-gguf (Linux CUDA). Others include vision-chat-muse-glimmer-30b, text-chat-laguna-s-2-1, text-chat-inkling-small, text-chat-bonsai-27b-1bit, text-chat-q36-nano, text-agent-ornith-9b, text-chat-gemma4[-12b|-12b-4bit|-turbo|-max|-nano], and text-chat-lfm25-a1b-8bit.")
    var model: String = TextChat.defaultChatModelId

    @Option(
        name: [.customLong("response-format")],
        help: "Response format: text or json_object. json_object uses constrained decoding on native MLX chat models."
    )
    var responseFormat: TextChatResponseFormat = .text

    @Option(name: [.customLong("lora")], help: "Optional cataloged adapter id or local LoRA .safetensors path for supported chat models.")
    var loraPath: String?

    @Option(name: [.customLong("lora-scale")], help: "LoRA adapter scale.")
    var loraScale: Double = 1.0

    @Flag(
        name: [.customLong("thinking"), .customLong("show-thinking")],
        inversion: .prefixedNo,
        help: "Show model reasoning output. Bonsai 27B and R1-style lanes (text-agent-ornith-*) generate with thinking enabled by default; pass --no-thinking to disable reasoning generation."
    )
    var thinking: Bool?

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

    @Flag(name: [.customLong("preflight")], help: "Inspect the chat request without loading or downloading a model.")
    var preflight: Bool = false

    @Flag(name: [.customLong("json")], help: "With --preflight, emit a structured JSON report.")
    var json: Bool = false

    @Flag(name: [.customLong("require-installed")], help: "Require an installed model and never download implicitly.")
    var requireInstalled: Bool = false

    func run() async throws {
        let normalizedModelId = model.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        try Self.validate(responseFormat: responseFormat, modelID: normalizedModelId)
        try Self.validateReasoningEffort(reasoningEffort, modelID: normalizedModelId)
        let installedModelPath = resolvedInstalledModelPath(modelID: normalizedModelId)
        if preflight {
            try emitPreflight(modelID: normalizedModelId, installedModelPath: installedModelPath)
            return
        }
        if requireInstalled {
            guard installedModelPath != nil else {
                throw ValidationError(
                    "Model '\(normalizedModelId)' is not installed. Run "
                        + "'mere.run model pull \(normalizedModelId)' explicitly."
                )
            }
        }
        let runtimeModelRoot = requireInstalled ? installedModelPath : modelRoot
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

        let lora: LoRA?
        if let loraPath, !loraPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            let resolved = try ManagedAdapterArgumentResolver.resolve(
                loraPath,
                baseModelID: model
            )
            lora = resolved.map { .local(path: $0, scale: loraScale) }
        } else {
            lora = nil
        }

        let recommendedSampling = Q35Resources.recommendedSampling(forModelId: model)
        let isLaguna = LagunaResources.handles(modelSpec: normalizedModelId)
        let isMuseGlimmer = MuseGlimmerResources.handles(modelSpec: normalizedModelId)
        let q35KVCacheMode = try resolveQ35KVCacheMode(for: normalizedModelId)
        if let contextSize, contextSize <= 0 {
            throw ValidationError("--context-size must be greater than zero.")
        }
        let requiresJSON = responseFormat == .jsonObject
        let request = ChatRequest(
            messages: messages,
            maxTokens: maxTokens,
            temperature: temperature
                ?? (isMuseGlimmer ? MuseGlimmerResources.recommendedTemperature : nil)
                ?? (isLaguna ? LagunaResources.recommendedTemperature : recommendedSampling?.temperature)
                ?? 0.7,
            topP: topP
                ?? (isMuseGlimmer ? MuseGlimmerResources.recommendedTopP : nil)
                ?? (isLaguna ? LagunaResources.recommendedTopP : recommendedSampling?.topP)
                ?? 0.9,
            topK: topK
                ?? (isMuseGlimmer ? MuseGlimmerResources.recommendedTopK : nil)
                ?? (isLaguna ? LagunaResources.recommendedTopK : recommendedSampling?.topK),
            minP: minP ?? (isLaguna ? LagunaResources.recommendedMinP : 0),
            reasoningEffort: reasoningEffort,
            showThinking: requiresJSON
                ? false
                : (thinking ?? (isMuseGlimmer ? false : Q35Resources.thinkingDefault(forModelId: model))),
            lora: lora,
            requiresJSON: requiresJSON,
            tools: toolDefs,
            kvCacheMode: q35KVCacheMode,
            maxContextTokens: contextSize
        )

        let streamingOutput = StreamingChatOutput(enabled: stream)
        let progressHandler = TextChatProgressHandler.make(
            quiet: quiet,
            streamingOutput: streamingOutput
        )

        let startTime = Date()
        if !quiet {
            CLIStderr.write("[runtime] text backend: \(Self.backendDescription(for: normalizedModelId))\n")
        }
        var lastGemma4MTPStats: Gemma4MTPStats?
        var lastLagunaDFlashStats: LagunaDFlashStats?
        var lastMuseDFlashStats: MuseGlimmerDFlashStats?
        let lagunaGenerator = isLaguna
            ? LagunaGenerator(
                dflashModelPath: LagunaResources.installedDFlashPath(
                    for: normalizedModelId
                )
            )
            : nil

        let chatOnce: (ChatRequest) async throws -> ChatResponse = { req in
            if normalizedModelId == Psi3ChatResources.defaultModelId {
                let generator = Psi3ChatGenerator(modelId: Psi3ChatResources.defaultModelId)
                return try await generator.chat(req, modelPath: runtimeModelRoot, progressHandler: progressHandler)
            } else if Gemma4Resources.handles(modelSpec: normalizedModelId) {
                let effectiveModelId = normalizedModelId.isEmpty ? Gemma4Resources.defaultModelId : normalizedModelId
                let kvQuantization = try self.resolveGemma4KVCacheQuantization(for: effectiveModelId)
                let generator = Gemma4Generator(
                    modelId: effectiveModelId,
                    kvCacheQuantization: kvQuantization
                )
                let response = try await generator.chat(req, modelPath: runtimeModelRoot, progressHandler: progressHandler)
                lastGemma4MTPStats = await generator.mtpStats()
                return response
            } else if LagunaResources.handles(modelSpec: normalizedModelId) {
                guard let lagunaModelPath = self.modelRoot ?? installedModelPath else {
                    let requestedID = LagunaResources.managedModelID(for: normalizedModelId)
                        ?? normalizedModelId
                    throw ValidationError(
                        "Model '\(requestedID)' is not installed. Run "
                            + "'mere.run model pull \(requestedID)' first."
                    )
                }
                guard let generator = lagunaGenerator else {
                    throw LagunaError.modelNotLoaded
                }
                let response = try await generator.chat(
                    req,
                    modelPath: lagunaModelPath,
                    progressHandler: progressHandler
                )
                lastLagunaDFlashStats = await generator.dflashStats()
                return response
            } else if ManagedModelCatalog.spec(for: normalizedModelId)?.validationKind == .codegenGGUF {
                // GGUF chat models run through the llama.cpp engine (the same path
                // `text code` uses). On Linux CUDA this is the GB10-optimized
                // llama.cpp runtime, which has fast quantized-MoE kernels MLX lacks.
                let generator = CodeGenGenerator(modelId: normalizedModelId)
                return try await generator.chat(req, modelPath: runtimeModelRoot, progressHandler: progressHandler)
            } else if InklingResources.handles(modelSpec: normalizedModelId) {
                let generator = InklingGenerator(modelID: normalizedModelId)
                return try await generator.chat(req, modelPath: runtimeModelRoot, progressHandler: progressHandler)
            } else if MuseGlimmerResources.handles(modelSpec: normalizedModelId) {
                let generator = MuseGlimmerGenerator(modelID: normalizedModelId)
                let response = try await generator.chat(
                    req,
                    modelPath: runtimeModelRoot,
                    progressHandler: progressHandler
                )
                lastMuseDFlashStats = await generator.dflashStats()
                return response
            } else if LFM2Resources.handles(modelSpec: normalizedModelId) {
                let effectiveModelId = normalizedModelId.isEmpty ? LFM2Resources.defaultModelId : normalizedModelId
                let generator = LFM2Generator(modelId: effectiveModelId)
                return try await generator.chat(req, modelPath: runtimeModelRoot, progressHandler: progressHandler)
            } else {
                let effectiveModelId = normalizedModelId.isEmpty ? Q35Resources.defaultModelId : normalizedModelId
                let generator = Q35Generator(modelId: effectiveModelId)
                return try await generator.chat(req, modelPath: runtimeModelRoot, progressHandler: progressHandler)
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
                        print(cleanResponse(result.response, showThinking: request.showThinking))
                    }
                    return
                }

                // Show the model's response (may contain text before/after tool calls)
                let textBeforeTools = result.response
                    .replacingOccurrences(of: "<\\|tool_call>.*?<tool_call\\|>", with: "", options: .regularExpression)
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                if !textBeforeTools.isEmpty {
                    CLIStderr.write(cleanResponse(textBeforeTools, showThinking: request.showThinking) + "\n")
                }

                loopMessages.append(ChatMessage(
                    role: .assistant,
                    content: result.response,
                    reasoningContent: result.reasoningContent
                ))

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
                    loopMessages.append(ChatMessage(role: .tool, content: output, name: call.name))
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
                    if let firstToken = timing.firstTokenSeconds,
                       let ttft = Self.ttftSeconds(for: timing) {
                        line += String(format: " ttft_s=%.3f first_token_s=%.3f", ttft, firstToken)
                    }
                    CLIStderr.write("\(line)\n")
                    if let mtp = lastGemma4MTPStats {
                        CLIStderr.write(Self.formatGemma4MTPStats(mtp) + "\n")
                    }
                    if let dflash = lastLagunaDFlashStats {
                        CLIStderr.write(Self.formatLagunaDFlashStats(dflash) + "\n")
                    }
                    if let dflash = lastMuseDFlashStats {
                        CLIStderr.write(Self.formatMuseDFlashStats(dflash) + "\n")
                    }
                } else {
                    let line = String(format: "time=%.2fs tokens=%d tps=%.2f", elapsed, result.tokensGenerated, e2eTps)
                    CLIStderr.write("\(line)\n")
                    if let mtp = lastGemma4MTPStats {
                        CLIStderr.write(Self.formatGemma4MTPStats(mtp) + "\n")
                    }
                    if let dflash = lastLagunaDFlashStats {
                        CLIStderr.write(Self.formatLagunaDFlashStats(dflash) + "\n")
                    }
                    if let dflash = lastMuseDFlashStats {
                        CLIStderr.write(Self.formatMuseDFlashStats(dflash) + "\n")
                    }
                }
            }

            if stream && streamingOutput.hasWritten {
                streamingOutput.finishLine()
            } else {
                print(cleanResponse(result.response, showThinking: request.showThinking))
            }
        }
    }

    private func resolvedInstalledModelPath(modelID: String) -> String? {
        if let modelRoot, !modelRoot.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            let url = URL(fileURLWithPath: modelRoot).standardizedFileURL
            return FileManager.default.fileExists(atPath: url.path) ? url.path : nil
        }
        return ManagedModelResolver.resolveInstalledModel(id: modelID)?.path
    }

    private func emitPreflight(modelID: String, installedModelPath: String?) throws {
        var diagnostics: [PreflightDiagnostic] = []
        if modelID.isEmpty || ManagedModelCatalog.spec(for: modelID) == nil {
            diagnostics.append(.init(
                id: "text_chat_model_unknown",
                severity: .blocker,
                title: "Unknown chat model",
                message: "No managed model is cataloged as '\(modelID)'."
            ))
        }
        if requireInstalled, installedModelPath == nil {
            diagnostics.append(.init(
                id: "text_chat_model_not_installed",
                severity: .blocker,
                title: "Chat model is not installed",
                message: "Install '\(modelID)' explicitly before running this workflow node."
            ))
        }
        if let image, !image.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
           !image.lowercased().hasPrefix("data:image/") {
            let imageURL = URL(fileURLWithPath: image).standardizedFileURL
            if !FileManager.default.fileExists(atPath: imageURL.path) {
                diagnostics.append(.init(
                    id: "text_chat_image_missing",
                    severity: .blocker,
                    title: "Chat image is missing",
                    message: "Image file not found: \(imageURL.path)"
                ))
            }
        }
        let report = TextChatPreflightReport(
            schemaVersion: 1,
            status: StructuredRunOutput.status(for: diagnostics),
            model: modelID,
            installed: installedModelPath != nil,
            modelPath: installedModelPath,
            diagnostics: diagnostics
        )
        if json {
            print(try StructuredRunOutput.encode(report))
        } else {
            print(report.summary)
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

    static func formatLagunaDFlashStats(_ stats: LagunaDFlashStats) -> String {
        String(
            format: "dflash=%@ proposals=%d routed=%d bypassed=%d drafted=%d accepted=%d acceptance=%.1f%% fallbacks=%d",
            stats.enabled ? "active" : "unavailable",
            stats.speculativeTokens,
            stats.routedRequests,
            stats.bypassedRequests,
            stats.draftedTokens,
            stats.acceptedDraftTokens,
            stats.acceptanceRate * 100,
            stats.adaptiveFallbacks
        )
    }

    static func formatMuseDFlashStats(_ stats: MuseGlimmerDFlashStats) -> String {
        let state: String
        if stats.active {
            state = "active"
        } else if stats.enabled {
            state = stats.assistantModelPath == nil ? "unavailable" : "fallback"
        } else {
            state = "disabled"
        }
        let reason = stats.reason.map { " reason=\($0)" } ?? ""
        return String(
            format: "muse_dflash=%@ proposals=%d rounds=%d drafted=%d accepted=%d acceptance=%.1f%% verify=%d recovery=%d fallback_forwards=%d adaptive_fallbacks=%d%@",
            state,
            stats.speculativeTokens,
            stats.rounds,
            stats.draftedTokens,
            stats.acceptedTokens,
            stats.acceptanceRate * 100,
            stats.targetVerificationForwards,
            stats.targetRecoveryForwards,
            stats.targetFallbackForwards,
            stats.adaptiveFallbacks,
            reason
        )
    }

    static func validate(responseFormat: TextChatResponseFormat, modelID: String) throws {
        guard responseFormat == .jsonObject else { return }
        if ManagedModelCatalog.spec(for: modelID)?.validationKind == .codegenGGUF {
            throw ValidationError(
                "--response-format json_object is not yet supported by the llama.cpp/GGUF chat runtime; use the native MLX text-chat-q36-nano model."
            )
        }
        if modelID == Psi3ChatResources.defaultModelId
            || LFM2Resources.handles(modelSpec: modelID)
            || InklingResources.handles(modelSpec: modelID)
            || MuseGlimmerResources.handles(modelSpec: modelID) {
            throw ValidationError(
                "--response-format json_object is supported by native Gemma4 and Qwen-family MLX chat models; Muse Glimmer supports structured tool schemas but not constrained JSON decoding."
            )
        }
        if LagunaResources.handles(modelSpec: modelID) {
            throw ValidationError(
                "--response-format json_object is not yet supported by the Laguna native runtime."
            )
        }
    }

    static func validateReasoningEffort(_ value: Double?, modelID: String) throws {
        guard let value else { return }
        guard InklingResources.handles(modelSpec: modelID)
                || MuseGlimmerResources.handles(modelSpec: modelID) else {
            throw ValidationError("--reasoning-effort is supported only for Inkling-Small and Muse Glimmer.")
        }
        let allowed = InklingResources.handles(modelSpec: modelID)
            ? (0...0.99).contains(value)
            : (0...1).contains(value)
        guard allowed else {
            let upper = InklingResources.handles(modelSpec: modelID) ? "0.99" : "1"
            throw ValidationError("--reasoning-effort must be between 0 and \(upper).")
        }
    }

    func cleanResponse(_ response: String, showThinking: Bool) -> String {
        guard !showThinking else { return response }
        return ChatReasoningMarkup.splitThinkBlocks(in: response).visibleContent
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

    func resolveQ35KVCacheMode(for modelId: String) throws -> RuntimeKVCacheMode? {
        guard Q35Resources.supportedModelIds.contains(modelId) else { return nil }

        guard let kvBits else {
            if kvQuantScheme != nil || kvGroupSize != nil || quantizedKVStart != nil {
                throw ValidationError("Qwen-family KV cache options require --kv-bits 4 or --kv-bits 8.")
            }
            return nil
        }

        guard kvBits == 4 || kvBits == 8 else {
            throw ValidationError("Qwen-family --kv-bits must be 4 or 8.")
        }
        if let kvQuantScheme,
           kvQuantScheme.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() != "uniform" {
            throw ValidationError("Qwen-family KV cache quantization uses the affine uniform scheme.")
        }
        if kvGroupSize != nil || quantizedKVStart != nil {
            throw ValidationError("Qwen-family KV cache group size and start offset are selected by the runtime.")
        }
        return kvBits == 4 ? .affine4 : .affine8
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

private struct TextChatPreflightReport: Codable, Equatable {
    let schemaVersion: Int
    let status: StructuredRunStatus
    let model: String
    let installed: Bool
    let modelPath: String?
    let diagnostics: [PreflightDiagnostic]

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case status
        case model
        case installed
        case modelPath = "model_path"
        case diagnostics
    }

    var summary: String {
        "\(status.rawValue): \(model) \(installed ? "is installed" : "is not installed")"
    }
}

enum TextChatProgressHandler {
    static func make(
        quiet: Bool,
        streamingOutput: StreamingChatOutput,
        diagnosticWriter: @escaping @Sendable (String) -> Void = CLIStderr.write
    ) -> (@Sendable (ChatProgress) -> Void)? {
        guard !quiet || streamingOutput.enabled else {
            return nil
        }

        return { progress in
            if streamingOutput.write(progress: progress) {
                return
            }
            guard !quiet else {
                return
            }
            diagnosticWriter("[\(progress.stage.rawValue)] \(progress.message ?? "")\n")
        }
    }
}

final class StreamingChatOutput: @unchecked Sendable {
    let enabled: Bool

    private let lock = NSLock()
    private var wroteOutput = false
    private let writer: @Sendable (String) -> Void

    init(
        enabled: Bool,
        writer: @escaping @Sendable (String) -> Void = CLIStdout.write
    ) {
        self.enabled = enabled
        self.writer = writer
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
        writer(text)
        wroteOutput = true
        return true
    }

    func finishLine() {
        lock.lock()
        defer { lock.unlock() }
        writer("\n")
    }
}
