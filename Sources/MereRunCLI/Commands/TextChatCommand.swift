import ArgumentParser
import Foundation
import MereRunCore

// MARK: - Text Chat Command

struct TextChat: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "chat",
        abstract: "Run local chat with text chat models.",
        discussion: """
        Auto-downloads the selected model on first use.
        Known model IDs:
          - text-chat-gemma4 (Gemma 4 default alias, currently 31B)
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
    var kvQuantScheme: String = Gemma4Resources.defaultKVQuantizationScheme.rawValue

    @Option(name: [.long], help: "Gemma4 KV cache quantization group size.")
    var kvGroupSize: Int = Gemma4Resources.defaultKVGroupSize

    @Option(name: [.long], help: "Gemma4 token offset at which KV cache quantization begins.")
    var quantizedKVStart: Int = Gemma4Resources.defaultQuantizedKVStart

    @Option(name: [.customShort("m"), .long], help: "Override model root directory (skips auto-download).")
    var modelRoot: String?

    @Option(name: [.long], help: "Canonical model id: text-chat-gemma4 (default alias), text-chat-gemma4-max, text-chat-gemma4-nano, text-chat-q35, text-chat-q35-nano, or text-chat-psi-agent.")
    var model: String = Gemma4Resources.defaultModelId

    @Flag(name: [.customLong("thinking"), .customLong("show-thinking")], help: "Show model reasoning output.")
    var thinking: Bool = false

    @Flag(name: [.customLong("stats")], help: "Print generation timing and tokens/sec.")
    var stats: Bool = false

    @Flag(name: [.short, .long], help: "Suppress progress output.")
    var quiet: Bool = false

    func run() async throws {
        try MLXBundleSupport.ensureAvailable(quiet: quiet)

        var messages: [ChatMessage] = []
        if let systemPrompt, !systemPrompt.isEmpty {
            messages.append(ChatMessage(role: .system, content: systemPrompt))
        }
        messages.append(ChatMessage(role: .user, content: prompt))

        let request = ChatRequest(
            messages: messages,
            maxTokens: maxTokens,
            temperature: temperature,
            topP: topP,
            showThinking: thinking
        )

        let progressHandler: (@Sendable (ChatProgress) -> Void)?
        if quiet {
            progressHandler = nil
        } else {
            progressHandler = { progress in
                fputs("[\(progress.stage.rawValue)] \(progress.message ?? "")\n", stderr)
            }
        }

        let startTime = Date()
        let normalizedModelId = model.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()

        let result: ChatResponse
        if normalizedModelId == Psi3ChatResources.defaultModelId {
            let generator = Psi3ChatGenerator(modelId: Psi3ChatResources.defaultModelId)
            result = try await generator.chat(request, modelPath: modelRoot, progressHandler: progressHandler)
        } else if Gemma4Resources.handles(modelSpec: normalizedModelId) {
            let effectiveModelId = normalizedModelId.isEmpty ? Gemma4Resources.defaultModelId : normalizedModelId
            let scheme = try parseGemma4KVQuantizationScheme(kvQuantScheme)
            let generator = Gemma4Generator(
                modelId: effectiveModelId,
                kvCacheQuantization: Gemma4KVCacheQuantization(
                    bits: kvBits,
                    scheme: scheme,
                    groupSize: kvGroupSize,
                    quantizedStart: quantizedKVStart
                )
            )
            result = try await generator.chat(request, modelPath: modelRoot, progressHandler: progressHandler)
        } else {
            let effectiveModelId = normalizedModelId.isEmpty ? Q35Resources.defaultModelId : normalizedModelId
            let generator = Q35Generator(modelId: effectiveModelId)
            result = try await generator.chat(request, modelPath: modelRoot, progressHandler: progressHandler)
        }

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
                fputs("\(line)\n", stderr)
            } else {
                let line = String(format: "time=%.2fs tokens=%d tps=%.2f", elapsed, result.tokensGenerated, e2eTps)
                fputs("\(line)\n", stderr)
            }
        }

        print(cleanResponse(result.response, showThinking: thinking))
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

    private func parseGemma4KVQuantizationScheme(_ raw: String) throws -> Gemma4KVQuantizationScheme {
        guard let scheme = Gemma4KVQuantizationScheme(
            rawValue: raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        ) else {
            throw ValidationError("Unsupported --kv-quant-scheme '\(raw)'. Expected 'uniform' or 'turboquant'.")
        }
        return scheme
    }
}
