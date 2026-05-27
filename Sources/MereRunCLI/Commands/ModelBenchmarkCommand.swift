import ArgumentParser
import Foundation
import MereRunCore
#if canImport(Darwin)
import Darwin
#endif

struct ModelBenchmark: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "benchmark",
        abstract: "Run focused local model benchmarks.",
        subcommands: [
            ModelBenchmarkGemma4KV.self,
        ]
    )
}

struct ModelBenchmarkGemma4KV: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "gemma4-kv",
        abstract: "Compare Gemma4 default KV cache decode against packed PolarKV."
    )

    @Option(name: [.long], help: "Gemma4 model id or repo id.")
    var model: String = Gemma4Resources.turboModelId

    @Option(name: [.customShort("m"), .long], help: "Override model root directory.")
    var modelRoot: String?

    @Option(name: [.customShort("p"), .long], help: "Benchmark prompt. Defaults to a repeated deterministic fixture.")
    var prompt: String?

    @Option(name: [.long], help: "Read the benchmark prompt from a UTF-8 file.")
    var promptFile: String?

    @Option(name: [.long], help: "Repeat count for the built-in fixture prompt.")
    var promptRepeat: Int = 220

    @Option(name: [.long], help: "Exact number of decode tokens to force per variant.")
    var decodeTokens: Int = 48

    @Option(name: [.long], help: "Temperature for benchmark sampling.")
    var temperature: Double = 0

    @Option(name: [.long], help: "Top-p for benchmark sampling.")
    var topP: Double = 1

    @Flag(name: [.long], help: "Emit machine-readable JSON.")
    var json: Bool = false

    func validate() throws {
        if prompt != nil && promptFile != nil {
            throw ValidationError("Specify either --prompt or --prompt-file, not both.")
        }
        guard promptRepeat > 0 else {
            throw ValidationError("--prompt-repeat must be greater than zero.")
        }
        guard decodeTokens > 0 else {
            throw ValidationError("--decode-tokens must be greater than zero.")
        }
        guard Gemma4Resources.handles(modelSpec: model) else {
            throw ValidationError("model benchmark gemma4-kv only supports Gemma4 model ids or repo ids.")
        }
    }

    func run() async throws {
        try MLXBundleSupport.ensureAvailable(quiet: json)
        let prompt = try resolvePrompt()
        let request = ChatRequest(
            messages: [ChatMessage(role: .user, content: prompt)],
            maxTokens: decodeTokens,
            temperature: temperature,
            topP: topP,
            showThinking: false,
            stopOnEOS: false
        )

        let variants = [
            Gemma4KVBenchmarkVariant.defaultTurbo(model: model),
            Gemma4KVBenchmarkVariant.polar2(model: model),
        ]
        var results: [Gemma4KVBenchmarkVariantResult] = []
        results.reserveCapacity(variants.count)

        for variant in variants {
            let result = try await runVariant(variant, request: request)
            guard result.generatedTokens == decodeTokens else {
                throw ValidationError(
                    "\(variant.name) generated \(result.generatedTokens) tokens, expected \(decodeTokens). Reduce prompt length or decode tokens."
                )
            }
            results.append(result)
        }

        let report = Gemma4KVBenchmarkReport(
            model: model,
            promptCharacters: prompt.count,
            requestedDecodeTokens: decodeTokens,
            variants: results
        )
        if json {
            print(try report.jsonString())
        } else {
            print(report.renderText())
        }
    }

    private func resolvePrompt() throws -> String {
        if let prompt {
            return prompt
        }
        if let promptFile {
            return try String(contentsOf: URL(fileURLWithPath: promptFile).standardizedFileURL, encoding: .utf8)
        }
        var lines: [String] = []
        lines.reserveCapacity(promptRepeat + 1)
        for index in 1...promptRepeat {
            lines.append(
                "Record \(index): Gemma turbo benchmark passage about local inference, packed key value cache memory, decode timing, and repeatable measurements. The checksum word is polar."
            )
        }
        lines.append("Question: Write one concise sentence naming the checksum word and benchmark subject.")
        return lines.joined(separator: "\n")
    }

    private func runVariant(
        _ variant: Gemma4KVBenchmarkVariant,
        request: ChatRequest
    ) async throws -> Gemma4KVBenchmarkVariantResult {
        let generator = Gemma4Generator(modelId: model, kvCacheQuantization: variant.quantization)
        let memoryBefore = ProcessMemorySnapshot.currentResidentBytes()
        let start = Date()
        let response = try await generator.chat(request, modelPath: modelRoot, progressHandler: nil)
        let elapsed = Date().timeIntervalSince(start)
        let memoryAfter = ProcessMemorySnapshot.currentResidentBytes()
        await generator.unload()

        guard let timing = response.timing else {
            throw ValidationError("\(variant.name) did not return timing data.")
        }
        let promptTokens = response.promptTokens ?? 0
        return Gemma4KVBenchmarkVariantResult(
            name: variant.name,
            kvScheme: variant.quantization.scheme.rawValue,
            kvBits: variant.quantization.bits,
            quantizedKVStart: variant.quantization.quantizedStart,
            promptTokens: promptTokens,
            generatedTokens: response.tokensGenerated,
            elapsedSeconds: elapsed,
            loadSeconds: timing.loadSeconds,
            prefillSeconds: timing.prefillSeconds,
            decodeSeconds: timing.decodeSeconds,
            firstTokenSeconds: timing.firstTokenSeconds,
            prefillTokensPerSecond: promptTokens > 0 && timing.prefillSeconds > 0
                ? Double(promptTokens) / timing.prefillSeconds
                : nil,
            decodeTokensPerSecond: timing.decodeSeconds > 0
                ? Double(response.tokensGenerated) / timing.decodeSeconds
                : nil,
            endToEndTokensPerSecond: elapsed > 0 ? Double(response.tokensGenerated) / elapsed : nil,
            residentMemoryBeforeBytes: memoryBefore,
            residentMemoryAfterBytes: memoryAfter
        )
    }
}

private struct Gemma4KVBenchmarkVariant {
    let name: String
    let quantization: Gemma4KVCacheQuantization

    static func defaultTurbo(model: String) -> Self {
        let usesTurboDefaults = Gemma4Resources.usesTurboDefaults(modelSpec: model)
        return Self(
            name: "default",
            quantization: Gemma4KVCacheQuantization(
                bits: usesTurboDefaults ? Gemma4Resources.defaultTurboKVBits : nil,
                scheme: usesTurboDefaults ? Gemma4Resources.defaultTurboKVQuantizationScheme : .uniform,
                groupSize: Gemma4Resources.defaultKVGroupSize,
                quantizedStart: usesTurboDefaults
                    ? Gemma4Resources.defaultTurboQuantizedKVStart
                    : Gemma4Resources.defaultQuantizedKVStart
            )
        )
    }

    static func polar2(model: String) -> Self {
        Self(
            name: "polar2",
            quantization: Gemma4KVCacheQuantization(
                bits: 2,
                scheme: .polar,
                groupSize: Gemma4Resources.defaultKVGroupSize,
                quantizedStart: 0
            )
        )
    }
}

private struct Gemma4KVBenchmarkReport: Encodable {
    let model: String
    let promptCharacters: Int
    let requestedDecodeTokens: Int
    let variants: [Gemma4KVBenchmarkVariantResult]

    func jsonString() throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(self)
        return String(decoding: data, as: UTF8.self)
    }

    func renderText() -> String {
        var lines = [
            "Gemma4 KV benchmark",
            "model: \(model)",
            "prompt characters: \(promptCharacters)",
            "decode tokens: \(requestedDecodeTokens)",
            "",
        ]
        for result in variants {
            lines.append(result.renderText())
        }
        return lines.joined(separator: "\n")
    }
}

private struct Gemma4KVBenchmarkVariantResult: Encodable {
    let name: String
    let kvScheme: String
    let kvBits: Double?
    let quantizedKVStart: Int
    let promptTokens: Int
    let generatedTokens: Int
    let elapsedSeconds: Double
    let loadSeconds: Double
    let prefillSeconds: Double
    let decodeSeconds: Double
    let firstTokenSeconds: Double?
    let prefillTokensPerSecond: Double?
    let decodeTokensPerSecond: Double?
    let endToEndTokensPerSecond: Double?
    let residentMemoryBeforeBytes: UInt64?
    let residentMemoryAfterBytes: UInt64?

    var ttftSeconds: Double? {
        guard let firstTokenSeconds else { return nil }
        return loadSeconds + prefillSeconds + firstTokenSeconds
    }

    enum CodingKeys: String, CodingKey {
        case name
        case kvScheme
        case kvBits
        case quantizedKVStart
        case promptTokens
        case generatedTokens
        case elapsedSeconds
        case loadSeconds
        case prefillSeconds
        case decodeSeconds
        case firstTokenSeconds
        case ttftSeconds
        case prefillTokensPerSecond
        case decodeTokensPerSecond
        case endToEndTokensPerSecond
        case residentMemoryBeforeBytes
        case residentMemoryAfterBytes
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(name, forKey: .name)
        try container.encode(kvScheme, forKey: .kvScheme)
        try container.encodeIfPresent(kvBits, forKey: .kvBits)
        try container.encode(quantizedKVStart, forKey: .quantizedKVStart)
        try container.encode(promptTokens, forKey: .promptTokens)
        try container.encode(generatedTokens, forKey: .generatedTokens)
        try container.encode(elapsedSeconds, forKey: .elapsedSeconds)
        try container.encode(loadSeconds, forKey: .loadSeconds)
        try container.encode(prefillSeconds, forKey: .prefillSeconds)
        try container.encode(decodeSeconds, forKey: .decodeSeconds)
        try container.encodeIfPresent(firstTokenSeconds, forKey: .firstTokenSeconds)
        try container.encodeIfPresent(ttftSeconds, forKey: .ttftSeconds)
        try container.encodeIfPresent(prefillTokensPerSecond, forKey: .prefillTokensPerSecond)
        try container.encodeIfPresent(decodeTokensPerSecond, forKey: .decodeTokensPerSecond)
        try container.encodeIfPresent(endToEndTokensPerSecond, forKey: .endToEndTokensPerSecond)
        try container.encodeIfPresent(residentMemoryBeforeBytes, forKey: .residentMemoryBeforeBytes)
        try container.encodeIfPresent(residentMemoryAfterBytes, forKey: .residentMemoryAfterBytes)
    }

    func renderText() -> String {
        [
            "\(name): \(kvScheme)\(kvBits.map { String(format: " %.1f-bit", $0) } ?? "") start=\(quantizedKVStart)",
            "  prompt_tokens=\(promptTokens) generated_tokens=\(generatedTokens)",
            "  time total=\(format(elapsedSeconds))s load=\(format(loadSeconds))s prefill=\(format(prefillSeconds))s decode=\(format(decodeSeconds))s ttft=\(formatOptional(ttftSeconds))s",
            "  throughput prefill=\(formatOptional(prefillTokensPerSecond)) tok/s decode=\(formatOptional(decodeTokensPerSecond)) tok/s e2e=\(formatOptional(endToEndTokensPerSecond)) tok/s",
            "  resident_memory before=\(formatBytes(residentMemoryBeforeBytes)) after=\(formatBytes(residentMemoryAfterBytes))",
            "",
        ].joined(separator: "\n")
    }

    private func format(_ value: Double) -> String {
        String(format: "%.3f", value)
    }

    private func formatOptional(_ value: Double?) -> String {
        guard let value else { return "n/a" }
        return format(value)
    }

    private func formatBytes(_ value: UInt64?) -> String {
        guard let value else { return "n/a" }
        return ByteCountFormatter.string(fromByteCount: Int64(value), countStyle: .memory)
    }
}

private enum ProcessMemorySnapshot {
    static func currentResidentBytes() -> UInt64? {
        #if canImport(Darwin)
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { rebound in
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), rebound, &count)
            }
        }
        guard result == KERN_SUCCESS else {
            return nil
        }
        return UInt64(info.resident_size)
        #else
        return nil
        #endif
    }
}
