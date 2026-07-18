import ArgumentParser
import Foundation
import MereRunCore
#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

struct ModelBenchmark: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "benchmark",
        abstract: "Run focused local model benchmarks.",
        subcommands: [
            ModelBenchmarkChat.self,
            ModelBenchmarkToolCalls.self,
            ModelBenchmarkToolContinuations.self,
            ModelBenchmarkCode.self,
            ModelBenchmarkGemma4KV.self,
            ModelBenchmarkGemma4MTP.self,
            ModelBenchmarkQ36MTP.self,
            ModelBenchmarkAPIWorkload.self,
            ModelBenchmarkVLM.self,
        ]
    )
}

struct ModelBenchmarkQ36MTP: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "q36-mtp",
        abstract: "Compare Qwen3.6 serial decode against adaptive and forced MTP speculative decode."
    )

    @Option(name: [.long], help: "Qwen3.6 model id.")
    var model: String = Q35Resources.q36NanoModelId

    @Option(name: [.customShort("m"), .long], help: "Override model root directory.")
    var modelRoot: String?

    @Option(name: [.customShort("p"), .long], help: "Benchmark prompt. Defaults to a repeated deterministic fixture.")
    var prompt: String?

    @Option(name: [.long], help: "Read the benchmark prompt from a UTF-8 file.")
    var promptFile: String?

    @Option(name: [.long], help: "Repeat count for the built-in fixture prompt.")
    var promptRepeat: Int = 150

    @Option(name: [.long], help: "Comma-separated repeat counts for a prompt-size benchmark matrix.")
    var promptRepeatValues: String?

    @Option(name: [.long], help: "Exact number of decode tokens to force per variant.")
    var decodeTokens: Int = 32

    @Option(name: [.long], help: "Comma-separated decode token counts for a decode-length benchmark matrix.")
    var decodeTokenValues: String?

    @Option(name: [.long], help: "Temperature for benchmark sampling.")
    var temperature: Double = 0

    @Option(name: [.long], help: "Comma-separated temperatures for a sampling matrix.")
    var temperatureValues: String?

    @Option(name: [.long], help: "Top-p for benchmark sampling.")
    var topP: Double = 0.9

    @Option(name: [.long], help: "Context size passed to Qwen3.6 generation.")
    var contextSize: Int = Q35Resources.defaultContextLength

    @Option(name: [.long], help: "Override MERERUN_Q35_MTP_BLOCK_SIZE for MTP variants.")
    var mtpBlockSize: Int?

    @Option(name: [.long], help: "MERERUN_Q35_MTP_MIN_PROMPT_TOKENS value for the forced MTP variant.")
    var forcedMTPMinPromptTokens: Int = 1

    @Flag(name: [.long], help: "Emit machine-readable JSON.")
    var json: Bool = false

    func validate() throws {
        if prompt != nil && promptFile != nil {
            throw ValidationError("Specify either --prompt or --prompt-file, not both.")
        }
        let supportedModels = [
            Q35Resources.q36NanoModelId,
            Q35Resources.ornith9BModelId,
            Q35Resources.ornith35BMLXModelId,
        ]
        guard supportedModels.contains(model) else {
            throw ValidationError(
                "model benchmark q36-mtp only supports \(supportedModels.joined(separator: ", "))."
            )
        }
        guard promptRepeat > 0 else {
            throw ValidationError("--prompt-repeat must be greater than zero.")
        }
        guard decodeTokens > 0 else {
            throw ValidationError("--decode-tokens must be greater than zero.")
        }
        guard contextSize > 0 else {
            throw ValidationError("--context-size must be greater than zero.")
        }
        guard forcedMTPMinPromptTokens >= 0 else {
            throw ValidationError("--forced-mtp-min-prompt-tokens must be non-negative.")
        }
        if let mtpBlockSize {
            guard mtpBlockSize >= 2 else {
                throw ValidationError("--mtp-block-size must be at least 2.")
            }
        }
        let promptRepeats = try parsedPositiveIntegers(
            promptRepeatValues,
            fallback: [promptRepeat],
            optionName: "--prompt-repeat-values"
        )
        _ = try parsedPositiveIntegers(
            decodeTokenValues,
            fallback: [decodeTokens],
            optionName: "--decode-token-values"
        )
        _ = try parsedTemperatures()
        guard (0...1).contains(topP), topP.isFinite else {
            throw ValidationError("--top-p must be finite and between 0 and 1.")
        }
        if prompt != nil || promptFile != nil {
            guard promptRepeats.count == 1 else {
                throw ValidationError("--prompt-repeat-values can only contain one value when using --prompt or --prompt-file.")
            }
        }
    }

    func run() async throws {
        try MLXBundleSupport.ensureAvailable(quiet: json)
        let promptRepeats = try parsedPositiveIntegers(
            promptRepeatValues,
            fallback: [promptRepeat],
            optionName: "--prompt-repeat-values"
        )
        let decodeTokenCounts = try parsedPositiveIntegers(
            decodeTokenValues,
            fallback: [decodeTokens],
            optionName: "--decode-token-values"
        )
        let temperatures = try parsedTemperatures()
        let variants = [
            Q36MTPBenchmarkVariant.baseline,
            Q36MTPBenchmarkVariant.adaptive,
            Q36MTPBenchmarkVariant.forced,
        ]

        var scenarios: [Q36MTPBenchmarkScenarioResult] = []
        scenarios.reserveCapacity(promptRepeats.count * decodeTokenCounts.count * temperatures.count)
        for repeatCount in promptRepeats {
            let prompt = try resolvePrompt(repeatCount: repeatCount)
            for decodeTokenCount in decodeTokenCounts {
                for temperature in temperatures {
                    let request = ChatRequest(
                        messages: [ChatMessage(role: .user, content: prompt)],
                        maxTokens: decodeTokenCount,
                        temperature: temperature,
                        topP: topP,
                        showThinking: false,
                        stopOnEOS: false,
                        maxContextTokens: contextSize
                    )
                    var results: [Q36MTPBenchmarkVariantResult] = []
                    results.reserveCapacity(variants.count)
                    for variant in variants {
                        let result = try await runVariant(variant, request: request)
                        results.append(result)
                    }
                    scenarios.append(
                        Q36MTPBenchmarkScenarioResult(
                            promptRepeat: self.prompt != nil || self.promptFile != nil ? nil : repeatCount,
                            promptCharacters: prompt.count,
                            requestedDecodeTokens: decodeTokenCount,
                            temperature: temperature,
                            contextSize: contextSize,
                            variants: results
                        )
                    )
                }
            }
        }

        let report = Q36MTPBenchmarkReport(model: model, scenarios: scenarios)
        if json {
            print(try report.jsonString())
        } else {
            print(report.renderText())
        }
    }

    private func resolvePrompt(repeatCount: Int) throws -> String {
        if let prompt {
            return prompt
        }
        if let promptFile {
            return try String(contentsOf: URL(fileURLWithPath: promptFile).standardizedFileURL, encoding: .utf8)
        }
        var lines: [String] = []
        lines.reserveCapacity(repeatCount + 1)
        for index in 1...repeatCount {
            lines.append(
                "Record \(index): Qwen3.6 MTP benchmark passage about local-first inference, cache pressure, speculative decoding, quantization, batching, and user-facing latency. The checksum word is velocity."
            )
        }
        lines.append("Question: Write one concise sentence naming the checksum word and benchmark subject.")
        return lines.joined(separator: "\n")
    }

    private func parsedPositiveIntegers(
        _ rawValue: String?,
        fallback: [Int],
        optionName: String
    ) throws -> [Int] {
        guard let rawValue else {
            return fallback
        }
        let values = rawValue
            .split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
        guard !values.isEmpty else {
            throw ValidationError("\(optionName) must contain at least one value.")
        }
        return try values.map { value in
            guard let parsed = Int(value), parsed > 0 else {
                throw ValidationError("\(optionName) values must be positive integers.")
            }
            return parsed
        }
    }

    private func parsedTemperatures() throws -> [Double] {
        guard let temperatureValues else {
            return [temperature]
        }
        let values = temperatureValues
            .split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
        guard !values.isEmpty else {
            throw ValidationError("--temperature-values must contain at least one value.")
        }
        return try values.map { value in
            guard let parsed = Double(value), (0...2).contains(parsed), parsed.isFinite else {
                throw ValidationError("--temperature-values values must be finite and between 0 and 2.")
            }
            return parsed
        }
    }

    private func runVariant(
        _ variant: Q36MTPBenchmarkVariant,
        request: ChatRequest
    ) async throws -> Q36MTPBenchmarkVariantResult {
        let generator = Q35Generator(modelId: model)
        let memoryBefore = ProcessMemorySnapshot.currentResidentBytes()
        let start = Date()
        let response = try await withTemporaryEnvironment(variant.environmentOverrides(self)) {
            try await generator.chat(request, modelPath: modelRoot, progressHandler: nil)
        }
        let elapsed = Date().timeIntervalSince(start)
        let memoryAfter = ProcessMemorySnapshot.currentResidentBytes()
        await generator.unload()

        guard let timing = response.timing else {
            throw ValidationError("\(variant.name) did not return timing data.")
        }
        let promptTokens = response.promptTokens ?? 0
        return Q36MTPBenchmarkVariantResult(
            name: variant.name,
            policy: variant.policy,
            expectedMTPActive: variant.expectedMTPActive(
                promptTokens: promptTokens,
                contextSize: contextSize,
                forcedThreshold: forcedMTPMinPromptTokens
            ),
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

private struct Q36MTPBenchmarkVariant {
    static let adaptiveThreshold = 6_144

    let name: String
    let policy: String

    static let baseline = Self(name: "baseline", policy: "disabled")
    static let adaptive = Self(name: "adaptive", policy: "adaptive")
    static let forced = Self(name: "forced", policy: "forced")

    func environmentOverrides(_ command: ModelBenchmarkQ36MTP) -> [String: String?] {
        var overrides: [String: String?] = [
            "MERERUN_Q35_MTP_SPECULATION": nil,
            "MERERUN_Q35_MTP_MIN_PROMPT_TOKENS": nil,
            "MERERUN_Q35_MTP_BLOCK_SIZE": nil,
        ]
        switch self.name {
        case Self.baseline.name:
            overrides["MERERUN_Q35_MTP_SPECULATION"] = "0"
        case Self.forced.name:
            overrides["MERERUN_Q35_MTP_SPECULATION"] = "1"
            overrides["MERERUN_Q35_MTP_MIN_PROMPT_TOKENS"] = String(command.forcedMTPMinPromptTokens)
        default:
            break
        }
        if self.name != Self.baseline.name, let mtpBlockSize = command.mtpBlockSize {
            overrides["MERERUN_Q35_MTP_BLOCK_SIZE"] = String(mtpBlockSize)
        }
        return overrides
    }

    func expectedMTPActive(
        promptTokens: Int,
        contextSize: Int,
        forcedThreshold: Int
    ) -> Bool {
        switch name {
        case Self.baseline.name:
            return false
        case Self.forced.name:
            return contextSize >= forcedThreshold
        default:
            return promptTokens >= Self.adaptiveThreshold && contextSize >= Self.adaptiveThreshold
        }
    }
}

private struct Q36MTPBenchmarkReport: Encodable {
    let model: String
    let scenarios: [Q36MTPBenchmarkScenarioResult]

    func jsonString() throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(self)
        return String(decoding: data, as: UTF8.self)
    }

    func renderText() -> String {
        var lines = [
            "Q36 MTP benchmark",
            "model: \(model)",
            "",
        ]
        for scenario in scenarios {
            lines.append(scenario.renderText())
        }
        return lines.joined(separator: "\n")
    }
}

private struct Q36MTPBenchmarkScenarioResult: Encodable {
    let promptRepeat: Int?
    let promptCharacters: Int
    let requestedDecodeTokens: Int
    let temperature: Double
    let contextSize: Int
    let variants: [Q36MTPBenchmarkVariantResult]

    var decodeSpeedups: [String: Double] {
        speedups(for: \.decodeTokensPerSecond)
    }

    var endToEndSpeedups: [String: Double] {
        speedups(for: \.endToEndTokensPerSecond)
    }

    enum CodingKeys: String, CodingKey {
        case promptRepeat
        case promptCharacters
        case requestedDecodeTokens
        case temperature
        case contextSize
        case decodeSpeedups
        case endToEndSpeedups
        case variants
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encodeIfPresent(promptRepeat, forKey: .promptRepeat)
        try container.encode(promptCharacters, forKey: .promptCharacters)
        try container.encode(requestedDecodeTokens, forKey: .requestedDecodeTokens)
        try container.encode(temperature, forKey: .temperature)
        try container.encode(contextSize, forKey: .contextSize)
        try container.encode(decodeSpeedups, forKey: .decodeSpeedups)
        try container.encode(endToEndSpeedups, forKey: .endToEndSpeedups)
        try container.encode(variants, forKey: .variants)
    }

    func renderText() -> String {
        var lines = [
            "scenario: prompt_repeat=\(promptRepeat.map(String.init) ?? "custom") prompt_characters=\(promptCharacters) decode_tokens=\(requestedDecodeTokens) temperature=\(format(temperature)) context_size=\(contextSize)",
            "speedup decode: \(formatSpeedups(decodeSpeedups))",
            "speedup e2e: \(formatSpeedups(endToEndSpeedups))",
        ]
        for result in variants {
            lines.append(result.renderText())
        }
        return lines.joined(separator: "\n")
    }

    private func speedups(for metric: KeyPath<Q36MTPBenchmarkVariantResult, Double?>) -> [String: Double] {
        guard
            let baseline = variants.first(where: { $0.policy == "disabled" })?[keyPath: metric],
            baseline > 0
        else {
            return [:]
        }
        return variants
            .filter { $0.policy != "disabled" }
            .reduce(into: [String: Double]()) { result, variant in
                if let value = variant[keyPath: metric] {
                    result[variant.name] = value / baseline
                }
            }
    }

    private func formatSpeedups(_ values: [String: Double]) -> String {
        guard !values.isEmpty else { return "n/a" }
        return values.keys.sorted().map { key in
            "\(key)=\(format(values[key] ?? 0))x"
        }.joined(separator: " ")
    }

    private func format(_ value: Double) -> String {
        String(format: "%.2f", value)
    }
}

private struct Q36MTPBenchmarkVariantResult: Encodable {
    let name: String
    let policy: String
    let expectedMTPActive: Bool
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
        case policy
        case expectedMTPActive
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
        try container.encode(policy, forKey: .policy)
        try container.encode(expectedMTPActive, forKey: .expectedMTPActive)
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
            "\(name): policy=\(policy) expected_mtp_active=\(expectedMTPActive)",
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

struct ModelBenchmarkGemma4MTP: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "gemma4-mtp",
        abstract: "Compare Gemma4 serial decode against verified MTP speculative decode."
    )

    @Option(name: [.long], help: "Gemma4 model id or repo id.")
    var model: String = Gemma4Resources.twelveB4BitModelId

    @Option(name: [.customShort("m"), .long], help: "Override model root directory.")
    var modelRoot: String?

    @Option(name: [.customShort("p"), .long], help: "Benchmark prompt. Defaults to a repeated deterministic fixture.")
    var prompt: String?

    @Option(name: [.long], help: "Read the benchmark prompt from a UTF-8 file.")
    var promptFile: String?

    @Option(name: [.long], help: "Repeat count for the built-in fixture prompt.")
    var promptRepeat: Int = 220

    @Option(name: [.long], help: "Comma-separated repeat counts for a prompt-size benchmark matrix.")
    var promptRepeatValues: String?

    @Option(name: [.long], help: "Exact number of decode tokens to force per variant.")
    var decodeTokens: Int = 48

    @Option(name: [.long], help: "Comma-separated decode token counts for a decode-length benchmark matrix.")
    var decodeTokenValues: String?

    @Option(name: [.long], help: "Override MERERUN_GEMMA4_MTP_BLOCK_SIZE for the MTP variant.")
    var mtpBlockSize: Int?

    @Option(name: [.long], help: "Override MERERUN_GEMMA4_MTP_MIN_PROMPT_TOKENS for the MTP variant.")
    var mtpMinPromptTokens: Int?

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
        let promptRepeats = try parsedPositiveIntegers(
            promptRepeatValues,
            fallback: [promptRepeat],
            optionName: "--prompt-repeat-values"
        )
        _ = try parsedPositiveIntegers(
            decodeTokenValues,
            fallback: [decodeTokens],
            optionName: "--decode-token-values"
        )
        if prompt != nil || promptFile != nil {
            guard promptRepeats.count == 1 else {
                throw ValidationError("--prompt-repeat-values can only contain one value when using --prompt or --prompt-file.")
            }
        }
        if let mtpBlockSize {
            guard mtpBlockSize >= 2 else {
                throw ValidationError("--mtp-block-size must be at least 2.")
            }
        }
        if let mtpMinPromptTokens {
            guard mtpMinPromptTokens >= 0 else {
                throw ValidationError("--mtp-min-prompt-tokens must be non-negative.")
            }
        }
        guard Gemma4Resources.handles(modelSpec: model) else {
            throw ValidationError("model benchmark gemma4-mtp only supports Gemma4 model ids or repo ids.")
        }
    }

    func run() async throws {
        try MLXBundleSupport.ensureAvailable(quiet: json)
        let promptRepeats = try parsedPositiveIntegers(
            promptRepeatValues,
            fallback: [promptRepeat],
            optionName: "--prompt-repeat-values"
        )
        let decodeTokenCounts = try parsedPositiveIntegers(
            decodeTokenValues,
            fallback: [decodeTokens],
            optionName: "--decode-token-values"
        )

        let variants = [
            Gemma4MTPBenchmarkVariant.baseline,
            Gemma4MTPBenchmarkVariant.mtp,
        ]

        var scenarios: [Gemma4MTPBenchmarkScenarioResult] = []
        scenarios.reserveCapacity(promptRepeats.count * decodeTokenCounts.count)
        for repeatCount in promptRepeats {
            let prompt = try resolvePrompt(repeatCount: repeatCount)
            for decodeTokenCount in decodeTokenCounts {
                let request = ChatRequest(
                    messages: [ChatMessage(role: .user, content: prompt)],
                    maxTokens: decodeTokenCount,
                    temperature: 0,
                    topP: 1,
                    showThinking: false,
                    stopOnEOS: false
                )
                var results: [Gemma4MTPBenchmarkVariantResult] = []
                results.reserveCapacity(variants.count)
                for variant in variants {
                    let result = try await runVariant(variant, request: request)
                    guard result.generatedTokens == decodeTokenCount else {
                        throw ValidationError(
                            "\(variant.name) generated \(result.generatedTokens) tokens, expected \(decodeTokenCount). Reduce prompt length or decode tokens."
                        )
                    }
                    results.append(result)
                }
                scenarios.append(
                    Gemma4MTPBenchmarkScenarioResult(
                        promptRepeat: self.prompt != nil || self.promptFile != nil ? nil : repeatCount,
                        promptCharacters: prompt.count,
                        requestedDecodeTokens: decodeTokenCount,
                        variants: results
                    )
                )
            }
        }

        let report = Gemma4MTPBenchmarkReport(
            model: model,
            scenarios: scenarios
        )
        if json {
            print(try report.jsonString())
        } else {
            print(report.renderText())
        }
    }

    private func resolvePrompt(repeatCount: Int) throws -> String {
        if let prompt {
            return prompt
        }
        if let promptFile {
            return try String(contentsOf: URL(fileURLWithPath: promptFile).standardizedFileURL, encoding: .utf8)
        }
        var lines: [String] = []
        lines.reserveCapacity(repeatCount + 1)
        for index in 1...repeatCount {
            lines.append(
                "Record \(index): Gemma4 MTP benchmark passage about local inference, speculative decode verification, draft acceptance, quantized weights, and repeatable measurements. The checksum word is swift."
            )
        }
        lines.append("Question: Write one concise sentence naming the checksum word and benchmark subject.")
        return lines.joined(separator: "\n")
    }

    private func parsedPositiveIntegers(
        _ rawValue: String?,
        fallback: [Int],
        optionName: String
    ) throws -> [Int] {
        guard let rawValue else {
            return fallback
        }
        let values = rawValue
            .split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
        guard !values.isEmpty else {
            throw ValidationError("\(optionName) must contain at least one value.")
        }
        return try values.map { value in
            guard let parsed = Int(value), parsed > 0 else {
                throw ValidationError("\(optionName) values must be positive integers.")
            }
            return parsed
        }
    }

    private func runVariant(
        _ variant: Gemma4MTPBenchmarkVariant,
        request: ChatRequest
    ) async throws -> Gemma4MTPBenchmarkVariantResult {
        let generator = Gemma4Generator(modelId: model)
        let memoryBefore = ProcessMemorySnapshot.currentResidentBytes()
        let start = Date()
        let response = try await withTemporaryEnvironment(variant.environmentOverrides(self)) {
            try await generator.chat(request, modelPath: modelRoot, progressHandler: nil)
        }
        let elapsed = Date().timeIntervalSince(start)
        let mtpStats = await generator.mtpStats()
        let memoryAfter = ProcessMemorySnapshot.currentResidentBytes()
        await generator.unload()

        guard let timing = response.timing else {
            throw ValidationError("\(variant.name) did not return timing data.")
        }
        let promptTokens = response.promptTokens ?? 0
        return Gemma4MTPBenchmarkVariantResult(
            name: variant.name,
            mtpEnabled: variant.enabled,
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
            residentMemoryAfterBytes: memoryAfter,
            mtpStats: mtpStats
        )
    }
}

private struct Gemma4MTPBenchmarkVariant {
    let name: String
    let enabled: Bool

    static let baseline = Self(name: "baseline", enabled: false)
    static let mtp = Self(name: "mtp", enabled: true)

    func environmentOverrides(_ command: ModelBenchmarkGemma4MTP) -> [String: String?] {
        var overrides: [String: String?] = [
            "MERERUN_GEMMA4_MTP": enabled ? "1" : "0",
        ]
        if let mtpBlockSize = command.mtpBlockSize {
            overrides["MERERUN_GEMMA4_MTP_BLOCK_SIZE"] = String(mtpBlockSize)
        }
        if let mtpMinPromptTokens = command.mtpMinPromptTokens {
            overrides["MERERUN_GEMMA4_MTP_MIN_PROMPT_TOKENS"] = String(mtpMinPromptTokens)
        }
        return overrides
    }
}

private struct Gemma4MTPBenchmarkReport: Encodable {
    let model: String
    let scenarios: [Gemma4MTPBenchmarkScenarioResult]

    func jsonString() throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(self)
        return String(decoding: data, as: UTF8.self)
    }

    func renderText() -> String {
        var lines = [
            "Gemma4 MTP benchmark",
            "model: \(model)",
            "",
        ]
        for scenario in scenarios {
            lines.append(scenario.renderText())
        }
        return lines.joined(separator: "\n")
    }
}

private struct Gemma4MTPBenchmarkScenarioResult: Encodable {
    let promptRepeat: Int?
    let promptCharacters: Int
    let requestedDecodeTokens: Int
    let variants: [Gemma4MTPBenchmarkVariantResult]

    var decodeSpeedup: Double? {
        speedup(for: \.decodeTokensPerSecond)
    }

    var endToEndSpeedup: Double? {
        speedup(for: \.endToEndTokensPerSecond)
    }

    enum CodingKeys: String, CodingKey {
        case promptRepeat
        case promptCharacters
        case requestedDecodeTokens
        case decodeSpeedup
        case endToEndSpeedup
        case variants
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encodeIfPresent(promptRepeat, forKey: .promptRepeat)
        try container.encode(promptCharacters, forKey: .promptCharacters)
        try container.encode(requestedDecodeTokens, forKey: .requestedDecodeTokens)
        try container.encodeIfPresent(decodeSpeedup, forKey: .decodeSpeedup)
        try container.encodeIfPresent(endToEndSpeedup, forKey: .endToEndSpeedup)
        try container.encode(variants, forKey: .variants)
    }

    func renderText() -> String {
        var lines = [
            "scenario: prompt_repeat=\(promptRepeat.map(String.init) ?? "custom") prompt_characters=\(promptCharacters) decode_tokens=\(requestedDecodeTokens)",
            "speedup: decode=\(formatOptional(decodeSpeedup))x e2e=\(formatOptional(endToEndSpeedup))x",
        ]
        for result in variants {
            lines.append(result.renderText())
        }
        return lines.joined(separator: "\n")
    }

    private func speedup(
        for metric: KeyPath<Gemma4MTPBenchmarkVariantResult, Double?>
    ) -> Double? {
        guard
            let baseline = variants.first(where: { !$0.mtpEnabled })?[keyPath: metric],
            let mtp = variants.first(where: { $0.mtpEnabled })?[keyPath: metric],
            baseline > 0
        else {
            return nil
        }
        return mtp / baseline
    }

    private func formatOptional(_ value: Double?) -> String {
        guard let value else { return "n/a" }
        return String(format: "%.2f", value)
    }
}

private struct Gemma4MTPBenchmarkVariantResult: Encodable {
    let name: String
    let mtpEnabled: Bool
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
    let mtpStats: Gemma4MTPStats

    var ttftSeconds: Double? {
        guard let firstTokenSeconds else { return nil }
        return loadSeconds + prefillSeconds + firstTokenSeconds
    }

    var mtpAcceptanceRate: Double? {
        guard mtpStats.draftedTokens > 0 else { return nil }
        return Double(mtpStats.acceptedTokens) / Double(mtpStats.draftedTokens)
    }

    var mtpAcceptedTokensPerRound: Double? {
        guard mtpStats.rounds > 0 else { return nil }
        return Double(mtpStats.acceptedTokens) / Double(mtpStats.rounds)
    }

    enum CodingKeys: String, CodingKey {
        case name
        case mtpEnabled
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
        case mtpStats
        case mtpAcceptanceRate
        case mtpAcceptedTokensPerRound
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(name, forKey: .name)
        try container.encode(mtpEnabled, forKey: .mtpEnabled)
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
        try container.encode(mtpStats, forKey: .mtpStats)
        try container.encodeIfPresent(mtpAcceptanceRate, forKey: .mtpAcceptanceRate)
        try container.encodeIfPresent(mtpAcceptedTokensPerRound, forKey: .mtpAcceptedTokensPerRound)
    }

    func renderText() -> String {
        [
            "\(name): mtp=\(mtpEnabled ? "enabled" : "disabled") state=\(mtpState)",
            "  prompt_tokens=\(promptTokens) generated_tokens=\(generatedTokens)",
            "  time total=\(format(elapsedSeconds))s load=\(format(loadSeconds))s prefill=\(format(prefillSeconds))s decode=\(format(decodeSeconds))s ttft=\(formatOptional(ttftSeconds))s",
            "  throughput prefill=\(formatOptional(prefillTokensPerSecond)) tok/s decode=\(formatOptional(decodeTokensPerSecond)) tok/s e2e=\(formatOptional(endToEndTokensPerSecond)) tok/s",
            "  mtp block=\(mtpStats.blockSize) threshold=\(mtpStats.threshold) rounds=\(mtpStats.rounds) drafted=\(mtpStats.draftedTokens) accepted=\(mtpStats.acceptedTokens) rejected=\(mtpStats.rejectedTokens) accept_rate=\(formatOptional(mtpAcceptanceRate)) accepted_per_round=\(formatOptional(mtpAcceptedTokensPerRound)) reason=\(mtpStats.reason ?? "n/a")",
            "  resident_memory before=\(formatBytes(residentMemoryBeforeBytes)) after=\(formatBytes(residentMemoryAfterBytes))",
            "",
        ].joined(separator: "\n")
    }

    private var mtpState: String {
        if mtpStats.active { return "active" }
        if mtpStats.available { return "available" }
        return "unavailable"
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

    @Option(name: [.long], help: "Comma-separated repeat counts for a prompt-size benchmark matrix.")
    var promptRepeatValues: String?

    @Option(name: [.long], help: "Exact number of decode tokens to force per variant.")
    var decodeTokens: Int = 48

    @Option(name: [.long], help: "Comma-separated decode token counts for a decode-length benchmark matrix.")
    var decodeTokenValues: String?

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
        let promptRepeats = try parsedPositiveIntegers(
            promptRepeatValues,
            fallback: [promptRepeat],
            optionName: "--prompt-repeat-values"
        )
        _ = try parsedPositiveIntegers(
            decodeTokenValues,
            fallback: [decodeTokens],
            optionName: "--decode-token-values"
        )
        if prompt != nil || promptFile != nil {
            guard promptRepeats.count == 1 else {
                throw ValidationError("--prompt-repeat-values can only contain one value when using --prompt or --prompt-file.")
            }
        }
        guard Gemma4Resources.handles(modelSpec: model) else {
            throw ValidationError("model benchmark gemma4-kv only supports Gemma4 model ids or repo ids.")
        }
    }

    func run() async throws {
        try MLXBundleSupport.ensureAvailable(quiet: json)
        let promptRepeats = try parsedPositiveIntegers(
            promptRepeatValues,
            fallback: [promptRepeat],
            optionName: "--prompt-repeat-values"
        )
        let decodeTokenCounts = try parsedPositiveIntegers(
            decodeTokenValues,
            fallback: [decodeTokens],
            optionName: "--decode-token-values"
        )

        let variants = [
            Gemma4KVBenchmarkVariant.defaultTurbo(model: model),
            Gemma4KVBenchmarkVariant.polar2(model: model),
        ]

        var scenarios: [Gemma4KVBenchmarkScenarioResult] = []
        scenarios.reserveCapacity(promptRepeats.count * decodeTokenCounts.count)
        for repeatCount in promptRepeats {
            let prompt = try resolvePrompt(repeatCount: repeatCount)
            for decodeTokenCount in decodeTokenCounts {
                let request = ChatRequest(
                    messages: [ChatMessage(role: .user, content: prompt)],
                    maxTokens: decodeTokenCount,
                    temperature: temperature,
                    topP: topP,
                    showThinking: false,
                    stopOnEOS: false
                )
                var results: [Gemma4KVBenchmarkVariantResult] = []
                results.reserveCapacity(variants.count)
                for variant in variants {
                    let result = try await runVariant(variant, request: request)
                    guard result.generatedTokens == decodeTokenCount else {
                        throw ValidationError(
                            "\(variant.name) generated \(result.generatedTokens) tokens, expected \(decodeTokenCount). Reduce prompt length or decode tokens."
                        )
                    }
                    results.append(result)
                }
                scenarios.append(
                    Gemma4KVBenchmarkScenarioResult(
                        promptRepeat: self.prompt != nil || self.promptFile != nil ? nil : repeatCount,
                        promptCharacters: prompt.count,
                        requestedDecodeTokens: decodeTokenCount,
                        variants: results
                    )
                )
            }
        }

        let report = Gemma4KVBenchmarkReport(
            model: model,
            scenarios: scenarios
        )
        if json {
            print(try report.jsonString())
        } else {
            print(report.renderText())
        }
    }

    private func resolvePrompt(repeatCount: Int) throws -> String {
        if let prompt {
            return prompt
        }
        if let promptFile {
            return try String(contentsOf: URL(fileURLWithPath: promptFile).standardizedFileURL, encoding: .utf8)
        }
        var lines: [String] = []
        lines.reserveCapacity(repeatCount + 1)
        for index in 1...repeatCount {
            lines.append(
                "Record \(index): Gemma turbo benchmark passage about local inference, packed key value cache memory, decode timing, and repeatable measurements. The checksum word is polar."
            )
        }
        lines.append("Question: Write one concise sentence naming the checksum word and benchmark subject.")
        return lines.joined(separator: "\n")
    }

    private func parsedPositiveIntegers(
        _ rawValue: String?,
        fallback: [Int],
        optionName: String
    ) throws -> [Int] {
        guard let rawValue else {
            return fallback
        }
        let values = rawValue
            .split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
        guard !values.isEmpty else {
            throw ValidationError("\(optionName) must contain at least one value.")
        }
        return try values.map { value in
            guard let parsed = Int(value), parsed > 0 else {
                throw ValidationError("\(optionName) values must be positive integers.")
            }
            return parsed
        }
    }

    private func runVariant(
        _ variant: Gemma4KVBenchmarkVariant,
        request: ChatRequest
    ) async throws -> Gemma4KVBenchmarkVariantResult {
        var request = request
        request.kvCacheMode = variant.requestKVCacheMode
        let generator = Gemma4Generator(modelId: model, kvCacheQuantization: variant.generatorQuantization)
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
            kvScheme: variant.displayQuantization.scheme.rawValue,
            kvBits: variant.displayQuantization.bits,
            quantizedKVStart: variant.displayQuantization.quantizedStart,
            promptTokens: promptTokens,
            generatedTokens: response.tokensGenerated,
            elapsedSeconds: elapsed,
            loadSeconds: timing.loadSeconds,
            prefillSeconds: timing.prefillSeconds,
            cacheConversionSeconds: timing.cacheConversionSeconds,
            decodeSeconds: timing.decodeSeconds,
            firstTokenSeconds: timing.firstTokenSeconds,
            kvCacheMode: timing.kvCacheMode,
            prefillKVCache: timing.prefillKVCache,
            decodeKVCache: timing.decodeKVCache,
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
    let displayQuantization: Gemma4KVCacheQuantization
    let generatorQuantization: Gemma4KVCacheQuantization
    let requestKVCacheMode: RuntimeKVCacheMode

    static func defaultTurbo(model: String) -> Self {
        let usesTurboDefaults = Gemma4Resources.usesTurboDefaults(modelSpec: model)
            && Gemma4Resources.supportsDefaultTurboKVQuantization
        let quantization = Gemma4KVCacheQuantization(
            bits: usesTurboDefaults ? Gemma4Resources.defaultTurboKVBits : nil,
            scheme: usesTurboDefaults ? Gemma4Resources.defaultTurboKVQuantizationScheme : .uniform,
            groupSize: Gemma4Resources.defaultKVGroupSize,
            quantizedStart: usesTurboDefaults
                ? Gemma4Resources.defaultTurboQuantizedKVStart
                : Gemma4Resources.defaultQuantizedKVStart
        )
        return Self(
            name: "default",
            displayQuantization: quantization,
            generatorQuantization: quantization,
            requestKVCacheMode: .default
        )
    }

    static func polar2(model: String) -> Self {
        let fallback = defaultTurbo(model: model).generatorQuantization
        return Self(
            name: "polar2",
            displayQuantization: Gemma4KVCacheQuantization(
                bits: 2,
                scheme: .polar,
                groupSize: Gemma4Resources.defaultKVGroupSize,
                quantizedStart: 0
            ),
            generatorQuantization: fallback,
            requestKVCacheMode: .polar2
        )
    }
}

private struct Gemma4KVBenchmarkReport: Encodable {
    let model: String
    let scenarios: [Gemma4KVBenchmarkScenarioResult]

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
            "",
        ]
        for scenario in scenarios {
            lines.append(scenario.renderText())
        }
        return lines.joined(separator: "\n")
    }
}

private struct Gemma4KVBenchmarkScenarioResult: Encodable {
    let promptRepeat: Int?
    let promptCharacters: Int
    let requestedDecodeTokens: Int
    let variants: [Gemma4KVBenchmarkVariantResult]

    func renderText() -> String {
        var lines = [
            "scenario: prompt_repeat=\(promptRepeat.map(String.init) ?? "custom") prompt_characters=\(promptCharacters) decode_tokens=\(requestedDecodeTokens)",
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
    let cacheConversionSeconds: Double?
    let decodeSeconds: Double
    let firstTokenSeconds: Double?
    let kvCacheMode: RuntimeKVCacheMode?
    let prefillKVCache: String?
    let decodeKVCache: String?
    let prefillTokensPerSecond: Double?
    let decodeTokensPerSecond: Double?
    let endToEndTokensPerSecond: Double?
    let residentMemoryBeforeBytes: UInt64?
    let residentMemoryAfterBytes: UInt64?

    var ttftSeconds: Double? {
        guard let firstTokenSeconds else { return nil }
        return loadSeconds + prefillSeconds + (cacheConversionSeconds ?? 0) + firstTokenSeconds
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
        case cacheConversionSeconds
        case decodeSeconds
        case firstTokenSeconds
        case ttftSeconds
        case kvCacheMode
        case prefillKVCache
        case decodeKVCache
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
        try container.encodeIfPresent(cacheConversionSeconds, forKey: .cacheConversionSeconds)
        try container.encode(decodeSeconds, forKey: .decodeSeconds)
        try container.encodeIfPresent(firstTokenSeconds, forKey: .firstTokenSeconds)
        try container.encodeIfPresent(ttftSeconds, forKey: .ttftSeconds)
        try container.encodeIfPresent(kvCacheMode, forKey: .kvCacheMode)
        try container.encodeIfPresent(prefillKVCache, forKey: .prefillKVCache)
        try container.encodeIfPresent(decodeKVCache, forKey: .decodeKVCache)
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
            "  kv mode=\(kvCacheMode?.rawValue ?? "default") prefill=\(prefillKVCache ?? "n/a") decode=\(decodeKVCache ?? "n/a")",
            "  time total=\(format(elapsedSeconds))s load=\(format(loadSeconds))s prefill=\(format(prefillSeconds))s kv_convert=\(formatOptional(cacheConversionSeconds))s decode=\(format(decodeSeconds))s ttft=\(formatOptional(ttftSeconds))s",
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

private func withTemporaryEnvironment<T>(
    _ overrides: [String: String?],
    operation: () async throws -> T
) async throws -> T {
    let previousValues = overrides.keys.reduce(into: [String: String?]()) { values, key in
        values.updateValue(ProcessInfo.processInfo.environment[key], forKey: key)
    }
    applyEnvironment(overrides)
    do {
        let result = try await operation()
        applyEnvironment(previousValues)
        return result
    } catch {
        applyEnvironment(previousValues)
        throw error
    }
}

private func applyEnvironment(_ values: [String: String?]) {
    for (key, value) in values {
        if let value {
            setenv(key, value, 1)
        } else {
            unsetenv(key)
        }
    }
}
