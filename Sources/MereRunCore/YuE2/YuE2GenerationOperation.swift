import AudioCore
import Foundation
import MLX
import MLXRandom

public struct YuE2Progress: Sendable {
    public enum Stage: String, Codable, Sendable { case loading, score, semantic, acoustic, decoding }
    public let stage: Stage
    public let completed: Int
    public let total: Int
}

public struct YuE2GenerationResult: Sendable {
    public let waveform: AudioWaveform
    public let abc: String?
    public let scoreTokens: [Int]
    public let codecTokens: [Int]
    public let scoreTruncated: Bool
    public let musicTruncated: Bool
}

/// Serial native Swift/MLX execution. The transformer is released before the
/// FP32 decoder is loaded; each request owns its seeds, caches, and artifacts.
public actor YuE2GenerationOperation {
    private let resources: YuE2Resources
    private let streams = Stream.Context()
    private var generating = false

    public init(resources: YuE2Resources) { self.resources = resources }

    public func generate(
        _ plan: YuE2GenerationPlan, progress: (@Sendable (YuE2Progress) -> Void)? = nil
    ) async throws -> YuE2GenerationResult {
        try plan.validate()
        guard !generating else { throw YuE2Error.invalidRequest("This YuE2 operation is already generating.") }
        generating = true
        defer { generating = false }
        return try await Stream.withDefaultStream(streams) {
            defer { streams.synchronize() }
            try Task.checkCancellation()
            let missing = resources.validate()
            guard missing.isEmpty else {
                throw YuE2Error.invalidWeights("Missing files: \(missing.map(\.path).joined(separator: ", ")).")
            }
            let config = try resources.configuration()
            let vaeConfig = try resources.vaeConfiguration()
            let tokenizer = try YuE2Tokenizer(url: resources.rootURL.appendingPathComponent("qwen.tiktoken"))
            let prepared = try Self.prepare(plan, tokenizer: tokenizer)
            progress?(YuE2Progress(stage: .loading, completed: 0, total: 1))
            let generated = try generateLatents(plan, prepared: prepared, tokenizer: tokenizer,
                                                configuration: config, progress: progress)
            streams.synchronize()
            MLX.Memory.clearCache()
            try Task.checkCancellation()
            let arrays = try SafetensorsStreamingLoader.loadArrays(
                url: resources.vaeURL.appendingPathComponent("model.safetensors"),
                where: { $0.hasPrefix("decoder.") }, dtype: .float32
            )
            let decoder = try YuE2Decoder(configuration: vaeConfig, arrays: arrays)
            let waveform = try decoder.decodeTiled(generated.latents) { completed, total in
                progress?(YuE2Progress(stage: .decoding, completed: completed, total: total))
            }
            return YuE2GenerationResult(
                waveform: waveform, abc: generated.abc, scoreTokens: generated.score,
                codecTokens: generated.codec, scoreTruncated: generated.scoreTruncated,
                musicTruncated: generated.musicTruncated
            )
        }
    }

    struct Prepared {
        let prefix: [Int]
        let score: [Int]?
    }

    static func prepare(_ plan: YuE2GenerationPlan, tokenizer: YuE2Tokenizer) throws -> Prepared {
        let score = try plan.abc.map { try tokenizer.encode($0) }
        let prefix = try YuE2Protocol.prefix(plan: plan, tokenizer: tokenizer, score: score)
        let budget = plan.planning == .off || score != nil
            ? plan.semanticSampling.maximumTokens : plan.scoreSampling.maximumTokens
        guard prefix.count + budget <= YuE2Protocol.context else {
            throw YuE2Error.invalidRequest("Prompt plus generation budget exceeds 24576 tokens; shorten the input.")
        }
        return Prepared(prefix: prefix, score: score)
    }

    private struct LatentResult {
        let latents: MLXArray
        let abc: String?
        let score: [Int]
        let codec: [Int]
        let scoreTruncated: Bool
        let musicTruncated: Bool
    }

    private func generateLatents(
        _ plan: YuE2GenerationPlan, prepared: Prepared, tokenizer: YuE2Tokenizer,
        configuration: YuE2Configuration, progress: (@Sendable (YuE2Progress) -> Void)?
    ) throws -> LatentResult {
        let model = try YuE2Model(configuration: configuration, arrays: SafetensorsStreamingLoader.loadArrays(
            url: resources.rootURL.appendingPathComponent("model.safetensors")
        ))
        progress?(YuE2Progress(stage: .loading, completed: 1, total: 1))
        let score: [Int]
        let scoreTruncated: Bool
        if plan.planning != .off, prepared.score == nil {
            (score, scoreTruncated) = try YuE2Sampler.generate(
                model: model, prefix: prepared.prefix, negative: nil, sampling: plan.scoreSampling,
                seed: plan.seed, phase: .score
            ) { progress?(YuE2Progress(stage: .score, completed: $0, total: plan.scoreSampling.maximumTokens)) }
        } else {
            score = prepared.score ?? []
            scoreTruncated = false
        }
        let prefix = try YuE2Protocol.prefix(plan: plan, tokenizer: tokenizer, score: score)
        // Validate acoustic capacity before spending the semantic token budget.
        _ = try YuE2Protocol.chunkRanges(frames: plan.semanticSampling.maximumTokens, prefixTokens: prefix.count)
        let negative = try plan.guidanceScale == 1 ? nil
            : YuE2Protocol.negative(plan: plan, tokenizer: tokenizer, score: score)
        let semantic = try YuE2Sampler.generate(
            model: model, prefix: prefix, negative: negative, sampling: plan.semanticSampling,
            seed: plan.seed, phase: .semantic, guidance: plan.guidanceScale, legacyOff: plan.planning == .off
        ) { progress?(YuE2Progress(stage: .semantic, completed: $0, total: plan.semanticSampling.maximumTokens)) }
        guard !semantic.tokens.isEmpty else { throw YuE2Error.invalidAudio("The model generated no music frames.") }
        let ranges = try YuE2Protocol.chunkRanges(frames: semantic.tokens.count, prefixTokens: prefix.count)
        // Draw the full-song noise once, then slice at the checkpoint's original context cuts.
        let random = MLXRandom.RandomState(seed: plan.seed)
        let noise = MLXRandom.normal([semantic.tokens.count, configuration.latentDim], key: random)
        MLX.eval(noise)
        var chunks: [MLXArray] = []
        for (index, range) in ranges.enumerated() {
            try Task.checkCancellation()
            let tokens = prefix + semantic.tokens[range] + [YuE2Protocol.musicEnd]
            let acoustic = try YuE2Acoustic(model: model, tokens: tokens)
            let chunk = try acoustic.solve(noise: noise[range], steps: plan.steps) {
                progress?(YuE2Progress(stage: .acoustic, completed: index * plan.steps + $0, total: ranges.count * plan.steps))
            }
            chunks.append(chunk)
        }
        let latents = concatenated(chunks, axis: 0)
        MLX.eval(latents)
        return LatentResult(latents: latents, abc: plan.planning == .off ? nil : try tokenizer.decode(score),
                            score: score, codec: semantic.tokens.map { $0 - YuE2Protocol.codecOffset },
                            scoreTruncated: scoreTruncated, musicTruncated: semantic.truncated)
    }
}
