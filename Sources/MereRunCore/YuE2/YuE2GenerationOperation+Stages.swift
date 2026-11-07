import AudioCore
import Foundation
import MLX
import MLXRandom

extension YuE2GenerationOperation {
    /// Advance exactly one stage. Existing destinations are refused, and source stages stay immutable.
    public func saveStage(
        _ stage: YuE2SavedStage.Stage, plan: YuE2GenerationPlan? = nil,
        resuming directory: URL? = nil, to output: URL,
        progress: (@Sendable (YuE2Progress) -> Void)? = nil
    ) async throws -> YuE2SavedStage {
        guard !savingGeneration else { throw YuE2Error.invalidRequest("This YuE2 operation is already generating.") }
        return try await advanceStage(stage, plan: plan, resuming: directory, to: output, progress: progress)
    }

    func advanceStage(
        _ stage: YuE2SavedStage.Stage, plan: YuE2GenerationPlan? = nil,
        resuming directory: URL? = nil, to output: URL,
        progress: (@Sendable (YuE2Progress) -> Void)? = nil
    ) async throws -> YuE2SavedStage {
        guard !generating else { throw YuE2Error.invalidRequest("This YuE2 operation is already generating.") }
        guard !FileManager.default.fileExists(atPath: output.path) else {
            throw YuE2Error.invalidRequest("Stage output already exists; choose a new directory.")
        }
        let saved = try directory.map(YuE2SavedStage.load)
        let request: YuE2GenerationPlan
        switch (stage, saved?.stage, plan) {
        case (.score, nil, .some(let value)): request = value
        case (.semantic, .some(.score), nil), (.acoustic, .some(.semantic), nil):
            guard let saved else { throw YuE2Error.invalidRequest("Missing saved stage.") }
            request = saved.plan
        default: throw YuE2Error.invalidRequest("Plan a score, then resume score → semantic → acoustic without changing its request.")
        }
        try request.validate()
        generating = true
        defer { generating = false }
        return try await Stream.withDefaultStream(streams) {
            defer { streams.synchronize() }
            try Task.checkCancellation()
            let tokenizer = try YuE2Tokenizer(url: resources.rootURL.appendingPathComponent("qwen.tiktoken"))
            try saved?.verify(resources: resources, tokenizer: tokenizer)
            let configuration = try resources.configuration()
            var score = try saved?.scoreTokens ?? request.abc.map { try tokenizer.encode($0) } ?? []
            var codec = saved?.codecTokens ?? []
            var scoreTruncated = saved?.scoreTruncated ?? false
            var musicTruncated = saved?.musicTruncated ?? false
            var latents: MLXArray?
            if stage == .score, request.planning != .off, request.abc == nil {
                let prepared = try Self.prepare(request, tokenizer: tokenizer)
                let model = try loadStageModel(configuration)
                (score, scoreTruncated) = try YuE2Sampler.generate(
                    model: model, prefix: prepared.prefix, negative: nil, sampling: request.scoreSampling,
                    seed: request.seed, phase: .score
                ) { progress?(.init(stage: .score, completed: $0, total: request.scoreSampling.maximumTokens)) }
            }
            if stage == .score, request.instrumental == true {
                guard !scoreTruncated else { throw YuE2Error.invalidRequest("Cannot convert a truncated instrumental score.") }
                let transfer = try YuE2Score(abc: tokenizer.decode(score)).instrumentalized(keepChords: request.planning == .full)
                score = try tokenizer.encode(transfer.abc)
            }
            let prefix = try YuE2Protocol.prefix(plan: request, tokenizer: tokenizer, score: score)
            if stage == .semantic {
                guard !scoreTruncated else { throw YuE2Error.invalidRequest("Finish the score before semantic generation.") }
                _ = try YuE2Protocol.chunkRanges(frames: request.semanticSampling.maximumTokens, prefixTokens: prefix.count)
                let negative = try request.guidanceScale == 1 ? nil
                    : YuE2Protocol.negative(plan: request, tokenizer: tokenizer, score: score)
                let result = try YuE2Sampler.generate(
                    model: loadStageModel(configuration), prefix: prefix, negative: negative, sampling: request.semanticSampling,
                    seed: request.seed, phase: .semantic, guidance: request.guidanceScale, legacyOff: request.planning == .off
                ) { progress?(.init(stage: .semantic, completed: $0, total: request.semanticSampling.maximumTokens)) }
                codec = result.tokens.map { $0 - YuE2Protocol.codecOffset }
                musicTruncated = result.truncated
            }
            if stage == .acoustic {
                latents = try synthesizeStage(codec, prefix: prefix, plan: request, configuration: configuration, progress: progress)
            }
            let result = try YuE2SavedStage(
                stage: stage, plan: request, abc: request.planning == .off ? nil : tokenizer.decode(score),
                scoreTokens: score, codecTokens: codec, scoreTruncated: scoreTruncated,
                musicTruncated: musicTruncated, resources: resources
            )
            try Task.checkCancellation()
            try result.save(to: output, latents: latents)
            return try YuE2SavedStage.load(from: output)
        }
    }

    /// Decode saved or imported NumPy latents without loading the song model or tokenizer.
    public func decodeLatents(
        at url: URL, vaeRoot: URL? = nil, coreFrames: Int? = nil, full: Bool = false,
        progress: (@Sendable (YuE2Progress) -> Void)? = nil
    ) async throws -> AudioWaveform {
        guard !savingGeneration else { throw YuE2Error.invalidRequest("This YuE2 operation is already generating.") }
        return try await decodeStageLatents(at: url, vaeRoot: vaeRoot, coreFrames: coreFrames, full: full, progress: progress)
    }

    func decodeStageLatents(
        at url: URL, vaeRoot: URL? = nil, coreFrames: Int? = nil, full: Bool = false,
        progress: (@Sendable (YuE2Progress) -> Void)? = nil
    ) async throws -> AudioWaveform {
        guard !generating else { throw YuE2Error.invalidRequest("This YuE2 operation is already generating.") }
        generating = true
        defer { generating = false }
        return try await Stream.withDefaultStream(streams) {
            defer { streams.synchronize() }
            let latents = try YuE2SavedStage.loadLatents(from: url)
            let root = vaeRoot ?? resources.vaeURL
            var config = try JSONDecoder().decode(YuE2VAEConfiguration.self,
                from: Data(contentsOf: root.appendingPathComponent("config.json")))
            try config.validateReleased(allowLegacy: true)
            if let coreFrames {
                guard (1...9000).contains(coreFrames) else { throw YuE2Error.invalidRequest("VAE core frames must be in 1...9000.") }
                config.decodeCoreFrames = coreFrames
            }
            let useResidentDecoder = memoryMode == .resident && coreFrames == nil
                && root.standardizedFileURL == resources.vaeURL.standardizedFileURL
            let decoder: YuE2Decoder
            if useResidentDecoder, let cached = residentDecoder { decoder = cached }
            else {
                let arrays = try SafetensorsStreamingLoader.loadArrays(url: root.appendingPathComponent("model.safetensors"),
                    where: { $0.hasPrefix("decoder.") }, dtype: .float32)
                decoder = try YuE2Decoder(configuration: config, arrays: arrays)
                if useResidentDecoder { residentDecoder = decoder }
            }
            if full {
                let audio = try decoder.decode(latents).reshaped(-1)
                guard all(isFinite(audio)).item(Bool.self) else { throw YuE2Error.invalidAudio("Non-finite decoder output.") }
                progress?(.init(stage: .decoding, completed: 1, total: 1))
                return try AudioWaveform(interleaved: audio.asArray(Float.self), channels: 2, sampleRate: 48000)
            }
            return try decoder.decodeTiled(latents) { progress?(.init(stage: .decoding, completed: $0, total: $1)) }
        }
    }

    private func loadStageModel(_ configuration: YuE2Configuration) throws -> YuE2Model {
        try Task.checkCancellation()
        return try loadModel(configuration)
    }

    private func synthesizeStage(
        _ codec: [Int], prefix: [Int], plan: YuE2GenerationPlan, configuration: YuE2Configuration,
        progress: (@Sendable (YuE2Progress) -> Void)?
    ) throws -> MLXArray {
        let ranges = try YuE2Protocol.chunkRanges(frames: codec.count, prefixTokens: prefix.count)
        let model = try loadStageModel(configuration)
        let random = MLXRandom.RandomState(seed: plan.seed)
        let noise = MLXRandom.normal([codec.count, configuration.latentDim], key: random)
        MLX.eval(noise)
        var chunks: [MLXArray] = []
        for (index, range) in ranges.enumerated() {
            let tokens = prefix + codec[range].map { $0 + YuE2Protocol.codecOffset } + [YuE2Protocol.musicEnd]
            let acoustic = try prepareAcoustic(model, tokens: tokens)
            chunks.append(try acoustic.solve(noise: noise[range], steps: plan.steps) {
                progress?(.init(stage: .acoustic, completed: index * plan.steps + $0, total: ranges.count * plan.steps))
            })
            if memoryMode == .lowMemory { model.acoustic = [] }
        }
        let latents = concatenated(chunks, axis: 0)
        MLX.eval(latents)
        return latents
    }
}
