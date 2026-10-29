import Foundation
import MLX
import MLXNN
import AudioSortformer

/// Native inference over original BF16 or locally quantized routed experts. Confine instances to one request stream.
public final class NemotronTranscribeModel {
    public let config: NemotronTranscribeConfig
    private let weights: TranscribeWeights
    private let speaker: Nemotron3DiarizationModel
    private let frontend: Nemotron3DiarizationRuntime
    private let language: TranscribeLanguage

    public init(root: URL) throws {
        config = try NemotronTranscribeConfig.load(at: root)
        weights = try TranscribeWeights.load(at: root)
        speaker = Nemotron3DiarizationModel()
        let prefix = "perception.encoder.diarization_model."
        let source = Dictionary(uniqueKeysWithValues: weights.arrays.compactMap { key, value in
            key.hasPrefix(prefix) ? (String(key.dropFirst(prefix.count)), value) : nil
        })
        try speaker.update(parameters: ModuleParameters.unflattened(Nemotron3DiarizationModel.compatibleWeights(source)), verify: .all)
        frontend = Nemotron3DiarizationRuntime(model: speaker,
            window: weights["perception.preprocessor.featurizer.window"], filterbank: weights["perception.preprocessor.featurizer.fb"])
        language = TranscribeLanguage(weights: weights, config: config.llmConfig)
    }

    /// NeMo's outer encoder chunking splits waveform rows before STFT; tiny tails join the preceding row.
    public static func audioChunks(sampleCount: Int, chunkSamples: Int = 480_000) -> [Range<Int>] {
        precondition(sampleCount >= 0 && chunkSamples >= 320)
        var ranges = stride(from: 0, to: sampleCount, by: chunkSamples).map { $0..<min($0 + chunkSamples, sampleCount) }
        if ranges.count > 1, ranges[ranges.count - 1].count < 320 {
            let last = ranges.removeLast()
            let previous = ranges.removeLast()
            ranges.append(previous.lowerBound..<last.upperBound)
        }
        return ranges
    }

    public func encodeAudio(_ samples: [Float], progress: (Int, Int) -> Void = { _, _ in }) throws -> MLXArray {
        guard (320...57_600_000).contains(samples.count), samples.allSatisfy(\.isFinite) else {
            throw NemotronTranscribeError.invalidCheckpoint("Transcribe expects finite 16 kHz audio between 20 ms and one hour.")
        }
        let encoder = TranscribeEncoder(weights: weights, prefix: "perception.encoder.asr_encoder",
                                        width: 1280, heads: 10, layers: 32, rotaryDimensions: 64, qkNorm: true)
        var chunks: [MLXArray] = []
        let ranges = Self.audioChunks(sampleCount: samples.count)
        for (index, range) in ranges.enumerated() {
            try Task.checkCancellation()
            let features = frontend.features(samples: Array(samples[range]))
            let validFrames = range.count / 160
            let raw = features[0..., 0..., 0..<validFrames]
            let normalized = NemotronTranscribeFusion.normalizeFeatures(raw.transposed(0, 2, 1))
            let encoded = try encoder(normalized.asType(.bfloat16))
            // Each 30 s row is independent in the released SALM encoder_chunk_size_seconds recipe.
            let embeddings = speaker.preEncode(raw)
            let highResolution = speaker.probabilities(preEncoded: embeddings)
            let speakers = highResolution.reshaped(1, encoded.dim(1), 8, 8).mean(axis: 2)
            let active = (speakers .> config.peEncoderConfig.speakerActivityThreshold).asType(encoded.dtype)
            let asr = weights.layerNorm(encoded, "perception.encoder.asr_norm")
            let diar = weights.layerNorm(active, "perception.encoder.diar_norm")
            let kernel = NemotronTranscribeFusion.speakerKernel(speakers: 8, dimensions: 1280).asType(diar.dtype)
            let fused = asr + config.peEncoderConfig.spkKernelScale * diar.matmul(kernel)
            let projected = weights.linear(fused, "perception.proj", bias: true)
            eval(projected)
            chunks.append(projected)
            progress(index + 1, ranges.count)
        }
        return concatenated(chunks, axis: 1)
    }

    public func inputEmbeddings(_ tokens: [Int]) -> MLXArray {
        language.inputEmbeddings(MLXArray(tokens.map(Int32.init)).reshaped(1, -1))
    }

    public func makeCache() -> NemotronTranscribeCache {
        NemotronTranscribeCache(count: config.llmConfig.layersBlockType.count)
    }

    public func prefill(embeddings: MLXArray, cache: NemotronTranscribeCache) throws -> MLXArray {
        var logits: MLXArray?
        for start in stride(from: 0, to: embeddings.dim(1), by: 128) {
            logits = try language(embeddings: embeddings[0..., start..<min(start + 128, embeddings.dim(1)), 0...], cache: cache)
            eval(logits!)
        }
        return logits!
    }

    public func decode(token: Int, cache: NemotronTranscribeCache) throws -> MLXArray {
        try language(embeddings: inputEmbeddings([token]), cache: cache)
    }
}
