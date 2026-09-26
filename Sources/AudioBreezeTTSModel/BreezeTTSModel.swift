// Adapted from Blaizzy/mlx-audio-swift at 01dec7c9bdce3088a6b6b7ab9f2e403458195efb.
// Copyright (c) 2025 Prince Canuma. Licensed under MIT; see THIRD_PARTY_NOTICES.md.
import AudioQwen3TTSModel
import Foundation
import MLX
import MLXNN
import MereRunDecode
import MereRunTensor
@preconcurrency import Tokenizers

public struct BreezeGenerationParameters: Sendable {
    public let maxFrames: Int
    public let temperature: Float
    public let topK: Int
    public let topP: Float
    public let cfgScale: Float

    public init(maxFrames: Int = 750, temperature: Float = 0.9, topK: Int = 50,
                topP: Float = 1, cfgScale: Float = 4) {
        self.maxFrames = maxFrames
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.cfgScale = cfgScale
    }
}

public enum BreezeTTSError: LocalizedError {
    case missingAsset(String)
    case invalidRequest(String)

    public var errorDescription: String? {
        switch self {
        case .missingAsset(let asset): "Breeze TTS asset missing: \(asset)"
        case .invalidRequest(let reason): reason
        }
    }
}

/// Native MLX text encoder, acoustic backbone, depth decoder, and Qwen speech codec.
public final class BreezeTTSModel: Module, @unchecked Sendable {
    let config: BreezeTTSConfig
    @ModuleInfo(key: "lm_head") var lmHead: Linear
    @ModuleInfo(key: "embed_text_tokens") var embedTextTokens: Embedding
    @ModuleInfo(key: "backbone_model") var backboneModel: BreezeBackbone
    @ModuleInfo(key: "depth_decoder") var depthDecoder: BreezeDepthDecoder
    @ModuleInfo(key: "text_encoder") var textEncoder: BreezeTTSTextEncoder
    @ModuleInfo(key: "text_encoder_proj") var textEncoderProjection: Linear

    private var tokenizer: (any Tokenizers.Tokenizer)?
    private var audioTokenizer: Qwen3TTSSpeechTokenizer?
    public var sampleRate: Int { config.sampleRate }

    init(config: BreezeTTSConfig) {
        self.config = config
        _lmHead.wrappedValue = Linear(config.backboneConfig.hiddenSize, config.audioVocabSize + 1, bias: false)
        _embedTextTokens.wrappedValue = Embedding(
            embeddingCount: config.textVocabSize, dimensions: config.backboneConfig.hiddenSize
        )
        _backboneModel.wrappedValue = BreezeBackbone(config: config)
        _depthDecoder.wrappedValue = BreezeDepthDecoder(config: config.depthDecoderConfig)
        _textEncoder.wrappedValue = BreezeTTSTextEncoder(config: config.textEncoderConfig)
        _textEncoderProjection.wrappedValue = Linear(
            config.textEncoderConfig.hiddenSize, config.backboneConfig.hiddenSize, bias: false
        )
    }

    public static func promptText(text: String, instruction: String?) -> String {
        guard let instruction, !instruction.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return "[S0]\(text)"
        }
        return "[S0]<ins_bos>\(instruction)<ins_eos>\(text)"
    }

    public func generate(
        text: String,
        instruction: String?,
        referenceAudio: MLXArray? = nil,
        referenceTranscript: String? = nil,
        parameters: BreezeGenerationParameters = .init(),
        onToken: ((Int) -> Void)? = nil,
        chunkFrameInterval: Int? = nil,
        onAudioChunk: (([Float]) -> Void)? = nil
    ) throws -> MLXArray {
        guard let tokenizer, let audioTokenizer else { throw BreezeTTSError.missingAsset("tokenizers") }
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw BreezeTTSError.invalidRequest("Speech text is empty.")
        }
        guard parameters.maxFrames > 0 else { throw BreezeTTSError.invalidRequest("Maximum frames must be positive.") }
        if let chunkFrameInterval, chunkFrameInterval <= 0 {
            throw BreezeTTSError.invalidRequest("Streaming chunk interval must be positive.")
        }
        guard referenceAudio == nil || referenceTranscript?.isEmpty == false else {
            throw BreezeTTSError.invalidRequest("Voice cloning requires the exact reference transcript.")
        }
        let conditionalPrompt = try promptEmbeddings(
            text: text, instruction: instruction, referenceAudio: referenceAudio,
            referenceTranscript: referenceTranscript, tokenizer: tokenizer, audioTokenizer: audioTokenizer
        )
        let guidanceEnabled = instruction?.isEmpty == false && parameters.cfgScale != 1
        let unconditionalPrompt = guidanceEnabled ? try promptEmbeddings(
            text: text, instruction: nil, referenceAudio: referenceAudio,
            referenceTranscript: referenceTranscript, tokenizer: tokenizer, audioTokenizer: audioTokenizer
        ) : nil

        let conditionalCache = backboneModel.makeCache()
        var conditionalHidden = backboneModel(inputEmbeddings: conditionalPrompt, cache: conditionalCache)[0..., -1, 0...]
        let unconditionalCache = unconditionalPrompt.map { _ in backboneModel.makeCache() }
        var unconditionalHidden: MLXArray?
        if let unconditionalPrompt, let unconditionalCache {
            unconditionalHidden = backboneModel(inputEmbeddings: unconditionalPrompt, cache: unconditionalCache)[0..., -1, 0...]
        }
        eval(conditionalHidden)
        if let unconditionalHidden { eval(unconditionalHidden) }

        let firstSampling = GenerationConfig(
            maxTokens: parameters.maxFrames, temperature: parameters.temperature,
            topK: parameters.topK, topP: parameters.topP, repetitionPenalty: 1,
            bannedTokens: Array(config.codecVocabSize..<config.audioVocabSize)
        )
        let depthSampling = GenerationConfig(
            maxTokens: config.numCodebooks, temperature: parameters.temperature,
            topK: parameters.topK, topP: parameters.topP, repetitionPenalty: 1,
            bannedTokens: Array(config.codecVocabSize..<config.audioVocabSize)
        )
        var frames = [[Int32]]()
        var streamedFrames = 0
        for _ in 0..<parameters.maxFrames {
            try Task.checkCancellation()
            var logits = lmHead(conditionalHidden)
            if let unconditionalHidden {
                let negative = lmHead(unconditionalHidden)
                logits = negative + parameters.cfgScale * (logits - negative)
            }
            let first = sampleToken(logits: logits.squeezed(), config: firstSampling,
                                    previousTokens: frames.map { Int($0[0]) })
            if first == config.audioVocabSize { break }
            onToken?(first)
            var frame = [Int32(first)]
            var depthInputs = [Int32(0), Int32(first)]
            while frame.count < config.numCodebooks {
                let ids = MLXArray(depthInputs).reshaped(1, depthInputs.count)
                var depthLogits = depthDecoder.nextLogits(tokenIDs: ids, backboneHiddenState: conditionalHidden)
                if let unconditionalHidden {
                    let negative = depthDecoder.nextLogits(tokenIDs: ids, backboneHiddenState: unconditionalHidden)
                    depthLogits = negative + parameters.cfgScale * (depthLogits - negative)
                }
                let next = sampleToken(logits: depthLogits.squeezed(), config: depthSampling, previousTokens: [])
                frame.append(Int32(next))
                depthInputs.append(Int32(next))
            }
            frames.append(frame)
            if let chunkFrameInterval, frames.count.isMultiple(of: chunkFrameInterval) {
                emitAudioChunk(frames: frames, from: streamedFrames, tokenizer: audioTokenizer, onAudioChunk: onAudioChunk)
                streamedFrames = frames.count
            }
            let codes = MLXArray(frame).reshaped(1, 1, config.numCodebooks)
            conditionalHidden = backboneModel(inputIDs: codes, cache: conditionalCache)[0..., -1, 0...]
            if let unconditionalCache {
                unconditionalHidden = backboneModel(inputIDs: codes, cache: unconditionalCache)[0..., -1, 0...]
            }
            eval(conditionalHidden)
            if let unconditionalHidden { eval(unconditionalHidden) }
        }
        guard !frames.isEmpty else { return MLXArray.zeros([0]) }
        if chunkFrameInterval != nil && streamedFrames < frames.count {
            emitAudioChunk(frames: frames, from: streamedFrames, tokenizer: audioTokenizer, onAudioChunk: onAudioChunk)
        }
        let codes = MLXArray(frames.flatMap { $0 }).reshaped(1, frames.count, config.numCodebooks)
        let decoded = audioTokenizer.decode(codes)
        let audio = decoded.audio.squeezed()
        let valid = decoded.lengths.asArray(Int32.self).first.map(Int.init) ?? audio.dim(0)
        let trimmed = valid < audio.dim(0) ? audio[..<valid] : audio
        eval(trimmed)
        return trimmed
    }

    private func emitAudioChunk(
        frames: [[Int32]], from firstNewFrame: Int,
        tokenizer: Qwen3TTSSpeechTokenizer, onAudioChunk: (([Float]) -> Void)?
    ) {
        guard let onAudioChunk else { return }
        let contextStart = max(0, firstNewFrame - 25)
        let chunkFrames = frames[contextStart...]
        let codes = MLXArray(chunkFrames.flatMap { $0 }).reshaped(1, chunkFrames.count, config.numCodebooks)
        let decoded = tokenizer.decode(codes)
        let audio = decoded.audio.squeezed()
        let start = (firstNewFrame - contextStart) * tokenizer.decodeUpsampleRate
        let valid = min(Int(decoded.lengths.asArray(Int32.self)[0]), audio.dim(0))
        if start < valid {
            let chunk = audio[start..<valid]
            eval(chunk)
            onAudioChunk(chunk.asArray(Float.self))
        }
    }

    private func promptEmbeddings(
        text: String, instruction: String?, referenceAudio: MLXArray?,
        referenceTranscript: String?, tokenizer: any Tokenizers.Tokenizer,
        audioTokenizer: Qwen3TTSSpeechTokenizer
    ) throws -> MLXArray {
        var segments = [MLXArray]()
        if let referenceAudio, let referenceTranscript {
            segments.append(textEmbeddings("[S0]\(referenceTranscript)", tokenizer: tokenizer))
            let input: MLXArray
            switch referenceAudio.ndim {
            case 1: input = referenceAudio.reshaped(1, 1, referenceAudio.dim(0))
            case 2: input = referenceAudio.expandedDimensions(axis: 1)
            case 3: input = referenceAudio
            default: throw BreezeTTSError.invalidRequest("Reference audio must be mono PCM.")
            }
            guard let encoder = audioTokenizer.encoderModel else {
                throw BreezeTTSError.invalidRequest("Reference audio requires the speech tokenizer encoder.")
            }
            let codes = encoder.encode(input).transposed(0, 2, 1)
            segments.append(backboneModel.embedTokens(codes))
            let eos = MLXArray([Int32](repeating: Int32(config.codebookEOSTokenID),
                                       count: config.numCodebooks)).reshaped(1, 1, config.numCodebooks)
            segments.append(backboneModel.embedTokens(eos))
        }
        segments.append(textEmbeddings(Self.promptText(text: text, instruction: instruction), tokenizer: tokenizer))
        return MLX.concatenated(segments, axis: 1)
    }

    private func textEmbeddings(_ text: String, tokenizer: any Tokenizers.Tokenizer) -> MLXArray {
        let ids = MLXArray(tokenizer.encode(text: text).map(Int32.init)).reshaped(1, -1)
        return textEncoderProjection(textEncoder(ids))
    }

    public static func fromModelDirectory(_ directory: URL) async throws -> BreezeTTSModel {
        let configURL = directory.appending(path: "config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            throw BreezeTTSError.missingAsset(configURL.path)
        }
        let config = try JSONDecoder().decode(BreezeTTSConfig.self, from: Data(contentsOf: configURL))
        guard config.modelType == "breeze" else {
            throw BreezeTTSError.invalidRequest("Expected a Breeze TTS checkpoint.")
        }
        let model = BreezeTTSModel(config: config)
        let indexURL = directory.appending(path: "model.safetensors.index.json")
        guard FileManager.default.fileExists(atPath: indexURL.path) else {
            throw BreezeTTSError.missingAsset(indexURL.path)
        }
        let index = try JSONDecoder().decode(HFSafetensorsIndex.self, from: Data(contentsOf: indexURL))
        var checkpointKeys = Set(index.weightMap.keys.filter {
            !$0.hasPrefix("codec_model.") && !$0.hasSuffix(".initialized")
        })
        if checkpointKeys.contains("depth_decoder.model.embed_tokens.weight") {
            checkpointKeys.insert("backbone_model.embed_tokens.embed_audio_tokens.weight")
        }
        let modelKeys = Set(model.parameters().flattened().map(\.0))
        let missing = modelKeys.subtracting(checkpointKeys)
        let unknown = checkpointKeys.subtracting(modelKeys)
        guard missing.isEmpty && unknown.isEmpty else {
            throw BreezeTTSError.invalidRequest(
                "Breeze checkpoint and native graph disagree: missing \(missing.sorted().prefix(3)), unknown \(unknown.sorted().prefix(3))."
            )
        }
        try HFSafetensorsWeightsLoader.applyShardedWeights(indexURL: indexURL, to: model, dtype: .bfloat16, mapper: {
            key, value in
            if key.hasPrefix("codec_model.") || key.hasSuffix(".initialized") { return [] }
            if key == "depth_decoder.model.embed_tokens.weight" {
                return [(key, value), ("backbone_model.embed_tokens.embed_audio_tokens.weight", value)]
            }
            return [(key, value)]
        })
        model.tokenizer = try await AutoTokenizer.from(modelFolder: directory)

        let audioDirectory = directory.appending(path: "audio_tokenizer", directoryHint: .isDirectory)
        let audioConfigURL = audioDirectory.appending(path: "config.json")
        let audioWeightsURL = audioDirectory.appending(path: "model.safetensors")
        guard FileManager.default.fileExists(atPath: audioConfigURL.path),
              FileManager.default.fileExists(atPath: audioWeightsURL.path) else {
            throw BreezeTTSError.missingAsset(audioDirectory.path)
        }
        let audioConfig = try JSONDecoder().decode(Qwen3TTSTokenizerConfig.self, from: Data(contentsOf: audioConfigURL))
        let audioTokenizer = Qwen3TTSSpeechTokenizer(config: audioConfig)
        let audioWeights = try MLX.loadArrays(url: audioWeightsURL)
        let sanitized = Qwen3TTSSpeechTokenizer.sanitize(audioWeights, config: audioConfig)
        try audioTokenizer.update(parameters: ModuleParameters.unflattened(sanitized.map { ($0.key, $0.value) }),
                                  verify: .none)
        model.audioTokenizer = audioTokenizer
        return model
    }
}
