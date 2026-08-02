import Foundation
import AudioCore
import MereRunCore

public struct Qwen3TTSResources: Sendable, Hashable {
    public static let defaultModelId = "speech-tts-qwen3-nano"
    public static let customVoiceModelId = "speech-tts-qwen3-customvoice"
    public static let sampleRate = 24000
    public static let voiceDesignRepoId = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    public static let customVoiceRepoId = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    public static let defaultRevision = "main"
    public static let supportedModelIds: Set<String> = [defaultModelId, customVoiceModelId]
    public static let snapshotPatterns = [
        "config.json",
        "generation_config.json",
        "merges.txt",
        "model.safetensors",
        "speech_tokenizer/*",
        "tokenizer_config.json",
        "vocab.json",
    ]

    // Special token IDs (from model config)
    public static let ttsBosTokenId = 151672
    public static let ttsEosTokenId = 151673
    public static let ttsPadTokenId = 151671

    // Codec token IDs (within codec vocab 0-3071)
    public static let codecBosId = 2149
    public static let codecEosTokenId = 2150

    // Speech tokenizer configuration (Qwen3TTSTokenizerV2, NOT SNAC)
    // Uses SplitResidualVectorQuantizer with 16 quantizers (1 semantic + 15 acoustic)
    public static let numQuantizers = 16
    public static let codebookSize = 2048
    public static let semanticCodebookSize = 4096

    public var rootURL: URL

    public init(rootURL: URL) {
        self.rootURL = rootURL
    }

    public static func hubFallbackConfig(for modelId: String) -> HubFallbackConfig? {
        switch modelId {
        case customVoiceModelId:
            return HubFallbackConfig(
                repoId: customVoiceRepoId,
                revision: defaultRevision,
                patterns: snapshotPatterns
            )
        case defaultModelId:
            return HubFallbackConfig(
                repoId: voiceDesignRepoId,
                revision: defaultRevision,
                patterns: snapshotPatterns
            )
        default:
            return nil
        }
    }

    // Model weights
    public var modelIndexURL: URL {
        rootURL.appending(path: "model.safetensors.index.json")
    }

    public var modelWeightsURL: URL {
        rootURL.appending(path: "model.safetensors")
    }

    // Speech tokenizer decoder weights (Qwen3TTSTokenizerV2)
    public var speechTokenizerDirURL: URL {
        rootURL.appending(path: "speech_tokenizer")
    }

    public var speakerEncoderDirURL: URL {
        rootURL.appending(path: "speaker_encoder")
    }

    public var speechTokenizerConfigURL: URL {
        speechTokenizerDirURL.appending(path: "config.json")
    }

    public var speechTokenizerWeightsURLs: [URL] {
        return (try? FileManager.default.contentsOfDirectoryResolvingSymlinks(
            at: speechTokenizerDirURL,
            includingPropertiesForKeys: nil
        ))?.filter { $0.pathExtension == "safetensors" } ?? []
    }

    public var speakerEncoderWeightsURLs: [URL] {
        return (try? FileManager.default.contentsOfDirectoryResolvingSymlinks(
            at: speakerEncoderDirURL,
            includingPropertiesForKeys: nil
        ))?.filter { $0.pathExtension == "safetensors" } ?? []
    }

    // Tokenizer - uses BPE format (vocab.json + merges.txt)
    public var tokenizerJSONURL: URL {
        rootURL.appending(path: "tokenizer.json")
    }

    public var vocabJSONURL: URL {
        rootURL.appending(path: "vocab.json")
    }

    public var mergesTxtURL: URL {
        rootURL.appending(path: "merges.txt")
    }

    public var tokenizerConfigURL: URL {
        rootURL.appending(path: "tokenizer_config.json")
    }

    public var addedTokensURL: URL {
        rootURL.appending(path: "added_tokens.json")
    }

    // Model config
    public var configURL: URL {
        rootURL.appending(path: "config.json")
    }

    public var generationConfigURL: URL {
        rootURL.appending(path: "generation_config.json")
    }

    public func validate(fileManager: FileManager = .default) -> [URL] {
        var missing: [URL] = []

        if !fileManager.fileExists(atPath: configURL.path) {
            missing.append(configURL)
        }

        let hasIndex = fileManager.fileExists(atPath: modelIndexURL.path)
        let hasSingle = fileManager.fileExists(atPath: modelWeightsURL.path)
        if !hasIndex && !hasSingle {
            missing.append(modelIndexURL)
        }

        if !fileManager.fileExists(atPath: speechTokenizerConfigURL.path) {
            missing.append(speechTokenizerConfigURL)
        }

        let tokenizerWeights = (try? fileManager.contentsOfDirectoryResolvingSymlinks(
            at: speechTokenizerDirURL,
            includingPropertiesForKeys: nil
        ))?.filter { $0.pathExtension == "safetensors" } ?? []

        if tokenizerWeights.isEmpty {
            missing.append(speechTokenizerDirURL)
        }

        let hasTokenizerJSON = fileManager.fileExists(atPath: tokenizerJSONURL.path)
        let hasVocab = fileManager.fileExists(atPath: vocabJSONURL.path)
        let hasMerges = fileManager.fileExists(atPath: mergesTxtURL.path)
        if !hasTokenizerJSON && !(hasVocab && hasMerges) {
            missing.append(tokenizerJSONURL)
        }

        if !fileManager.fileExists(atPath: tokenizerConfigURL.path) {
            missing.append(tokenizerConfigURL)
        }

        return missing
    }

    public func validate(for voiceMode: TTSVoiceMode, fileManager: FileManager = .default) -> [URL] {
        var missing = validate(fileManager: fileManager)
        if voiceMode == .clone {
            missing.append(contentsOf: validateCloneAssets(fileManager: fileManager))
        }
        return missing
    }

    public func validateCloneAssets(fileManager: FileManager = .default) -> [URL] {
        var missing: [URL] = []

        // Clone path expects tokenizer encoder_config in speech tokenizer config.
        let hasTokenizerEncoderConfig: Bool = {
            guard let data = try? Data(contentsOf: speechTokenizerConfigURL),
                  let config = try? JSONDecoder().decode(Qwen3TTSTokenizerConfig.self, from: data) else {
                return false
            }
            return config.encoderConfig != nil
        }()
        if !hasTokenizerEncoderConfig {
            missing.append(speechTokenizerConfigURL)
        }

        return missing
    }
}

// MARK: - Model Configuration

/// Root config for Qwen3-TTS model
public struct Qwen3TTSConfig: Decodable, Sendable, Hashable {
    public let talkerConfig: TalkerConfig
    public let quantization: QuantizationConfig?

    enum CodingKeys: String, CodingKey {
        case talkerConfig = "talker_config"
        case quantization
    }

    public var vocabSize: Int { talkerConfig.vocabSize }
    public var hiddenSize: Int { talkerConfig.hiddenSize }
    public var numHiddenLayers: Int { talkerConfig.numHiddenLayers }
    public var numAttentionHeads: Int { talkerConfig.numAttentionHeads }
    public var numKeyValueHeads: Int { talkerConfig.numKeyValueHeads }
    public var intermediateSize: Int { talkerConfig.intermediateSize }
    public var maxPositionEmbeddings: Int { talkerConfig.maxPositionEmbeddings }
    public var ropeTheta: Float { talkerConfig.ropeTheta }
    public var rmsNormEps: Float { talkerConfig.rmsNormEps }
    public var headDim: Int { talkerConfig.headDim }
    public var textVocabSize: Int { talkerConfig.textVocabSize }

    public var computedHeadDim: Int { headDim }

    public static func load(from url: URL) throws -> Qwen3TTSConfig {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(Qwen3TTSConfig.self, from: data)
    }

    /// Talker config with the actual model parameters
    public struct TalkerConfig: Decodable, Sendable, Hashable {
        public let vocabSize: Int
        public let hiddenSize: Int
        public let numHiddenLayers: Int
        public let numAttentionHeads: Int
        public let numKeyValueHeads: Int
        public let intermediateSize: Int
        public let maxPositionEmbeddings: Int
        public let ropeTheta: Float
        public let rmsNormEps: Float
        public let headDim: Int
        public let textVocabSize: Int
        public let textHiddenSize: Int

        enum CodingKeys: String, CodingKey {
            case vocabSize = "vocab_size"
            case hiddenSize = "hidden_size"
            case numHiddenLayers = "num_hidden_layers"
            case numAttentionHeads = "num_attention_heads"
            case numKeyValueHeads = "num_key_value_heads"
            case intermediateSize = "intermediate_size"
            case maxPositionEmbeddings = "max_position_embeddings"
            case ropeTheta = "rope_theta"
            case rmsNormEps = "rms_norm_eps"
            case headDim = "head_dim"
            case textVocabSize = "text_vocab_size"
            case textHiddenSize = "text_hidden_size"
        }
    }

    public struct QuantizationConfig: Decodable, Sendable, Hashable {
        public let groupSize: Int
        public let bits: Int

        enum CodingKeys: String, CodingKey {
            case groupSize = "group_size"
            case bits
        }
    }
}
