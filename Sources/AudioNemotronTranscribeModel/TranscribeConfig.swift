import Foundation

/// Inference fields from NVIDIA's NeMo SALM configuration. Training-only fields are ignored.
public struct NemotronTranscribeConfig: Decodable, Equatable, Sendable {
    public static let repository = "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B"
    public static let revision = "ef905575a106fdad904f3b0ce661dccad654607d"
    public static let modelID = "speech-asr-nemotron-35-transcribe"
    public static let weightBytes: Int64 = 67_496_013_734
    public static let maximumContext = 65_536

    public let modelType: String
    let dtype: String
    let promptFormat: String
    let spkKernelScale: Float
    let architectures: [String]
    let audioLocatorTag: String
    let encoderChunkSizeSeconds: Double
    let perception: Perception
    let peEncoderConfig: ParallelEncoder
    let llmConfig: Language
    public let eosTokenId: [Int]

    struct Frontend: Decodable, Equatable, Sendable {
        let sampleRate: Int
        let features: Int
        let nFft: Int
        let windowSize: Double
        let windowStride: Double
        let window: String
        let normalize: String?
        let padTo: Int
        let padValue: Float
    }

    struct Transformer: Decodable, Equatable, Sendable {
        let featIn: Int
        let dModel: Int
        let nHeads: Int
        let nLayers: Int
        let qkvBias: Bool
        let qkNorm: Bool
        let ffExpansion: Float
        let preBlockNorm: Bool
        let subsamplingFactor: Int
        let attnMode: String
        let selfAttentionModel: String
        let ropeBase: Float?
        let rotaryFraction: Float?
    }

    struct Perception: Decodable, Equatable, Sendable {
        let outputDim: Int
        let preprocessor: Frontend
        let encoder: Transformer
        let modalityAdapter: Connector
        struct Connector: Decodable, Equatable, Sendable { let dModel: Int }
    }

    struct ParallelEncoder: Decodable, Equatable, Sendable {
        let asrEncoderType: String
        let asrEncoderCfg: Transformer
        let asrNormalizeType: String
        let diarNormalizeType: String
        let speakerFeatureConfigVersion: Int
        let speakerFeatureMode: String
        let speakerActivityThreshold: Float
        let spkKernelScale: Float
        let chunkSizeSeconds: Double
        let diarizationModelCfg: Diarization
        struct Diarization: Decodable, Equatable, Sendable {
            let sampleRate: Int
            let maxNumOfSpks: Int
            let highResolution: Bool
            let outputSubsamplingFactor: Int
            let encoder: Transformer
            let modelDefaults: Dimensions
            struct Dimensions: Decodable, Equatable, Sendable { let fcDModel: Int; let tfDModel: Int }
        }
    }

    struct Language: Decodable, Equatable, Sendable {
        let modelType: String
        let vocabSize: Int
        let hiddenSize: Int
        let layersBlockType: [String]
        let numAttentionHeads: Int
        let numKeyValueHeads: Int
        let headDim: Int
        let normEps: Float
        let mambaHeadDim: Int
        let mambaNumHeads: Int
        let ssmStateSize: Int
        let nGroups: Int
        let convKernel: Int
        let timeStepMin: Float
        let nRoutedExperts: Int
        let numExpertsPerTok: Int
        let moeIntermediateSize: Int
        let moeSharedExpertIntermediateSize: Int
        let routedScalingFactor: Float
        let normTopkProb: Bool
        let nGroup: Int
        let topkGroup: Int
        let mlpHiddenAct: String
        let timeStepLimit: [TimeLimit]
        let rescalePrenormResidual: Bool

        enum TimeLimit: Decodable, Equatable, Sendable {
            case finite(Float)
            case infinity
            private struct Infinity: Decodable { let value: String; enum CodingKeys: String, CodingKey { case value = "__float__" } }
            init(from decoder: Decoder) throws {
                let container = try decoder.singleValueContainer()
                if let value = try? container.decode(Float.self) { self = .finite(value) }
                else {
                    let value = try container.decode(Infinity.self)
                    guard value.value == "Infinity" else {
                        throw DecodingError.dataCorruptedError(in: container, debugDescription: "Expected positive infinite time-step limit.")
                    }
                    self = .infinity
                }
            }
        }
    }

    public static func load(at root: URL) throws -> Self {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let config = try decoder.decode(Self.self, from: Data(contentsOf: root.appendingPathComponent("config.json")))
        let expected = try decoder.decode(Self.self, from: Data(contentsOf: Bundle.module.url(forResource: "config", withExtension: "json")!))
        guard config == expected else {
            throw NemotronTranscribeError.invalidCheckpoint("Unsupported Transcribe architecture or precision configuration.")
        }
        return config
    }
}

public enum NemotronTranscribeError: LocalizedError {
    case invalidCheckpoint(String)
    public var errorDescription: String? {
        switch self { case .invalidCheckpoint(let message): message }
    }
}
