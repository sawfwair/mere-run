import Foundation

public struct RoFormerConfiguration: Codable, Equatable, Sendable {
    public let modelID: String
    public let repository: String
    public let revision: String
    public let license: String
    public let sampleRate: Int
    public let audioChannels: Int
    public let chunkSize: Int
    public let overlap: Int
    public let dim: Int
    public let depth: Int
    public let numStems: Int
    public let timeTransformerDepth: Int
    public let frequencyTransformerDepth: Int
    public let heads: Int
    public let dimHead: Int
    public let dimensionFrequencies: Int
    public let stftNFFT: Int
    public let stftHopLength: Int
    public let stftWindowLength: Int
    public let stftNormalized: Bool
    public let zeroDC: Bool
    public let maskEstimatorDepth: Int
    public let mlpExpansionFactor: Int
    public let frequenciesPerBand: [Int]

    public var frequencyBandInputDimensions: [Int] {
        frequenciesPerBand.map { $0 * audioChannels * 2 }
    }

    public func validate() throws {
        guard modelID == RoFormerResources.modelID else {
            throw RoFormerError.invalidConfiguration("unexpected model id \(modelID)")
        }
        guard repository == RoFormerResources.repository, revision == RoFormerResources.revision else {
            throw RoFormerError.invalidConfiguration("source repository or revision is not pinned")
        }
        guard license == "MIT" else {
            throw RoFormerError.invalidConfiguration("expected MIT model license")
        }
        guard sampleRate == 44_100, audioChannels == 2, numStems == 1 else {
            throw RoFormerError.invalidConfiguration("ViperX requires 44.1 kHz stereo audio and one target stem")
        }
        guard dim == heads * dimHead else {
            throw RoFormerError.invalidConfiguration("model dimension must equal heads times head dimension")
        }
        guard frequenciesPerBand.count == 62,
              frequenciesPerBand.reduce(0, +) == dimensionFrequencies,
              dimensionFrequencies == stftNFFT / 2 + 1 else {
            throw RoFormerError.invalidConfiguration("frequency bands do not cover the STFT bins")
        }
        guard chunkSize > 0, overlap > 0, chunkSize.isMultiple(of: overlap) else {
            throw RoFormerError.invalidConfiguration("chunk size must be divisible by overlap")
        }
        guard stftWindowLength == stftNFFT, stftHopLength > 0 else {
            throw RoFormerError.invalidConfiguration("unsupported STFT geometry")
        }
    }

    private enum CodingKeys: String, CodingKey {
        case modelID = "model_id"
        case repository
        case revision
        case license
        case sampleRate = "sample_rate"
        case audioChannels = "audio_channels"
        case chunkSize = "chunk_size"
        case overlap
        case dim
        case depth
        case numStems = "num_stems"
        case timeTransformerDepth = "time_transformer_depth"
        case frequencyTransformerDepth = "frequency_transformer_depth"
        case heads
        case dimHead = "dim_head"
        case dimensionFrequencies = "dimension_frequencies"
        case stftNFFT = "stft_n_fft"
        case stftHopLength = "stft_hop_length"
        case stftWindowLength = "stft_window_length"
        case stftNormalized = "stft_normalized"
        case zeroDC = "zero_dc"
        case maskEstimatorDepth = "mask_estimator_depth"
        case mlpExpansionFactor = "mlp_expansion_factor"
        case frequenciesPerBand = "frequencies_per_band"
    }
}

public enum RoFormerError: Error, Equatable, LocalizedError, Sendable {
    case missingBundledConfiguration
    case invalidConfiguration(String)
    case invalidCheckpointInventory(expected: Int, actual: Int)
    case checkpointKeyMismatch(missing: [String], unexpected: [String])
    case checkpointShapeMismatch(key: String, expected: [Int], actual: [Int])
    case unsupportedAudio(sampleRate: Int, channels: Int)
    case invalidOverlap(Int)
    case invalidAudioBuffer

    public var errorDescription: String? {
        switch self {
        case .missingBundledConfiguration:
            "Bundled ViperX RoFormer configuration is missing."
        case .invalidConfiguration(let detail):
            "Invalid ViperX RoFormer configuration: \(detail)."
        case .invalidCheckpointInventory(let expected, let actual):
            "ViperX checkpoint contains \(actual) tensors; expected \(expected)."
        case .checkpointKeyMismatch(let missing, let unexpected):
            "ViperX checkpoint keys do not match the native model (missing: \(missing.joined(separator: ", ")); unexpected: \(unexpected.joined(separator: ", ")))."
        case .checkpointShapeMismatch(let key, let expected, let actual):
            "ViperX checkpoint tensor \(key) has shape \(actual); expected \(expected)."
        case .unsupportedAudio(let sampleRate, let channels):
            "ViperX separation requires 44.1 kHz stereo audio; received \(sampleRate) Hz with \(channels) channels."
        case .invalidOverlap(let overlap):
            "RoFormer overlap must be a positive divisor of 352800; received \(overlap)."
        case .invalidAudioBuffer:
            "RoFormer input must contain complete interleaved stereo frames."
        }
    }
}
