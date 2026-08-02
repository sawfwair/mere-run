import Foundation

public struct UniverSRConfiguration: Codable, Equatable, Sendable {
    public struct Path: Codable, Equatable, Sendable {
        public let sigmaMinimum: Float

        enum CodingKeys: String, CodingKey {
            case sigmaMinimum = "sigma_min"
        }
    }

    public struct Transform: Codable, Equatable, Sendable {
        public let window: String
        public let nFFT: Int
        public let samplingRate: Int
        public let hopLength: Int
        public let alpha: Float
        public let beta: Float
        public let compressionEpsilon: Float

        enum CodingKeys: String, CodingKey {
            case window
            case nFFT = "n_fft"
            case samplingRate = "sampling_rate"
            case hopLength = "hop_length"
            case alpha
            case beta
            case compressionEpsilon = "compression_epsilon"
        }
    }

    public struct Model: Codable, Equatable, Sendable {
        public let inputChannels: Int
        public let outputChannels: Int
        public let dimensions: [Int]
        public let depths: [Int]
        public let timeDimension: Int
        public let conditioningDimension: Int
        public let totalFrequencyBins: Int
        public let highFrequencyBins: Int
        public let conditioningLayers: Int
        public let inputRateBins: [String: Int]

        enum CodingKeys: String, CodingKey {
            case inputChannels = "input_channels"
            case outputChannels = "output_channels"
            case dimensions
            case depths
            case timeDimension = "time_dimension"
            case conditioningDimension = "conditioning_dimension"
            case totalFrequencyBins = "total_frequency_bins"
            case highFrequencyBins = "high_frequency_bins"
            case conditioningLayers = "conditioning_layers"
            case inputRateBins = "input_rate_bins"
        }

        public func frequencyBins(for inputRateKHz: Int) -> Int? {
            inputRateBins[String(inputRateKHz)]
        }
    }

    public struct Inference: Codable, Equatable, Sendable {
        public let minimumSamples: Int
        public let odeMethod: String
        public let odeSteps: Int
        public let guidanceScale: Float
        public let chunkSeconds: Int

        enum CodingKeys: String, CodingKey {
            case minimumSamples = "minimum_samples"
            case odeMethod = "ode_method"
            case odeSteps = "ode_steps"
            case guidanceScale = "guidance_scale"
            case chunkSeconds = "chunk_seconds"
        }
    }

    public let seed: UInt64
    public let path: Path
    public let transform: Transform
    public let model: Model
    public let inference: Inference

    public var supportedInputRates: [Int] {
        model.inputRateBins.keys.compactMap(Int.init).map { $0 * 1_000 }.sorted()
    }

    public func validate() throws {
        guard path.sigmaMinimum == 1e-4 else {
            throw UniverSRError.invalidConfiguration("expected sigma_min=1e-4")
        }
        guard transform.window == "hann",
              transform.nFFT == 1_024,
              transform.samplingRate == 48_000,
              transform.hopLength == 512,
              transform.alpha == 0.2,
              transform.beta == 1,
              transform.compressionEpsilon == 1e-4 else {
            throw UniverSRError.invalidConfiguration("the complex STFT profile differs from the pinned checkpoint")
        }
        guard model.inputChannels == 2,
              model.outputChannels == 2,
              model.dimensions == [96, 192, 384, 768],
              model.depths == [2, 2, 4, 2],
              model.timeDimension == 256,
              model.conditioningDimension == 384,
              model.totalFrequencyBins == 512,
              model.highFrequencyBins == 432,
              model.conditioningLayers == 4,
              model.inputRateBins == ["8": 80, "12": 128, "16": 170, "24": 256] else {
            throw UniverSRError.invalidConfiguration("the ConvNeXt U-Net graph differs from the pinned checkpoint")
        }
        guard inference.minimumSamples == 32_768,
              inference.odeMethod == "midpoint",
              inference.odeSteps == 4,
              inference.guidanceScale == 1.5,
              inference.chunkSeconds >= 3 else {
            throw UniverSRError.invalidConfiguration("the inference defaults differ from the published profile")
        }
    }
}

public enum UniverSRODEMethod: String, CaseIterable, Codable, Sendable {
    case euler
    case midpoint
    case rk4
}

public enum UniverSRError: Error, Equatable, LocalizedError, Sendable {
    case missingBundledConfiguration
    case invalidConfiguration(String)
    case invalidAudioBuffer
    case unsupportedInputRate(Int)
    case invalidInferenceOption(String)
    case invalidCheckpointInventory(tensors: Int, scalars: Int)
    case checkpointKeyMismatch(missing: [String], unexpected: [String])
    case checkpointShapeMismatch(key: String, expected: [Int], actual: [Int])
    case nonFiniteOutput

    public var errorDescription: String? {
        switch self {
        case .missingBundledConfiguration:
            "Bundled UniverSR configuration is missing."
        case .invalidConfiguration(let detail):
            "Invalid UniverSR configuration: \(detail)."
        case .invalidAudioBuffer:
            "UniverSR requires a non-empty mono floating-point audio buffer."
        case .unsupportedInputRate(let rate):
            "UniverSR supports effective input rates of 8000, 12000, 16000, or 24000 Hz; received \(rate) Hz."
        case .invalidInferenceOption(let detail):
            "Invalid UniverSR inference option: \(detail)."
        case .invalidCheckpointInventory(let tensors, let scalars):
            "UniverSR checkpoint has \(tensors) tensors/\(scalars) scalars; expected "
                + "\(UniverSRResources.expectedTensorCount)/\(UniverSRResources.expectedScalarCount)."
        case .checkpointKeyMismatch(let missing, let unexpected):
            "UniverSR checkpoint keys differ from the native graph; missing=\(missing), unexpected=\(unexpected)."
        case .checkpointShapeMismatch(let key, let expected, let actual):
            "UniverSR checkpoint tensor '\(key)' has shape \(actual); expected \(expected)."
        case .nonFiniteOutput:
            "UniverSR produced non-finite audio samples."
        }
    }
}
