import Foundation

public struct APBWEConfiguration: Codable, Equatable, Sendable {
    public let channels: Int
    public let layers: Int
    public let nFFT: Int
    public let hopSize: Int
    public let winSize: Int
    public let highSampleRate: Int
    public let lowSampleRate: Int
    public let chunkSize: Int
    public let overlap: Int

    enum CodingKeys: String, CodingKey {
        case channels
        case layers
        case nFFT = "n_fft"
        case hopSize = "hop_size"
        case winSize = "win_size"
        case highSampleRate = "high_sample_rate"
        case lowSampleRate = "low_sample_rate"
        case chunkSize = "chunk_size"
        case overlap
    }

    public var frequencyBins: Int { nFFT / 2 + 1 }

    public func validate() throws {
        guard channels == 512, layers == 8 else {
            throw APBWEError.invalidConfiguration(
                "expected 512 channels and 8 ConvNeXt layers"
            )
        }
        guard nFFT == 1_024, hopSize == 80, winSize == 320 else {
            throw APBWEError.invalidConfiguration(
                "expected n_fft=1024, hop_size=80, and win_size=320"
            )
        }
        guard highSampleRate == 48_000, lowSampleRate == 16_000 else {
            throw APBWEError.invalidConfiguration(
                "expected the pinned 16 kHz to 48 kHz profile"
            )
        }
        guard chunkSize > nFFT,
              chunkSize.isMultiple(of: hopSize),
              overlap > 0,
              chunkSize.isMultiple(of: overlap) else {
            throw APBWEError.invalidConfiguration(
                "chunk_size must exceed n_fft and be divisible by hop_size and overlap"
            )
        }
    }
}

public enum APBWEError: Error, Equatable, LocalizedError, Sendable {
    case missingBundledConfiguration
    case invalidConfiguration(String)
    case invalidAudioBuffer
    case unsupportedAudio(sampleRate: Int, channels: Int)
    case invalidCheckpointInventory(tensors: Int, scalars: Int)
    case checkpointKeyMismatch(missing: [String], unexpected: [String])
    case checkpointShapeMismatch(key: String, expected: [Int], actual: [Int])
    case checkpointIdentityChanged

    public var errorDescription: String? {
        switch self {
        case .missingBundledConfiguration:
            "Bundled AP-BWE configuration is missing."
        case .invalidConfiguration(let detail):
            "Invalid AP-BWE configuration: \(detail)."
        case .invalidAudioBuffer:
            "AP-BWE requires a non-empty mono floating-point audio buffer."
        case .unsupportedAudio(let sampleRate, let channels):
            "AP-BWE requires 48 kHz mono narrowband audio; received \(sampleRate) Hz with \(channels) channels."
        case .invalidCheckpointInventory(let tensors, let scalars):
            "AP-BWE checkpoint has \(tensors) tensors/\(scalars) scalars; expected "
                + "\(APBWEResources.expectedTensorCount)/\(APBWEResources.expectedScalarCount)."
        case .checkpointKeyMismatch(let missing, let unexpected):
            "AP-BWE checkpoint keys differ from the native graph; missing=\(missing), unexpected=\(unexpected)."
        case .checkpointShapeMismatch(let key, let expected, let actual):
            "AP-BWE checkpoint tensor '\(key)' has shape \(actual); expected \(expected)."
        case .checkpointIdentityChanged:
            "AP-BWE checkpoint identity changed after preflight verification."
        }
    }
}
