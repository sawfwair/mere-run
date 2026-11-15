import Foundation
import AudioSOVTranscribeModel

/// Synchronized resident timings. Decode includes prompt assembly, prefill, and generation.
public struct SOVMeasuredTranscription: Codable, Sendable {
    public let text: String
    public let tokens: [Int]
    public let featureSeconds: Double
    public let encoderProjectorSeconds: Double
    public let decodeSeconds: Double
    public let prefillSeconds: Double
    public let generationSeconds: Double
    public let cleanupSeconds: Double
    public let totalSeconds: Double
    public let verificationRounds: Int
    public let acceptedDraftTokens: Int
    public let rejectedRounds: Int
    public let reusedVerificationRounds: Int?
    /// Raw agreeing prefixes, including any suffix after EOS. Kept separate from emitted drafts.
    public let acceptedLengths: [Int]?
    public let emittedDraftTokens: Int?
    public let proposedLengths: [Int]?
    public let fullAudioTokens: Int?
    public let audioTokens: Int?
    public let promptTokens: Int?
    public let audioTokenMargin: Int?
    public let decodeRounds: [SOVDecodeRoundTiming]?
}
