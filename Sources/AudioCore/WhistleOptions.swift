import Foundation

/// Controls for the native Whistle decoder. Other ASR families do not consume these options.
public struct WhistleOptions: Sendable, Hashable, Codable {
    public enum Weights: String, Sendable, Hashable, Codable { case cactus, fp32 }
    public var weights: Weights
    public var beamSize: Int
    public var decoderDepth: Int
    public var keywords: [String]
    public var wordTimestamps: Bool

    public init(weights: Weights = .cactus, beamSize: Int = 5, decoderDepth: Int = 8,
                keywords: [String] = [], wordTimestamps: Bool = true) {
        self.weights = weights
        self.beamSize = beamSize
        self.decoderDepth = decoderDepth
        self.keywords = keywords
        self.wordTimestamps = wordTimestamps
    }

    public func validate() throws {
        guard (1...8).contains(beamSize), (2...8).contains(decoderDepth) else {
            throw SpeechTranscriptionIssue("invalid_whistle_options", "Whistle beam size must be 1...8 and decoder depth 2...8.")
        }
        guard keywords.count <= 64, keywords.allSatisfy({ !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && $0.unicodeScalars.count <= 128 }) else {
            throw SpeechTranscriptionIssue("invalid_whistle_keywords", "Use at most 64 nonempty keywords or phrases, each at most 128 characters.")
        }
    }
}
