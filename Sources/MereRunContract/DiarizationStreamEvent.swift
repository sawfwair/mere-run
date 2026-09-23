import Foundation

/// Versioned JSON Lines contract for incremental speaker activity.
public struct DiarizationStreamEvent: Codable, Equatable, Sendable {
    public enum Kind: String, Codable, Sendable {
        case ready
        case activity
        case final
        case error
    }

    public struct Segment: Codable, Equatable, Sendable {
        public let speaker: String
        public let speakerIndex: Int
        public let startSeconds: Double
        public let endSeconds: Double

        public init(speakerIndex: Int, startSeconds: Double, endSeconds: Double) {
            self.speaker = "speaker_\(speakerIndex)"
            self.speakerIndex = speakerIndex
            self.startSeconds = startSeconds
            self.endSeconds = endSeconds
        }

        enum CodingKeys: String, CodingKey {
            case speaker
            case speakerIndex = "speaker_index"
            case startSeconds = "start_seconds"
            case endSeconds = "end_seconds"
        }
    }

    public let schemaVersion: Int
    public let type: Kind
    public let model: String?
    public let latency: String?
    public let sampleRate: Int?
    public let inputFormat: String?
    public let startSeconds: Double?
    public let endSeconds: Double?
    public let audioSeconds: Double?
    public let speakerCount: Int?
    public let segments: [Segment]?
    public let reason: String?
    public let code: String?
    public let message: String?

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case type, model, latency
        case sampleRate = "sample_rate"
        case inputFormat = "input_format"
        case startSeconds = "start_seconds"
        case endSeconds = "end_seconds"
        case audioSeconds = "audio_seconds"
        case speakerCount = "speaker_count"
        case segments, reason, code, message
    }

    public init(
        type: Kind,
        model: String? = nil,
        latency: String? = nil,
        sampleRate: Int? = nil,
        inputFormat: String? = nil,
        startSeconds: Double? = nil,
        endSeconds: Double? = nil,
        audioSeconds: Double? = nil,
        speakerCount: Int? = nil,
        segments: [Segment]? = nil,
        reason: String? = nil,
        code: String? = nil,
        message: String? = nil
    ) {
        schemaVersion = 1
        self.type = type
        self.model = model
        self.latency = latency
        self.sampleRate = sampleRate
        self.inputFormat = inputFormat
        self.startSeconds = startSeconds
        self.endSeconds = endSeconds
        self.audioSeconds = audioSeconds
        self.speakerCount = speakerCount
        self.segments = segments
        self.reason = reason
        self.code = code
        self.message = message
    }
}
