import Foundation

/// The final stdout record from speech converse, shared with Studio for replay and history.
package struct HaruConversationSummary: Codable, Sendable {
    package let output: String?
    package let frames: Int
    package let sampleRate: Int
    package let transcript: String
    package let segments: [HaruTranscriptSegment]?
    package let inputTranscriptionError: String?

    package init(
        output: String?, frames: Int, sampleRate: Int, transcript: String,
        segments: [HaruTranscriptSegment]? = nil, inputTranscriptionError: String? = nil
    ) {
        self.output = output
        self.frames = frames
        self.sampleRate = sampleRate
        self.transcript = transcript
        self.segments = segments
        self.inputTranscriptionError = inputTranscriptionError
    }
}

package struct HaruTranscriptSegment: Codable, Equatable, Sendable {
    package enum Speaker: String, Codable, Sendable {
        case user
        case haru

        package var title: String { self == .user ? "You" : "Haru" }
    }

    package let speaker: Speaker
    package let startMs: Int
    package var text: String

    package init(speaker: Speaker, startMs: Int, text: String) {
        self.speaker = speaker
        self.startMs = startMs
        self.text = text
    }

    package static func chronological(_ segments: [Self]) -> [Self] {
        segments.sorted { $0.startMs < $1.startMs }
    }
}

/// Groups reply tokens into speech spans. A playback transition to quiet closes the span
/// after any text from that frame has been appended.
package struct HaruReplyTranscript: Equatable, Sendable {
    package private(set) var segments: [HaruTranscriptSegment] = []
    private var startsNewSegment = true
    private var wasSpeaking = false

    package init() {}

    package mutating func append(_ text: String, at startMs: Int, speaking: Bool) {
        if !text.isEmpty {
            if startsNewSegment {
                segments.append(.init(speaker: .haru, startMs: startMs, text: text))
                startsNewSegment = false
            } else {
                segments[segments.count - 1].text += text
            }
        }
        if wasSpeaking, !speaking { startsNewSegment = true }
        wasSpeaking = speaking
    }
}

/// Studio opts into live events and microphone transcription through the child environment.
/// An ordinary invocation prints one JSON summary on stdout.
package struct HaruConversationEvent: Codable, Sendable {
    package static let environmentKey = "MERERUN_STUDIO_HARU_EVENTS"
    package static let inputTranscriptionEnvironmentKey = "MERERUN_STUDIO_HARU_INPUT_TRANSCRIPT"
    package static let inputTranscriptionModelID = "speech-asr-parakeet"

    package enum Kind: String, Codable, Sendable {
        case capturing
        case frame
        case inputPartial
        case inputCommit
        case inputUnavailable
    }

    package let haruEvent: Kind
    package let speaking: Bool
    package let text: String
    package let startMs: Int?

    package init(_ kind: Kind, speaking: Bool = false, text: String = "", startMs: Int? = nil) {
        haruEvent = kind
        self.speaking = speaking
        self.text = text
        self.startMs = startMs
    }
}
