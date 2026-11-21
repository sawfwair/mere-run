import Foundation

/// The final stdout record from speech converse, shared with Studio for replay and history.
package struct HaruConversationSummary: Codable, Sendable {
    package let output: String?
    package let frames: Int
    package let sampleRate: Int
    package let transcript: String

    package init(output: String?, frames: Int, sampleRate: Int, transcript: String) {
        self.output = output
        self.frames = frames
        self.sampleRate = sampleRate
        self.transcript = transcript
    }
}

/// Studio opts into live status on stdout through the child environment. The ordinary CLI
/// continues to print one JSON summary on stdout and live diagnostics on stderr.
package struct HaruConversationEvent: Codable, Sendable {
    package static let environmentKey = "MERERUN_STUDIO_HARU_EVENTS"

    package enum Kind: String, Codable, Sendable {
        case capturing
        case frame
    }

    package let haruEvent: Kind
    package let speaking: Bool
    package let text: String

    package init(_ kind: Kind, speaking: Bool = false, text: String = "") {
        haruEvent = kind
        self.speaking = speaking
        self.text = text
    }
}
