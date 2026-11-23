import MereRunContract

package struct StudioHaruLiveOutput: Equatable {
    package private(set) var isCapturing = false
    package private(set) var isSpeaking = false
    package private(set) var transcript = ""
    package private(set) var inputTranscriptionError: String?
    package private(set) var partialInput: HaruTranscriptSegment?
    private var userSegments: [HaruTranscriptSegment] = []
    private var reply = HaruReplyTranscript()

    package var segments: [HaruTranscriptSegment] {
        HaruTranscriptSegment.chronological(userSegments + reply.segments)
    }

    mutating func apply(_ event: HaruConversationEvent) {
        switch event.haruEvent {
        case .capturing:
            isCapturing = true
        case .frame:
            isSpeaking = event.speaking
            transcript += event.text
            if let startMs = event.startMs {
                reply.append(event.text, at: startMs, speaking: event.speaking)
            }
        case .inputPartial:
            guard let startMs = event.startMs else { return }
            partialInput = .init(speaker: .user, startMs: startMs, text: event.text)
        case .inputCommit:
            partialInput = nil
            if let startMs = event.startMs, !event.text.isEmpty {
                userSegments.append(.init(speaker: .user, startMs: startMs, text: event.text))
            }
        case .inputUnavailable:
            partialInput = nil
            inputTranscriptionError = event.text
        }
    }
}
