import MereRunContract

package struct StudioHaruLiveOutput: Equatable {
    package private(set) var isCapturing = false
    package private(set) var isSpeaking = false
    package private(set) var transcript = ""

    mutating func apply(_ event: HaruConversationEvent) {
        switch event.haruEvent {
        case .capturing:
            isCapturing = true
        case .frame:
            isSpeaking = event.speaking
            transcript += event.text
        }
    }
}
