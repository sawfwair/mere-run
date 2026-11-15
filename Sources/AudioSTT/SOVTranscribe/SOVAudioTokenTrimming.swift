import AudioCore

/// Research policy: retain every complete window and trim only the final projected tail.
/// The released processor includes the full padded window, so this changes model input.
enum SOVAudioTokenTrimming {
    static func validate(margin: Int?) throws {
        guard margin.map({ (0...375).contains($0) }) ?? true else {
            throw SpeechTranscriptionIssue("invalid_audio_token_margin", "Audio token margin must be between 0 and 375; omit it to retain complete windows.")
        }
    }

    static func retainedCount(sampleCount: Int, tokensPerWindow: Int, margin: Int?) throws -> Int {
        let windows = try SOVTranscribeGenerator.windowCount(sampleCount: sampleCount)
        try validate(margin: margin)
        guard let margin else { return windows * tokensPerWindow }
        let finalSamples = (sampleCount - 1) % 480_000 + 1
        let coveredTokens = (finalSamples * tokensPerWindow + 479_999) / 480_000
        return (windows - 1) * tokensPerWindow + min(tokensPerWindow, coveredTokens + margin)
    }
}
