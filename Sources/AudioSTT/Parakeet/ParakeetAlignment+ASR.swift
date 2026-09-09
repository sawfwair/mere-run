import AudioCore
import AudioParakeetModel

extension ParakeetAlignment {
    public static func toASRTokenAlignments(_ tokens: [ParakeetAlignedToken]) -> [ASRTokenAlignment] {
        tokens.map {
            ASRTokenAlignment(
                id: $0.id,
                text: $0.text,
                startSeconds: $0.start,
                durationSeconds: $0.duration,
                endSeconds: $0.end
            )
        }
    }

    public static func toASRSentenceAlignments(_ sentences: [ParakeetAlignedSentence]) -> [ASRSentenceAlignment] {
        sentences.map { sentence in
            ASRSentenceAlignment(
                text: sentence.text,
                startSeconds: sentence.start,
                durationSeconds: sentence.duration,
                endSeconds: sentence.end,
                tokens: toASRTokenAlignments(sentence.tokens)
            )
        }
    }

}
