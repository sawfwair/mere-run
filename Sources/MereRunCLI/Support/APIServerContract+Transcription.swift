import Foundation
import MereRunCore
import AudioCore
import AudioSTT

extension APIServerContract {
    static let maxTranscriptionTokens = 4_096
    static let defaultTranscriptionModelID = ParakeetResources.defaultModelId

    struct TranscriptionPlan: Equatable, Sendable {
        let modelID: String
        let language: String?
        let responseFormat: String
        let task: ASRTask
        let maxTokens: Int
    }

    static func transcriptionPlan(from form: MultipartFormData) throws -> TranscriptionPlan {
        let modelID = normalizedTranscriptionModelID(form.field("model"))
        return TranscriptionPlan(
            modelID: modelID,
            language: normalizedOptional(form.field("language")),
            responseFormat: try transcriptionResponseFormat(form.field("response_format")),
            task: try transcriptionTask(form.field("task")),
            maxTokens: try transcriptionMaxTokens(form.field("max_tokens"))
        )
    }

    static func transcriptionResponse(
        from result: ASRResult,
        verbose: Bool
    ) -> OpenAIAudioTranscriptionResponse {
        OpenAIAudioTranscriptionResponse(
            text: result.text,
            language: result.language,
            duration: verbose ? result.duration : nil,
            segments: verbose ? transcriptionSegments(from: result) : nil
        )
    }

    static func transcriptionSubtitle(from result: ASRResult, format: String) -> String {
        let segments = transcriptionSegments(from: result) ?? [
            OpenAIAudioTranscriptionSegment(
                id: 0,
                start: 0,
                end: max(result.duration, 0.001),
                text: result.text
            ),
        ]
        switch format {
        case "srt":
            return segments.enumerated()
                .map { index, segment in
                    let start = subtitleTimestamp(segment.start, separator: ",")
                    let end = subtitleTimestamp(max(segment.end, segment.start + 0.001), separator: ",")
                    return "\(index + 1)\n\(start) --> \(end)\n\(segment.text)"
                }
                .joined(separator: "\n\n") + "\n"
        default:
            let body = segments
                .map { segment in
                    let start = subtitleTimestamp(segment.start, separator: ".")
                    let end = subtitleTimestamp(max(segment.end, segment.start + 0.001), separator: ".")
                    return "\(start) --> \(end)\n\(segment.text)"
                }
                .joined(separator: "\n\n")
            return body.isEmpty ? "WEBVTT\n" : "WEBVTT\n\n\(body)\n"
        }
    }

    private static func transcriptionResponseFormat(_ rawValue: String?) throws -> String {
        let value = normalizedOptional(rawValue)?.lowercased() ?? "json"
        guard ["json", "text", "verbose_json", "srt", "vtt"].contains(value) else {
            throw APIRequestValidationError.invalidField(
                "response_format",
                "expected json, text, verbose_json, srt, or vtt"
            )
        }
        return value
    }

    private static func transcriptionTask(_ rawValue: String?) throws -> ASRTask {
        let value = normalizedOptional(rawValue)?.lowercased() ?? ASRTask.transcribe.rawValue
        guard let task = ASRTask(rawValue: value) else {
            throw APIRequestValidationError.invalidField("task", "expected transcribe or translate")
        }
        return task
    }

    private static func transcriptionMaxTokens(_ rawValue: String?) throws -> Int {
        guard let rawValue = normalizedOptional(rawValue) else {
            return 448
        }
        guard let value = Int(rawValue), (1...maxTranscriptionTokens).contains(value) else {
            throw APIRequestValidationError.invalidField(
                "max_tokens",
                "must be between 1 and \(maxTranscriptionTokens)"
            )
        }
        return value
    }

    private static func normalizedTranscriptionModelID(_ rawValue: String?) -> String {
        let modelID = normalizedModelID(rawValue, defaultID: defaultTranscriptionModelID)
        switch modelID.lowercased() {
        case "whisper-1", "gpt-4o-transcribe", "gpt-4o-mini-transcribe":
            return defaultTranscriptionModelID
        default:
            return modelID
        }
    }

    private static func transcriptionSegments(
        from result: ASRResult
    ) -> [OpenAIAudioTranscriptionSegment]? {
        if let sentences = result.sentenceAlignments, !sentences.isEmpty {
            return sentences.enumerated().map { index, sentence in
                OpenAIAudioTranscriptionSegment(
                    id: index,
                    start: sentence.startSeconds,
                    end: sentence.endSeconds,
                    text: sentence.text
                )
            }
        }
        if let tokens = result.tokenAlignments, !tokens.isEmpty {
            return tokens.enumerated().map { index, token in
                OpenAIAudioTranscriptionSegment(
                    id: index,
                    start: token.startSeconds,
                    end: token.endSeconds,
                    text: token.text
                )
            }
        }
        return nil
    }

    private static func subtitleTimestamp(_ seconds: Double, separator: String) -> String {
        let milliseconds = max(0, Int((seconds * 1_000).rounded()))
        let hours = milliseconds / 3_600_000
        let minutes = (milliseconds % 3_600_000) / 60_000
        let secs = (milliseconds % 60_000) / 1_000
        let millis = milliseconds % 1_000
        return String(format: "%02d:%02d:%02d%@%03d", hours, minutes, secs, separator, millis)
    }
}
