import Foundation

public enum MiniMaxMusic3Prompt {
    public static let audioEndTokenID = 151_670
    public static let audioCFGTokenID = 151_654
    public static let audioCodeOffset = 151_675
    public static let semanticVocabularySize = 16_384
    public static let maxPromptTokens = 5_000
    public static let maxAudioFrames = 9_000
    public static let frameRate = 25

    public static func assemble(caption: String, lyrics: String) -> String {
        "<|im_start|><|caption_start|>\(cleanCaption(caption))<|caption_end|>"
            + "<|lyrics_start|>\(normalizeLyrics(lyrics))<|lyrics_end|><|im_end|><|audio_start|>"
    }

    public static func cleanCaption(_ caption: String) -> String {
        var text = caption
        if let specialPattern = try? NSRegularExpression(pattern: #"<\|([^|]*)\|>"#) {
            let range = NSRange(text.startIndex..<text.endIndex, in: text)
            for match in specialPattern.matches(in: text, range: range).reversed() {
                guard let capture = Range(match.range(at: 1), in: text),
                      let whole = Range(match.range(at: 0), in: text) else { continue }
                let inner = text[capture].trimmingCharacters(in: .whitespacesAndNewlines)
                let parts = inner.split(maxSplits: 1, whereSeparator: { $0.isWhitespace })
                let replacement = parts.count == 2 ? "\(parts[0]) is \(parts[1])" : inner
                text.replaceSubrange(whole, with: replacement)
            }
        }
        text = text.replacingOccurrences(of: "\r\n", with: "\n")
            .replacingOccurrences(of: "\r", with: "\n")
        var sourceLines = text.components(separatedBy: "\n")
        if sourceLines.last?.isEmpty == true {
            sourceLines.removeLast()
        }
        let lines = sourceLines.map { line -> String in
            var value = line
            value = value.replacingOccurrences(of: #"^\s{0,3}#{1,6}\s+"#, with: "", options: .regularExpression)
            value = value.replacingOccurrences(of: #"^\s*[*+-]\s+"#, with: "", options: .regularExpression)
            value = value.replacingOccurrences(of: #"^\s*\*\s+"#, with: "", options: .regularExpression)
            while value.contains("**") {
                let updated = value.replacingOccurrences(
                    of: #"\*\*([^*]+)\*\*"#,
                    with: "$1",
                    options: .regularExpression
                )
                if updated == value {
                    break
                }
                value = updated
            }
            value = value.replacingOccurrences(
                of: #"(?<!\*)\*([^*\n]+)\*(?!\*)"#,
                with: "$1",
                options: .regularExpression
            )
            while value.last.map({ $0 == " " || $0 == "\t" }) == true {
                value.removeLast()
            }
            return value
        }
        text = lines.joined(separator: "\n")
        text = text.replacingOccurrences(of: #"(?m)^\s*[-*_]{3,}\s*$"#, with: "", options: .regularExpression)
        text = text.replacingOccurrences(of: "• ", with: "").replacingOccurrences(of: "    ", with: "")
        return text.replacingOccurrences(of: #"\n{2,}"#, with: "\n", options: .regularExpression)
    }

    public static func normalizeLyrics(_ lyrics: String) -> String {
        let pattern = try? NSRegularExpression(pattern: #"^[ \t]*((?:\[[^\]]+\][ \t]*)+)"#)
        let lines = lyrics.components(separatedBy: "\n").map { line -> String in
            let range = NSRange(line.startIndex..<line.endIndex, in: line)
            guard let match = pattern?.firstMatch(in: line, range: range),
                  let capture = Range(match.range(at: 1), in: line) else {
                return line
            }
            return String(line[capture]).trimmingCharacters(in: .whitespaces)
        }
        var text = lines.joined(separator: "\n")
        text = text.replacingOccurrences(of: "] ", with: "]\n")
        text = text.replacingOccurrences(of: " [", with: "\n[")
        text = text.replacingOccurrences(of: " ^ ", with: "\n")
        if let tagPattern = try? NSRegularExpression(pattern: #"\[([^\]]+)\]"#) {
            let fullRange = NSRange(text.startIndex..<text.endIndex, in: text)
            for match in tagPattern.matches(in: text, range: fullRange).reversed() {
                guard let capture = Range(match.range(at: 1), in: text),
                      let whole = Range(match.range(at: 0), in: text) else { continue }
                text.replaceSubrange(whole, with: "[\(text[capture].lowercased())]")
            }
        }
        return "[start]\n\(text)"
    }

    public static func chunkStarts(frameCount: Int, chunkFrames: Int = 200, hop: Int = 100) -> [Int] {
        guard frameCount > chunkFrames else { return [0] }
        return Array(stride(from: 0, to: frameCount - hop, by: hop))
    }

    public static func latentLength(
        frameCount: Int,
        inputSamplingRate: Int = 24_000,
        inputHopLength: Int = 960,
        outputSamplingRate: Int = 44_100,
        outputHopLength: Int = 512
    ) -> Int {
        max(1, frameCount * outputSamplingRate * inputHopLength / inputSamplingRate / outputHopLength)
    }

    public static func decodedSampleCount(
        frameCount: Int,
        inputSamplingRate: Int = 24_000,
        inputHopLength: Int = 960,
        outputSamplingRate: Int = 44_100,
        outputHopLength: Int = 512
    ) -> Int {
        latentLength(
            frameCount: frameCount,
            inputSamplingRate: inputSamplingRate,
            inputHopLength: inputHopLength,
            outputSamplingRate: outputSamplingRate,
            outputHopLength: outputHopLength
        ) * outputHopLength
    }

    /// Returns the first semantic frame count whose decoded waveform is at
    /// least the requested duration. The vocoder emits whole 512-sample latent
    /// hops, so `duration * 25` can otherwise undershoot by a fraction of a hop.
    public static func minimumFrameCount(forDurationSeconds durationSeconds: Float) -> Int {
        let minimumSamples = Int(
            Foundation.ceil(Double(durationSeconds) * 44_100)
        )
        var frames = max(1, Int(Foundation.ceil(Double(durationSeconds) * Double(frameRate))))
        while decodedSampleCount(frameCount: frames) < minimumSamples {
            frames += 1
        }
        return frames
    }
}
