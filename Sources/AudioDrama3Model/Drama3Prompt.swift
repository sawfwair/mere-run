import Foundation

public enum Drama3Prompt {
    public struct Reference: Sendable {
        public let transcript: String
        public let codes: [[Int32]]
        public init(transcript: String, codes: [[Int32]]) { self.transcript = transcript; self.codes = codes }
    }

    private static let structural = ["<|im_start|>", "<|im_end|>", "<|voice|>", "<|text|>", "<|interleave|>", "<|semantic:"]
    private static let whitespace = CharacterSet(charactersIn:
        "\u{0009}\u{000A}\u{000B}\u{000C}\u{000D}\u{0020}\u{0085}\u{00A0}\u{1680}\u{2000}\u{2001}\u{2002}\u{2003}\u{2004}\u{2005}\u{2006}\u{2007}\u{2008}\u{2009}\u{200A}\u{2028}\u{2029}\u{202F}\u{205F}\u{3000}")

    public static func validate(_ text: String) throws {
        guard !text.trimmingCharacters(in: whitespace).isEmpty,
              !structural.contains(where: text.contains) else {
            throw Drama3Error.invalid("text is empty or contains prompt structural tokens")
        }
        var remaining = text[...]
        while let marker = remaining.range(of: "<|speaker") {
            let suffix = remaining[marker.lowerBound...]
            guard (0...15).contains(where: { suffix.hasPrefix("<|speaker:\($0)|>") }) else {
                throw Drama3Error.invalid("speaker tags must be <|speaker:N|> with N from 0 through 15")
            }
            remaining = remaining[marker.upperBound...]
        }
    }

    private static func tagged(_ text: String) -> String {
        let trimmed = text.trimmingCharacters(in: whitespace)
        return trimmed.hasPrefix("<|speaker:") ? text : "<|speaker:0|> " + text
    }

    /// Tokenize each segment independently: BPE merges across seams change ids.
    public static func rows(text: String, references: [Reference], config: Drama3Config,
                            encode: (String) -> [Int]) throws -> [[Int32]] {
        try validate(text)
        var result: [[Int32]] = []
        func append(_ segment: String) throws {
            let ids = encode(segment)
            guard ids.allSatisfy({ (0..<config.text.vocabSize).contains($0) }) else {
                throw Drama3Error.invalid("tokenizer id outside the checkpoint vocabulary")
            }
            result += ids.map { [Int32($0)] + Array(repeating: Int32(0), count: config.audio.codebooks) }
        }
        try append("<|im_start|>system\n")
        try append(references.isEmpty ? "convert the provided text to speech" :
            "convert the provided text to speech reference to the following:")
        for (i, reference) in references.enumerated() {
            try validate(reference.transcript)
            guard reference.codes.count == config.audio.codebooks,
                  let count = reference.codes.first?.count, count > 0,
                  reference.codes.allSatisfy({ $0.count == count && $0.allSatisfy({ $0 >= 0 && $0 < config.audio.vocab }) }) else {
                throw Drama3Error.invalid("reference needs ten codebook rows with valid nonempty frames")
            }
            // A leading speaker tag must directly follow Text:, preserving its whitespace after the tag.
            let transcript = tagged(reference.transcript)
            let body = transcript.drop(while: { char in char.unicodeScalars.allSatisfy(whitespace.contains) })
            let leading = transcript.prefix(transcript.count - body.count)
            guard let end = body.range(of: "|>") else { throw Drama3Error.invalid("missing reference speaker") }
            let reordered = String(body[..<end.upperBound]) + leading + body[end.upperBound...]
            try append("\n\n[Reference \(i + 1)]\nText: \(reordered)\nSpeech: ")
            for frame in 0..<count {
                let codes = reference.codes.map { $0[frame] }
                result.append([Int32(config.semanticStart) + codes[0]] + codes)
            }
        }
        try append("<|im_end|>\n")
        try append("<|im_start|>user\n")
        try append(tagged(text))
        try append("<|im_end|>\n")
        try append("<|im_start|>assistant\n<|voice|>")
        guard !result.isEmpty else { throw Drama3Error.invalid("tokenizer produced an empty prompt") }
        return result
    }
}
