/// Emits only the stable suffix of a cumulatively decoded token sequence.
///
/// Byte-fallback tokenizers can decode an incomplete multibyte scalar as
/// U+FFFD when a single token or partial sequence is decoded. The replacement
/// disappears once the remaining byte tokens arrive, so streaming must wait
/// for a stable cumulative prefix before exposing it.
struct IncrementalTokenTextDecoder {
    private var emittedUTF8: [UInt8] = []

    mutating func append(decodedText: String) -> String {
        var stableText = decodedText
        while stableText.last == "\u{FFFD}" {
            stableText.removeLast()
        }

        let stableUTF8 = Array(stableText.utf8)
        guard stableUTF8.starts(with: emittedUTF8) else {
            return ""
        }

        let delta = stableUTF8.dropFirst(emittedUTF8.count)
        emittedUTF8 = stableUTF8
        return String(decoding: delta, as: UTF8.self)
    }
}
