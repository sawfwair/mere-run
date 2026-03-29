import Foundation

enum ParakeetTokenizer {
    static func decode(tokens: [Int], vocabulary: [String]) -> String {
        var text = ""
        text.reserveCapacity(tokens.count * 2)
        for token in tokens {
            guard token >= 0, token < vocabulary.count else { continue }
            text += vocabulary[token].replacingOccurrences(of: "▁", with: " ")
        }
        return text
    }
}
