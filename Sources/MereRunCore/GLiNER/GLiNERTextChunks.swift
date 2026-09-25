import Foundation

struct GLiNERTextChunk {
    let text: String
    let start: Int
}

enum GLiNERTextChunks {
    static func make(text: String, size: Int = 384, overlap: Int = 64,
                     fits: (String) throws -> Bool) throws -> [GLiNERTextChunk] {
        guard size > 0, overlap >= 0, overlap < size else {
            throw GLiNERClassificationError.invalidRequest("Chunk size must exceed the overlap.")
        }
        let expression = try NSRegularExpression(pattern:
            #"(?:https?://[^\s]+|www\.[^\s]+)|[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}|@[a-z0-9_]+|\w+(?:[-_]\w+)*|\S"#,
            options: [.caseInsensitive])
        let source = text as NSString
        let words = expression.matches(in: text, range: NSRange(location: 0, length: source.length)).map(\.range)
        guard !words.isEmpty else {
            throw GLiNERClassificationError.invalidRequest("Provide text containing at least one word.")
        }
        var chunks: [GLiNERTextChunk] = []
        var first = 0
        while first < words.count {
            var last = min(first + size, words.count)
            var chunkText = ""
            while last > first {
                let range = NSRange(location: words[first].location,
                                    length: NSMaxRange(words[last - 1]) - words[first].location)
                chunkText = source.substring(with: range)
                if try fits(chunkText) { break }
                last -= 1
            }
            guard last > first else {
                throw GLiNERClassificationError.invalidRequest("A single text word exceeds the checkpoint token limit.")
            }
            let index = String.Index(utf16Offset: words[first].location, in: text)
            let start = text.unicodeScalars.distance(from: text.startIndex, to: index)
            chunks.append(GLiNERTextChunk(text: chunkText, start: start))
            if last == words.count { break }
            first = max(first + 1, last - min(overlap, last - first - 1))
        }
        return chunks
    }
}
