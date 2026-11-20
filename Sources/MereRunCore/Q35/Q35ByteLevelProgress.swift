/// ByteLevel decoding with cleanup disabled concatenates token bytes between added tokens.
/// Complete UTF-8 pieces can be appended directly. If a suffix ends in a replacement
/// scalar, the tokenizer decodes that suffix again with the next token.
struct Q35ByteLevelProgress {
    private var committed = ""
    private var pendingTokens: [Int] = []
    private var pendingText = ""

    var text: String { pendingTokens.isEmpty ? committed : committed + pendingText }

    mutating func append(token: Int, piece: String, decode: ([Int]) -> String) {
        if pendingTokens.isEmpty && piece.unicodeScalars.last?.value != 0xfffd {
            committed += piece
            return
        }
        pendingTokens.append(token)
        pendingText = pendingTokens.count == 1 ? piece : decode(pendingTokens)
        if pendingText.unicodeScalars.last?.value != 0xfffd {
            committed += pendingText
            pendingTokens.removeAll(keepingCapacity: true)
            pendingText = ""
        }
    }
}
