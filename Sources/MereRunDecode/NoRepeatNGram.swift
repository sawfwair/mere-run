/// Generated-token n-gram bans with the same sliding-window semantics as the
/// upstream Unlimited-OCR serving processor. Prompt tokens stay outside this history.
public enum NoRepeatNGram {
    public static func bannedTokens(
        in generatedTokens: [Int], size: Int, window: Int, allowedTokens: Set<Int> = []
    ) -> [Int] {
        guard size > 0, generatedTokens.count >= size else { return [] }
        let start = max(0, generatedTokens.count - window)
        let end = generatedTokens.count - size + 1
        guard start < end else { return [] }
        let prefix = generatedTokens.suffix(size - 1)
        var banned: Set<Int> = []
        for offset in start..<end {
            if generatedTokens[offset..<(offset + size - 1)].elementsEqual(prefix) {
                banned.insert(generatedTokens[offset + size - 1])
            }
        }
        return banned.subtracting(allowedTokens).sorted()
    }
}
