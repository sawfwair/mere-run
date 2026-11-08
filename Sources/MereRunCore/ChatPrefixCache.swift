import Foundation

public struct ChatCacheBoundary: Hashable, Sendable {
    public let messageIndex: Int
    public let blockCount: Int
    public let ttlSeconds: Int
    public init(messageIndex: Int, blockCount: Int, ttlSeconds: Int) {
        self.messageIndex = messageIndex; self.blockCount = blockCount; self.ttlSeconds = ttlSeconds
    }
}
public struct ChatToolCacheBoundary: Hashable, Sendable {
    public let throughTool: String
    public let ttlSeconds: Int
    public init(throughTool: String, ttlSeconds: Int) { self.throughTool = throughTool; self.ttlSeconds = ttlSeconds }
}

public struct ChatPrefixCachePolicy: Hashable, Sendable {
    public let scope: String
    public let boundaries: [ChatCacheBoundary]
    public let wholePromptTTL: Int?
    public let toolPrefixTTL: Int?
    public let toolBoundaries: [ChatToolCacheBoundary]
    public init(scope: String, boundaries: [ChatCacheBoundary], wholePromptTTL: Int?, toolPrefixTTL: Int? = nil,
                toolBoundaries: [ChatToolCacheBoundary] = []) {
        self.scope = scope; self.boundaries = boundaries; self.wholePromptTTL = wholePromptTTL; self.toolPrefixTTL = toolPrefixTTL
        self.toolBoundaries = toolBoundaries
    }
}

/// Holds actual runtime cache values; ownership remains inside the resident model.
/// Total retained tokens and entry count bound GPU storage as well as metadata.
struct ChatNativePrefixCache<Value> {
    struct Entry {
        let identity: String
        let tokens: [Int]
        let value: Value
        var expires: Date
        var ttlSeconds: Int
        var lastAccess: UInt64
    }
    var entries: [Entry] = []
    let maximumEntries: Int
    let maximumTokens: Int
    private var sequence: UInt64 = 0

    init(maximumEntries: Int = 2, maximumTokens: Int = 8192) {
        self.maximumEntries = maximumEntries; self.maximumTokens = maximumTokens
    }
    mutating func lookup(identity: String, tokens: [Int], ttls: [Int: Int] = [:], now: Date = Date(),
                         isCompatible: (Value, Int) -> Bool = { _, _ in true }) -> Entry? {
        entries.removeAll { $0.expires <= now }
        let boundaryLimit = ttls.keys.max() ?? Int.max
        guard let index = entries.indices.filter({ entries[$0].identity == identity && entries[$0].tokens.count <= boundaryLimit
            && tokens.starts(with: entries[$0].tokens) && isCompatible(entries[$0].value, entries[$0].tokens.count) })
            .max(by: { entries[$0].tokens.count < entries[$1].tokens.count }) else { return nil }
        sequence += 1; entries[index].lastAccess = sequence
        if let requested = ttls[entries[index].tokens.count], [300, 3600].contains(requested) {
            entries[index].ttlSeconds = max(entries[index].ttlSeconds, requested)
        }
        entries[index].expires = now.addingTimeInterval(Double(entries[index].ttlSeconds))
        return entries[index]
    }
    @discardableResult
    mutating func store(identity: String, tokens: [Int], value: Value, ttl: Int, now: Date = Date()) -> Bool {
        guard !tokens.isEmpty, tokens.count <= maximumTokens, maximumEntries > 0, [300, 3600].contains(ttl) else { return false }
        entries.removeAll { $0.expires <= now || ($0.identity == identity && $0.tokens == tokens) }
        while entries.count >= maximumEntries || entries.reduce(tokens.count, { $0 + $1.tokens.count }) > maximumTokens {
            guard let index = entries.indices.min(by: { entries[$0].lastAccess < entries[$1].lastAccess }) else { break }
            entries.remove(at: index)
        }
        sequence += 1
        entries.append(Entry(identity: identity, tokens: tokens, value: value, expires: now.addingTimeInterval(Double(ttl)), ttlSeconds: ttl, lastAccess: sequence))
        return true
    }
    mutating func removeAll() { entries.removeAll() }
}
