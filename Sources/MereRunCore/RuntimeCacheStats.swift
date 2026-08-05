import Foundation

public struct PrefixKVCacheStats: Codable, Sendable, Hashable {
    public let enabled: Bool
    public let entries: Int
    public let maxEntries: Int
    public let hits: Int
    public let misses: Int
    public let storedPrefixes: Int
    public let reusedTokens: Int
    public let storedTokens: Int

    public init(
        enabled: Bool,
        entries: Int,
        maxEntries: Int,
        hits: Int,
        misses: Int,
        storedPrefixes: Int,
        reusedTokens: Int,
        storedTokens: Int
    ) {
        self.enabled = enabled
        self.entries = entries
        self.maxEntries = maxEntries
        self.hits = hits
        self.misses = misses
        self.storedPrefixes = storedPrefixes
        self.reusedTokens = reusedTokens
        self.storedTokens = storedTokens
    }
}

public struct RuntimeDecodeBatchingStats: Codable, Sendable, Hashable {
    public let enabled: Bool
    public let activeRows: Int
    public let queuedRows: Int
    public let batchedDecodeSteps: Int
    public let samePositionBatchedSteps: Int
    public let variablePositionBatchedSteps: Int
    public let singleDecodeSteps: Int
    public let totalBatchedRows: Int
    public let maxBatchSize: Int

    enum CodingKeys: String, CodingKey {
        case enabled
        case activeRows
        case queuedRows
        case batchedDecodeSteps
        case samePositionBatchedSteps
        case variablePositionBatchedSteps
        case singleDecodeSteps
        case totalBatchedRows
        case maxBatchSize
    }

    public init(
        enabled: Bool,
        activeRows: Int,
        queuedRows: Int,
        batchedDecodeSteps: Int,
        samePositionBatchedSteps: Int = 0,
        variablePositionBatchedSteps: Int = 0,
        singleDecodeSteps: Int,
        totalBatchedRows: Int,
        maxBatchSize: Int
    ) {
        self.enabled = enabled
        self.activeRows = activeRows
        self.queuedRows = queuedRows
        self.batchedDecodeSteps = batchedDecodeSteps
        self.samePositionBatchedSteps = samePositionBatchedSteps
        self.variablePositionBatchedSteps = variablePositionBatchedSteps
        self.singleDecodeSteps = singleDecodeSteps
        self.totalBatchedRows = totalBatchedRows
        self.maxBatchSize = maxBatchSize
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.enabled = try container.decode(Bool.self, forKey: .enabled)
        self.activeRows = try container.decode(Int.self, forKey: .activeRows)
        self.queuedRows = try container.decode(Int.self, forKey: .queuedRows)
        self.batchedDecodeSteps = try container.decode(Int.self, forKey: .batchedDecodeSteps)
        self.samePositionBatchedSteps = try container.decodeIfPresent(Int.self, forKey: .samePositionBatchedSteps) ?? 0
        self.variablePositionBatchedSteps = try container.decodeIfPresent(Int.self, forKey: .variablePositionBatchedSteps) ?? 0
        self.singleDecodeSteps = try container.decode(Int.self, forKey: .singleDecodeSteps)
        self.totalBatchedRows = try container.decode(Int.self, forKey: .totalBatchedRows)
        self.maxBatchSize = try container.decode(Int.self, forKey: .maxBatchSize)
    }
}

enum RuntimePrefillCheckpointPlanner {
    static func nextEnd(
        processed: Int,
        total: Int,
        chunkSize: Int,
        checkpoints: Set<Int>
    ) -> Int {
        let chunkEnd = min(processed + max(1, chunkSize), total)
        guard let checkpoint = checkpoints
            .filter({ $0 > processed && $0 < chunkEnd && $0 < total })
            .min() else {
            return chunkEnd
        }
        return checkpoint
    }

    static func normalizedCheckpoints(_ counts: [Int], total: Int) -> Set<Int> {
        Set(counts.filter { $0 > 0 && $0 < total })
    }

    static func storagePriority(
        tokenCount: Int,
        total: Int,
        semanticCheckpoints: Set<Int>
    ) -> RuntimePrefixCacheEntryPriority? {
        if semanticCheckpoints.contains(tokenCount) {
            return .semantic
        }
        if tokenCount == total {
            return .chunk
        }
        return nil
    }
}

enum RuntimePrefixCacheEntryPriority: Int, Sendable {
    case chunk = 0
    case semantic = 1
}

struct RuntimePrefixCacheRetentionMetadata: Sendable, Hashable {
    let priority: RuntimePrefixCacheEntryPriority
    let lastAccess: Date
}

enum RuntimePrefixCacheRetentionPlanner {
    static func keyToPrune<Key: Hashable>(
        entries: [Key: RuntimePrefixCacheRetentionMetadata]
    ) -> Key? {
        entries
            .sorted { lhs, rhs in
                if lhs.value.priority != rhs.value.priority {
                    return lhs.value.priority.rawValue < rhs.value.priority.rawValue
                }
                return lhs.value.lastAccess < rhs.value.lastAccess
            }
            .first?
            .key
    }
}

enum RuntimeDecodeBatchPositionKind {
    static func variablePositionBatchCount(_ positions: [Int]) -> Int {
        Set(positions).count > 1 ? 1 : 0
    }
}

struct RuntimeDecodeBatchRowMetadata<Row: Hashable>: Sendable where Row: Sendable {
    let row: Row
    let signature: String
    let position: Int

    init(row: Row, signature: String, position: Int) {
        self.row = row
        self.signature = signature
        self.position = position
    }
}

enum RuntimeDecodeBatchPlanner {
    static func selectRows<Row: Hashable & Sendable>(
        _ rows: [RuntimeDecodeBatchRowMetadata<Row>]
    ) -> [Row] {
        guard rows.count > 1 else {
            return rows.map(\.row)
        }

        let grouped = Dictionary(grouping: rows, by: \.signature)
        let compatibleBatches = grouped.values.filter { $0.count > 1 }
        let earliest = rows.map(\.position).min() ?? 0
        let earliestBatches = compatibleBatches.filter { earliestPosition($0) == earliest }
        if let batch = earliestBatches
            .sorted(by: { lhs, rhs in
                if lhs.count != rhs.count {
                    return lhs.count > rhs.count
                }
                return earliestPosition(lhs) < earliestPosition(rhs)
            })
            .first {
            return batch.map(\.row)
        }

        let earliestRows = rows.filter { $0.position == earliest }
        guard let catchUpRow = earliestRows.min(by: { lhs, rhs in
            String(describing: lhs.row) < String(describing: rhs.row)
        }) else {
            return []
        }
        return [catchUpRow.row]
    }

    private static func earliestPosition<Row: Hashable & Sendable>(
        _ rows: [RuntimeDecodeBatchRowMetadata<Row>]
    ) -> Int {
        rows.map(\.position).min() ?? 0
    }
}
