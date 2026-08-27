import MLX
import MLXFast

/// The learned indexer's unpooled keys and rotary positions belong to the same
/// request snapshot as the target KV. Never keep them on the shared model.
final class Q38QSACache: KVCache {
    let attention: KVCache
    private let indexer: KVCache

    init(attention: KVCache = KVCacheSimple(), indexer: KVCache = KVCacheSimple()) {
        self.attention = attention
        self.indexer = indexer
    }

    var offset: Int { attention.offset }
    var rowOffsets: [Int]? { attention.rowOffsets }
    var supportsVariablePositionBatching: Bool { attention.supportsVariablePositionBatching }
    var canRollback: Bool { attention is KVCacheSimple && indexer is KVCacheSimple }

    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        attention.update(keys: keys, values: values)
    }

    func updateIndexer(keys: MLXArray, positions: MLXArray) -> (MLXArray, MLXArray) {
        indexer.update(keys: keys, values: positions)
    }

    func makeMask(n: Int) -> MLXFast.ScaledDotProductAttentionMaskMode {
        attention.makeMask(n: n)
    }

    func fork() -> KVCache {
        Q38QSACache(attention: attention.fork(), indexer: indexer.fork())
    }

    func rollback(toOffset offset: Int) {
        precondition(canRollback)
        (attention as? KVCacheSimple)?.rollback(toOffset: offset)
        (indexer as? KVCacheSimple)?.rollback(toOffset: offset)
    }

    func batched(with caches: [KVCache]) -> KVCache? {
        guard let caches = caches as? [Q38QSACache], !caches.isEmpty,
              let main = attention.batched(with: caches.map(\.attention)),
              let side = indexer.batched(with: caches.map(\.indexer)) else { return nil }
        return Q38QSACache(attention: main, indexer: side)
    }

    func unbatchedRows(count: Int) -> [KVCache]? {
        guard let main = attention.unbatchedRows(count: count),
              let side = indexer.unbatchedRows(count: count) else { return nil }
        return zip(main, side).map { Q38QSACache(attention: $0, indexer: $1) }
    }
}
