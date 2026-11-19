import MLX
import MLXFast

/// Each batch item is one query and its own committed prefix plus ancestry.
/// Padding is masked only within a serial Metal reduction regime.
func q35TreeAttention(
    q: MLXArray, k: MLXArray, v: MLXArray, tree: Q35VerificationTree, scale: Float
) -> MLXArray {
    #if canImport(Metal)
    if let indexed = Q35IndexedTreeAttention.apply(q: q, k: k, v: v, tree: tree, scale: scale) {
        return indexed
    }
    #endif
    let prefix = k.dim(2) - tree.count
    if Device.defaultDevice().deviceType == .cpu {
        // Preserve the serial CPU gather and query layout as well as its
        // length: BF16 reductions can round differently after layout changes.
        return concatenated(tree.parents.indices.map { row in
            let indices = Array(0..<prefix) + tree.paths[row].map { prefix + $0 }
            return MLXFast.scaledDotProductAttention(
                queries: q[0..., 0..., row..<(row + 1), 0...],
                keys: Q35VerificationTree.take(k, rows: indices, axis: 2),
                values: Q35VerificationTree.take(v, rows: indices, axis: 2),
                scale: scale, mask: .none
            )
        }, axis: 2)
    }
    let gqa = q.dim(1) / k.dim(1)
    let specializedGQA = (gqa == 8 && [64, 128].contains(q.dim(3)))
        || ([12, 16].contains(gqa) && q.dim(3) == 128)
    let groups: [[Int]]
    if specializedGQA && prefix + tree.depths.max()! + 1 >= 8_192 {
        // MLX's long-context GQA kernel requires an unmasked query. Retain
        // exact-length groups rather than changing its reduction arithmetic.
        groups = tree.groups
    } else {
        let boundaries = [1_024, 1_025, 4_096, 8_193, 16_384, 32_769, 65_536, 65_537]
        groups = Dictionary(grouping: tree.parents.indices) { row in
            boundaries.filter { $0 <= prefix + tree.paths[row].count }.count
        }.sorted { $0.key < $1.key }.map(\.value)
    }
    let queries = q.transposed(2, 1, 0, 3)
    let rows = groups.map { group in
        let width = group.map { tree.paths[$0].count }.max()!
        let indices = MLXArray(group.flatMap { row in
            tree.paths[row].map { Int32(prefix + $0) }
                + Array(repeating: Int32(prefix), count: width - tree.paths[row].count)
        }).reshaped(group.count, width)
        func history(_ x: MLXArray) -> MLXArray {
            let committed = broadcast(x[0..., 0..., 0..<prefix, 0...],
                                      to: [group.count, x.dim(1), prefix, x.dim(3)])
            let branch = take(x, indices, axis: 2).squeezed(axis: 0).transposed(1, 0, 2, 3)
            return concatenated([committed, branch], axis: 2)
        }
        let mask: MLXFast.ScaledDotProductAttentionMaskMode
        if group.allSatisfy({ tree.paths[$0].count == width }) {
            mask = .none
        } else {
            mask = .array(MLXArray(group.flatMap { row in
                Array(repeating: true, count: prefix)
                    + (0..<width).map { $0 < tree.paths[row].count }
            }).reshaped(group.count, 1, 1, prefix + width))
        }
        return MLXFast.scaledDotProductAttention(
            queries: Q35VerificationTree.take(queries, rows: group, axis: 0),
            keys: history(k), values: history(v), scale: scale, mask: mask
        ).transposed(2, 1, 0, 3)
    }
    if groups.count == 1 { return rows[0] }
    let order = groups.flatMap { $0 }
    let inverse = tree.parents.indices.map { order.firstIndex(of: $0)! }
    return Q35VerificationTree.take(concatenated(rows, axis: 2), rows: inverse, axis: 2)
}
