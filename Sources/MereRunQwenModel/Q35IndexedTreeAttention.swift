#if canImport(Metal)
import Foundation
import MereRunEnvironment
import MLX

enum Q35IndexedTreeAttention {
    @TaskLocal static var enabled =
        MereRunEnvironmentSwitch.isEnabled("MERERUN_Q38_TREE_INDEXED_ATTENTION", default: false)

    /// Uses native vector-SDPA reduction geometry with indexed KV loads. Prefix
    /// buffers stay shared; only compact queries, paths, and partials are owned.
    static func apply(q: MLXArray, k: MLXArray, v: MLXArray,
                      tree: Q35VerificationTree, scale: Float) -> MLXArray? {
        guard enabled, Device.defaultDevice().deviceType == .gpu,
              q.dim(0) == 1, q.dim(3) == 256, v.dim(3) == 256,
              q.dim(1) / k.dim(1) <= 8,
              q.dtype == .bfloat16, k.dtype == q.dtype, v.dtype == q.dtype else { return nil }
        let prefix = k.dim(2) - tree.count, heads = q.dim(1), kvHeads = k.dim(1)
        let gqa = heads / kvHeads
        let architecture = GPU.deviceInfo().architecture.last
        let groups = Dictionary(grouping: tree.parents.indices) { row in
            blocks(length: prefix + tree.paths[row].count, gqa: gqa, architecture: architecture)
        }.sorted { $0.key < $1.key }
        let paths = tree.devicePlan?.paths16
            ?? MLXArray(tree.paths.flatMap { $0.map(Int32.init) + Array(repeating: 0, count: 16 - $0.count) })
        let lengths = tree.devicePlan?.lengths ?? MLXArray(tree.paths.map { Int32($0.count) })
        let queries = q.transposed(2, 1, 0, 3)
        var outputs: [MLXArray] = []
        for (blocks, rows) in groups {
            let width = rows.map { tree.paths[$0].count }.max()!
            let selected = Q35VerificationTree.take(queries, rows: rows, axis: 0).contiguous()
            let inputs = [selected, k, v, paths, MLXArray(rows.map(Int32.init)), lengths,
                          MLXArray([Int32(prefix + width), Int32(prefix), Int32(heads), Int32(kvHeads)]),
                          MLXArray([scale])]
            let output: MLXArray
            if blocks == 0 {
                output = Q35IndexedTreeKernels.single(inputs,
                    template: [("T", q.dtype), ("D", 256), ("V", 256), ("G", gqa)],
                    grid: (heads * rows.count * 1_024, 1, 1), threadGroup: (1_024, 1, 1),
                    outputShapes: [[rows.count, heads, 1, 256]], outputDTypes: [q.dtype])[0]
            } else {
                let partial = Q35IndexedTreeKernels.partial(inputs,
                    template: [("T", q.dtype), ("D", 256), ("V", 256), ("G", gqa), ("Blocks", blocks)],
                    grid: (kvHeads * 32, rows.count * gqa, blocks), threadGroup: (32, gqa, 1),
                    outputShapes: [[rows.count, heads, blocks, 256], [rows.count, heads, blocks],
                                   [rows.count, heads, blocks]], outputDTypes: [q.dtype, .float32, .float32])
                output = Q35IndexedTreeKernels.merge(partial,
                    template: [("T", q.dtype), ("D", 256), ("Blocks", blocks)],
                    grid: (rows.count * heads * 1_024, 1, 1), threadGroup: (1_024, 1, 1),
                    outputShapes: [[rows.count, heads, 1, 256]], outputDTypes: [q.dtype])[0]
            }
            outputs.append(output.transposed(2, 1, 0, 3))
        }
        if outputs.count == 1 { return outputs[0] }
        let order = groups.flatMap(\.value)
        let inverse = tree.parents.indices.map { order.firstIndex(of: $0)! }
        return Q35VerificationTree.take(concatenated(outputs, axis: 2), rows: inverse, axis: 2)
    }

    static func blocks(length: Int, gqa: Int, architecture: Character?) -> Int {
        let threshold = architecture == "s" || architecture == "d" ? 1_024 : 4_096
        guard length >= threshold else { return 0 }
        if let raw = ProcessInfo.processInfo.environment["MLX_SDPA_BLOCKS"], let count = Int(raw), count > 0 {
            return ((count + 31) / 32) * 32
        }
        if architecture == "s" {
            if length > 1_024 && gqa > 4 {
                if length <= 8_192 { return 128 }
                if length <= 32_768 { return 256 }
                if length <= 65_536 { return 512 }
                return 1_024
            }
            return 64
        }
        if architecture == "d" {
            if gqa <= 2 && length > 8_192 { return 256 }
            if gqa >= 6 {
                if length >= 65_536 { return 1_024 }
                if length >= 16_384 { return 512 }
            }
            return 128
        }
        return gqa >= 4 ? 64 : 32
    }
}
#endif
