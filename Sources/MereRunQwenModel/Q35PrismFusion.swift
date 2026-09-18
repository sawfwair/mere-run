import Foundation
import MLX
import MLXNN

/// Keeps fusion outside the parameter tree and rebuilds it after module replacement.
package final class Q35PrismFusion {
    // Retain sources so allocator reuse cannot make a replacement look identical.
    private var sources: [Linear?] = []
    private var packed: [Q35PrismLinear] = []
    private var projection: Q35PrismLinear?
    private var boundaries: [Int] = []
    package static let mode = ProcessInfo.processInfo.environment["MERERUN_Q35_PRISM_FUSION"] ?? "shared"
    private let fuseWeights: Bool

    package init(fuseWeights: Bool = mode == "1") {
        self.fuseWeights = fuseWeights
    }

    package func callSplit(_ input: MLXArray, projections: [Linear?]) -> [MLXArray]? {
        guard Self.mode != "0" else { return nil }
        let ids = projections.map { $0.map(ObjectIdentifier.init) }
        if ids != sources.map({ $0.map(ObjectIdentifier.init) }) {
            sources = projections
            projection = nil
            packed = []
            boundaries = []
            prepare(projections)
        }
        if let projection {
            return MLX.split(projection(input), indices: boundaries, axis: -1)
        }
        guard let first = packed.first else { return nil }
        let value = first.signs.map { Q35PrismTransform.apply(input, block: first.block, signs: $0) } ?? input
        return packed.map { $0.projectTransformed(value) }
    }

    private func prepare(_ sources: [Linear?]) {
        let packed = sources.compactMap { $0 as? Q35PrismLinear }
        guard packed.count == sources.count, packed.count > 1, let first = packed.first else { return }
        // Equal widths alone do not establish equal rotations. Check the actual
        // signs once during preparation, never in the token loop.
        for next in packed.dropFirst() {
            guard next.block == first.block,
                  next.weight.dim(-1) == first.weight.dim(-1),
                  next.scales.dtype == first.scales.dtype,
                  next.biases.dtype == first.biases.dtype else { return }
            switch (first.signs, next.signs) {
            case (nil, nil): break
            case let (lhs?, rhs?):
                guard lhs.shape == rhs.shape, MLX.all(lhs .== rhs).item(Bool.self) else { return }
            default: return
            }
        }
        self.packed = packed
        guard fuseWeights else { return }
        var total = 0
        for source in packed.dropLast() {
            total += source.weight.dim(0)
            boundaries.append(total)
        }
        let weight = MLX.concatenated(packed.map(\.weight), axis: 0)
        let scales = MLX.concatenated(packed.map(\.scales), axis: 0)
        let biases = MLX.concatenated(packed.map(\.biases), axis: 0)
        MLX.eval(weight, scales, biases)
        projection = Q35PrismLinear(
            weight: weight, scales: scales, biases: biases, signs: first.signs, block: first.block
        )
    }
}
