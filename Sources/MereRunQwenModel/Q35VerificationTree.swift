import MLX

/// Row zero is the already chosen anchor. Every later row has an earlier parent.
/// Row indices are storage identities; depths are positions in the token stream.
package struct Q35VerificationTree: Sendable {
    package enum Invalid: Error { case parents }
    package let parents: [Int]
    package let paths: [[Int]]
    package let depths: [Int]
    let recurrentSlots: Q35TreeStateSlots
    let devicePlan: Q35TreeDevicePlan?

    package init(parents: [Int]) throws {
        guard (1...16).contains(parents.count), parents.first == -1,
              parents.dropFirst().enumerated().allSatisfy({ $0.element >= 0 && $0.element <= $0.offset }) else {
            throw Invalid.parents
        }
        self.parents = parents
        var paths = [[0]]
        for row in 1..<parents.count { paths.append(paths[parents[row]] + [row]) }
        self.paths = paths
        let depths = paths.map { $0.count - 1 }
        let slots = Q35TreeStateSlots(parents: parents)
        self.depths = depths
        self.recurrentSlots = slots
        self.devicePlan = Q35TreePipeline.enabled ? Q35TreeDevicePlan(paths: paths, depths: depths, slots: slots) : nil
    }

    var count: Int { parents.count }
    var groups: [[Int]] {
        (0...depths.max()!).map { depth in parents.indices.filter { depths[$0] == depth } }
    }

    /// Negative entries address the committed convolution window before the root.
    func convolutionRows(keep: Int) -> [[Int]] {
        paths.map { path in
            Array((Array(-keep..<0) + path).suffix(keep + 1))
        }
    }

    static func take(_ array: MLXArray, rows: [Int], axis: Int = 1) -> MLXArray {
        // Accepted paths often occupy the leading rows. A view avoids a gather
        // dispatch for each saved GDN input and confirmed draft-context tap.
        if axis == 1, rows.enumerated().allSatisfy({ $0.element == $0.offset }) {
            return array[0..., 0..<rows.count, .ellipsis]
        }
        return MLX.take(array, MLXArray(rows.map(Int32.init)), axis: axis)
    }

    /// Follows target choices, never draft scores, and returns the consumed rows.
    package func matchingPath(tokens: [Int], targetPicks: [Int], budget: Int, eos: Set<Int>) -> [Int] {
        precondition(tokens.count == count && targetPicks.count == count && budget > 0)
        var path = [0]
        while path.count < budget {
            let parent = path.last!
            let token = targetPicks[parent]
            guard !eos.contains(token),
                  let child = parents.indices.dropFirst().first(where: { parents[$0] == parent && tokens[$0] == token }) else {
                break
            }
            path.append(child)
        }
        return path
    }
}

/// Immutable metadata shared by every layer of one verification round. The
/// round owns the arrays until all submitted consumers have finished.
final class Q35TreeDevicePlan: @unchecked Sendable {
    let paths: MLXArray
    let paths16: MLXArray
    let lengths: MLXArray
    let depths: MLXArray
    let convolution3: MLXArray
    let slotReads: MLXArray
    let slotWrites: MLXArray
    let slotOrder: MLXArray

    init(paths: [[Int]], depths: [Int], slots: Q35TreeStateSlots) {
        self.paths = MLXArray(paths.flatMap { $0 + Array(repeating: 0, count: paths.count - $0.count) }.map(Int32.init))
        self.paths16 = MLXArray(paths.flatMap { $0 + Array(repeating: 0, count: 16 - $0.count) }.map(Int32.init))
        self.lengths = MLXArray(paths.map { Int32($0.count) })
        self.depths = MLXArray(depths.map(Int32.init))
        self.convolution3 = MLXArray(paths.flatMap { Array(([-3, -2, -1] + $0).suffix(4)) }.map(Int32.init))
        self.slotReads = MLXArray(slots.reads.map(Int32.init))
        self.slotWrites = MLXArray(slots.writes.map(Int32.init))
        self.slotOrder = MLXArray(slots.order.map(Int32.init))
    }
}

enum Q35TreeVerificationContext {
    @TaskLocal static var current: Q35VerificationTree?
}
