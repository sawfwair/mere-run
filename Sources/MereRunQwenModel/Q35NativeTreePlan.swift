#if os(macOS)
import Metal

/// One round's immutable metadata, shared by every native layer. A causal
/// prompt shares one identity path across all queries.
final class Q35NativeTreePlan {
    enum Failure: Error { case allocation }
    let count: Int
    let branchRows: Int
    let pathStride: Int
    let slotCount: Int
    let pathLengths: [Int]
    let ancestry: any MTLBuffer
    let reads: any MTLBuffer
    let writes: any MTLBuffer
    let order: any MTLBuffer
    let paths: any MTLBuffer
    let lengths: any MTLBuffer
    let depths: any MTLBuffer

    convenience init(tree: Q35VerificationTree, device: any MTLDevice) throws {
        try self.init(
            device: device, branchRows: 16, pathStride: 16, slotCount: max(1, tree.recurrentSlots.count),
            ancestry: tree.convolutionRows(keep: 3).flatMap { $0 }, reads: tree.recurrentSlots.reads,
            writes: tree.recurrentSlots.writes, order: tree.recurrentSlots.order,
            paths: tree.paths.flatMap { $0 + Array(repeating: 0, count: 16 - $0.count) },
            depths: tree.depths
        )
    }

    convenience init(causalRows: Int, device: any MTLDevice) throws {
        precondition(causalRows > 0)
        let rows = Array(0..<causalRows)
        try self.init(
            device: device, branchRows: ((causalRows + 15) / 16) * 16, pathStride: 0, slotCount: 1,
            ancestry: rows.flatMap { Array(($0 - 3)...$0) },
            reads: [-1] + Array(repeating: 0, count: causalRows - 1),
            writes: Array(repeating: 0, count: causalRows), order: rows, paths: rows, depths: rows
        )
    }

    private init(
        device: any MTLDevice, branchRows: Int, pathStride: Int, slotCount: Int,
        ancestry: [Int], reads: [Int], writes: [Int], order: [Int], paths: [Int], depths: [Int]
    ) throws {
        func buffer(_ values: [Int]) throws -> any MTLBuffer {
            try values.map(Int32.init).withUnsafeBytes {
                guard let buffer = device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared) else {
                    throw Failure.allocation
                }
                return buffer
            }
        }
        count = depths.count
        self.branchRows = branchRows
        self.pathStride = pathStride
        self.slotCount = slotCount
        pathLengths = depths.map { $0 + 1 }
        self.ancestry = try buffer(ancestry)
        self.reads = try buffer(reads)
        self.writes = try buffer(writes)
        self.order = try buffer(order)
        self.paths = try buffer(paths)
        lengths = try buffer(pathLengths)
        self.depths = try buffer(depths)
    }
}
#endif
