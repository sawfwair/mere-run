#if os(macOS)
import Foundation
import Metal
import MLX

/// Immutable after a round completes. Views share one backing allocation and
/// one MLX ownership bridge; retained taps use a separate arena from replay.
@available(macOS 26.0, *)
final class Q35NativeArena {
    enum Failure: Error { case allocation }
    struct Slice {
        fileprivate let owner: Q35NativeArena
        let offset: Int
        let byteCount: Int
        var buffer: any MTLBuffer { owner.buffer }
        func array(_ shape: [Int]) -> MLXArray { owner.array(offset: offset, byteCount: byteCount, shape: shape) }
    }
    private let buffer: any MTLBuffer
    private let dtype: DType
    private var cursor = 0
    private let exportLock = NSLock()
    private var mlxBacking: MLXArray?

    init(device: any MTLDevice, bytes: Int, dtype: DType) throws {
        self.dtype = dtype
        guard let buffer = device.makeBuffer(length: max(256, bytes),
            options: [.storageModeShared, .hazardTrackingModeUntracked]) else { throw Failure.allocation }
        self.buffer = buffer
    }

    func allocate(_ count: Int) -> Slice {
        let offset = (cursor + 255) / 256 * 256, bytes = count * dtype.size
        precondition(count > 0 && offset + bytes <= buffer.length)
        cursor = offset + bytes
        return Slice(owner: self, offset: offset, byteCount: bytes)
    }

    private func array(offset: Int, byteCount: Int, shape: [Int]) -> MLXArray {
        exportLock.lock()
        defer { exportLock.unlock() }
        let count = shape.reduce(1, *)
        precondition(count * dtype.size <= byteCount && offset.isMultiple(of: dtype.size))
        if mlxBacking == nil {
            let backing = buffer
            mlxBacking = MLXArray(rawPointer: backing.contents(), [backing.length / dtype.size], dtype: dtype) { _ = backing }
        }
        let first = offset / dtype.size
        return mlxBacking![first..<(first + count)].reshaped(shape)
    }
}
#endif
