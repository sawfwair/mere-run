#if os(macOS)
import Cmlx
import Darwin
import Metal
import MLX

/// A byte view for an evaluated MLX array, including sliced and strided arrays.
/// The Metal buffer retains its MLX backing owner. Callers must order GPU work
/// and keep the source immutable while the view is in use.
package final class M5NativeArrayBuffer {
    enum Failure: Error { case bufferView }
    private final class Owner: @unchecked Sendable {
        let array: MLXArray
        init(_ array: MLXArray) { self.array = array }
    }

    package let buffer: any MTLBuffer
    package let offset: Int
    package let shape: [Int]
    package let strides: [Int]
    package let dtype: DType

    /// An existing native allocation needs no new Metal alias or MLX evaluation.
    package init(buffer: any MTLBuffer, offset: Int, shape: [Int], dtype: DType) {
        precondition(!shape.isEmpty && shape.allSatisfy { $0 > 0 })
        var stride = 1
        var reversed: [Int] = []
        for dimension in shape.reversed() { reversed.append(stride); stride *= dimension }
        precondition(offset >= 0 && offset + stride * dtype.size <= buffer.length)
        self.buffer = buffer; self.offset = offset; self.shape = shape; self.dtype = dtype
        strides = Array(reversed.reversed())
    }

    package init(_ array: MLXArray, device: any MTLDevice) throws {
        array.eval()
        shape = array.shape
        dtype = array.dtype
        let sourceStrides = mlx_array_strides(array.ctx)!
        strides = (0..<array.ndim).map { Int(sourceStrides[$0]) }
        precondition(array.size > 0 && strides.allSatisfy { $0 >= 0 })
        let pointer = UnsafeMutableRawPointer(mutating: mlx_array_data_uint8(array.ctx))!
        let address = Int(bitPattern: pointer), page = Int(getpagesize())
        let aligned = address / page * page
        offset = address - aligned
        let extent = zip(shape, strides).reduce(1) { $0 + ($1.0 - 1) * $1.1 } * array.itemSize
        let length = (offset + extent + page - 1) / page * page
        let owner = Owner(array)
        guard let buffer = device.makeBuffer(bytesNoCopy: UnsafeMutableRawPointer(bitPattern: aligned)!,
            length: length, options: [.storageModeShared, .hazardTrackingModeUntracked], deallocator: { [owner] _, _ in _ = owner }) else {
            throw Failure.bufferView
        }
        self.buffer = buffer
    }
}
#endif
