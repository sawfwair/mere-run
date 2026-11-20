#if os(macOS)
import Metal
import MereRunTensor
import MLX

/// The four KV heads retain their committed prefix in a request-owned buffer.
@available(macOS 26.0, *)
struct Q35NativeKVBank {
    let keys: M5NativeArrayBuffer
    let values: M5NativeArrayBuffer
    let capacity: Int

    init(device: any MTLDevice, capacity: Int, exportable: Bool = false) throws {
        func allocate() throws -> M5NativeArrayBuffer {
            guard let buffer = device.makeBuffer(length: 4 * capacity * 256 * 2, options: exportable ? .storageModeShared : .storageModePrivate) else {
                throw Q35NativeVerification.Failure.allocation
            }
            return M5NativeArrayBuffer(buffer: buffer, offset: 0, shape: [1, 4, capacity, 256], dtype: .bfloat16)
        }
        self.capacity = capacity
        keys = try allocate()
        values = try allocate()
    }

    /// The bank stops receiving writes before these immutable MLX views escape.
    func suffix(_ range: Range<Int>) -> (MLXArray, MLXArray) {
        func view(_ tensor: M5NativeArrayBuffer) -> MLXArray {
            let buffer = tensor.buffer
            let array = MLXArray(rawPointer: buffer.contents(), tensor.shape, dtype: .bfloat16) { _ = buffer }
            return array[0..., 0..., range, 0...]
        }
        return (view(keys), view(values))
    }

    func importPrefix(_ prefix: (MLXArray, MLXArray), count: Int, device: any MTLDevice,
                      into encoder: any MTLBlitCommandEncoder) throws {
        for (array, destination) in [(prefix.0, keys), (prefix.1, values)] {
            guard array.shape == [1, 4, count, 256], array.dtype == .bfloat16 else {
                throw Q35NativeVerification.Failure.cacheFamily
            }
            var source = try M5NativeArrayBuffer(array, device: device)
            if source.strides[2] != 256 || source.strides[3] != 1 {
                source = try M5NativeArrayBuffer(contiguous(array), device: device)
            }
            for head in 0..<4 {
                encoder.copy(from: source.buffer, sourceOffset: source.offset + head * source.strides[1] * 2,
                             to: destination.buffer, destinationOffset: head * capacity * 256 * 2, size: count * 256 * 2)
            }
        }
    }
}

@available(macOS 26.0, *)
final class Q35NativeKVCommit {
    private let pipeline: any MTLComputePipelineState

    init(device: any MTLDevice) throws {
        let library = try M5NativeFFN.makeLibrary(device: device, source: Self.source)
        guard let function = library.makeFunction(name: "q35_append_accepted_kv") else {
            throw Q35NativeVerification.Failure.geometry
        }
        pipeline = try device.makeComputePipelineState(function: function)
    }

    func encode(
        into encoder: any MTLComputeCommandEncoder, keys: Q35NativeArena.Slice, values: Q35NativeArena.Slice,
        bank: Q35NativeKVBank, selection: any MTLBuffer, offset: Int, rows: Int, branchRows: Int
    ) {
        precondition(offset + rows <= bank.capacity)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(keys.buffer, offset: keys.offset, index: 0)
        encoder.setBuffer(values.buffer, offset: values.offset, index: 1)
        encoder.setBuffer(bank.keys.buffer, offset: 0, index: 2)
        encoder.setBuffer(bank.values.buffer, offset: 0, index: 3)
        encoder.setBuffer(selection, offset: 0, index: 4)
        [UInt32(offset), UInt32(bank.capacity), UInt32(branchRows), 0].withUnsafeBytes {
            encoder.setBytes($0.baseAddress!, length: $0.count, index: 5)
        }
        encoder.dispatchThreads(MTLSize(width: 64, height: rows, depth: 4),
            threadsPerThreadgroup: MTLSize(width: 64, height: 2, depth: 1))
    }

    private static let source = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void q35_append_accepted_kv(
            const device ushort4* k [[buffer(0)]], const device ushort4* v [[buffer(1)]],
            device ushort4* keys [[buffer(2)]], device ushort4* values [[buffer(3)]],
            const device int* selection [[buffer(4)]], constant uint4& layout [[buffer(5)]],
            uint3 id [[thread_position_in_grid]]) {
            if (int(id.y) >= selection[0]) return;
            uint from = (id.z * layout.z + uint(selection[2 + id.y])) * 64 + id.x;
            uint to = (id.z * layout.y + layout.x + id.y) * 64 + id.x;
            keys[to] = k[from];
            values[to] = v[from];
        }
        """
}
#endif
